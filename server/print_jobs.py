"""Print jobs that outlive the request that started them.

A job's state lives in a key-value store that every container can read (a modal.Dict on Modal,
memory locally), so the app can drop the POST, come back later and pick the job up from any
container. Keys: `<id>` holds the state dict, `<id>:stl:<n>` the finished STL in chunks and
`<id>:gcode:<n>` its K1C G-code, when it was sliced.
"""

import logging
import os
import threading
import time

TTL_S = 24 * 3600
TOUCH_S = 5.0  # a running job rewrites its state this often...
STALE_S = 90.0  # ...so one silent for this long died with its container
QUEUED_STALE_S = 10 * 60.0
CHUNK = 32 * 1024 * 1024  # modal.Dict values are capped at 100 MiB, so big STLs go in pieces


class MemoryStore:
    def __init__(self):
        self._d, self._lock = {}, threading.Lock()

    def get(self, key):
        with self._lock:
            return self._d.get(key)

    def put(self, key, value, skip_if_exists=False):
        with self._lock:
            if skip_if_exists and key in self._d:
                return False
            self._d[key] = value
            return True

    def prune(self, now):
        with self._lock:
            old = {k: v for k, v in self._d.items()
                   if isinstance(v, dict) and now - v["created"] > TTL_S}
            for k, v in old.items():
                self._d.pop(k, None)
                for kind in ("stl", "gcode"):
                    for i in range(v.get(f"{kind}_chunks", 0)):
                        self._d.pop(f"{k}:{kind}:{i}", None)


class ModalDictStore:
    """modal.Dict drops idle entries itself after 7 days; `get_state` enforces the 24 h."""

    def __init__(self, d):
        self._d = d

    def get(self, key):
        return self._d.get(key)

    def put(self, key, value, skip_if_exists=False):
        return self._d.put(key, value, skip_if_exists=skip_if_exists)

    def prune(self, now):
        pass


store = MemoryStore()


def get_state(job_id):
    """The job's state, or None when unknown or expired. A job whose container died is an error."""
    state = store.get(job_id)
    now = time.time()
    if state is None or now - state["created"] > TTL_S:
        return None
    queued = state.get("stage") == "queued"
    if state["type"] == "progress" and now - state["updated"] > (QUEUED_STALE_S if queued else STALE_S):
        state = dict(state, type="error", status=500,
                     detail=("no print server GPU became available within 10 minutes; send it again" if queued
                             else "the print server restarted during this print; send it again"))
    return state


def claim(job_id, filename, stage="received"):
    """Reserve an id before dispatch; retries join the existing job."""
    now = time.time()
    store.prune(now)
    state = dict(type="progress", stage=stage, history=[stage], since=now, created=now, updated=now,
                 filename=filename)
    if not store.put(job_id, state, skip_if_exists=True):
        old = store.get(job_id)
        if old is not None and now - old["created"] <= TTL_S:
            return False
        store.put(job_id, state)  # that id's job expired: start afresh
    return True


def fail(job_id, status, detail):
    state = store.get(job_id)
    if state is not None:
        store.put(job_id, dict(state, type="error", status=status, detail=detail, updated=time.time()))


def run_existing(job_id, run):
    """Execute an already-reserved job in this process and publish its result."""
    state = get_state(job_id)
    if state is None or state["type"] != "progress":
        return

    lock = threading.Lock()

    def save(**fields):
        with lock:
            state.update(fields, updated=time.time())
            store.put(job_id, dict(state))

    def work():
        from fastapi import HTTPException

        try:
            if state["stage"] == "queued":
                save(stage="received", since=time.time(), history=state["history"] + ["received"])
            body, report, *rest = run(lambda stage: save(stage=stage, since=time.time(),
                                                         history=state["history"] + [stage]))
            gcode = rest[0] if rest else None
            result = {}
            if os.environ.get("PRINT_S3_BUCKET"):
                from server import print_uploads

                for kind, label, data, media in (("stl", "STL", body, "model/stl"),
                                                 ("gcode", "G-code", gcode, "text/x-gcode")):
                    if data is None:
                        continue
                    key = f"results/{job_id}/model.{kind}"
                    try:
                        print_uploads.client().put_object(Bucket=print_uploads.bucket(), Key=key,
                                                          Body=data, ContentType=media)
                        result[f"{kind}_file_id"] = key
                    except Exception as exc:
                        logging.getLogger(__name__).warning("could not upload %s result job_id=%s error=%s",
                                                           label, job_id, type(exc).__name__)
            result["stl_chunks"] = _put_chunks(f"{job_id}:stl", body)
            if gcode is not None and "gcode_file_id" not in result:  # served from storage when uploaded
                result["gcode_chunks"] = _put_chunks(f"{job_id}:gcode", gcode)
            save(type="done", report=report, **result)
        except HTTPException as exc:
            save(type="error", status=exc.status_code, detail=exc.detail)
        except Exception as exc:  # a bug, not the client's fault
            save(type="error", status=500, detail=f"{type(exc).__name__}: {exc}")

    finished = threading.Event()

    def keepalive():
        while not finished.wait(TOUCH_S):
            save()

    threading.Thread(target=keepalive, daemon=True).start()
    try:
        work()
    finally:
        finished.set()


def start(job_id, run, filename):
    """Start a local job in a thread; retries join the existing job."""
    if not claim(job_id, filename):
        return False
    threading.Thread(target=run_existing, args=(job_id, run), daemon=True).start()
    return True


def _put_chunks(prefix, data):
    chunks = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)] or [b""]
    for i, chunk in enumerate(chunks):
        store.put(f"{prefix}:{i}", chunk)
    return len(chunks)


def _get_chunks(job_id, kind):
    n = store.get(job_id)[f"{kind}_chunks"]
    return b"".join(store.get(f"{job_id}:{kind}:{i}") for i in range(n))


def get_stl(job_id):
    return _get_chunks(job_id, "stl")


def get_gcode(job_id):
    return _get_chunks(job_id, "gcode")
