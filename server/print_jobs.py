"""Print jobs that outlive the request that started them.

A job's state lives in a key-value store that every container can read (a modal.Dict on Modal,
memory locally), so the app can drop the POST, come back later and pick the job up from any
container. Keys: `<id>` holds the state dict, `<id>:stl:<n>` the finished STL in chunks.
"""

import threading
import time

TTL_S = 24 * 3600
TOUCH_S = 5.0  # a running job rewrites its state this often...
STALE_S = 90.0  # ...so one silent for this long died with its container
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
                for i in range(v.get("stl_chunks", 0)):
                    self._d.pop(f"{k}:stl:{i}", None)


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
    if state["type"] == "progress" and now - state["updated"] > STALE_S:
        state = dict(state, type="error", status=500,
                     detail="the print server restarted during this print; send it again")
    return state


def start(job_id, run, filename):
    """Start `run(progress) -> (stl bytes, report)` as job `job_id` unless that id exists.

    Returns True if this call started it, False if it joined a job already there.
    """
    now = time.time()
    store.prune(now)
    state = dict(type="progress", stage="received", history=["received"], since=now, created=now, updated=now,
                 filename=filename)
    if not store.put(job_id, state, skip_if_exists=True):
        old = store.get(job_id)
        if old is not None and now - old["created"] <= TTL_S:
            return False
        store.put(job_id, state)  # that id's job expired: start afresh

    lock = threading.Lock()

    def save(**fields):
        with lock:
            state.update(fields, updated=time.time())
            store.put(job_id, dict(state))

    def work():
        from fastapi import HTTPException

        try:
            body, report = run(lambda stage: save(stage=stage, since=time.time(),
                                                  history=state["history"] + [stage]))
            chunks = [body[i:i + CHUNK] for i in range(0, len(body), CHUNK)] or [b""]
            for i, chunk in enumerate(chunks):
                store.put(f"{job_id}:stl:{i}", chunk)
            save(type="done", report=report, stl_chunks=len(chunks))
        except HTTPException as exc:
            save(type="error", status=exc.status_code, detail=exc.detail)
        except Exception as exc:  # a bug, not the client's fault
            save(type="error", status=500, detail=f"{type(exc).__name__}: {exc}")

    def keepalive(worker):
        while worker.is_alive():
            worker.join(TOUCH_S)
            if worker.is_alive():
                save()

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    threading.Thread(target=keepalive, args=(worker,), daemon=True).start()
    return True


def get_stl(job_id):
    n = store.get(job_id)["stl_chunks"]
    return b"".join(store.get(f"{job_id}:stl:{i}") for i in range(n))
