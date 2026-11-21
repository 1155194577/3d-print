"""A persistent, spawn-based mesh worker, separate from the ASGI interpreter/GIL."""

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import multiprocessing
import os
import threading
import time

from fastapi import HTTPException

_pool = None
_lock = threading.Lock()


def _run_job(job, progress_path):
    from server.print_api import _make_stl

    with open(progress_path, "w", buffering=1) as events:
        def progress(stage):
            events.write(stage + "\n")

        try:
            body, report = _make_stl(job, progress)
            return body, report, None, None
        except HTTPException as exc:
            return None, None, exc.status_code, exc.detail


def execute_job(job, progress):
    global _pool

    progress_path = os.path.join(job["workdir"], "progress.txt")
    with open(progress_path, "w"):
        pass
    with _lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
        pool = _pool
    try:
        future = pool.submit(_run_job, job, progress_path)
        with open(progress_path) as events:
            while True:
                done = future.done()
                for stage in events:
                    progress(stage.rstrip("\n"))
                if done:
                    break
                time.sleep(0.05)
        body, report, status, detail = future.result()
        if status is not None:
            raise HTTPException(status, detail)
        return body, report
    except BrokenProcessPool as exc:
        with _lock:
            if _pool is pool:
                _pool = None
                pool.shutdown(wait=False)
        raise HTTPException(500, "the mesh worker stopped during this print; send it again") from exc
