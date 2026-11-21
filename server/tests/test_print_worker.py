import asyncio
from concurrent.futures import ProcessPoolExecutor
import ctypes
import json
import multiprocessing
import os
import sys
import time

from fastapi import HTTPException
import pytest
import trimesh

from server import print_api as api
from server import print_jobs, print_worker


def hold_gil():
    ctypes.PyDLL(None).sleep(5)
    return os.getpid()


@pytest.fixture
def pool(monkeypatch):
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as executor:
        monkeypatch.setattr(print_worker, "_pool", executor)
        yield executor


@pytest.fixture
def job(tmp_path):
    src = tmp_path / "mesh.obj"
    trimesh.creation.box(extents=[.1, .2, .3]).export(src)
    return dict(src=str(src), workdir=str(tmp_path), is_mesh=True, frame=None,
                gravity=None, mode="object", height=30, bed=None, base_plate=False,
                base_thickness=3, units_to_mm=1000, up_axis="y")


async def collect_events(accepted, with_stl=True):
    events, times = [], []
    async for line in api._events("test-job", accepted, with_stl):
        times.append(time.monotonic())
        events.append(json.loads(line))
    assert max(b - a for a, b in zip(times, times[1:])) <= 3
    assert len([e for e in events if e["type"] == "progress"]) >= 2
    return events


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX sleep via PyDLL holds the child GIL")
@pytest.mark.parametrize("accepted", [True, False])
def test_spawned_gil_heavy_job_keeps_post_and_resume_alive(pool, job, monkeypatch, accepted):
    monkeypatch.setattr(print_jobs, "store", print_jobs.MemoryStore())
    busy = pool.submit(hold_gil)
    print_jobs.start("test-job", lambda progress: print_worker.execute_job(job, progress), "mesh.stl")
    events = asyncio.run(collect_events(accepted, with_stl=accepted))
    assert busy.result() != os.getpid()
    assert events[-1]["type"] == "done"
    assert events[-1]["report"]["watertight"]
    assert ("stl_base64" in events[-1]) == accepted


@pytest.mark.parametrize("accepted", [True, False])
def test_slow_missing_state_read_still_sends_heartbeats(monkeypatch, accepted):
    def missing(_):
        time.sleep(5)
        return None

    monkeypatch.setattr(print_jobs, "get_state", missing)
    events = asyncio.run(collect_events(accepted))
    assert events[-1]["type"] == "error" and events[-1]["status"] == 404
    progress = [event for event in events if event["type"] == "progress"]
    assert progress[-1]["elapsed_s"] > progress[0]["elapsed_s"]


def test_slow_stl_read_still_sends_heartbeats(monkeypatch):
    now = time.time()
    monkeypatch.setattr(print_jobs, "get_state", lambda _: dict(
        type="done", history=[], created=now, updated=now, report={}, filename="a.stl"))

    def stl(_):
        time.sleep(5)
        return b"stl"

    monkeypatch.setattr(print_jobs, "get_stl", stl)
    events = asyncio.run(collect_events(True))
    assert events[-1]["type"] == "done" and events[-1]["stl_base64"] == "c3Rs"


def test_worker_preserves_http_error(pool, job):
    job["src"] = os.path.join(job["workdir"], "missing.obj")
    with pytest.raises(HTTPException) as exc:
        print_worker.execute_job(job, lambda _: None)
    assert exc.value.status_code == 422


def test_completed_job_heartbeats_repeat_last_replayed_stage(monkeypatch):
    now = time.time()
    monkeypatch.setattr(api, "HEARTBEAT_S", 0.03)
    monkeypatch.setattr(print_jobs, "get_state", lambda _: dict(
        type="done", history=list(api.STAGES), created=now - 30, updated=now,
        report={}, filename="a.stl"))

    def stl(_):
        time.sleep(0.2)
        return b"stl"

    monkeypatch.setattr(print_jobs, "get_stl", stl)
    events = asyncio.run(collect_events(True))
    progress = [event for event in events if event["type"] == "progress"]
    values = [event["progress"] for event in progress]
    assert values == sorted(values)
    assert all(event["stage"] == "check" for event in progress[len(api.STAGES):])
    assert all(event["elapsed_s"] >= 30 for event in progress)
    assert events[-1]["type"] == "done"


def test_native_worker_exit_is_a_recoverable_error(pool, job):
    crashed = pool.submit(os._exit, 17)
    with pytest.raises(Exception):
        crashed.result()
    with pytest.raises(HTTPException) as exc:
        print_worker.execute_job(job, lambda _: None)
    assert exc.value.status_code == 500
    assert print_worker._pool is None
