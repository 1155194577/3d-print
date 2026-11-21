import io
import json
from pathlib import Path
import pickle
import shutil
import tempfile
import threading
import time
import uuid

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pytest
import trimesh

from server import print_api as api
from server import print_jobs, print_spec
from depth_fusion import Frame


@pytest.fixture
def store(monkeypatch):
    value = print_jobs.MemoryStore()
    monkeypatch.setattr(print_jobs, "store", value)
    return value


def test_queue_has_a_separate_scheduling_deadline(store):
    now = time.time()
    state = dict(type="progress", stage="queued", history=["queued"], created=now - 120,
                 since=now - 120, updated=now - 120, filename="a.stl")
    store.put("queued", state)
    assert print_jobs.get_state("queued")["type"] == "progress"
    store.put("running", dict(state, stage="repair"))
    assert "restarted" in print_jobs.get_state("running")["detail"]
    store.put("expired", dict(state, created=now - 601, updated=now - 601))
    assert "GPU" in print_jobs.get_state("expired")["detail"]


@pytest.mark.parametrize("depth", [False, True])
@pytest.mark.parametrize("mode", ["object", "face"])
def test_spec_roundtrip_has_identical_report(tmp_path, monkeypatch, depth, mode):
    class FakePhoto:
        def __call__(self, source, output, **kwargs):
            assert Path(source).read_bytes() == b"photo-bytes"
            trimesh.creation.cylinder(radius=.05, height=.2).export(output)

        def foreground_mask(self, _):
            return np.full((4, 4), 255, dtype=np.uint8)

    monkeypatch.setattr(api, "reconstruct_photo", FakePhoto())
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"photo-bytes")
    frame = None
    if depth:
        frame = Frame(np.full((4, 4), np.nan, dtype="<f4"), np.eye(3),
                      confidence=np.ones((4, 4), dtype=np.uint8),
                      camera_to_world=np.eye(4), image_orientation=6,
                      photo_size=(1280, 720), taps=np.array([[1., 2.]]))
        frame.depth_to_photo = np.eye(4)
        frame.depth_to_photo[0, 3] = .01
        frame.photo_K = np.diag([500., 500., 1.])
        (tmp_path / "depth.f32").write_bytes(frame.depth.tobytes())
        (tmp_path / "confidence.bin").write_bytes(frame.confidence.tobytes())
    job = dict(src=str(source), workdir=str(tmp_path), is_mesh=False, frame=frame,
               gravity=None, height=42.5, bed=None, base_plate=True, base_thickness=3,
               units_to_mm=1000., up_axis="y", mode=mode,
               face_to_world=np.eye(4), image_orientation=6)
    spec = print_spec.pack(job)
    directory = tmp_path / "remote"
    directory.mkdir()
    remote = print_spec.unpack(spec, directory)
    assert spec["files"]["source"] == b"photo-bytes"
    if depth:
        assert (directory / "depth.f32").read_bytes() == frame.depth.tobytes()
        assert (directory / "confidence.bin").read_bytes() == frame.confidence.tobytes()
        for key, value in vars(frame).items():
            np.testing.assert_equal(getattr(remote["frame"], key), value)
    _, local_report = api._make_stl(job, lambda _: None)
    _, remote_report = api._make_stl(remote, lambda _: None)
    assert remote_report == local_report


@pytest.mark.parametrize("accept", ["application/x-ndjson", "model/stl"])
def test_pluggable_queued_runner_and_retry(store, monkeypatch, accept):
    monkeypatch.delenv("PRINT_API_TOKEN", raising=False)
    monkeypatch.setattr(api, "POLL_S", .01)
    monkeypatch.setattr(api, "HEARTBEAT_S", .03)
    dispatched = []

    def pipeline(job, progress):
        assert Path(job["src"]).read_bytes() == b"mesh-bytes"
        progress("repair")
        return b"stl", {"watertight": True}

    def submit(job_id, job, filename):
        spec = print_spec.pack(job)
        shutil.rmtree(job["workdir"])
        if not print_jobs.claim(job_id, filename, stage="queued"):
            return False
        dispatched.append(spec)

        def gpu():
            time.sleep(.15)
            with tempfile.TemporaryDirectory() as directory:
                restored = print_spec.unpack(spec, directory)
                print_jobs.run_existing(job_id, lambda progress: pipeline(restored, progress))

        threading.Thread(target=gpu, daemon=True).start()
        return True

    monkeypatch.setattr(api, "submit_job", submit)
    jid = str(uuid.uuid4()).upper()
    with TestClient(api.create_print_app()) as client:
        kwargs = dict(files={"mesh": ("scan.obj", b"mesh-bytes")}, data={"job_id": jid}, headers={"Accept": accept})
        response = client.post("/print", **kwargs)
        assert response.status_code == 200
        if accept == "application/x-ndjson":
            events = [json.loads(line) for line in response.text.splitlines()]
            assert events[0] == {"type": "accepted", "job_id": jid.lower()}
            assert any(event.get("stage") == "queued" for event in events)
            assert events[-1]["type"] == "done"
        else:
            assert response.content == b"stl"
            assert json.loads(response.headers["X-Print-Report"]) == {"watertight": True}
        assert client.post("/print", **kwargs).status_code == 200
        assert len(dispatched) == 1
        assert store.get(jid.lower())["history"] == ["queued", "received", "repair"]


@pytest.mark.parametrize("body", [b"", b"photo-bytes"])
def test_spec_preserves_source_bytes(tmp_path, body):
    source = tmp_path / "source.jpg"
    source.write_bytes(body)
    spec = print_spec.pack(dict(src=str(source), workdir=str(tmp_path), frame=None, mode="face"))
    assert Path(print_spec.unpack(spec, tmp_path)["src"]).read_bytes() == body


@pytest.mark.parametrize("with_frame", [False, True])
def test_spec_relocates_extra_view_photos(tmp_path, with_frame):
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    source, extra = local / "photo.jpg", local / "view_1_photo.jpg"
    source.write_bytes(b"front")
    extra.write_bytes(b"side")
    frame = None
    if with_frame:
        frame = Frame(np.ones((8, 8)), np.eye(3), camera_to_world=np.eye(4),
                      confidence=np.ones((8, 8), dtype=np.uint8),
                      taps=np.array([[1., 2.]]))
        frame.depth_to_photo = np.eye(4)
        frame.photo_K = np.eye(3)
    mask = np.ones((8, 8), dtype=bool)
    job = dict(src=str(source), workdir=str(local), frame=frame,
               face_to_world=np.eye(4), gravity=(np.array([0., 1., 0.]), np.int64(1)),
               views=[dict(src=str(extra), frame=frame, face_to_world=np.eye(4), mask=mask)])
    spec = print_spec.pack(job)

    class BuiltinsOnlyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            raise AssertionError(f"spec requires importing {module}.{name}")

    spec = BuiltinsOnlyUnpickler(io.BytesIO(pickle.dumps(spec))).load()
    shutil.rmtree(local)
    restored = print_spec.unpack(spec, remote)
    view = restored["views"][0]
    assert Path(view["src"]).read_bytes() == b"side"
    assert Path(view["src"]).parent == remote
    if with_frame:
        for key, value in vars(frame).items():
            np.testing.assert_equal(getattr(view["frame"], key), value)
    else:
        assert view["frame"] is None
    np.testing.assert_equal(view["face_to_world"], np.eye(4))
    np.testing.assert_equal(view["mask"], mask)
    np.testing.assert_equal(restored["face_to_world"], np.eye(4))
    np.testing.assert_equal(restored["gravity"], job["gravity"])


def test_queued_worker_error_and_expired_job(store):
    def broken(_):
        raise HTTPException(422, "bad shape")

    assert print_jobs.claim("job", "a.stl", stage="queued")
    print_jobs.run_existing("job", broken)
    assert print_jobs.get_state("job")["status"] == 422
    print_jobs.run_existing("missing", lambda _: pytest.fail("must not execute an unknown job"))
    print_jobs.run_existing("job", lambda _: pytest.fail("must not execute a terminal job"))
