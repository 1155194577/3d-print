"""Streamed /print progress: Accept: application/x-ndjson."""

import base64
import io
import json
import time

import pytest
import trimesh
from fastapi.testclient import TestClient

from server import print_api as api
from server.app import Settings, create_app

NDJSON = {"Accept": "application/x-ndjson"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv("PRINT_API_TOKEN", raising=False)
    monkeypatch.setattr(api, "HEARTBEAT_S", 0.05)
    with TestClient(create_app(Settings(jobs_dir=tmp_path / "jobs"))) as c:
        yield c


def lines(resp):
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(x) for x in resp.text.splitlines()]


def slow_triposr(image_path, out_path, progress=None, **_):
    for stage in ("background", "reconstruct", "mesh"):
        progress(stage)
        time.sleep(0.12)
    trimesh.creation.cylinder(radius=0.3, height=0.8).export(out_path)


def test_photo_streams_stages_heartbeats_and_stl(client, monkeypatch):
    monkeypatch.setattr(api, "reconstruct_photo", slow_triposr)
    out = lines(client.post("/print", files={"photo": ("can.jpg", b"jpeg")},
                            data={"height_mm": "100"}, headers=NDJSON))
    *progress, done = out
    stages = [p["stage"] for p in progress]
    assert all(p["type"] == "progress" and 0 <= p["progress"] < 1 and p["message"] for p in progress)
    assert [s for i, s in enumerate(stages) if i == 0 or s != stages[i - 1]] == [
        "received", "background", "reconstruct", "mesh", "repair", "export"]
    assert stages.count("reconstruct") > 1  # heartbeats repeat the stage
    values = [p["progress"] for p in progress]
    assert values == sorted(values)
    assert done["type"] == "done" and done["progress"] == 1 and done["filename"] == "can.stl"
    stl = trimesh.load(io.BytesIO(base64.b64decode(done["stl_base64"])), file_type="stl")
    assert stl.extents[2] == pytest.approx(102.5, abs=0.5)
    assert done["report"]["watertight"]


def test_failure_is_an_error_line(client, monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(api, "reconstruct_photo", broken)
    out = lines(client.post("/print", files={"photo": ("can.jpg", b"jpeg")}, headers=NDJSON))
    assert out[-1]["type"] == "error" and out[-1]["status"] == 422
    assert "CUDA out of memory" in out[-1]["detail"]
    assert sum(o["type"] in ("done", "error") for o in out) == 1


def test_mesh_streams(client):
    data = trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(file_type="obj")
    out = lines(client.post("/print", files={"mesh": ("scan.obj", data)}, headers=NDJSON))
    assert out[0]["stage"] == "received" and out[-1]["type"] == "done"


def test_validation_errors_stay_plain_http(client, monkeypatch):
    resp = client.post("/print", files={"photo": ("a.heic", b"x")}, headers=NDJSON)
    assert resp.status_code == 415 and "detail" in resp.json()
    monkeypatch.setenv("PRINT_API_TOKEN", "s3cret")
    resp = client.post("/print", files={"photo": ("a.jpg", b"x")}, headers=NDJSON)
    assert resp.status_code == 401


def test_without_accept_it_is_still_stl(client, monkeypatch):
    monkeypatch.setattr(api, "reconstruct_photo", slow_triposr)
    resp = client.post("/print", files={"photo": ("can.jpg", b"jpeg")})
    assert resp.status_code == 200 and resp.headers["content-type"] == "model/stl"
