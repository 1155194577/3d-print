"""Streamed /print progress: Accept: application/x-ndjson."""

import base64
import io
import json
import time
import uuid

import pytest
import trimesh
from fastapi.testclient import TestClient

from server import print_api as api
from server import print_jobs
from server.app import Settings, create_app

NDJSON = {"Accept": "application/x-ndjson"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv("PRINT_API_TOKEN", raising=False)
    monkeypatch.setattr(api, "HEARTBEAT_S", 0.05)
    monkeypatch.setattr(api, "POLL_S", 0.01)
    monkeypatch.setattr(print_jobs, "store", print_jobs.MemoryStore())
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
    accepted, *progress, done = out
    assert accepted["type"] == "accepted" and uuid.UUID(accepted["job_id"])
    stages = [p["stage"] for p in progress]
    assert all(p["type"] == "progress" and 0 <= p["progress"] < 1 and p["message"] for p in progress)
    assert [s for i, s in enumerate(stages) if i == 0 or s != stages[i - 1]] == [
        "received", "background", "reconstruct", "mesh", "repair", "export", "check", "slice"]
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
    assert out[1]["stage"] == "received" and out[-1]["type"] == "done"


def test_validation_errors_stay_plain_http(client, monkeypatch):
    resp = client.post("/print", files={"photo": ("a.heic", b"x")}, headers=NDJSON)
    assert resp.status_code == 415 and "detail" in resp.json()
    monkeypatch.setenv("PRINT_API_TOKEN", "s3cret")
    resp = client.post("/print", files={"photo": ("a.jpg", b"x")}, headers=NDJSON)
    assert resp.status_code == 401


@pytest.mark.parametrize("status,method,path,kwargs,detail", [
    (400, "POST", "/print", {}, "send exactly one"),
    (401, "POST", "/print", {}, "missing or wrong bearer token"),
    (404, "GET", "/print/unknown-job", {}, "no such print job"),
    (405, "PUT", "/print", {}, "Method Not Allowed"),
    (415, "POST", "/print", {"files": {"photo": ("a.heic", b"private-image")}}, "unsupported file type"),
    (422, "POST", "/print", {"data": {"height_mm": "private-invalid-number"}}, "float_parsing"),
])
def test_print_4xx_logs_detail_not_credentials_or_body(client, monkeypatch, caplog,
                                                      status, method, path, kwargs, detail):
    if status == 401:
        monkeypatch.setenv("PRINT_API_TOKEN", "private-server-token")
    response = client.request(method, path, headers={"Authorization": "Bearer private-client-token"}, **kwargs)
    assert response.status_code == status
    records = [r.message for r in caplog.records if "print request rejected" in r.message]
    assert len(records) == 1
    assert f"status={status}" in records[0] and detail in records[0]
    assert "private-" not in records[0]


def test_without_accept_it_is_still_stl(client, monkeypatch):
    monkeypatch.setattr(api, "reconstruct_photo", slow_triposr)
    resp = client.post("/print", files={"photo": ("can.jpg", b"jpeg")})
    assert resp.status_code == 200 and resp.headers["content-type"] == "model/stl"


def test_accept_q0_is_not_ndjson(client, monkeypatch):
    monkeypatch.setattr(api, "reconstruct_photo", slow_triposr)
    resp = client.post("/print", files={"photo": ("can.jpg", b"jpeg")},
                       headers={"Accept": "Application/X-NDJSON;q=0, model/stl"})
    assert resp.headers["content-type"] == "model/stl"
    assert api._wants_ndjson("model/stl, APPLICATION/x-ndjson")


def test_resume_and_result(client, monkeypatch):
    monkeypatch.setattr(api, "reconstruct_photo", slow_triposr)
    jid = str(uuid.uuid4())
    out = lines(client.post("/print", files={"photo": ("can.jpg", b"jpeg")},
                            data={"height_mm": "100", "job_id": jid}, headers=NDJSON))
    assert out[0] == {"type": "accepted", "job_id": jid} and out[-1]["type"] == "done"
    resumed = lines(client.get(f"/print/{jid}", headers=NDJSON))
    assert resumed == [{k: v for k, v in out[-1].items() if k != "stl_base64"}]
    assert client.get(f"/print/{jid}").json()["type"] == "done"
    res = client.get(f"/print/{jid}/result")
    assert res.status_code == 200 and res.headers["content-type"] == "model/stl"
    assert res.content == base64.b64decode(out[-1]["stl_base64"])
    assert json.loads(res.headers["X-Print-Report"])["watertight"]


def test_job_runs_without_the_client_and_retry_joins(client, monkeypatch):
    gate, calls = __import__("threading").Event(), []

    def gated(image_path, out_path, progress=None, **_):
        calls.append(1)
        progress("reconstruct")
        gate.wait(5)
        trimesh.creation.box(extents=[0.1, 0.1, 0.1]).export(out_path)

    monkeypatch.setattr(api, "reconstruct_photo", gated)
    jid = str(uuid.uuid4())
    form = dict(files={"photo": ("a.jpg", b"jpeg")}, data={"job_id": jid}, headers=NDJSON)
    with client.stream("POST", "/print", **form) as resp:  # the app goes away mid-print
        it = resp.iter_lines()
        assert json.loads(next(it))["type"] == "accepted"
    deadline = time.time() + 5
    while client.get(f"/print/{jid}").json().get("stage") != "reconstruct" and time.time() < deadline:
        time.sleep(0.01)
    resumed = client.stream("GET", f"/print/{jid}", headers=NDJSON)
    with resumed as r:
        it = r.iter_lines()
        first = json.loads(next(it))
        assert first["type"] == "progress" and first["stage"] == "reconstruct"
        gate.set()
        last = [json.loads(x) for x in it][-1]
    assert last["type"] == "done" and "stl_base64" not in last
    retry = lines(client.post("/print", **form))  # same job_id: joins, no second print
    assert retry[0]["job_id"] == jid and retry[-1]["type"] == "done" and len(calls) == 1


def test_failed_job_result_and_unknown_ids(client, monkeypatch):
    def broken(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(api, "reconstruct_photo", broken)
    jid = lines(client.post("/print", files={"photo": ("a.jpg", b"x")}, headers=NDJSON))[0]["job_id"]
    res = client.get(f"/print/{jid}/result")
    assert res.status_code == 422 and "boom" in res.json()["detail"]
    for path in (f"/print/{uuid.uuid4()}", f"/print/{uuid.uuid4()}/result", "/print/nope"):
        assert client.get(path).status_code == 404
    assert client.post("/print", files={"photo": ("a.jpg", b"x")}, data={"job_id": "nope"},
                       headers=NDJSON).status_code == 400


def test_job_routes_need_the_token(client, monkeypatch):
    monkeypatch.setenv("PRINT_API_TOKEN", "s3cret")
    assert client.get(f"/print/{uuid.uuid4()}/result").status_code == 401


def test_a_job_whose_container_died_is_an_error(client):
    jid, old = str(uuid.uuid4()), time.time() - 200
    print_jobs.store.put(jid, dict(type="progress", stage="mesh", since=old, created=old,
                                   updated=old, filename="a.stl"))
    out = lines(client.get(f"/print/{jid}", headers=NDJSON))
    assert out[-1]["type"] == "error" and "restarted" in out[-1]["detail"]


def test_big_stl_is_chunked_and_expired_ids_restart(client, monkeypatch):
    monkeypatch.setattr(print_jobs, "CHUNK", 1000)
    data = trimesh.creation.icosphere(subdivisions=3).export(file_type="obj")
    jid = str(uuid.uuid4())
    old = time.time() - print_jobs.TTL_S - 10
    print_jobs.store.put(jid, dict(type="done", created=old, updated=old, filename="x.stl",
                                   report={}, stl_chunks=0))
    out = lines(client.post("/print", files={"mesh": ("scan.obj", data)}, data={"job_id": jid},
                            headers=NDJSON))
    assert out[-1]["type"] == "done" and print_jobs.store.get(jid)["stl_chunks"] > 1
    assert client.get(f"/print/{jid}/result").content == base64.b64decode(out[-1]["stl_base64"])
