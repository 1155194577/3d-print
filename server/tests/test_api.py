"""API tests for the photo → STL service.

The real pipeline takes ~90 s and needs a GPU, so the tests point the app at
``fake_triposr.py`` — same log lines, same STL contract, milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import Settings, create_app, metrics_from_log, stage_for_line

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FAKE = HERE / "fake_triposr.py"
SAMPLE_IMAGE = (REPO / "examples" / "sample_can.jpg").read_bytes()


@pytest.fixture()
def app_client(tmp_path):
    settings = Settings(
        triposr_bin=FAKE,
        jobs_dir=tmp_path / "jobs",
        preview_script=REPO / "tools" / "preview_stl.py",
        max_upload_bytes=5 * 1024 * 1024,
        timeout_s=60,
    )
    with TestClient(create_app(settings)) as client:
        yield client, settings


def collect_events(client: TestClient, job_id: str) -> list[dict]:
    events: list[dict] = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):].strip()
            if payload and payload != "{}":
                events.append(json.loads(payload))
    return events


# --------------------------------------------------------------------- units
def test_stage_mapping_prefers_finished_variants():
    assert stage_for_line("Running model ...") == ("generating geometry", 45)
    assert stage_for_line("Running model finished in 22s") == ("generating geometry", 70)
    assert stage_for_line("Initializing model ...") == ("loading model", 10)
    assert stage_for_line("Initializing model finished in 5s") == ("loading model", 15)
    assert stage_for_line("unrelated chatter") is None


def test_metrics_parsed_from_report():
    lines = [
        "  1. x.stl",
        "     12.0 × 10.0 × 8.0 mm   (4 faces, max dim → 12 mm)",
        "     watertight: ✓ yes   boundary edges: 0   non-manifold: 0   parts: 1",
        "     volume: 160 cm³  (~198 g PLA solid)",
    ]
    metrics = metrics_from_log(lines)
    assert metrics["dimensions_mm"] == [12.0, 10.0, 8.0]
    assert metrics["watertight"] is True
    assert metrics["volume_cm3"] == 160
    assert metrics["faces"] == 4


# ---------------------------------------------------------------- endpoints
def test_health_reports_pipeline(app_client):
    client, settings = app_client
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["pipeline_ready"] is True
    assert body["pipeline"] == str(settings.triposr_bin)


def test_index_page_served(app_client):
    client, _ = app_client
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Drop an image" in response.text


def test_upload_streams_progress_and_serves_model(app_client):
    client, _ = app_client
    created = client.post(
        "/api/jobs",
        files={"image": ("can.jpg", SAMPLE_IMAGE, "image/jpeg")},
    )
    assert created.status_code == 202
    job_id = created.json()["id"]

    events = collect_events(client, job_id)
    assert events, "no SSE events received"
    assert events[-1]["status"] == "done", events[-1]
    assert events[-1]["progress"] == 100

    progress = [event["progress"] for event in events]
    assert progress == sorted(progress), "progress must never go backwards"
    stages = [event["stage"] for event in events]
    assert "generating geometry" in stages

    model = client.get(f"/api/jobs/{job_id}/model")
    assert model.status_code == 200
    assert model.headers["content-type"] == "model/stl"
    assert model.content.startswith(b"solid")

    preview = client.get(f"/api/jobs/{job_id}/preview.png")
    assert preview.status_code == 200
    assert preview.content[:8] == b"\x89PNG\r\n\x1a\n"

    state = client.get(f"/api/jobs/{job_id}").json()
    assert state["status"] == "done"
    assert state["metrics"]["watertight"] is True
    assert state["metrics"]["dimensions_mm"] == [12.0, 10.0, 8.0]


def test_two_jobs_queue_and_both_complete(app_client):
    client, _ = app_client
    ids = [
        client.post("/api/jobs", files={"image": (f"a{i}.jpg", SAMPLE_IMAGE, "image/jpeg")}).json()["id"]
        for i in range(2)
    ]
    for job_id in ids:
        events = collect_events(client, job_id)
        assert events[-1]["status"] == "done"
        assert client.get(f"/api/jobs/{job_id}/model").status_code == 200


# ----------------------------------------------------------------- validation
def test_rejects_unsupported_extension(app_client):
    client, _ = app_client
    response = client.post("/api/jobs", files={"image": ("scan.heic", b"\x00" * 64, "image/heic")})
    assert response.status_code == 415
    assert "heic" in response.json()["detail"].lower()


def test_rejects_payload_that_is_not_an_image(app_client):
    client, _ = app_client
    response = client.post(
        "/api/jobs",
        files={"image": ("evil.jpg", b"#!/bin/sh\nrm -rf /\n", "image/jpeg")},
    )
    assert response.status_code == 415
    assert "does not look like" in response.json()["detail"]


def test_rejects_empty_upload(app_client):
    client, _ = app_client
    response = client.post("/api/jobs", files={"image": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


def test_rejects_too_large_upload(tmp_path):
    from fastapi.testclient import TestClient as TC

    settings = Settings(
        triposr_bin=FAKE,
        jobs_dir=tmp_path / "jobs",
        preview_script=REPO / "tools" / "preview_stl.py",
        max_upload_bytes=1024,
        timeout_s=10,
    )
    with TC(create_app(settings)) as client:
        response = client.post(
            "/api/jobs",
            files={"image": ("big.jpg", SAMPLE_IMAGE, "image/jpeg")},
        )
        assert response.status_code == 413


def test_unknown_job_returns_404(app_client):
    client, _ = app_client
    assert client.get("/api/jobs/doesnotexist").status_code == 404
    assert client.get("/api/jobs/doesnotexist/model").status_code == 404
    assert client.get("/api/jobs/doesnotexist/events").status_code == 404
