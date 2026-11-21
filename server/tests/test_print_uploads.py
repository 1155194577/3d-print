import json
from urllib.parse import parse_qs, unquote, urlparse
import uuid

import boto3
from fastapi.testclient import TestClient
from moto import mock_aws
import numpy as np
import pytest

from server import print_api as api, print_jobs
from server.tests.test_face import FakeFace, face_frame, photo_bytes


@pytest.fixture
def s3_client(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("PRINT_S3_BUCKET", "print-test-uploads")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("PRINT_API_TOKEN", raising=False)
    monkeypatch.setattr(print_jobs, "store", print_jobs.MemoryStore())
    monkeypatch.setattr(api, "POLL_S", .01)
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="print-test-uploads")
        with TestClient(api.create_print_app()) as client:
            yield s3, client


def register(client, files, job_id=None):
    payload = {"files": {name: {"size": len(body), "content_type": content_type, "filename": filename}
                         for name, (filename, body, content_type) in files.items()}}
    if job_id is not None:
        payload["job_id"] = job_id
    response = client.post("/print/uploads", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def put(s3, registration, name, body):
    upload = registration["uploads"][name]
    key = unquote(urlparse(upload["url"]).path).lstrip("/")
    s3.put_object(Bucket="print-test-uploads", Key=key, Body=body,
                  ContentType=upload["headers"]["Content-Type"])
    return key


def keys(s3, job_id):
    return [entry["Key"] for entry in s3.list_objects_v2(
        Bucket="print-test-uploads", Prefix=f"uploads/{job_id}/").get("Contents", [])]


def test_presign_shape_and_repeat_same_job(s3_client):
    s3, client = s3_client
    files = {"photo": ("face.jpg", b"first", "image/jpeg")}
    first = register(client, files, str(uuid.uuid4()).upper())
    assert first["expires_in"] == 900
    assert first["job_id"] == first["job_id"].lower()
    upload = first["uploads"]["photo"]
    assert upload["method"] == "PUT"
    assert upload["headers"] == {"Content-Type": "image/jpeg"}
    query = parse_qs(urlparse(upload["url"]).query)
    assert query["X-Amz-Expires"] == ["900"]
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert "content-type" in query["X-Amz-SignedHeaders"][0].split(";")
    assert "content-length" in query["X-Amz-SignedHeaders"][0].split(";")
    key = put(s3, first, "photo", b"first")
    second = register(client, files, first["job_id"])
    assert second["job_id"] == first["job_id"]
    assert put(s3, second, "photo", b"other") == key
    assert keys(s3, first["job_id"]) == [key]
    assert s3.get_object(Bucket="print-test-uploads", Key=key)["Body"].read() == b"other"


def test_presign_uses_opt_in_regional_endpoint(s3_client, monkeypatch):
    _, client = s3_client
    monkeypatch.setenv("AWS_REGION", "ap-east-1")
    response = client.post("/print/uploads", json={"files": {
        "photo": {"size": 1, "content_type": "image/jpeg"}}})
    assert response.status_code == 200
    url = urlparse(response.json()["uploads"]["photo"]["url"])
    assert url.hostname == "print-test-uploads.s3.ap-east-1.amazonaws.com"
    assert "/ap-east-1/s3/" in parse_qs(url.query)["X-Amz-Credential"][0]
    assert url.path.endswith("/photo/photo.jpg")


@pytest.mark.parametrize("payload", [
    {}, {"files": {}}, {"files": {"other": {"size": 1, "content_type": "image/jpeg"}}},
    {"job_id": "bad", "files": {"photo": {"size": 1, "content_type": "image/jpeg"}}},
    *[{"files": {"photo": {"size": size, "content_type": "image/jpeg"}}}
      for size in (0, -1, True, "2", api.MAX_UPLOAD_BYTES + 1)],
    {"files": {"photo": {"size": 1, "content_type": "bad\r\nheader"}}},
])
def test_bad_registration(s3_client, payload):
    _, client = s3_client
    assert client.post("/print/uploads", json=payload).status_code == 400


def test_missing_preserves_uploaded_parts_for_retry(s3_client):
    s3, client = s3_client
    files = {"photo": ("a.jpg", b"jpeg", "image/jpeg"), "depth": ("depth.f32", b"depth", "application/octet-stream")}
    registration = register(client, files)
    key = put(s3, registration, "photo", b"jpeg")
    response = client.post("/print", data={"job_id": registration["job_id"], "uploaded": "photo,depth"},
                           headers={"Accept": "application/x-ndjson"})
    assert response.status_code == 409
    assert response.json() == {"detail": "upload missing: depth", "missing": ["depth"]}
    assert keys(s3, registration["job_id"]) == [key]
    assert print_jobs.get_state(registration["job_id"]) is None


@pytest.mark.parametrize("accept", ["application/x-ndjson", "model/stl"])
def test_two_step_face_matches_multipart(s3_client, monkeypatch, accept):
    s3, client = s3_client
    mesh, depth, mask, intr = face_frame()
    monkeypatch.setattr(api, "reconstruct_photo", FakeFace(mesh, mask))
    files = {"photo": ("face.png", photo_bytes(), "image/png"),
             "depth": ("depth.f32", depth.tobytes(), "application/octet-stream"),
             "face_vertices": ("vertices.f32", np.zeros((3, 3), dtype="<f4").tobytes(), "application/octet-stream")}
    data = dict(mode="face", depth_width=depth.shape[1], depth_height=depth.shape[0],
                intrinsics=json.dumps(intr), image_orientation="6", height_mm="100.0", face_vertex_count="3")
    legacy = client.post("/print", data=data, files=files)
    assert legacy.status_code == 200
    registration = register(client, files)
    for name, (_, body, _) in files.items():
        put(s3, registration, name, body)
    response = client.post("/print", data={**data, "job_id": registration["job_id"],
                           "uploaded": ",".join(files)}, headers={"Accept": accept})
    assert response.status_code == 200, response.text
    report = ([json.loads(line) for line in response.text.splitlines()][-1]["report"]
              if accept == "application/x-ndjson" else json.loads(response.headers["X-Print-Report"]))
    assert report == json.loads(legacy.headers["X-Print-Report"])
    assert keys(s3, registration["job_id"]) == []


@pytest.mark.parametrize("data", [{"mode": "face", "face_transform": "bad"},
                                   {"mode": "not-a-mode"}, {"uploaded": "unknown"}])
def test_validation_failure_removes_objects(s3_client, data):
    s3, client = s3_client
    registration = register(client, {"photo": ("a.jpg", b"jpeg", "image/jpeg")})
    put(s3, registration, "photo", b"jpeg")
    response = client.post("/print", data={"job_id": registration["job_id"], "uploaded": "photo", **data})
    assert response.status_code in (400, 422)
    assert keys(s3, registration["job_id"]) == []


def test_conflicting_part_rejected(s3_client):
    s3, client = s3_client
    registration = register(client, {"photo": ("a.jpg", b"jpeg", "image/jpeg")})
    put(s3, registration, "photo", b"jpeg")
    response = client.post("/print", data={"job_id": registration["job_id"], "uploaded": "photo"},
                           files={"photo": ("a.jpg", b"jpeg")})
    assert response.status_code == 400
    assert "not both" in response.json()["detail"]
    assert keys(s3, registration["job_id"]) == []


def test_actual_size_is_checked(s3_client):
    s3, client = s3_client
    registration = register(client, {"photo": ("a.jpg", b"jpeg", "image/jpeg")})
    put(s3, registration, "photo", b"more bytes than declared")
    response = client.post("/print", data={"job_id": registration["job_id"], "uploaded": "photo"})
    assert response.status_code == 400
    assert keys(s3, registration["job_id"]) == []


def test_registration_requires_auth(s3_client, monkeypatch):
    _, client = s3_client
    monkeypatch.setenv("PRINT_API_TOKEN", "test-token")
    assert client.post("/print/uploads", json={}).status_code == 401
