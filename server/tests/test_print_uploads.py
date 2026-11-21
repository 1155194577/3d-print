import base64
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, unquote, urlparse
import uuid

import boto3
from fastapi.testclient import TestClient
from moto import mock_aws
from botocore.exceptions import ClientError
import numpy as np
import pytest
import requests

from server import print_api as api, print_jobs, print_uploads
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
    assert upload["file_id"] == key
    assert requests.get(upload["get_url"]).content == b"first"
    assert parse_qs(urlparse(upload["get_url"]).query)["X-Amz-Expires"] == ["3600"]
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
    assert keys(s3, registration["job_id"])
    original = print_jobs.get_state(registration["job_id"])
    retry = client.post("/print", data={**data, "job_id": registration["job_id"],
                        "uploaded": ",".join(files)}, headers={"Accept": "application/x-ndjson"})
    assert retry.status_code == 200, retry.text
    events = [json.loads(line) for line in retry.text.splitlines()]
    assert events[0] == {"type": "accepted", "job_id": registration["job_id"]}
    assert events[-1]["type"] == "done"
    assert events[-1]["report"] == report
    assert print_jobs.get_state(registration["job_id"]) == original
    assert keys(s3, registration["job_id"])


def test_conflicting_part_rejected(s3_client):
    s3, client = s3_client
    registration = register(client, {"photo": ("a.jpg", b"jpeg", "image/jpeg")})
    put(s3, registration, "photo", b"jpeg")
    response = client.post("/print", data={"job_id": registration["job_id"], "uploaded": "photo"},
                           files={"photo": ("a.jpg", b"jpeg")})
    assert response.status_code == 400
    assert "not both" in response.json()["detail"]
    assert keys(s3, registration["job_id"])


def test_actual_size_is_checked(s3_client):
    s3, client = s3_client
    registration = register(client, {"photo": ("a.jpg", b"jpeg", "image/jpeg")})
    put(s3, registration, "photo", b"more bytes than declared")
    response = client.post("/print", data={"job_id": registration["job_id"], "uploaded": "photo"})
    assert response.status_code == 400
    assert keys(s3, registration["job_id"])


def test_registration_requires_auth(s3_client, monkeypatch):
    _, client = s3_client
    monkeypatch.setenv("PRINT_API_TOKEN", "test-token")
    assert client.post("/print/uploads", json={}).status_code == 401


def test_preview_only_existing_manifest_objects(s3_client):
    s3, client = s3_client
    files = {"photo": ("a.jpg", b"jpeg", "image/jpeg"), "depth": ("a.f32", b"1234", "application/octet-stream")}
    registration = register(client, files)
    key = put(s3, registration, "photo", b"jpeg")
    response = client.get("/print/uploads/" + registration["job_id"])
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == registration["job_id"]
    assert set(data["files"]) == {"photo"}
    entry = data["files"]["photo"]
    assert set(entry) == {"file_id", "size", "get_url"}
    assert entry["file_id"] == key and entry["size"] == 4
    assert parse_qs(urlparse(entry["get_url"]).query)["X-Amz-Expires"] == ["3600"]
    assert requests.get(entry["get_url"]).content == b"jpeg"
    assert client.get("/print/uploads/" + str(uuid.uuid4())).status_code == 404
    assert client.get("/print/uploads/bad").status_code == 400


@pytest.mark.parametrize("accept", ["application/x-ndjson", "model/stl"])
def test_additive_result_stores_stl_and_reissues_links(s3_client, monkeypatch, accept):
    s3, client = s3_client
    body, report = b"solid model\nendsolid model\n", {"watertight": True}
    monkeypatch.setattr(api, "execute_job", lambda job, progress: (body, report))
    registration = register(client, {"photo": ("a.jpg", b"jpeg", "image/jpeg")})
    put(s3, registration, "photo", b"jpeg")
    data = dict(job_id=registration["job_id"], uploaded="photo")
    response = client.post("/print", data=data, headers={"Accept": accept})
    assert response.status_code == 200
    if accept == "application/x-ndjson":
        result = json.loads(response.text.splitlines()[-1])
        assert result["report"] == report
        assert base64.b64decode(result["stl_base64"]) == body
    else:
        assert response.content == body
        assert json.loads(response.headers["X-Print-Report"]) == report
        result = {"stl_url": response.headers["X-STL-URL"],
                  "stl_file_id": response.headers["X-STL-File-ID"],
                  "stl_expires_in": int(response.headers["X-STL-Expires-In"])}
    assert result["stl_expires_in"] == 3600
    assert result["stl_file_id"] == f"results/{registration['job_id']}/model.stl"
    stored = s3.get_object(Bucket="print-test-uploads", Key=result["stl_file_id"])
    assert stored["Body"].read() == body
    assert stored["ContentType"] == "model/stl"
    with patch("botocore.auth.get_current_datetime", return_value=datetime(2030, 1, 1, tzinfo=timezone.utc)):
        reissued = client.get("/print/" + registration["job_id"]).json()
        binary = client.get("/print/" + registration["job_id"] + "/result")
        replay = client.get("/print/" + registration["job_id"], headers={"Accept": "application/x-ndjson"})
    assert reissued["type"] == "done"
    assert reissued["stl_url"] != result["stl_url"]
    assert parse_qs(urlparse(reissued["stl_url"]).query)["X-Amz-Expires"] == ["3600"]
    assert binary.content == body and binary.headers["X-STL-URL"] == reissued["stl_url"]
    assert json.loads(replay.text.splitlines()[-1])["stl_url"] == reissued["stl_url"]
    retry = client.post("/print", data={"job_id": registration["job_id"], "uploaded": "photo"})
    assert retry.status_code == 200 and retry.content == body
    assert retry.headers["X-STL-File-ID"] == result["stl_file_id"]
    assert keys(s3, registration["job_id"])


def test_lifecycle_merges_existing_rules(s3_client):
    s3, _ = s3_client
    old = dict(ID="keep-me", Status="Enabled", Filter={"Prefix": "other/"}, Expiration={"Days": 30})
    s3.put_bucket_lifecycle_configuration(Bucket="print-test-uploads", LifecycleConfiguration={"Rules": [old]})
    assert print_uploads.configure_lifecycle()["configured"]
    rules = s3.get_bucket_lifecycle_configuration(Bucket="print-test-uploads")["Rules"]
    assert old in rules
    assert {rule["Filter"]["Prefix"]: rule["Expiration"]["Days"] for rule in rules} == {
        "other/": 30, "uploads/": 7, "results/": 7}


def test_lifecycle_access_denied_is_nonfatal(monkeypatch):
    s3 = Mock()
    s3.get_bucket_lifecycle_configuration.side_effect = ClientError({"Error": {"Code": "AccessDenied"}},
                                                                   "GetBucketLifecycleConfiguration")
    monkeypatch.setattr(print_uploads, "client", lambda: s3)
    monkeypatch.setenv("PRINT_S3_BUCKET", "print-test-uploads")
    assert print_uploads.configure_lifecycle()["error"] == "AccessDenied"
    s3.put_bucket_lifecycle_configuration.assert_not_called()


@pytest.mark.parametrize("accept", ["application/x-ndjson", "model/stl"])
def test_result_upload_error_keeps_legacy_payload(s3_client, monkeypatch, caplog, accept):
    _, client = s3_client
    monkeypatch.setattr(api, "execute_job", lambda job, progress: (b"stl", {}))
    s3 = Mock()
    s3.put_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")
    monkeypatch.setattr(print_uploads, "client", lambda: s3)
    response = client.post("/print", files={"photo": ("a.jpg", b"jpeg")}, headers={"Accept": accept})
    assert response.status_code == 200
    if accept == "application/x-ndjson":
        done = json.loads(response.text.splitlines()[-1])
        assert base64.b64decode(done["stl_base64"]) == b"stl"
        assert not {"stl_url", "stl_file_id", "stl_expires_in"} & done.keys()
    else:
        assert response.content == b"stl" and "X-STL-URL" not in response.headers
    assert "could not upload STL result" in caplog.text


def test_unconfigured_s3_keeps_legacy_stream(s3_client, monkeypatch):
    _, client = s3_client
    monkeypatch.delenv("PRINT_S3_BUCKET")
    monkeypatch.setattr(api, "execute_job", lambda job, progress: (b"stl", {}))
    s3 = Mock()
    monkeypatch.setattr(print_uploads, "client", s3)
    response = client.post("/print", files={"photo": ("a.jpg", b"jpeg")},
                           headers={"Accept": "application/x-ndjson"})
    done = json.loads(response.text.splitlines()[-1])
    assert set(done) == {"type", "report", "stl_base64", "elapsed_s", "filename", "progress"}
    assert done["type"] == "done" and done["report"] == {}
    assert base64.b64decode(done["stl_base64"]) == b"stl"
    assert done["filename"] == "a.stl" and done["progress"] == 1.0
    s3.assert_not_called()
