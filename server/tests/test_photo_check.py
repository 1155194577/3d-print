"""POST /photo-check tests with the VLM swapped for canned answers, so no GPU or weights."""

import io
import json
import math

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from server import photo_check as pc
from server import print_api as api


def jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(buf, "JPEG")
    return buf.getvalue()


def pitched(deg):
    """ARKit camera-to-world, column-major, camera tilted `deg` below the horizon."""
    a = math.radians(deg)
    # Camera X stays world X; its view direction (-Z) is (0, -sin a, -cos a).
    x, y, z = [1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]
    return json.dumps(x + [0] + y + [0] + z + [0] + [0.1, 1.4, 0.2, 1])


class FakeVLM:
    def __init__(self, **yes):
        self.yes, self.asked = yes, []

    def __call__(self, data, questions):
        self.asked.append(sorted(questions))
        return {k: "Yes." if self.yes.get(k) else "No." for k in questions}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PRINT_API_TOKEN", "s3cret")
    with TestClient(api.create_print_app()) as c:
        yield c


def check(client, vlm, monkeypatch, data=None, token="s3cret"):
    monkeypatch.setattr(pc, "vlm", vlm)
    return client.post("/photo-check", files={"photo": ("shot.jpg", jpeg())}, data=data or {},
                       headers={"Authorization": f"Bearer {token}"})


def codes(resp):
    assert resp.status_code == 200, resp.text
    return [i["code"] for i in resp.json()["issues"]]


def test_clean_shot_passes(client, monkeypatch):
    resp = check(client, FakeVLM(), monkeypatch)
    assert codes(resp) == [] and resp.json()["ok"] is True
    assert resp.json()["angle_source"] == "model"


def test_model_flags_every_issue_with_a_message(client, monkeypatch):
    vlm = FakeVLM(steep_angle=True, cluttered_background=True, cut_off=True)
    resp = check(client, vlm, monkeypatch)
    assert codes(resp) == ["steep_angle", "cluttered_background", "cut_off"]
    assert resp.json()["ok"] is False and all(i["message"] for i in resp.json()["issues"])


@pytest.mark.parametrize("deg,steep", [(0, False), (15, False), (45, True), (54, True), (90, True)])
def test_measured_pitch_decides_the_angle(client, monkeypatch, deg, steep):
    vlm = FakeVLM(steep_angle=not steep)
    resp = check(client, vlm, monkeypatch, {"camera_transform": pitched(deg)})
    assert ("steep_angle" in codes(resp)) is steep
    assert resp.json()["angle_source"] == "camera"
    assert resp.json()["camera_pitch_deg"] == pytest.approx(deg, abs=0.1)
    assert vlm.asked == [["cluttered_background", "cut_off"]]


def test_auth_and_bad_requests(client, monkeypatch):
    assert check(client, FakeVLM(), monkeypatch, token="nope").status_code == 401
    monkeypatch.setattr(pc, "vlm", FakeVLM())
    auth = {"Authorization": "Bearer s3cret"}
    post = lambda files, data=None: client.post("/photo-check", files=files, data=data or {}, headers=auth)
    assert post({"photo": ("shot.gif", jpeg())}).status_code == 415
    assert post({"photo": ("shot.jpg", b"")}).status_code == 400
    assert post({"photo": ("shot.jpg", b"not an image")}).status_code == 422
    assert post({"photo": ("shot.jpg", jpeg())}, {"camera_transform": "[1, 2]"}).status_code == 400


def test_unparseable_model_answer_is_502(client, monkeypatch):
    resp = check(client, lambda data, qs: {k: "maybe" for k in qs}, monkeypatch)
    assert resp.status_code == 502
