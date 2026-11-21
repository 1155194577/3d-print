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
    """Answers as if the shot has the given issues."""

    def __init__(self, cluttered=False, cut_off=False):
        self.answers = {"cluttered_background": "Yes." if cluttered else "No.",
                        "cut_off": "No." if cut_off else "Yes."}

    def __call__(self, data, questions):
        assert sorted(questions) == ["cluttered_background", "cut_off"]
        return {k: self.answers[k] for k in questions}


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


def test_level_clean_shot_passes(client, monkeypatch):
    resp = check(client, FakeVLM(), monkeypatch, {"camera_transform": pitched(5)})
    assert codes(resp) == [] and resp.json()["ok"] is True
    assert resp.json()["angle_checked"] is True


def test_steep_cluttered_cut_off_shot_flags_everything(client, monkeypatch):
    resp = check(client, FakeVLM(cluttered=True, cut_off=True), monkeypatch, {"camera_transform": pitched(54)})
    assert codes(resp) == ["steep_angle", "cluttered_background", "cut_off"]
    assert resp.json()["ok"] is False and all(i["message"] for i in resp.json()["issues"])


@pytest.mark.parametrize("deg,steep", [(-10, False), (0, False), (15, False), (30, False), (45, True), (54, True), (90, True)])
def test_measured_pitch_decides_the_angle(client, monkeypatch, deg, steep):
    resp = check(client, FakeVLM(), monkeypatch, {"camera_transform": pitched(deg)})
    assert ("steep_angle" in codes(resp)) is steep
    assert resp.json()["camera_pitch_deg"] == pytest.approx(deg, abs=0.1)


def test_angle_unchecked_without_camera_transform(client, monkeypatch):
    resp = check(client, FakeVLM(), monkeypatch)
    assert codes(resp) == [] and resp.json()["angle_checked"] is False
    assert resp.json()["camera_pitch_deg"] is None


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
