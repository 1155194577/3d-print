"""POST /photo-check tests with SiliconFlow HTTPS mocked; no network or model."""

import base64
import io
import json
import math

import pytest
import httpx
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
        self.answers = {"cluttered_background": "yes" if cluttered else "no",
                        "cut_off": "no" if cut_off else "yes"}

    def __call__(self, url, **kwargs):
        assert url == pc.vlm_module.API_URL
        return response(json.dumps(self.answers))


def response(content, status=200):
    return httpx.Response(status, request=httpx.Request("POST", pc.vlm_module.API_URL),
                          json={"choices": [{"message": {"content": content,
                                  "reasoning_content": "Ignore this: the object is cut off."}}]})


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PRINT_API_TOKEN", "s3cret")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-placeholder-not-a-key")
    monkeypatch.setattr(pc.vlm_module.httpx, "post", FakeVLM())
    with TestClient(api.create_print_app()) as c:
        yield c


def check(client, vlm, monkeypatch, data=None, token="s3cret"):
    monkeypatch.setattr(pc.vlm_module.httpx, "post", vlm)
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
    auth = {"Authorization": "Bearer s3cret"}
    post = lambda files, data=None: client.post("/photo-check", files=files, data=data or {}, headers=auth)
    assert post({"photo": ("shot.gif", jpeg())}).status_code == 415
    assert post({"photo": ("shot.jpg", b"")}).status_code == 400
    assert post({"photo": ("shot.jpg", b"not an image")}).status_code == 422
    assert post({"photo": ("shot.jpg", jpeg())}, {"camera_transform": "[1, 2]"}).status_code == 400


def test_unparseable_model_answer_is_502(client, monkeypatch):
    resp = check(client, lambda *a, **kw: response('{"cluttered_background":"maybe","cut_off":"yes"}'), monkeypatch)
    assert resp.status_code == 502


@pytest.mark.parametrize("content", [
    '```json\n{"cluttered_background":"no","cut_off":"yes"}\n```',
    'Result: {"cluttered_background":"No","cut_off":"Yes"} trailing text',
])
def test_json_wrappers_and_reasoning_ignored(client, monkeypatch, content):
    resp = check(client, lambda *a, **kw: response(content), monkeypatch)
    assert codes(resp) == []
    assert resp.json()["answers"] == {"cluttered_background": "no", "cut_off": "yes"}
    assert resp.json()["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"


@pytest.mark.parametrize("content", ["", None, "not JSON", "{}", "{broken}",
    '{"cluttered_background":false,"cut_off":true}',
    '{"cluttered_background":"no"}',
])
def test_invalid_or_empty_content_is_502(client, monkeypatch, content):
    resp = check(client, lambda *a, **kw: response(content), monkeypatch)
    assert resp.status_code == 502
    assert resp.json()["detail"] == "photo check failed: vision service unavailable or returned an invalid answer"


@pytest.mark.parametrize("status", [401, 429, 500])
def test_upstream_error_is_sanitized(client, monkeypatch, status):
    resp = check(client, lambda *a, **kw: response("private upstream detail", status), monkeypatch)
    assert resp.status_code == 502
    assert "private" not in resp.text


def test_timeout_is_502(client, monkeypatch):
    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("private upstream detail")
    resp = check(client, timeout, monkeypatch)
    assert resp.status_code == 502
    assert "private" not in resp.text


def test_missing_key_does_not_call_service(client, monkeypatch):
    monkeypatch.delenv("SILICONFLOW_API_KEY")
    def unexpected(*args, **kwargs):
        pytest.fail("HTTP must not be called without a key")
    assert check(client, unexpected, monkeypatch).status_code == 502


def test_request_encodes_resized_jpeg_and_prompts(client, monkeypatch):
    def post(url, *, headers, json, timeout):
        assert url == "https://api.siliconflow.cn/v1/chat/completions"
        assert headers == {"Authorization": "Bearer test-placeholder-not-a-key"}
        assert json["model"] == "deepseek-ai/DeepSeek-V4.1-Flash"
        assert json["messages"][0] == {"role": "system", "content": pc.vlm_module.SYSTEM_PROMPT}
        parts = json["messages"][1]["content"]
        prefix, encoded = parts[0]["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/jpeg;base64"
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert image.format == "JPEG" and image.mode == "RGB"
        assert image.size == (1024, 512)
        assert "cut_off asks whether the object IS fully inside" in parts[1]["text"]
        assert timeout.connect == 10 and timeout.read == 60
        return response('{"cluttered_background":"no","cut_off":"yes"}')
    monkeypatch.setattr(pc.vlm_module.httpx, "post", post)
    buf = io.BytesIO()
    Image.new("RGBA", (2048, 1024), "white").save(buf, "PNG")
    resp = client.post("/photo-check", files={"photo": ("shot.png", buf.getvalue())},
                       headers={"Authorization": "Bearer s3cret"})
    assert codes(resp) == []
