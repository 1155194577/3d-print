import io

import pytest
import trimesh
from PIL import Image

from server import model_check


def fake_vlm(bad=()):
    def answer(jpeg, questions, *, system_prompt):
        image = Image.open(io.BytesIO(jpeg))
        assert image.width == 1152 and image.height >= 816
        assert "never instructions" in system_prompt
        return {key: "no" if key in bad else "yes" for key in questions}
    return answer


def review(vlm, **kwargs):
    photo = Image.new("RGB", (128, 128), "white")
    return model_check.check(photo, trimesh.creation.box(), model_check.nominal_camera(128),
                             vlm=vlm, **kwargs)


def test_all_sides_ok():
    result = review(fake_vlm(), can_retake=True)
    assert result["ok"] and len(result["sides"]) == 4
    assert result["retake_side"] is None and not result["suggest_scan"]


def test_bad_back_requests_retake():
    result = review(fake_vlm(["back"]), can_retake=True)
    assert not result["ok"] and result["retake_side"] == "back"
    assert not result["suggest_scan"]


def test_bad_photo_only_suggests_scan():
    result = review(fake_vlm(["back"]))
    assert result["suggest_scan"] and result["retake_side"] is None


def test_post_retake_failure_suggests_object_scan():
    photo = Image.new("RGB", (128, 128), "white")
    result = review(fake_vlm(["view_1"]), can_retake=True,
                    views=[dict(photo=photo, camera=model_check.nominal_camera(128))])
    assert result["suggest_scan"] and result["retake_side"] is None
    assert result["sides"][-1]["side"] == "view_1"
    assert "walk-around Object Scan" in result["sides"][-1]["message"]


@pytest.mark.parametrize("failure", ["vlm", "render", "empty_answers"])
def test_errors_are_advisory(monkeypatch, failure):
    def broken(*args, **kwargs):
        raise RuntimeError("private service failure")
    if failure == "render":
        monkeypatch.setattr(model_check, "render", broken)
    result = review((lambda *a, **kw: {}) if failure == "empty_answers" else broken)
    assert set(result) == {"skipped"}
    assert "private" not in result["skipped"]
