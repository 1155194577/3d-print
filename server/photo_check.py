"""POST /photo-check: before a Photo Print is sent to /print, a vision-language model
looks at the shot and flags what makes TripoSR rebuild it badly, so the app can ask
for a retake. Contract: server/README.md.

Issue codes (stable; messages are for the user):
    steep_angle            camera tilted too far down (TripoSR wants a level side view)
    cluttered_background   other things around or behind the object
    cut_off                part of the object is outside the frame

The angle comes from the phone's measured pitch (ARKit `camera_transform`), not the
model: Qwen2.5-VL 3B and 7B both called Johnny's 54-degree top-down purifier shot "eye
level", while they answer clutter and cut-off reliably. Without a transform the angle
is not checked.
"""

import json
import math
import os
import re

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

from server import vlm as vlm_module

MAX_PITCH_DEG = float(os.environ.get("PHOTO_CHECK_MAX_PITCH_DEG", "30"))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# ISSUE_IF is the yes/no answer that flags each issue.
QUESTIONS = {
    "cluttered_background": "Apart from the main object and the floor, wall or table it stands on, are other "
                            "objects (furniture, chairs, clothes, boxes, other items) visible in the photo? "
                            "Answer yes or no.",
    "cut_off": "Is the whole main object inside the photo, with none of it cut off by the edges? "
               "Answer yes or no.",
}
ISSUE_IF = {"cluttered_background": True, "cut_off": False}
CODES = ["steep_angle", *QUESTIONS]

MESSAGES = {
    "steep_angle": "The camera is pointing down at the object. Hold the phone level with its middle "
                   "and shoot it from the side.",
    "cluttered_background": "There are other things around the object. Move it to a clear spot or "
                            "clear the background.",
    "cut_off": "Part of the object is outside the photo. Step back so all of it is in the frame.",
}

vlm = vlm_module.SiliconFlowVLM()
router = APIRouter()


def is_yes(answer: str) -> bool:
    match = re.match(r"\W*(yes|no)\b", answer, re.I)
    if not match:
        raise ValueError(f"model answered {answer[:80]!r}, not yes or no")
    return match.group(1).lower() == "yes"


def camera_pitch_deg(camera_transform: str) -> float:
    """Degrees the camera looks below the horizon (90 = straight down, negative = up),
    from a 16-float column-major camera-to-world matrix with world +Y up (ARKit).
    The camera looks down its -Z axis, so the view direction is minus column 2."""
    try:
        m = [float(v) for v in json.loads(camera_transform)]
    except (TypeError, ValueError):
        m = []
    if len(m) != 16 or not all(math.isfinite(v) for v in m):
        raise ValueError("camera_transform wants a JSON list of 16 numbers")
    z = m[8:11]
    norm = math.sqrt(sum(v * v for v in z))
    if norm < 1e-6:
        raise ValueError("camera_transform has a zero Z axis")
    return math.degrees(math.asin(max(-1.0, min(1.0, z[1] / norm))))


def check(jpeg: bytes, pitch_deg: float | None = None) -> dict:
    answers = vlm(jpeg, QUESTIONS)
    flagged = {k: is_yes(answers[k]) == ISSUE_IF[k] for k in QUESTIONS}
    flagged["steep_angle"] = pitch_deg is not None and pitch_deg > MAX_PITCH_DEG
    codes = [k for k in CODES if flagged[k]]
    return {
        "ok": not codes,
        "issues": [{"code": c, "message": MESSAGES[c]} for c in codes],
        "angle_checked": pitch_deg is not None,
        "camera_pitch_deg": None if pitch_deg is None else round(pitch_deg, 1),
        "max_pitch_deg": MAX_PITCH_DEG,
        "model": vlm_module.VLM_MODEL,
        "answers": answers,
    }


@router.post("/photo-check")
def photo_check(
    photo: UploadFile = File(..., description="the shot about to be printed: jpg/png/webp"),
    camera_transform: str | None = Form(None, description="16 floats, column-major camera-to-world (ARKit, +Y up)"),
    authorization: str | None = Header(None),
):
    from server.print_api import PHOTO_EXTS, _check_auth

    _check_auth(authorization)
    ext = os.path.splitext(photo.filename or "")[1].lower()
    if ext not in PHOTO_EXTS:
        raise HTTPException(415, f"photo: unsupported file type {ext or '(none)'}; use one of {', '.join(PHOTO_EXTS)}")
    data = photo.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "photo is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"photo is over {MAX_UPLOAD_BYTES >> 20} MB")
    try:
        vlm_module.load_image(data)
    except Exception:
        raise HTTPException(422, "photo is not a readable image")
    pitch = None
    if camera_transform:
        try:
            pitch = camera_pitch_deg(camera_transform)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    try:
        return check(data, pitch)
    except Exception as exc:
        raise HTTPException(502, f"photo check failed: {exc}")
