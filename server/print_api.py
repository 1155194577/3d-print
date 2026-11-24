"""POST /print: a LiDAR-scanned mesh, or a photo plus its measured height, in;
a printable STL out, in one synchronous request. Contract: server/README.md.

The local runner executes in-process; Modal dispatches validated uploads to a
GPU function, keeping its GIL and scheduling delays away from the CPU frontend.
"""

import asyncio
import base64
import hmac
from http import HTTPStatus
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid

from fastapi import APIRouter, Body, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPOSR_DIR = os.path.join(ROOT, "research", "triposr")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import depth_fusion  # noqa: E402
import img2mesh  # noqa: E402
import photo_level  # noqa: E402
import printability  # noqa: E402
import topology  # noqa: E402

from server import print_gcode, print_jobs, print_uploads  # noqa: E402

logger = logging.getLogger(__name__)


def _log_rejection(request, status, detail):
    request.state.print_rejection_logged = True
    route = request.scope.get("route")
    logger.warning("print request rejected status=%s route=%s detail=%s", status,
                   getattr(route, "path", "/print"), json.dumps(detail, ensure_ascii=True))


class PrintRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def logged(request):
            try:
                _check_auth(request.headers.get("authorization"))
                async with print_uploads.resolve(request, MAX_UPLOAD_BYTES):
                    return await handler(request)
            except print_uploads.MissingUploads as exc:
                _log_rejection(request, exc.status_code, exc.detail)
                return JSONResponse({"detail": exc.detail, "missing": exc.missing}, status_code=409)
            except StarletteHTTPException as exc:
                if 400 <= exc.status_code < 500:
                    _log_rejection(request, exc.status_code, exc.detail)
                raise
            except RequestValidationError as exc:
                _log_rejection(request, 422, [
                    {key: error[key] for key in ("loc", "type", "msg")}
                    for error in exc.errors()])
                raise

        return logged

MAX_UPLOAD_BYTES = int(os.environ.get("PRINT_MAX_UPLOAD_MB", "200")) * 1024 * 1024
PHOTO_EXTS = (".jpg", ".jpeg", ".png", ".webp")
DEPTH_EXTS = (".f32", ".bin", ".raw", ".png")
CONFIDENCE_EXTS = (".u8", ".bin", ".raw")
MESH_EXTS = (".obj", ".ply", ".usdz", ".usd", ".usdc", ".usda", ".glb", ".stl")
TRIPOSR_MODEL = os.environ.get("TRIPOSR_MODEL", "stabilityai/TripoSR")


class PhotoReconstructor:
    """TripoSR kept loaded between requests; one inference at a time."""

    def __init__(self, model_id=TRIPOSR_MODEL, chunk_size=8192):
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        if TRIPOSR_DIR not in sys.path:
            sys.path.insert(0, TRIPOSR_DIR)
        import rembg
        import torch
        from tsr.system import TSR

        self.device = os.environ.get("TRIPOSR_DEVICE") or (
            "cuda:0" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")
        model = TSR.from_pretrained(self.model_id, config_name="config.yaml",
                                    weight_name="model.ckpt")
        model.renderer.set_chunk_size(self.chunk_size)
        model.to(self.device)
        self.rembg_session = rembg.new_session()
        self.model = model

    def foreground_mask(self, image_path):
        """Object mask of the photo as stored (no EXIF rotation), uint8 0-255."""
        import numpy as np
        import rembg
        from PIL import Image

        with self.lock:
            self.load()
            img = Image.open(image_path).convert("RGB")
            return np.asarray(rembg.remove(img, session=self.rembg_session, only_mask=True))

    def __call__(self, image_path, out_path, mc_resolution=256, foreground_ratio=0.85,
                 quarter_turns=None, progress=None):
        progress = progress or (lambda *a: None)
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        with self.lock:
            self.load()
            from tsr.utils import remove_background, resize_foreground

            if quarter_turns is None:  # iPhone JPEGs are stored sideways with an EXIF tag
                image = ImageOps.exif_transpose(Image.open(image_path))
            else:  # sensor-orientation frame: turn it upright from gravity
                image = Image.open(image_path).convert("RGB").rotate(90 * quarter_turns, expand=True)
            progress("background")
            image = remove_background(image, self.rembg_session)
            image = resize_foreground(image, foreground_ratio)
            arr = np.array(image).astype(np.float32) / 255.0
            arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
            image = Image.fromarray((arr * 255.0).astype(np.uint8))
            with torch.no_grad():
                progress("reconstruct")
                codes = self.model([image], device=self.device)
                progress("mesh")
                meshes = self.model.extract_mesh(codes, True, resolution=mc_resolution)
            meshes[0].export(out_path)


if os.environ.get("TRIPOSR_BACKEND") == "modal":
    from server.remote_triposr import ModalReconstructor

    reconstruct_photo = ModalReconstructor()
else:
    reconstruct_photo = PhotoReconstructor()
router = APIRouter(route_class=PrintRoute)


def _check_auth(authorization):
    token = os.environ.get("PRINT_API_TOKEN")
    if not token:
        return
    given = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(given.encode(), token.encode()):
        raise HTTPException(401, "missing or wrong bearer token")


async def reject_unauthorized_print(request: Request, call_next):
    """Check the token before the body is read, so anonymous uploads never spool."""
    if request.url.path in ("/print", "/photo-check") or request.url.path.startswith("/print/"):
        try:
            _check_auth(request.headers.get("authorization"))
        except HTTPException as exc:
            if request.url.path == "/print" or request.url.path.startswith("/print/"):
                _log_rejection(request, exc.status_code, exc.detail)
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    response = await call_next(request)
    if (request.url.path == "/print" or request.url.path.startswith("/print/")) and (
            400 <= response.status_code < 500 and not getattr(request.state, "print_rejection_logged", False)):
        _log_rejection(request, response.status_code, HTTPStatus(response.status_code).phrase)
    return response


def create_print_app():
    """POST /print, POST /photo-check and a health check: what the serverless deploy exposes."""
    from server.photo_check import router as photo_check_router

    app = FastAPI(title="3d-print /print")
    app.middleware("http")(reject_unauthorized_print)
    app.include_router(router)
    app.include_router(photo_check_router)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    return app


def _save_upload(upload, allowed, workdir, name):
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(415, f"{name}: unsupported file type {ext or '(none)'}; "
                                 f"use one of {', '.join(allowed)}")
    path = os.path.join(workdir, f"{name}{ext}")
    size = 0
    with open(path, "wb") as fh:
        while chunk := upload.file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{name} is over {MAX_UPLOAD_BYTES >> 20} MB")
            fh.write(chunk)
    if size == 0:
        raise HTTPException(400, f"{name} is empty")
    return path


def _parse_bed(fit_bed):
    if not fit_bed:
        return None
    try:
        bed = tuple(float(x) for x in fit_bed.lower().split("x"))
        if len(bed) != 3 or min(bed) <= 0:
            raise ValueError
        return bed
    except ValueError:
        raise HTTPException(400, "fit_bed wants WxDxH in mm, e.g. 220x220x250")


def _face_anchor(value):
    import numpy as np

    if value is None:
        return None
    try:
        vals = np.asarray(json.loads(value), dtype=float)
        if vals.shape != (16,) or not np.isfinite(vals).all():
            raise ValueError
        return vals.reshape(4, 4).T
    except (ValueError, TypeError):
        raise ValueError("face_transform wants a JSON array of 16 finite floats")


def _load_face_views(form, count, workdir):
    views = []
    for i in range(1, count):
        prefix = f"view_{i}_"
        get = lambda name: form.get(prefix + name)
        try:
            anchor = _face_anchor(get("face_transform"))
            photo, depth = get("photo"), get("depth")
            if photo is None or depth is None:
                raise ValueError(f"{prefix}photo and {prefix}depth are required")
            src = _save_upload(photo, PHOTO_EXTS, workdir, prefix + "photo")
            dpath = _save_upload(depth, DEPTH_EXTS, workdir, prefix + "depth")
            frame = depth_fusion.Frame.load(
                dpath, int(get("depth_width")), int(get("depth_height")),
                intrinsics=get("intrinsics"), camera_transform=get("camera_transform"),
                image_orientation=int(get("image_orientation") or 1),
                depth_intrinsics=get("depth_intrinsics"), depth_extrinsics=get("depth_extrinsics"))
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"{prefix}: {exc}")
        views.append(dict(src=src, frame=frame, face_to_world=anchor))
    return views


@router.post("/print/uploads")
def print_upload_urls(payload: object = Body(...), authorization: str | None = Header(None)):
    _check_auth(authorization)
    return print_uploads.register(payload, MAX_UPLOAD_BYTES)


@router.get("/print/uploads/{job_id}")
def print_upload_preview(job_id: str):
    return print_uploads.preview(job_id)


@router.post("/print", response_class=Response,
          responses={200: {"content": {"model/stl": {}}}})
def print_stl(
    request: Request,
    mesh: UploadFile | None = File(None, description="LiDAR scan: usdz/obj/ply/glb/stl"),
    photo: UploadFile | None = File(None, description="photo: jpg/png/webp"),
    mode: str = Form("object", pattern="^(object|face)$"),
    face_transform: str | None = Form(None, description="16 floats, column-major ARFaceAnchor transform"),
    height_mm: float | None = Form(None, gt=0, description="measured object height, mm"),
    units: str = Form("m", pattern="^(m|cm|mm)$"),
    up_axis: str = Form("y", pattern="^(y|z)$"),
    fit_bed: str | None = Form(None),
    base_plate: bool = Form(True),
    optimise: bool = Form(False),
    depth: UploadFile | None = File(None, description="ARKit sceneDepth: float32 LE metres (or 16-bit PNG, mm)"),
    depth_width: int | None = Form(None, gt=0),
    depth_height: int | None = Form(None, gt=0),
    depth_confidence: UploadFile | None = File(None, description="ARKit confidenceMap: uint8 0/1/2"),
    intrinsics: str | None = Form(None, description='{"fx","fy","cx","cy","width","height"} in photo px'),
    depth_intrinsics: str | None = Form(None, description="depth camera intrinsics JSON"),
    depth_extrinsics: str | None = Form(None, description="16 floats, column-major depth-to-photo, OpenCV axes"),
    camera_transform: str | None = Form(None, description="16 floats, column-major camera-to-world"),
    image_orientation: int | None = Form(None),
    object_points: str | None = Form(None, description="[[u, v], ...] taps on the object, photo px"),
    job_id: str | None = Form(None, description="client-chosen UUID; a retry with it joins the job"),
    uploaded: str | None = Form(None, description="comma-separated names uploaded with /print/uploads"),
    authorization: str | None = Header(None),
    accept: str | None = Header(None),
):
    _check_auth(authorization)
    join_id = getattr(request.state, "print_join_job_id", None)
    if join_id is not None:
        return _job_response(join_id, accept)
    if (mesh is None) == (photo is None):
        raise HTTPException(400, "send exactly one of `mesh` or `photo`")
    if mode == "face" and photo is None:
        raise HTTPException(400, "mode=face requires `photo`")
    form = request._form
    try:
        view_count = int(form.get("view_count", "1"))
        if not 1 <= view_count <= 5:
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(400, "view_count must be an integer between 1 and 5")
    if view_count > 1 and (mode != "face" or depth is None):
        raise HTTPException(400, "multiple views require mode=face and front depth")
    face_to_world = None
    if mode == "face" and face_transform is not None:
        import numpy as np

        try:
            vals = np.asarray(json.loads(face_transform), dtype=float)
            if vals.shape != (16,) or not np.isfinite(vals).all():
                raise ValueError
            face_to_world = vals.reshape(4, 4).T
        except (ValueError, TypeError):
            raise HTTPException(400, "face_transform wants a JSON array of 16 finite floats")
    bed = _parse_bed(fit_bed)
    if depth is not None and photo is None:
        raise HTTPException(400, "`depth` goes with `photo`")
    if image_orientation is not None and image_orientation not in (1, 3, 6, 8):
        raise HTTPException(400, "image_orientation must be 1, 3, 6 or 8")
    try:
        job_id = str(uuid.UUID(job_id)) if job_id else str(uuid.uuid4())
    except ValueError:
        raise HTTPException(400, "`job_id` must be a UUID")
    base_thickness = 3.0
    plate_height = max(base_thickness - 0.5, 0.0) if base_plate else 0.0
    # height_mm is the object itself; prepare_mesh's height includes the plate.
    object_height = height_mm if height_mm is not None else (100 if mode == "face" else None)
    height = object_height + plate_height if object_height else None

    workdir = tempfile.mkdtemp(prefix="print_")
    try:
        frame = None
        views = []
        gravity = None
        if mesh is not None:
            src = _save_upload(mesh, MESH_EXTS, workdir, "mesh")
            stem = mesh.filename
        else:
            src = _save_upload(photo, PHOTO_EXTS, workdir, "photo")
            stem = photo.filename
            if mode == "object" and camera_transform and depth is None:
                try:
                    gravity = photo_level.photo_up(camera_transform, image_orientation or 1)
                except ValueError as exc:
                    raise HTTPException(400, str(exc))
            views = _load_face_views(form, view_count, workdir)
            if depth is not None:
                dpath = _save_upload(depth, DEPTH_EXTS, workdir, "depth")
                cpath = (_save_upload(depth_confidence, CONFIDENCE_EXTS, workdir, "confidence")
                         if depth_confidence is not None else None)
                try:
                    frame = depth_fusion.Frame.load(
                        dpath, depth_width, depth_height, cpath, intrinsics,
                        camera_transform, image_orientation or 1, object_points,
                        depth_intrinsics, depth_extrinsics)
                except ValueError as exc:
                    raise HTTPException(400, str(exc))
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    name = os.path.splitext(os.path.basename(stem or "model"))[0] or "model"
    job = dict(src=src, is_mesh=mesh is not None, frame=frame, gravity=gravity, height=height, bed=bed,
               base_plate=base_plate, base_thickness=base_thickness, optimise=optimise,
               units_to_mm=img2mesh.UNITS_TO_MM[units], up_axis=up_axis, workdir=workdir,
               mode=mode, face_to_world=face_to_world, image_orientation=image_orientation,
               views=views, name=name)

    try:
        if not submit_job(job_id, job, f"{name}.stl"):
            shutil.rmtree(workdir, ignore_errors=True)  # a retry: join the job already running
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    return _job_response(job_id, accept)


def _job_response(job_id, accept):
    if _wants_ndjson(accept):
        return _ndjson(_events(job_id, accepted=True, with_stl=True))
    state = print_jobs.get_state(job_id)
    while state is not None and state["type"] == "progress":
        time.sleep(POLL_S)
        state = print_jobs.get_state(job_id)
    if state is None:
        raise HTTPException(404, "the print job expired")
    if state["type"] == "error":
        raise HTTPException(state["status"], state["detail"])
    return _stl_response(print_jobs.get_stl(job_id), state["report"], state["filename"],
                         print_uploads.result_fields(state))


def execute_job(job, progress):
    body, report = _make_stl(job, progress)
    progress("slice")
    gcode, report["gcode"] = print_gcode.slice_stl(body, report["extents_mm"],
                                                   f"{job.get('name', 'model')}.gcode")
    return body, report, gcode


def submit_job(job_id, job, filename):
    """Local runner; the Modal CPU frontend replaces this with GPU dispatch."""
    def run(progress):
        try:
            return execute_job(job, progress)
        finally:
            shutil.rmtree(job["workdir"], ignore_errors=True)

    return print_jobs.start(job_id, run, filename)


def _stl_response(body, summary, filename, result=None):
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Print-Report": json.dumps(summary, ensure_ascii=True),
        "Access-Control-Expose-Headers": "X-Print-Report, Content-Disposition",
    }
    if result and "stl_url" in result:
        headers.update({"X-STL-URL": result["stl_url"], "X-STL-File-ID": result["stl_file_id"],
                        "X-STL-Expires-In": str(result["stl_expires_in"])})
        headers["Access-Control-Expose-Headers"] += ", X-STL-URL, X-STL-File-ID, X-STL-Expires-In"
    return Response(body, media_type="model/stl", headers=headers)


def _wants_ndjson(accept):
    for part in (accept or "").split(","):
        media, *params = [p.strip() for p in part.split(";")]
        if media.lower() == "application/x-ndjson":
            q = next((p.split("=", 1)[1] for p in params if p.lower().startswith("q=")), "1")
            try:
                return float(q) > 0
            except ValueError:
                return False
    return False


def _job_state(job_id):
    try:
        job_id = str(uuid.UUID(job_id))
    except ValueError:
        raise HTTPException(404, "no such print job")
    state = print_jobs.get_state(job_id)
    if state is None:
        raise HTTPException(404, "no such print job (finished jobs are kept 24 h)")
    return job_id, state


@router.get("/print/{job_id}")
async def print_job(job_id: str, accept: str | None = Header(None)):
    """The job's latest line; with Accept: application/x-ndjson, that line then the rest."""
    job_id, state = await run_in_threadpool(_job_state, job_id)
    if _wants_ndjson(accept):
        return _ndjson(_events(job_id, accepted=False, with_stl=False))
    line = _line(state)
    if state["type"] == "done":
        line.update(await run_in_threadpool(print_uploads.result_fields, state))
    return line


@router.get("/print/{job_id}/result", response_class=Response,
            responses={200: {"content": {"model/stl": {}}}})
async def print_result(job_id: str):
    """Held open until the job ends: the STL, or the job's error as {"detail"}."""
    job_id, state = await run_in_threadpool(_job_state, job_id)
    while state["type"] == "progress":
        await asyncio.sleep(POLL_S)
        job_id, state = await run_in_threadpool(_job_state, job_id)
    if state["type"] == "error":
        return JSONResponse({"detail": state["detail"]}, status_code=state["status"])
    body = await run_in_threadpool(print_jobs.get_stl, job_id)
    return _stl_response(body, state["report"], state["filename"],
                         await run_in_threadpool(print_uploads.result_fields, state))


@router.get("/print/{job_id}/gcode", response_class=Response,
            responses={200: {"content": {"text/x-gcode": {}}}})
async def print_gcode_result(job_id: str):
    """Held open until the job ends: K1C PLA G-code, or why there is none as {"detail"}."""
    job_id, state = await run_in_threadpool(_job_state, job_id)
    while state["type"] == "progress":
        await asyncio.sleep(POLL_S)
        job_id, state = await run_in_threadpool(_job_state, job_id)
    if state["type"] == "error":
        return JSONResponse({"detail": state["detail"]}, status_code=state["status"])
    info = state["report"].get("gcode") or {}
    if not state.get("gcode_chunks"):
        signed = await run_in_threadpool(print_uploads.result_fields, state)
        if "gcode_url" in signed:
            return RedirectResponse(signed["gcode_url"], status_code=307)
        return JSONResponse({"detail": "no G-code for this print: " + info.get("skipped", "not sliced")},
                            status_code=404)
    body = await run_in_threadpool(print_jobs.get_gcode, job_id)
    return Response(body, media_type="text/x-gcode", headers={
        "Content-Disposition": print_uploads.attachment(info.get("filename", "model.gcode"))})


# stage -> (progress when it starts, message); the order is the order they run in
STAGES = {
    "queued": (0.01, "Waiting for a print server GPU"),
    "received": (0.02, "Upload received"),
    "background": (0.08, "Removing the background"),
    "reconstruct": (0.15, "Reconstructing the shape"),
    "mesh": (0.45, "Extracting the mesh"),
    "fuse": (0.60, "Fitting to the LiDAR depth"),
    "repair": (0.75, "Repairing and adding the base"),
    "optimise": (0.80, "Removing unloaded material"),
    "export": (0.88, "Writing the STL"),
    "check": (0.92, "Checking it will print"),
    "slice": (0.95, "Slicing G-code for the MakerSpace K1C"),
}
HEARTBEAT_S = 2.0


def _make_stl(job, progress):
    """The slow part of /print. Returns (stl bytes, report); raises HTTPException."""
    src, frame, height = job["src"], job["frame"], job["height"]
    gravity = job["gravity"]
    mode = job["mode"]
    fusion = None
    if job["is_mesh"]:
        kwargs = dict(units_to_mm=job["units_to_mm"], up_axis=job["up_axis"])
    else:
        img, src = src, os.path.join(job["workdir"], "raw.obj")
        try:
            mask = None
            if frame is None:
                orientation = job["image_orientation"]
                turns = gravity[1] if gravity else {1: 0, 3: 2, 6: 3, 8: 1}.get(orientation)
                reconstruct_photo(img, src, quarter_turns=turns, progress=progress)
            else:
                photo_up = frame.depth_to_photo[:3, :3] @ frame.up_camera()
                turns = depth_fusion.upright_turns(photo_up)
                reconstruct_photo(img, src, quarter_turns=turns, progress=progress)
                progress("fuse")
                mask = reconstruct_photo.foreground_mask(img)
        except Exception as exc:
            raise HTTPException(422, f"could not reconstruct the photo: {exc}")
        kwargs = {}
        if frame is not None:
            import trimesh

            try:
                raw = trimesh.load(src, force="mesh")
                if mode == "face":
                    views = [dict(view, mask=reconstruct_photo.foreground_mask(view["src"]))
                             for view in job.get("views", [])]
                    fused, fusion = depth_fusion.fuse_face(
                        raw, frame, mask, job["face_to_world"], views=views)
                else:
                    fused, fusion = depth_fusion.fuse(raw, frame, mask)
                src = os.path.join(job["workdir"], "fused.ply")
                fused.export(src)
                kwargs = dict(units_to_mm=1000.0, up_axis="z")
                if mode != "face":
                    height = None
            except depth_fusion.FusionError as exc:
                fusion = {"skipped": str(exc)}
                if frame.camera_to_world is not None:
                    gravity = (photo_up, turns)
        if mode == "object" and (not fusion or "skipped" in fusion):
            kwargs.update(photo_level=True, gravity=gravity)
        if mode == "face" and (frame is None or "skipped" in (fusion or {})):
            import trimesh

            raw = trimesh.load(src, force="mesh")
            bust = depth_fusion.cut_bust(raw, raw.bounds[0, 2] + 0.2 * raw.extents[2])
            src = os.path.join(job["workdir"], "bust.ply")
            bust.export(src)
    progress("repair")
    try:
        out, report = img2mesh.prepare_mesh(
            src, height=height, fit_bed=job["bed"], base_plate=job["base_plate"],
            base_thickness=job["base_thickness"], **kwargs)
    except SystemExit as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(422, f"could not process the mesh: {exc}")
    optimisation = None
    if job.get("optimise", False):
        progress("optimise")
        out, optimisation = topology.optimise(out)
        report.update(watertight=out.is_watertight, faces=len(out.faces), volume_cm3=out.volume / 1000)
    progress("export")
    body = out.export(file_type="stl")
    progress("check")
    body, check = _check_printable(body, job["bed"])

    ex = report["extents"]
    return body, {
        **({"mode": "face", "views_used": (fusion or {}).get("views_used", 1)}
           if mode == "face" else {}),
        "extents_mm": [round(float(v), 1) for v in ex],
        "watertight": bool(report["watertight"]),
        **({"fusion": fusion} if fusion else {}),
        **({"optimise": optimisation} if optimisation is not None else {}),
        "faces": int(report["faces"]),
        **({"level": report["level"]} if "level" in report else {}),
        "volume_cm3": round(report["volume_cm3"], 1) if report["volume_cm3"] else None,
        "printability": check,
        "scale": report["scale_note"],
    }


def _check_printable(body, bed):
    """Check the STL as written; repair once if needed. Returns (stl, check) or raises 422."""
    import io

    import trimesh

    def load(data):
        return trimesh.load(io.BytesIO(data), file_type="stl", force="mesh")

    check = printability.check(load(body), bed)
    if check["repairable"]:
        fixed = printability.fix(load(body)).export(file_type="stl")
        recheck = printability.check(load(fixed), bed)
        if recheck["printable"] or not check["printable"]:
            body, check = fixed, {**recheck, "repaired": check["errors"] + check["warnings"]}
    if not check["printable"]:
        raise HTTPException(422, "the model can't be printed: " + "; ".join(check["errors"]))
    return body, check


POLL_S = 0.5


def _ndjson(lines):
    return StreamingResponse(lines, media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _line(state):
    """A job state as the line the app sees, minus the STL."""
    now = time.time()
    if state["type"] == "progress":
        stage = state["stage"]
        start, message = STAGES[stage]
        order = list(STAGES)
        nxt = ([STAGES[s][0] for s in order[order.index(stage) + 1:]] + [1.0])[0]
        # creep toward the next stage so the bar keeps moving, never reaching it
        frac = start + 0.9 * (nxt - start) * (1 - math.exp(-(now - state["since"]) / 20))
        return {"type": "progress", "stage": stage, "progress": round(frac, 3),
                "message": message, "elapsed_s": round(now - state["created"], 1)}
    elapsed = round(state["updated"] - state["created"], 1)
    if state["type"] == "done":
        return {"type": "done", "progress": 1.0, "filename": state["filename"],
                **({"views_used": state["report"]["views_used"]}
                   if "views_used" in state["report"] else {}),
                "report": state["report"], "elapsed_s": elapsed}
    return {"type": "error", "status": state["status"], "detail": state["detail"],
            "elapsed_s": elapsed}


async def _wait_with_heartbeats(awaitable, progress, sent_at):
    task = asyncio.create_task(awaitable)
    try:
        while True:
            remaining = max(0, HEARTBEAT_S - (time.monotonic() - sent_at))
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if done:
                yield True, task.result()
                return
            sent_at = time.monotonic()
            yield False, _line(progress)
    finally:
        task.cancel()


async def _events(job_id, accepted, with_stl):
    """NDJSON: [accepted,] progress lines (repeated every HEARTBEAT_S), then done or error."""
    def out(obj):
        return (json.dumps(obj, ensure_ascii=True) + "\n").encode()

    if accepted:
        yield out({"type": "accepted", "job_id": job_id})
    # every stage is sent once, in order, even ones that ended between two polls;
    # a resume starts from the current stage
    sent, sent_at = (0 if accepted else None), time.monotonic()
    now = time.time()
    last_progress = dict(type="progress", stage="received", since=now, created=now)
    while True:
        async for ready, value in _wait_with_heartbeats(
                run_in_threadpool(print_jobs.get_state, job_id), last_progress, sent_at):
            if not ready:
                sent_at = time.monotonic()
                yield out(value)
            else:
                state = value
        if state is None:
            yield out({"type": "error", "status": 404, "detail": "the print job expired"})
            return
        history, running = state.get("history", []), state["type"] == "progress"
        if sent is None:
            sent = len(history) - running
        for stage in history[sent:len(history) - running]:
            last_progress = dict(state, type="progress", stage=stage, since=time.time())
            yield out(_line(last_progress))
            sent_at = time.monotonic()
        if running:
            last_progress = state
        sent = max(sent, len(history) - running)
        line = _line(state)
        if state["type"] == "done":
            async for ready, value in _wait_with_heartbeats(
                    run_in_threadpool(print_uploads.result_fields, state), last_progress, sent_at):
                if not ready:
                    sent_at = time.monotonic()
                    yield out(value)
                else:
                    line.update(value)
            if with_stl:
                def encoded_stl():
                    return base64.b64encode(print_jobs.get_stl(job_id)).decode()

                async for ready, value in _wait_with_heartbeats(
                        run_in_threadpool(encoded_stl), last_progress, sent_at):
                    if not ready:
                        sent_at = time.monotonic()
                        yield out(value)
                    else:
                        line["stl_base64"] = value
            yield out(line)
            return
        if state["type"] == "error":
            yield out(line)
            return
        if sent < len(history) or time.monotonic() - sent_at >= HEARTBEAT_S:
            sent, sent_at = len(history), time.monotonic()
            yield out(line)
        await asyncio.sleep(POLL_S)
