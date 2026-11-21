"""POST /print: a LiDAR-scanned mesh, or a photo plus its measured height, in;
a printable STL out, in one synchronous request. Contract: server/README.md.

Unlike the /api/jobs queue, which spawns bin/triposr per job, this keeps
TripoSR loaded in-process, which is what a serverless GPU container wants.
"""

import hmac
import json
import os
import shutil
import sys
import tempfile
import threading

from fastapi import APIRouter, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIPOSR_DIR = os.path.join(ROOT, "research", "triposr")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import depth_fusion  # noqa: E402
import img2mesh  # noqa: E402

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
                 quarter_turns=None):
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
            image = remove_background(image, self.rembg_session)
            image = resize_foreground(image, foreground_ratio)
            arr = np.array(image).astype(np.float32) / 255.0
            arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
            image = Image.fromarray((arr * 255.0).astype(np.uint8))
            with torch.no_grad():
                codes = self.model([image], device=self.device)
                meshes = self.model.extract_mesh(codes, True, resolution=mc_resolution)
            meshes[0].export(out_path)


reconstruct_photo = PhotoReconstructor()
router = APIRouter()


def _check_auth(authorization):
    token = os.environ.get("PRINT_API_TOKEN")
    if not token:
        return
    given = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(given.encode(), token.encode()):
        raise HTTPException(401, "missing or wrong bearer token")


async def reject_unauthorized_print(request: Request, call_next):
    """Check the token before the body is read, so anonymous uploads never spool."""
    if request.url.path == "/print":
        try:
            _check_auth(request.headers.get("authorization"))
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    return await call_next(request)


def create_print_app():
    """Just POST /print and a health check: what the serverless deploy exposes."""
    app = FastAPI(title="3d-print /print")
    app.middleware("http")(reject_unauthorized_print)
    app.include_router(router)

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


@router.post("/print", response_class=Response,
          responses={200: {"content": {"model/stl": {}}}})
def print_stl(
    mesh: UploadFile | None = File(None, description="LiDAR scan: usdz/obj/ply/glb/stl"),
    photo: UploadFile | None = File(None, description="photo: jpg/png/webp"),
    mode: str = Form("object", pattern="^(object|face)$"),
    face_transform: str | None = Form(None, description="16 floats, column-major ARFaceAnchor transform"),
    height_mm: float | None = Form(None, gt=0, description="measured object height, mm"),
    units: str = Form("m", pattern="^(m|cm|mm)$"),
    up_axis: str = Form("y", pattern="^(y|z)$"),
    fit_bed: str | None = Form(None),
    base_plate: bool = Form(True),
    depth: UploadFile | None = File(None, description="ARKit sceneDepth: float32 LE metres (or 16-bit PNG, mm)"),
    depth_width: int | None = Form(None, gt=0),
    depth_height: int | None = Form(None, gt=0),
    depth_confidence: UploadFile | None = File(None, description="ARKit confidenceMap: uint8 0/1/2"),
    intrinsics: str | None = Form(None, description='{"fx","fy","cx","cy","width","height"} in photo px'),
    depth_intrinsics: str | None = Form(None, description="depth camera intrinsics JSON"),
    depth_extrinsics: str | None = Form(None, description="16 floats, column-major depth-to-photo, OpenCV axes"),
    camera_transform: str | None = Form(None, description="16 floats, column-major camera-to-world"),
    image_orientation: int = Form(1),
    object_points: str | None = Form(None, description="[[u, v], ...] taps on the object, photo px"),
    authorization: str | None = Header(None),
):
    _check_auth(authorization)
    if (mesh is None) == (photo is None):
        raise HTTPException(400, "send exactly one of `mesh` or `photo`")
    if mode == "face" and photo is None:
        raise HTTPException(400, "mode=face requires `photo`")
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
    fusion = None

    base_thickness = 3.0
    plate_height = max(base_thickness - 0.5, 0.0) if base_plate else 0.0
    # height_mm is the object itself; prepare_mesh's height includes the plate.
    object_height = height_mm if height_mm is not None else (100 if mode == "face" else None)
    height = object_height + plate_height if object_height else None

    workdir = tempfile.mkdtemp(prefix="print_")
    try:
        if mesh is not None:
            src = _save_upload(mesh, MESH_EXTS, workdir, "mesh")
            stem = mesh.filename
            kwargs = dict(units_to_mm=img2mesh.UNITS_TO_MM[units], up_axis=up_axis)
        else:
            img = _save_upload(photo, PHOTO_EXTS, workdir, "photo")
            stem = photo.filename
            src = os.path.join(workdir, "raw.obj")
            frame = None
            if depth is not None:
                dpath = _save_upload(depth, DEPTH_EXTS, workdir, "depth")
                cpath = (_save_upload(depth_confidence, CONFIDENCE_EXTS, workdir, "confidence")
                         if depth_confidence is not None else None)
                try:
                    frame = depth_fusion.Frame.load(
                        dpath, depth_width, depth_height, cpath, intrinsics,
                        camera_transform, image_orientation, object_points,
                        depth_intrinsics, depth_extrinsics)
                except ValueError as exc:
                    raise HTTPException(400, str(exc))
            try:
                mask = None
                if frame is None:
                    reconstruct_photo(img, src)
                else:
                    photo_up = frame.depth_to_photo[:3, :3] @ frame.up_camera()
                    turns = depth_fusion.upright_turns(photo_up)
                    reconstruct_photo(img, src, quarter_turns=turns)
                    mask = reconstruct_photo.foreground_mask(img)
            except Exception as exc:
                raise HTTPException(422, f"could not reconstruct the photo: {exc}")
            kwargs = {}
            if frame is not None:
                import trimesh

                try:
                    raw = trimesh.load(src, force="mesh")
                    if mode == "face":
                        fused, fusion = depth_fusion.fuse_face(raw, frame, mask, face_to_world)
                    else:
                        fused, fusion = depth_fusion.fuse(raw, frame, mask)
                    src = os.path.join(workdir, "fused.ply")
                    fused.export(src)
                    kwargs = dict(units_to_mm=1000.0, up_axis="z")
                    if mode != "face":
                        height = None
                except depth_fusion.FusionError as exc:
                    fusion = {"skipped": str(exc)}
            if mode == "face" and (frame is None or "skipped" in (fusion or {})):
                import trimesh

                raw = trimesh.load(src, force="mesh")
                bust = depth_fusion.cut_bust(raw, raw.bounds[0, 2] + 0.2 * raw.extents[2])
                src = os.path.join(workdir, "bust.ply")
                bust.export(src)
        try:
            out, report = img2mesh.prepare_mesh(
                src, height=height, fit_bed=bed, base_plate=base_plate,
                base_thickness=base_thickness, **kwargs)
        except SystemExit as exc:
            raise HTTPException(422, str(exc))
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, f"could not process the mesh: {exc}")
        body = out.export(file_type="stl")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    ex = report["extents"]
    summary = {
        **({"mode": "face"} if mode == "face" else {}),
        "extents_mm": [round(float(v), 1) for v in ex],
        "watertight": bool(report["watertight"]),
        **({"fusion": fusion} if fusion else {}),
        "faces": int(report["faces"]),
        "volume_cm3": round(report["volume_cm3"], 1) if report["volume_cm3"] else None,
        "scale": report["scale_note"],
    }
    name = os.path.splitext(os.path.basename(stem or "model"))[0] or "model"
    return Response(body, media_type="model/stl", headers={
        "Content-Disposition": f'attachment; filename="{name}.stl"',
        "X-Print-Report": json.dumps(summary, ensure_ascii=True),
        "Access-Control-Expose-Headers": "X-Print-Report, Content-Disposition",
    })
