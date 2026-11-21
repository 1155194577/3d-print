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
import img2mesh  # noqa: E402

MAX_UPLOAD_BYTES = int(os.environ.get("PRINT_MAX_UPLOAD_MB", "200")) * 1024 * 1024
PHOTO_EXTS = (".jpg", ".jpeg", ".png", ".webp")
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

    def __call__(self, image_path, out_path, mc_resolution=256, foreground_ratio=0.85):
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        with self.lock:
            self.load()
            from tsr.utils import remove_background, resize_foreground

            # iPhone JPEGs are stored sideways with an EXIF rotation tag.
            image = ImageOps.exif_transpose(Image.open(image_path))
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
    height_mm: float | None = Form(None, gt=0, description="measured object height, mm"),
    units: str = Form("m", pattern="^(m|cm|mm)$"),
    up_axis: str = Form("y", pattern="^(y|z)$"),
    fit_bed: str | None = Form(None),
    base_plate: bool = Form(True),
    authorization: str | None = Header(None),
):
    _check_auth(authorization)
    if (mesh is None) == (photo is None):
        raise HTTPException(400, "send exactly one of `mesh` or `photo`")
    bed = _parse_bed(fit_bed)

    base_thickness = 3.0
    plate_height = max(base_thickness - 0.5, 0.0) if base_plate else 0.0
    # height_mm is the object itself; prepare_mesh's height includes the plate.
    height = height_mm + plate_height if height_mm else None

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
            try:
                reconstruct_photo(img, src)
            except Exception as exc:
                raise HTTPException(422, f"could not reconstruct the photo: {exc}")
            kwargs = {}
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
        "extents_mm": [round(float(v), 1) for v in ex],
        "watertight": bool(report["watertight"]),
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
