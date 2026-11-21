"""FastAPI service: photo in, printable STL out.

Wraps the ``bin/triposr`` pipeline in a single-worker job queue. TripoSR loads a
~1.5 GB model onto the GPU per invocation, so running jobs in parallel would only
thrash unified memory — one job at a time, queued, is both faster and safer.

Endpoints
    GET  /                            single-page uploader
    POST /api/jobs                    multipart image upload  -> 202 {id}
    GET  /api/jobs/{id}               current job state
    GET  /api/jobs/{id}/events        Server-Sent Events progress stream
    GET  /api/jobs/{id}/model         finished STL
    GET  /api/jobs/{id}/preview.png   three-view render of the result
    GET  /api/health                  liveness + pipeline availability

Configuration (all optional, via environment)
    TRIPOSR_BIN       path to the triposr CLI        (default: <repo>/bin/triposr)
    TRIPOSR_JOBS_DIR  where uploads and outputs live (default: server/jobs)
    TRIPOSR_MAX_UPLOAD_MB  upload cap                (default: 20)
    TRIPOSR_JOB_TTL_MIN    minutes before a job is pruned (default: 120)
    TRIPOSR_TIMEOUT_S      hard limit per generation (default: 900)
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TERMINAL = {"done", "error"}

_DIMS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*mm")
_VOLUME_RE = re.compile(r"volume:\s*([\d,]+)\s*cm³")
_FACES_RE = re.compile(r"([\d,]+)\s*faces")


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, injected so tests never touch the real pipeline."""

    triposr_bin: Path = REPO_ROOT / "bin" / "triposr"
    jobs_dir: Path = Path(__file__).resolve().parent / "jobs"
    preview_script: Path = REPO_ROOT / "tools" / "preview_stl.py"
    max_upload_bytes: int = 20 * 1024 * 1024
    job_ttl_s: int = 120 * 60
    timeout_s: int = 900

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            triposr_bin=Path(os.environ.get("TRIPOSR_BIN", REPO_ROOT / "bin" / "triposr")),
            jobs_dir=Path(os.environ.get("TRIPOSR_JOBS_DIR", Path(__file__).resolve().parent / "jobs")),
            max_upload_bytes=int(float(os.environ.get("TRIPOSR_MAX_UPLOAD_MB", "20")) * 1024 * 1024),
            job_ttl_s=int(os.environ.get("TRIPOSR_JOB_TTL_MIN", "120")) * 60,
            timeout_s=int(os.environ.get("TRIPOSR_TIMEOUT_S", "900")),
        )


@dataclass
class Job:
    """Mutable job state; every mutation happens under the manager's lock."""

    id: str
    filename: str
    status: str = "queued"          # queued | running | done | error
    stage: str = "queued"
    progress: int = 0
    message: str = "waiting for the GPU"
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    finished: float | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    input_path: Path | None = None
    output_path: Path | None = None
    preview_path: Path | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "metrics": self.metrics,
            "log": self.log[-8:],
            "elapsed_s": round((self.finished or time.time()) - self.created, 1),
            "model_url": f"/api/jobs/{self.id}/model" if self.status == "done" else None,
            "preview_url": (
                f"/api/jobs/{self.id}/preview.png"
                if self.status == "done" and self.preview_path
                else None
            ),
        }


def stage_for_line(line: str) -> tuple[str, int] | None:
    """Map a pipeline log line to (stage, percent). ``None`` = not a milestone.

    Order matters: the "…finished" variants must be tested before their parents.
    """
    low = line.lower()
    if "triposr failed" in low or "traceback" in low:
        return ("failed", 0)
    if "triposr:" in low:
        return ("generating", 5)
    if "initializing model finished" in low:
        return ("loading model", 15)
    if "initializing model" in low:
        return ("loading model", 10)
    if "processing images finished" in low:
        return ("preparing image", 40)
    if "processing images" in low:
        return ("preparing image", 22)
    if "running model finished" in low:
        return ("generating geometry", 70)
    if "running model" in low:
        return ("generating geometry", 45)
    if "extracting mesh finished" in low:
        return ("extracting mesh", 85)
    if "extracting mesh" in low:
        return ("extracting mesh", 78)
    if "exporting mesh finished" in low:
        return ("exporting", 92)
    if "exporting mesh" in low:
        return ("exporting", 88)
    if "watertight:" in low:
        return ("finalising", 95)
    return None


def metrics_from_log(lines: list[str]) -> dict[str, Any]:
    """Pull the human-readable report values out of the pipeline output."""
    metrics: dict[str, Any] = {}
    for line in lines:
        if "watertight" in line.lower():
            metrics["watertight"] = "✓ yes" in line or "yes" in line.lower().split("watertight")[-1]
        m = _DIMS_RE.search(line)
        if m:
            metrics["dimensions_mm"] = [float(g) for g in m.groups()]
        v = _VOLUME_RE.search(line)
        if v:
            metrics["volume_cm3"] = int(v.group(1).replace(",", ""))
        f = _FACES_RE.search(line)
        if f and "faces" not in metrics:
            metrics["faces"] = int(f.group(1).replace(",", ""))
    return metrics


class JobManager:
    """Single background worker draining a FIFO queue of jobs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._prune()
        threading.Thread(target=self._worker, name="triposr-worker", daemon=True).start()

    # ---------------------------------------------------------------- public
    def create(self, filename: str, suffix: str) -> tuple[Job, Path]:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.settings.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        dest = job_dir / f"input{suffix}"
        job = Job(id=job_id, filename=filename, input_path=dest)
        with self._lock:
            self._jobs[job_id] = job
        return job, dest

    def enqueue(self, job: Job) -> None:
        """Queue a job *after* its input file is complete on disk."""
        self._queue.put(job.id)

    def reject(self, job: Job, message: str) -> None:
        """Drop a job whose upload failed validation, removing its directory."""
        self._fail(job, message)
        if job.input_path is not None:
            shutil.rmtree(job.input_path.parent, ignore_errors=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def queue_depth(self) -> int:
        return self._queue.qsize()

    # --------------------------------------------------------------- worker
    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            try:
                self._run(job)
            except Exception as exc:  # never let the worker die
                self._fail(job, f"internal error: {exc}")
            finally:
                self._queue.task_done()

    def _run(self, job: Job) -> None:
        assert job.input_path is not None
        job_dir = job.input_path.parent
        out_path = job_dir / "model.stl"
        self._update(job, status="running", stage="starting", progress=3,
                     message="spawning TripoSR")

        if not self.settings.triposr_bin.exists():
            self._fail(job, f"pipeline not found at {self.settings.triposr_bin}")
            return

        cmd = [str(self.settings.triposr_bin), str(job.input_path), "-o", str(out_path)]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore"}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        deadline = time.time() + self.settings.timeout_s
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._ingest(job, line)
                if time.time() > deadline:
                    proc.kill()
                    self._fail(job, f"timed out after {self.settings.timeout_s}s")
                    return
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()

        if code != 0 or not out_path.exists():
            tail = "\n".join(job.log[-6:]) or "no output"
            self._fail(job, f"pipeline exited with code {code}\n{tail}")
            return

        self._update(job, stage="rendering preview", progress=96, message="rendering preview")
        preview = self._render_preview(out_path, job_dir)
        with self._lock:
            job.output_path = out_path
            job.preview_path = preview
        self._update(job, status="done", stage="done", progress=100,
                     message="model ready", finished=time.time())

    def _ingest(self, job: Job, line: str) -> None:
        text = line.rstrip("\n")
        with self._lock:
            job.log.append(text)
            if len(job.log) > 200:
                del job.log[:-200]
        milestone = stage_for_line(text)
        if milestone:
            stage, progress = milestone
            if progress:
                self._update(job, stage=stage, progress=max(job.progress, progress),
                             message=stage)
        metrics = metrics_from_log([text])
        if metrics:
            with self._lock:
                job.metrics.update(metrics)

    def _render_preview(self, stl: Path, job_dir: Path) -> Path | None:
        out = job_dir / "preview.png"
        try:
            subprocess.run(
                [sys.executable, str(self.settings.preview_script), str(stl),
                 "-o", str(out), "--size", "700"],
                check=True, capture_output=True, timeout=180,
            )
            return out if out.exists() else None
        except Exception:
            return None  # preview is a nicety; never fail the job for it

    # ---------------------------------------------------------------- state
    def _update(self, job: Job, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated = time.time()

    def _fail(self, job: Job, message: str) -> None:
        self._update(job, status="error", stage="failed", message=message.splitlines()[0],
                     error=message, finished=time.time())
        # keep only the inputs' job dir tidy if generation never produced output
        if job.input_path and not job.output_path:
            for stray in job.input_path.parent.glob("*.stl"):
                stray.unlink(missing_ok=True)

    def _prune(self) -> None:
        cutoff = time.time() - self.settings.job_ttl_s
        with self._lock:
            stale = [j for j in self._jobs.values() if j.updated < cutoff]
            for job in stale:
                self._jobs.pop(job.id, None)
        for job_dir in self.settings.jobs_dir.iterdir() if self.settings.jobs_dir.exists() else []:
            try:
                if job_dir.is_dir() and job_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(job_dir, ignore_errors=True)
            except OSError:
                continue


def _sniff_image(head: bytes) -> bool:
    """Cheap magic-number check so a renamed .exe never reaches the pipeline."""
    return (
        head.startswith(b"\xff\xd8\xff")            # JPEG
        or head.startswith(b"\x89PNG\r\n\x1a\n")    # PNG
        or (head.startswith(b"RIFF") and head[8:12] == b"WEBP")
        or head.startswith(b"BM")                   # BMP
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    manager = JobManager(settings)

    app = FastAPI(title="3d-print", description="Photo → printable STL", version="1.0.0")
    app.state.settings = settings
    app.state.manager = manager

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(500, "index.html missing")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "pipeline": str(settings.triposr_bin),
            "pipeline_ready": settings.triposr_bin.exists(),
            "queued": manager.queue_depth(),
            "jobs": len(manager.all()),
        }

    @app.post("/api/jobs", status_code=202)
    async def create_job(image: UploadFile = File(...)) -> dict[str, Any]:
        suffix = Path(image.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                415, f"unsupported file type {suffix or '(none)'}; "
                     f"convert to one of {', '.join(sorted(ALLOWED_SUFFIXES))}"
            )
        job, dest = manager.create(image.filename or f"upload{suffix}", suffix)
        size = 0
        head = b""
        try:
            with dest.open("wb") as fh:
                while chunk := await image.read(1 << 20):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(
                            413, f"file larger than {settings.max_upload_bytes // (1024 * 1024)} MB")
                    if len(head) < 16:
                        head += chunk[: 16 - len(head)]
                    fh.write(chunk)
            if size == 0:
                raise HTTPException(400, "empty upload")
            if not _sniff_image(head):
                raise HTTPException(415, "file does not look like a JPEG/PNG/WebP/BMP image")
        except HTTPException as exc:
            manager.reject(job, f"upload rejected: {exc.detail}")
            raise
        manager.enqueue(job)
        return {"id": job.id, "status": job.status,
                "events_url": f"/api/jobs/{job.id}/events"}

    @app.get("/api/jobs/{job_id}")
    async def job_state(job_id: str) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.snapshot()

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")

        async def stream():
            last = None
            while True:
                payload = json.dumps(job.snapshot(), ensure_ascii=False)
                if payload != last:
                    yield f"data: {payload}\n\n"
                    last = payload
                if job.status in TERMINAL:
                    break
                await asyncio.sleep(0.1)
            yield "event: end\ndata: {}\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/jobs/{job_id}/model")
    async def job_model(job_id: str) -> FileResponse:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if job.status != "done" or not job.output_path or not job.output_path.exists():
            raise HTTPException(409, "model not ready")
        name = f"{Path(job.filename).stem or 'model'}.stl"
        return FileResponse(job.output_path, media_type="model/stl", filename=name)

    @app.get("/api/jobs/{job_id}/preview.png")
    async def job_preview(job_id: str) -> FileResponse:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        if not job.preview_path or not job.preview_path.exists():
            raise HTTPException(404, "no preview for this job")
        return FileResponse(job.preview_path, media_type="image/png")

    return app


app = create_app()
