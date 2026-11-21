"""Deploy POST /print, POST /photo-check (and GET /api/health) on Modal (serverless GPU).

Only the synchronous routes are exposed, behind PRINT_API_TOKEN; the browser
uploader and its in-memory /api/jobs queue stay local-only. /photo-check's VLM
needs a newer transformers than TripoSR, so it runs in its own GPU container
(PhotoCheckVLM) that the web function calls.

    modal secret create 3d-print-api PRINT_API_TOKEN=<random token>
    modal deploy deploy/modal_app.py

Prints the URL, https://<workspace>--3d-print-web.modal.run. Model weights
are baked into the image, so a cold start only loads them onto the GPU.
"""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = "/root/app"
MODELS = "/models"
SKIP = ["*.pyc", "__pycache__"]


def download_weights():
    import rembg
    from huggingface_hub import hf_hub_download

    for name in ("config.yaml", "model.ckpt"):
        hf_hub_download("stabilityai/TripoSR", name)
    rembg.new_session()


image = (
    # CUDA devel base (matching torch's cu13.2 wheels) so torchmcubes builds with GPU support.
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "libgl1", "libglib2.0-0")
    .env({"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "8.9", "CC": "gcc", "CXX": "g++"})
    .pip_install_from_requirements(str(ROOT / "research" / "triposr_requirements.txt"))
    .pip_install_from_requirements(str(ROOT / "server" / "requirements-web.txt"))
    .run_commands("pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git")
    .env({"HF_HOME": f"{MODELS}/hf", "U2NET_HOME": f"{MODELS}/rembg",
          "TOKENIZERS_PARALLELISM": "false", "PYTHONWARNINGS": "ignore",
          "TRIPOSR_PYTHON": "/usr/local/bin/python", "TRIPOSR_JOBS_DIR": "/tmp/jobs"})
    .run_function(download_weights)
    .add_local_dir(ROOT / "bin", f"{APP_DIR}/bin")
    .add_local_dir(ROOT / "tools", f"{APP_DIR}/tools", ignore=SKIP)
    .add_local_dir(ROOT / "server", f"{APP_DIR}/server", ignore=SKIP + ["jobs", "tests"])
    .add_local_dir(ROOT / "research" / "triposr", f"{APP_DIR}/research/triposr",
                   ignore=SKIP + [".git", "examples", "figures"])
)


def download_vlm():
    from huggingface_hub import snapshot_download

    from server.vlm import VLM_MODEL

    snapshot_download(VLM_MODEL)


vlm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "server" / "requirements-vlm.txt"))
    .env({"HF_HOME": f"{MODELS}/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONWARNINGS": "ignore",
          "PYTHONPATH": APP_DIR})
    .add_local_file(ROOT / "server" / "vlm.py", f"{APP_DIR}/server/vlm.py", copy=True)
    .add_local_file(ROOT / "server" / "__init__.py", f"{APP_DIR}/server/__init__.py", copy=True)
    .run_function(download_vlm)
)

app = modal.App("3d-print", image=image)


@app.cls(image=vlm_image, gpu="L4", timeout=120, scaledown_window=300)
@modal.concurrent(max_inputs=4)
class PhotoCheckVLM:
    @modal.enter()
    def load(self):
        from server.vlm import QwenVLM

        self.vlm = QwenVLM()
        self.vlm.load()

    @modal.method()
    def ask(self, jpeg: bytes, questions: dict) -> dict:
        return self.vlm(jpeg, questions)


@app.function(
    gpu="L4",
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("3d-print-api")],
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, APP_DIR)
    from server import photo_check
    from server.print_api import create_print_app, reconstruct_photo
    from server import print_jobs

    print_jobs.store = print_jobs.ModalDictStore(
        modal.Dict.from_name("3d-print-jobs", create_if_missing=True))

    reconstruct_photo.load()
    photo_check.vlm = PhotoCheckVLM().ask.remote
    return create_print_app()
