"""Deploy the /print API on Modal (serverless GPU).

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
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "libgl1", "libglib2.0-0")
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

app = modal.App("3d-print", image=image)


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
    from server.app import app as api
    from server.print_api import reconstruct_photo

    reconstruct_photo.load()
    return api
