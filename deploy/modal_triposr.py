"""TripoSR + rembg inference on a Modal GPU: the only part of /print that runs on Modal.

Everything else (auth, uploads, job state, fusion, repair, printability, S3) runs on
the print server (deploy/box), which calls this with TRIPOSR_BACKEND=modal.

    modal deploy deploy/modal_triposr.py
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
          "TOKENIZERS_PARALLELISM": "false", "PYTHONWARNINGS": "ignore"})
    .run_function(download_weights)
    .add_local_dir(ROOT / "tools", f"{APP_DIR}/tools", ignore=SKIP)
    .add_local_dir(ROOT / "server", f"{APP_DIR}/server", ignore=SKIP + ["jobs", "tests"])
    .add_local_dir(ROOT / "research" / "triposr", f"{APP_DIR}/research/triposr",
                   ignore=SKIP + [".git", "examples", "figures"])
)

app = modal.App("3d-print-triposr", image=image)


def _write(directory, data, suffix):
    path = Path(directory) / f"photo{suffix}"
    path.write_bytes(data)
    return str(path)


def _png(mask):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return buf.getvalue()


@app.cls(gpu="L4", timeout=600, scaledown_window=300)
class TripoSR:
    @modal.enter()
    def load(self):
        import sys

        sys.path.insert(0, APP_DIR)
        from server.print_api import PhotoReconstructor

        self.model = PhotoReconstructor()
        self.model.load()

    @modal.method()
    def reconstruct(self, image, suffix, mc_resolution=256, foreground_ratio=0.85,
                    quarter_turns=None, with_mask=False, with_input=False):
        """Yields ("stage", name) as it goes, then ("result", {"mesh": obj bytes, "mask": png|None})."""
        import queue
        import tempfile
        import threading

        events = queue.Queue()
        with tempfile.TemporaryDirectory() as directory:
            src, out = _write(directory, image, suffix), str(Path(directory) / "raw.obj")
            input_path = str(Path(directory) / "input.png") if with_input else None

            def work():
                try:
                    self.model(src, out, mc_resolution=mc_resolution, foreground_ratio=foreground_ratio,
                               quarter_turns=quarter_turns, progress=lambda s: events.put(("stage", s)),
                               input_path=input_path)
                    mask = _png(self.model.foreground_mask(src)) if with_mask else None
                    events.put(("result", {"mesh": Path(out).read_bytes(), "mask": mask,
                                           "input": Path(input_path).read_bytes() if input_path else None}))
                except BaseException as exc:
                    events.put(("error", exc))

            threading.Thread(target=work, daemon=True).start()
            while True:
                kind, value = events.get()
                if kind == "error":
                    raise value
                yield kind, value
                if kind == "result":
                    return

    @modal.method()
    def foreground_mask(self, image, suffix):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            return _png(self.model.foreground_mask(_write(directory, image, suffix)))
