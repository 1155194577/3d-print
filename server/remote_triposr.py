"""TripoSR inference on Modal's GPU, called from the CPU-only print server.

Same interface as print_api.PhotoReconstructor. Enabled with TRIPOSR_BACKEND=modal;
the GPU side is deploy/modal_triposr.py.
"""

import io
import os
import threading

APP_NAME = os.environ.get("TRIPOSR_MODAL_APP", "3d-print-triposr")
CLASS_NAME = "TripoSR"
MASK_CACHE = 8  # only depth prints read the mask back; photo-only prints leave theirs


class ModalReconstructor:
    def __init__(self, app_name=APP_NAME, class_name=CLASS_NAME):
        self.app_name, self.class_name = app_name, class_name
        self._remote = None
        self._masks = {}
        self._lock = threading.Lock()

    def _service(self):
        if self._remote is None:
            import modal

            self._remote = modal.Cls.from_name(self.app_name, self.class_name)()
        return self._remote

    def __call__(self, image_path, out_path, mc_resolution=256, foreground_ratio=0.85,
                 quarter_turns=None, progress=None, input_path=None):
        progress = progress or (lambda *a: None)
        with open(image_path, "rb") as fh:
            image = fh.read()
        result = None
        for kind, value in self._service().reconstruct.remote_gen(
                image, os.path.splitext(image_path)[1], mc_resolution=mc_resolution,
                foreground_ratio=foreground_ratio, quarter_turns=quarter_turns, with_mask=True,
                **({"with_input": True} if input_path is not None else {})):
            if kind == "stage":
                progress(value)
            elif kind == "result":
                result = value
        if result is None:
            raise RuntimeError("TripoSR on Modal returned no mesh")
        with open(out_path, "wb") as fh:
            fh.write(result["mesh"])
        if input_path is not None and result.get("input") is not None:
            with open(input_path, "wb") as fh:
                fh.write(result["input"])
        if result.get("mask") is not None:
            with self._lock:
                self._masks[os.path.abspath(image_path)] = result["mask"]
                while len(self._masks) > MASK_CACHE:
                    self._masks.pop(next(iter(self._masks)))

    def foreground_mask(self, image_path):
        """Object mask of the photo as stored (no EXIF rotation), uint8 0-255."""
        import numpy as np
        from PIL import Image

        with self._lock:
            png = self._masks.pop(os.path.abspath(image_path), None)
        if png is None:
            with open(image_path, "rb") as fh:
                png = self._service().foreground_mask.remote(fh.read(), os.path.splitext(image_path)[1])
        return np.asarray(Image.open(io.BytesIO(png)))
