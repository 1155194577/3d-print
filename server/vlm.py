"""Qwen2.5-VL answering short questions about one photo, for POST /photo-check.

Needs a newer transformers than TripoSR's pin (server/requirements-vlm.txt), so the
Modal deploy runs it in its own GPU container (deploy/modal_app.py).
"""

import io
import os
import threading

VLM_MODEL = os.environ.get("PHOTO_CHECK_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
MAX_SIDE_PX = 768


def load_image(data: bytes):
    """Upright RGB (EXIF honoured), longest side at most MAX_SIDE_PX."""
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    image.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX))
    return image


class QwenVLM:
    """Kept loaded between requests; one generation at a time."""

    def __init__(self, model_id=VLM_MODEL):
        self.model_id = model_id
        self.model = None
        self.lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id, dtype=torch.bfloat16).to(self.device).eval()

    def __call__(self, jpeg: bytes, questions: dict[str, str]) -> dict[str, str]:
        """{name: question} -> {name: answer}, each asked about the photo on its own."""
        import torch

        image = load_image(jpeg)
        answers = {}
        with self.lock:
            self.load()
            for name, question in questions.items():
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
                answers[name] = self.processor.batch_decode(
                    out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return answers
