"""TripoSR split into two on-device models, weight quantization, and a reference pipeline.

The phone runs TripoSR as two Core ML models plus marching cubes:

  encoder  image (1x3x512x512, RGB in [0, 1], background composited onto 0.5 grey)
           -> triplanes (3x40x64x64)
  decoder  triplanes + points (Nx3, in [-0.87, 0.87]) -> density_logit (Nx1)

density = exp(density_logit); the surface is density == 25 (TripoSR's default threshold).
The decoder returns the logit rather than the density because exp() overflows float16
past ~11 and Core ML computes in float16 on the GPU and Neural Engine.

`reference_mesh` is the same pipeline in PyTorch, which the evaluation compares across
weight quantizations (`fake_quantize`) to pick what ships.
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research", "triposr"))

IMAGE_SIZE = 512
RADIUS = 0.87
DENSITY_THRESHOLD = 25.0
FOREGROUND_RATIO = 0.85


def load_tsr(model_dir_or_repo="stabilityai/TripoSR"):
    try:
        import torchmcubes  # noqa: F401
    except ImportError:  # only TSR.extract_mesh needs it; mesh_from_logits uses scikit-image
        import types
        sys.modules["torchmcubes"] = types.SimpleNamespace(marching_cubes=None)
    from tsr.system import TSR

    model = TSR.from_pretrained(model_dir_or_repo, config_name="config.yaml", weight_name="model.ckpt")
    model.eval()
    assert model.renderer.cfg.radius == RADIUS and model.cfg.cond_image_size == IMAGE_SIZE
    return model


def preprocess(path, rembg_session=None):
    """Same input TripoSR's run.py builds: cut out, centre at 85 % of the frame, grey background."""
    import rembg
    from PIL import Image
    from tsr.utils import remove_background, resize_foreground

    image = remove_background(Image.open(path), rembg_session)
    image = resize_foreground(image, FOREGROUND_RATIO)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    rgba = np.asarray(image).astype(np.float32) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:4] + (1 - rgba[..., 3:4]) * 0.5
    return torch.from_numpy(rgb).permute(2, 0, 1)[None].contiguous()


class Encoder(nn.Module):
    """Image -> triplanes: DINO ViT-B/16, the 16-layer triplane transformer and the upsampler."""

    def __init__(self, tsr):
        super().__init__()
        self.image_tokenizer = tsr.image_tokenizer
        self.tokenizer = tsr.tokenizer
        self.backbone = tsr.backbone
        self.post_processor = tsr.post_processor
        vit = self.image_tokenizer.model
        emb = vit.embeddings
        # Bake the 224 -> 512 position-embedding interpolation (bicubic, unsupported by Core ML)
        # into a constant so the traced graph has none.
        with torch.no_grad():
            n_patches = (IMAGE_SIZE // vit.config.patch_size) ** 2
            dummy = torch.zeros(1, n_patches + 1, vit.config.hidden_size)
            pos = emb.interpolate_pos_encoding(dummy, IMAGE_SIZE, IMAGE_SIZE)
        self.register_buffer("vit_pos", pos.clone(), persistent=False)

    def forward(self, image):
        vit = self.image_tokenizer.model
        x = (image - self.image_tokenizer.image_mean[0]) / self.image_tokenizer.image_std[0]
        emb = vit.embeddings
        x = emb.patch_embeddings(x, interpolate_pos_encoding=True)
        cls = emb.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.vit_pos
        x = vit.encoder(x).last_hidden_state
        image_tokens = vit.layernorm(x)
        tokens = self.tokenizer(image.shape[0])
        tokens = self.backbone(tokens, encoder_hidden_states=image_tokens)
        return self.post_processor(self.tokenizer.detokenize(tokens))[0]


class Decoder(nn.Module):
    """Triplanes + points -> density logit (density = exp(logit))."""

    def __init__(self, tsr):
        super().__init__()
        self.mlp = tsr.decoder.layers
        self.density_bias = float(tsr.renderer.cfg.density_bias)
        assert tsr.renderer.cfg.density_activation == "exp"
        assert tsr.renderer.cfg.feature_reduction == "concat"

    def forward(self, triplanes, points):
        x = points / RADIUS
        px, py, pz = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        grid = torch.stack(
            (torch.cat((px, py), 1), torch.cat((px, pz), 1), torch.cat((py, pz), 1)), 0
        )  # 3 x N x 2
        feats = F.grid_sample(triplanes, grid.unsqueeze(1), mode="bilinear",
                              padding_mode="zeros", align_corners=False)  # 3 x C x 1 x N
        feats = feats.squeeze(2).permute(2, 0, 1).reshape(points.shape[0], -1)  # N x 3C
        return self.mlp(feats)[:, 0:1] + self.density_bias


def grid_points(resolution):
    """Grid vertices in [-RADIUS, RADIUS]^3, x-major, as TripoSR's MarchingCubeHelper builds them."""
    t = torch.linspace(-RADIUS, RADIUS, resolution)
    x, y, z = torch.meshgrid(t, t, t, indexing="ij")
    return torch.stack((x, y, z), -1).reshape(-1, 3)


@torch.no_grad()
def density_logits(decoder, triplanes, resolution, chunk=262144):
    pts = grid_points(resolution).to(triplanes.dtype)
    out = [decoder(triplanes, pts[i:i + chunk]).float() for i in range(0, len(pts), chunk)]
    return torch.cat(out).reshape(resolution, resolution, resolution)


def mesh_from_logits(logits, threshold=DENSITY_THRESHOLD):
    """Marching cubes on density - threshold (TripoSR's level set), vertices in model units."""
    import trimesh
    from skimage.measure import marching_cubes

    level = torch.exp(logits.double()).numpy() - threshold
    res = level.shape[0]
    verts, faces, _, _ = marching_cubes(level, 0.0)
    verts = verts / (res - 1) * (2 * RADIUS) - RADIUS
    # skimage orients faces the other way from torchmcubes for a field that is positive inside.
    return trimesh.Trimesh(vertices=verts, faces=faces[:, ::-1], process=False)


def quantizable(module):
    """Encoder weights the Core ML export compresses: linear/conv with >= 2048 elements.

    The decoder MLP (45 K weights) stays float16.
    """
    for name, m in module.named_modules():
        if name.startswith("decoder"):
            continue
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)) and m.weight.numel() >= 2048:
            yield name, m


@torch.no_grad()
def fake_quantize(module, scheme):
    """Round weights in place the way coremltools' weight compression does, to measure quality in PyTorch.

    int8          per-output-channel symmetric int8 (linear_quantize_weights, iOS 16+)
    int4-block32  symmetric int4 with one scale per 32 input weights on linear layers, int8 on
                  the two convolutions (iOS 18+)
    """
    if scheme == "fp16":
        for _, m in quantizable(module):
            m.weight.copy_(m.weight.half().float())
        return
    for _, m in quantizable(module):
        w = m.weight
        is_conv = isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
        bits, block = (8, None) if is_conv else ({"int8": (8, None), "int4-block32": (4, 32)}[scheme])
        qmax = 2 ** (bits - 1) - 1
        # coremltools treats axis 0 as the output channel for linear/conv, axis 1 for conv_transpose.
        wt = w.transpose(0, 1) if isinstance(m, nn.ConvTranspose2d) else w
        flat = wt.reshape(wt.shape[0], -1)
        if block:
            assert flat.shape[1] % block == 0, (m, flat.shape)
            g = flat.reshape(flat.shape[0], -1, block)
        else:
            g = flat.unsqueeze(1)
        scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
        q = torch.clamp(torch.round(g / scale), -qmax - 1, qmax) * scale
        q = q.reshape(wt.shape)
        w.copy_(q.transpose(0, 1) if isinstance(m, nn.ConvTranspose2d) else q)


def weight_bytes(module, scheme):
    total = 0
    for p in module.state_dict().values():
        total += p.numel() * 2  # float16
    if scheme in ("fp32",):
        return total * 2
    q = sum(m.weight.numel() for _, m in quantizable(module))
    bits = {"fp16": 16, "int8": 8, "int4-block32": 4}[scheme]
    scales = 0
    if scheme == "int8":
        scales = sum(m.weight.shape[0] for _, m in quantizable(module)) * 2
    elif scheme == "int4-block32":
        scales = q // 32 * 2
    return total - q * 2 + q * bits // 8 + scales
