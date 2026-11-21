#!/usr/bin/env python3
"""Export TripoSR to two Core ML models with quantized weights, for on-device (offline) inference.

    python edge/export_coreml.py --model-dir <dir with config.yaml + model.ckpt> --out build/coreml
    python edge/export_coreml.py --quantization int4-block32 ...   # smaller, needs iOS 18

Writes TripoSREncoder.mlpackage (image -> triplanes, quantized) and TripoSRDecoder.mlpackage
(triplanes + 65,536 points -> density logits, float16). See edge/README.md.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triposr_edge as te  # noqa: E402

DECODER_POINTS = 65536


def quantize(mlmodel, scheme):
    import coremltools.optimize as cto

    int8 = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                              granularity="per_channel", weight_threshold=2048)
    if scheme == "int8":
        linear = int8
    elif scheme == "int4-block32":
        linear = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4",
                                                    granularity="per_block", block_size=32,
                                                    weight_threshold=2048)
    else:
        raise ValueError(scheme)
    # By consumer op: only linear/conv weights are compressed. The learned triplane tokens and
    # the position embeddings (consumed by add) keep float16, as in te.fake_quantize.
    config = cto.coreml.OptimizationConfig(
        global_config=None,
        op_type_configs={"linear": linear, "conv": int8, "conv_transpose": int8},
    )
    return cto.coreml.linear_quantize_weights(mlmodel, config)


def main():
    import coremltools as ct

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="stabilityai/TripoSR")
    ap.add_argument("--out", default="build/coreml")
    ap.add_argument("--quantization", choices=["int8", "int4-block32", "none"], default="int8")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    tsr = te.load_tsr(args.model_dir)
    os.makedirs(args.out, exist_ok=True)
    target = ct.target.iOS18 if args.quantization == "int4-block32" else ct.target.iOS17
    meta = {
        "com.github.1155194577.triposr.radius": str(te.RADIUS),
        "com.github.1155194577.triposr.density_threshold": str(te.DENSITY_THRESHOLD),
        "com.github.1155194577.triposr.quantization": args.quantization,
    }

    encoder = te.Encoder(tsr).eval()
    image = torch.rand(1, 3, te.IMAGE_SIZE, te.IMAGE_SIZE)
    traced = torch.jit.trace(encoder, image)
    enc = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=image.shape, scale=1 / 255.0,
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="triplanes", dtype=np.float32)],
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    if args.quantization != "none":
        enc = quantize(enc, args.quantization)
    enc.short_description = ("TripoSR image -> triplanes. Input: 512x512 RGB, object centred at 85% of "
                             "the frame on a 50% grey background.")
    enc.user_defined_metadata.update(meta)
    enc.save(os.path.join(args.out, "TripoSREncoder.mlpackage"))

    decoder = te.Decoder(tsr).eval()
    planes = torch.randn(3, 40, 64, 64)
    points = (torch.rand(DECODER_POINTS, 3) * 2 - 1) * te.RADIUS
    traced = torch.jit.trace(decoder, (planes, points))
    dec = ct.convert(
        traced,
        inputs=[ct.TensorType(name="triplanes", shape=planes.shape, dtype=np.float32),
                ct.TensorType(name="points", shape=points.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="density_logit", dtype=np.float32)],
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    dec.short_description = (f"TripoSR triplanes + {DECODER_POINTS} points in [-{te.RADIUS}, {te.RADIUS}]^3 -> "
                             f"density logit; density = exp(logit), surface at density {te.DENSITY_THRESHOLD:g}.")
    dec.user_defined_metadata.update(meta)
    dec.save(os.path.join(args.out, "TripoSRDecoder.mlpackage"))

    for name in ("TripoSREncoder", "TripoSRDecoder"):
        path = os.path.join(args.out, f"{name}.mlpackage")
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
        print(f"{path}: {size / 2**20:.0f} MB")


if __name__ == "__main__":
    main()
