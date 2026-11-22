#!/usr/bin/env python3
"""Compare TripoSR meshes under each weight quantization against the float32 model.

    python edge/evaluate.py --model-dir <dir with config.yaml + model.ckpt> img1.png img2.jpg ...

For every image and scheme it reports the triplane error, the occupancy IoU of the density
grid at --resolution, and the Chamfer distance between the meshes as % of the float32
mesh's bounding-box diagonal. `+fp16act` runs the whole model in float16, like Core ML on
the GPU / Neural Engine.
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triposr_edge as te  # noqa: E402

SCHEMES = ["fp32", "fp16", "int8", "int4-block32", "int8+fp16act", "int4-block32+fp16act"]


def chamfer_pct(a, b, n=30000, seed=0):
    from scipy.spatial import cKDTree

    pa = a.sample(n, seed=seed) if hasattr(a, "sample") else a
    pb = b.sample(n, seed=seed)
    d = (cKDTree(pb).query(pa)[0].mean() + cKDTree(pa).query(pb)[0].mean()) / 2
    return 100 * d / np.linalg.norm(a.bounds[1] - a.bounds[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model-dir", default="stabilityai/TripoSR")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--schemes", nargs="+", default=SCHEMES)
    ap.add_argument("--out", default=None, help="write the results as JSON here")
    ap.add_argument("--save-meshes", default=None, help="directory to write each mesh as .obj")
    args = ap.parse_args()

    import rembg

    torch.set_grad_enabled(False)
    base = te.load_tsr(args.model_dir)
    session = rembg.new_session()
    images = {p: te.preprocess(p, session) for p in dict.fromkeys(args.images)}

    models = {}
    # fp32 runs first: it is the reference every other scheme is compared with.
    for scheme in ["fp32"] + [s for s in args.schemes if s != "fp32"]:
        weights, _, act = scheme.partition("+")
        m = base if weights == "fp32" else copy.deepcopy(base)
        if weights != "fp32":
            te.fake_quantize(m, weights)
        enc, dec = te.Encoder(m).eval(), te.Decoder(m).eval()
        if act == "fp16act":
            enc, dec = enc.half(), dec.half()
        models[scheme] = (enc, dec, torch.float16 if act else torch.float32)

    rows = []
    for index, (name, img) in enumerate(images.items()):
        ref = None
        for scheme, (enc, dec, dtype) in models.items():
            t0 = time.time()
            planes = enc(img.to(dtype))
            t1 = time.time()
            logits = te.density_logits(dec, planes, args.resolution)
            t2 = time.time()
            mesh = te.mesh_from_logits(logits)
            occ = logits.numpy() > np.log(te.DENSITY_THRESHOLD)
            finite = bool(torch.isfinite(planes).all() and torch.isfinite(logits).all())
            if ref is None:
                ref = (planes.float(), occ, mesh)
            row = {
                "image": name,
                "scheme": scheme,
                "triplane_rel_err": float((planes.float() - ref[0]).norm() / ref[0].norm()),
                "occupancy_iou": float((occ & ref[1]).sum() / max(1, (occ | ref[1]).sum())),
                "chamfer_pct_diag": float(chamfer_pct(ref[2], mesh)) if len(mesh.faces) else float("nan"),
                "faces": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
                "finite": finite,
                "encoder_s": round(t1 - t0, 2),
                "decoder_s": round(t2 - t1, 2),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            if args.save_meshes:
                os.makedirs(args.save_meshes, exist_ok=True)
                mesh.export(os.path.join(args.save_meshes, f"{index:02d}_{os.path.splitext(os.path.basename(name))[0]}_{scheme}.obj"))

    print("\nscheme                  IoU     chamfer%  triplane err   weights")
    for scheme in args.schemes:
        r = [x for x in rows if x["scheme"] == scheme]
        weights = scheme.partition("+")[0]
        mb = te.weight_bytes(base, weights) / 2**20
        print(f"{scheme:22s} {np.mean([x['occupancy_iou'] for x in r]):.4f}  "
              f"{np.nanmean([x['chamfer_pct_diag'] for x in r]):8.3f}  "
              f"{np.mean([x['triplane_rel_err'] for x in r]):12.4f}   {mb:7.0f} MB")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
