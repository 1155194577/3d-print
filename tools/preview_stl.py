#!/usr/bin/env python3
"""preview_stl — render an STL to three orthographic PNG views.

A quick way to eyeball a generated model before slicing: front (X-Z),
side (Y-Z) and top (X-Y), with any flat base plate at the bottom
highlighted in red. No GPU, no OpenGL — just a painter's-algorithm
rasteriser so it works over SSH and inside sandboxes.

    python tools/preview_stl.py model.stl -o model_views.png
    python tools/preview_stl.py model.stl --size 900 --no-plate
"""

import argparse
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw


def load(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    if len(mesh.faces) == 0:
        raise SystemExit(f"error: no faces in {path}")
    return mesh


def render_view(vertices, faces, shade, plate, pa, pb, pz, flip_b, title, size):
    x, y = vertices[:, pa].copy(), vertices[:, pb].copy()
    if flip_b:
        y = -y
    span = max(np.ptp(x), np.ptp(y)) * 1.1 or 1.0
    px = (x - (x.max() + x.min()) / 2) * (size / span) + size / 2
    py = size / 2 - (y - (y.max() + y.min()) / 2) * (size / span)

    img = Image.new("RGB", (size, size), (250, 250, 252))
    draw = ImageDraw.Draw(img)
    order = np.argsort(vertices[faces].mean(axis=1)[:, pz])

    for i in order:
        if plate is not None and plate[i]:
            base = np.array([205.0, 70.0, 70.0])
        else:
            base = np.array([150.0, 168.0, 196.0])
        rgb = tuple(int(c) for c in np.clip(base * shade[i], 0, 255))
        tri = faces[i]
        draw.polygon(
            [(px[tri[0]], py[tri[0]]), (px[tri[1]], py[tri[1]]), (px[tri[2]], py[tri[2]])],
            fill=rgb,
        )
    draw.text((12, 10), title, fill=(20, 20, 20))
    return img


def main():
    ap = argparse.ArgumentParser(prog="preview_stl", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stl", help="input mesh (.stl/.obj/.glb)")
    ap.add_argument("-o", "--output", default="preview.png", help="output PNG (default: preview.png)")
    ap.add_argument("--size", type=int, default=760, help="pixels per view (default 760)")
    ap.add_argument("--plate", type=float, default=3.6,
                    help="highlight faces within N mm of the floor as the base plate (default 3.6, 0 = off)")
    ap.add_argument("--view", choices=["all", "front", "side", "top"], default="all")
    args = ap.parse_args()

    mesh = load(args.stl)
    v = mesh.vertices
    v = v - v.mean(axis=0)

    normals = np.cross(v[mesh.faces[:, 1]] - v[mesh.faces[:, 0]],
                       v[mesh.faces[:, 2]] - v[mesh.faces[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0] = 1.0
    normals = normals / lengths[:, None]
    light = np.array([0.35, -0.55, 0.75])
    light /= np.linalg.norm(light)
    shade = np.clip(np.abs(normals @ light), 0.35, 1.0)

    plate = None
    if args.plate > 0:
        zmin = v[:, 2].min()
        plate = mesh.triangles_center[:, 2] < zmin + args.plate

    views = {
        "front": (0, 2, 1, False, "FRONT  (X-Z)"),
        "side": (1, 2, 0, False, "SIDE   (Y-Z)"),
        "top": (0, 1, 2, False, "TOP    (X-Y)"),
    }
    chosen = views if args.view == "all" else {args.view: views[args.view]}

    tiles = [render_view(v, mesh.faces, shade, plate, *spec, size=args.size)
             for spec in chosen.values()]

    canvas = Image.new("RGB", (sum(t.width for t in tiles), max(t.height for t in tiles)),
                       (255, 255, 255))
    x = 0
    for tile in tiles:
        canvas.paste(tile, (x, 0))
        x += tile.width
    canvas.save(args.output)
    print(f"{args.output}  ({canvas.width}x{canvas.height}, {len(mesh.faces):,} faces)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
