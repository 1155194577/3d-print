#!/usr/bin/env python3
"""
img2mesh — turn any photo into a printable 3D model.

Runs TripoSR (single-image → 3D) and then prepares the mesh for printing:
scale to real mm, flatten the base, add a printable base plate, repair, export.

Invoked through the `triposr` wrapper in bin/, which sets the model caches
and the Python 3.11 venv. You can also run it directly:

    research/triposr_env311/bin/python tools/img2mesh.py photo.jpg --height 120
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRIPOSR_DIR = os.path.join(ROOT, "research", "triposr")
TRIPOSR_PY = sys.executable  # the wrapper puts us on the right interpreter


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def run_triposr(images, workdir, device, mc_res, texture, tex_res, no_remove_bg,
                foreground_ratio, chunk_size):
    """Call TripoSR's run.py as a subprocess (it does work at import time)."""
    cmd = [
        TRIPOSR_PY, os.path.join(TRIPOSR_DIR, "run.py"),
        *images,
        "--output-dir", workdir,
        "--device", device,
        "--mc-resolution", str(mc_res),
        "--chunk-size", str(chunk_size),
        "--foreground-ratio", str(foreground_ratio),
    ]
    if texture:
        cmd += ["--bake-texture", "--texture-resolution", str(tex_res)]
    if no_remove_bg:
        cmd += ["--no-remove-bg"]

    log(f"TripoSR: {len(images)} image(s) → 3D  (device={device}, mc-res={mc_res})")
    # Stream child output, but keep a ring buffer so a failure is never silent.
    from collections import deque
    tail = deque(maxlen=40)
    proc = subprocess.Popen(cmd, cwd=TRIPOSR_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        tail.append(line.rstrip())
        if any(k in line for k in ("Initializing", "Processing", "Running",
                                   "Extracting", "Exporting")):
            log(line.strip())
    code = proc.wait()
    if code != 0:
        print("\n--- TripoSR output (last lines) ---", file=sys.stderr)
        for line in tail:
            print(line, file=sys.stderr)
        raise SystemExit(f"TripoSR failed with exit code {code}")


def find_mesh(workdir, index):
    """TripoSR writes <output-dir>/<index>/mesh.obj (or .glb)."""
    for ext in ("obj", "glb"):
        hits = sorted(glob.glob(os.path.join(workdir, str(index), f"mesh.{ext}")))
        if hits:
            return hits[0]
    return None


USD_EXTS = (".usdz", ".usd", ".usdc", ".usda")
MESH_EXTS = (".obj", ".ply", ".usdz", ".usd", ".usdc", ".usda", ".glb", ".gltf", ".stl", ".off")
UNITS_TO_MM = {"m": 1000.0, "cm": 10.0, "mm": 1.0}


def is_mesh_file(path):
    return os.path.splitext(path)[1].lower() in MESH_EXTS


def _load_usd(path):
    """Read every UsdGeom.Mesh in a USD/USDZ stage, in metres, Z-up."""
    import numpy as np
    import trimesh
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        raise SystemExit("USD/USDZ input needs the usd-core package (pip install usd-core)")

    stage = Usd.Stage.Open(path)
    if stage is None:
        raise SystemExit(f"Could not open USD file: {path}")
    metres = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    cache = UsdGeom.XformCache()
    parts = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        xf = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        pts = pts @ xf[:3, :3] + xf[3, :3]  # USD matrices are row-vector
        idx = np.asarray(idx, dtype=np.int64)
        faces, start = [], 0
        for c in counts:  # fan-triangulate quads/ngons
            for k in range(1, c - 1):
                faces.append((idx[start], idx[start + k], idx[start + k + 1]))
            start += c
        parts.append(trimesh.Trimesh(vertices=pts, faces=faces, process=False))
    if not parts:
        raise SystemExit(f"No mesh geometry in {path}")
    m = trimesh.util.concatenate(parts)
    m.apply_scale(metres)
    up = UsdGeom.GetStageUpAxis(stage)
    return m, ("z" if up == UsdGeom.Tokens.z else "y")


def load_mesh(src_path, up_axis="z"):
    """Load any supported mesh and return it Z-up (the pipeline's convention).

    USD/USDZ carry their own units and up axis, which override `up_axis`;
    the result is then in metres. Other formats keep their file units.
    """
    import numpy as np
    import trimesh

    ext = os.path.splitext(src_path)[1].lower()
    if ext in USD_EXTS:
        m, up_axis = _load_usd(src_path)
    else:
        m = trimesh.load(src_path, force="mesh")
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate(m.dump())
    if up_axis == "y":  # ARKit / glTF convention: rotate Y-up to Z-up
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    elif up_axis != "z":
        raise SystemExit(f"up axis must be y or z, got {up_axis!r}")
    return m


def prepare_mesh(src_path, height=None, size=None, fit_bed=None,
                 base_plate=True, base_thickness=3.0, target_format="stl",
                 units_to_mm=None, up_axis="z", photo_level=False, gravity=None):
    """Scale, flatten, base-plate, repair. Returns (mesh, report).

    With `units_to_mm` (e.g. 1000 for a LiDAR scan in metres) and no
    `height`/`size`, the mesh keeps its real size instead of the 200 mm default.
    """
    import numpy as np
    import trimesh

    m = load_mesh(src_path, up_axis)
    level = None
    if photo_level:
        from photo_level import level_mesh
        level = level_mesh(m, gravity)
    if units_to_mm and os.path.splitext(src_path)[1].lower() in USD_EXTS:
        units_to_mm = 1000.0  # load_mesh already converted the stage to metres
    if len(m.faces) == 0:
        raise SystemExit(f"Empty mesh: {src_path}")

    raw_faces = len(m.faces)
    raw_extents = m.extents.copy()

    # ---- 1. scale ------------------------------------------------
    plate_height = max(base_thickness - 0.5, 0.0) if base_plate else 0.0
    if height is not None:
        cur = m.extents[2]
        if cur <= 0:
            raise SystemExit("Degenerate mesh (zero height)")
        if height <= 0:
            raise SystemExit("Height must be positive")
        if base_plate and height <= base_thickness:
            raise SystemExit("Height must exceed the base plate's thickness")
        m.apply_scale((height - plate_height) / cur)
        scale_note = f"height → {height:g} mm"
    elif units_to_mm and not size:
        m.apply_scale(units_to_mm)
        scale_note = f"real size (×{units_to_mm:g} to mm)"
    else:
        target = size if size else 200.0
        factor = target / m.extents.max()
        m.apply_scale(factor)
        scale_note = f"max dim → {target:g} mm"

    # ---- 2. flatten the bottom so it sits flat on the bed ----------
    # Do this BEFORE repair: clamping vertices can create degenerate faces,
    # and the repair pass must see that geometry to fix it.
    v = m.vertices.copy()
    f = m.faces.copy()
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    z_range = max(z_max - z_min, 1e-6)

    floor_mask = v[:, 2] < z_min + z_range * 0.03
    v[floor_mask, 2] = z_min
    m = trimesh.Trimesh(vertices=v, faces=f, process=True)

    # ---- 3. repair into a closed solid ---------------------------
    # Marching-cubes output from TripoSR has small holes, non-manifold edges
    # and floating fragments. pymeshfix closes holes and drops the debris.
    repair_note = "no repair"
    try:
        import pymeshfix
        bodies = m.split(only_watertight=False)
        main = max(bodies, key=lambda b: len(b.faces)) if len(bodies) > 1 else m
        mf = pymeshfix.MeshFix(main.vertices, main.faces)
        mf.repair()
        m = trimesh.Trimesh(vertices=mf.points, faces=mf.faces, process=True)
        repair_note = (f"merged {len(bodies)} parts, pymeshfix"
                       if len(bodies) > 1 else "pymeshfix")
    except Exception as exc:  # keep going: slicers have their own repair
        repair_note = f"pymeshfix unavailable ({type(exc).__name__})"
        m.merge_vertices()
        try:
            m.update_faces(m.unique_faces())
            m.fill_holes()
        except Exception:
            pass
    try:
        trimesh.repair.fix_normals(m)
    except Exception:
        pass

    if height is not None:
        m.apply_scale((height - plate_height) / m.extents[2])
    if fit_bed:
        bx, by, bz = fit_bed
        ex, ey, ez = m.extents
        if base_plate:
            if bz <= base_thickness:
                raise SystemExit("Bed height must exceed the base plate's thickness")
            factor = min((bx - 4.0) / ex, (by - 4.0) / ey,
                         bx / (ex + 0.06 * ez), by / (ey + 0.06 * ez),
                         (bz - plate_height) / ez)
        else:
            factor = min(bx / ex, by / ey, bz / ez)
        if factor <= 0:
            raise SystemExit("Bed is too small for the base plate")
        if factor < 1.0:
            m.apply_scale(factor)
            scale_note += f", shrunk ×{factor:.3f} to fit {bx:g}×{by:g}×{bz:g} bed"

    # ---- 4. fuse a base plate -----------------------------------
    if base_plate:
        z_range = m.extents[2]
        margin = max(z_range * 0.03, 2.0)
        x0, x1 = m.vertices[:, 0].min() - margin, m.vertices[:, 0].max() + margin
        y0, y1 = m.vertices[:, 1].min() - margin, m.vertices[:, 1].max() + margin
        plate = trimesh.creation.box(extents=[x1 - x0, y1 - y0, base_thickness])
        # sink it half a mm into the floor so the union definitely fuses
        plate.apply_translation([
            (x0 + x1) / 2, (y0 + y1) / 2,
            m.vertices[:, 2].min() - base_thickness / 2 + 0.5,
        ])
        fused = None
        try:
            fused = trimesh.boolean.union([m, plate])
        except Exception:
            fused = None
        if fused is not None and len(fused.faces):
            m = fused
            repair_note += ", base fused"
        else:
            # boolean unavailable: repair the concatenation once more
            m = trimesh.util.concatenate([m, plate])
            try:
                import pymeshfix
                mf = pymeshfix.MeshFix(m.vertices, m.faces)
                mf.repair()
                m = trimesh.Trimesh(vertices=mf.points, faces=mf.faces, process=True)
                repair_note += ", base repaired"
            except Exception:
                repair_note += ", base appended"

    # ---- 4. report ----------------------------------------------
    e = m.edges_sorted
    _, counts = np.unique(e, axis=0, return_counts=True)
    parts = len(m.split(only_watertight=False))
    overhang = float((m.face_normals[:, 2] < -0.5).mean()) if len(m.faces) else 0.0
    report = {
        "raw_faces": raw_faces,
        "faces": len(m.faces),
        "verts": len(m.vertices),
        "extents": m.extents,
        "raw_extents": raw_extents,
        "scale_note": scale_note,
        "repair": repair_note,
        "watertight": m.is_watertight,
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "parts": parts,
        "overhang_frac": overhang,
        "volume_cm3": m.volume / 1000.0 if m.is_watertight else None,
    }
    if level is not None:
        report["level"] = level
    return m, report


def print_report(report, path, n):
    ex = report["extents"]
    print()
    print(f"  {n}. {os.path.basename(path)}")
    print(f"     {ex[0]:.1f} × {ex[1]:.1f} × {ex[2]:.1f} mm   "
          f"({report['faces']:,} faces, {report['scale_note']}, {report['repair']})")
    ok = report["watertight"]
    print(f"     watertight: {'✓ yes' if ok else '✗ no'}"
          f"   boundary edges: {report['boundary_edges']}"
          f"   non-manifold: {report['nonmanifold_edges']}"
          f"   parts: {report['parts']}")
    if report["volume_cm3"]:
        print(f"     volume: {report['volume_cm3']:.0f} cm³  (~{report['volume_cm3']*1.24:.0f} g PLA solid)")
    ov = report["overhang_frac"]
    note = "supports recommended" if ov > 0.15 else "should print without supports"
    print(f"     overhang faces: {ov*100:.0f}%  → {note}")
    if not report["watertight"]:
        print("     ⚠ slicer repair may be needed (Cura/PrusaSlicer handle this automatically)")


def main():
    ap = argparse.ArgumentParser(
        prog="triposr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Turn a photo into a printable 3D model (TripoSR + mesh prep).",
        epilog="""examples:
  triposr chair.jpg                       # → chair.stl, 200 mm max dimension
  triposr chair.jpg --height 800          # → chair.stl, exactly 800 mm tall
  triposr a.jpg b.jpg c.jpg               # one model per image
  triposr mug.jpg -o /tmp/mug.stl --size 90 --no-base
  triposr shoe.jpg --mc-resolution 320 --texture    # finer geometry + texture
  triposr part.jpg --fit-bed 220x220x250  # shrink to a K1C build plate
  triposr scan.usdz                       # LiDAR scan → real-size STL, no TripoSR
  triposr scan.ply --units m --up y       # same for OBJ/PLY (these are the defaults)
""")
    ap.add_argument("images", nargs="+", metavar="input",
                    help="input image(s) (jpg/png/webp), or scanned mesh(es) "
                         "(obj/ply/usdz/glb/stl), which skip TripoSR")
    ap.add_argument("-o", "--output", help="output path (single image only)")
    ap.add_argument("--outdir", default=".", help="output directory (default: .)")
    ap.add_argument("--height", type=float, help="scale to this total height including the base plate (mm)")
    ap.add_argument("--size", type=float, help="scale so the largest dimension is this (mm, default 200)")
    ap.add_argument("--fit-bed", type=str, metavar="WxDxH",
                    help="shrink the scaled model including its base plate to fit a printer bed, e.g. 220x220x250")
    ap.add_argument("--no-base", action="store_true", help="skip the flat base plate")
    ap.add_argument("--base-thickness", type=float, default=3.0, help="base plate thickness in mm (default 3)")
    ap.add_argument("--mc-resolution", type=int, default=256,
                    help="marching-cubes resolution: 256 default, 320-512 = more detail, slower")
    ap.add_argument("--chunk-size", type=int, default=8192, help="lower this if you run out of memory")
    ap.add_argument("--texture", action="store_true", help="bake a UV texture (colors preserved in obj/glb)")
    ap.add_argument("--texture-resolution", type=int, default=2048, help="texture atlas size (default 2048)")
    ap.add_argument("--no-remove-bg", action="store_true",
                    help="skip background removal (use for pre-cut, gray-background images)")
    ap.add_argument("--foreground-ratio", type=float, default=0.85,
                    help="how much of the frame the subject should fill (default 0.85)")
    ap.add_argument("--format", default="stl", choices=["stl", "obj", "glb"],
                    help="output format (default stl; obj/glb keep vertex colors)")
    ap.add_argument("--units", default="m", choices=sorted(UNITS_TO_MM),
                    help="units of a mesh input without --height/--size (default m, as LiDAR "
                         "scans are); kept at real size. USDZ uses its own units")
    ap.add_argument("--up", default="y", choices=["y", "z"],
                    help="up axis of a mesh input (default y, as ARKit exports); USDZ uses its own")
    ap.add_argument("--device", default=None, help="torch device (default: auto mps/cuda/cpu)")
    ap.add_argument("--keep-obj", action="store_true", help="keep TripoSR's raw mesh.obj next to the output")
    ap.add_argument("--quiet", action="store_true", help="only print the output paths")

    args = ap.parse_args()

    # ---- validate inputs ----------------------------------------
    # Absolute paths: TripoSR runs in its own directory (see run_triposr).
    args.images = [os.path.abspath(p) for p in args.images]
    for img in args.images:
        if not os.path.isfile(img):
            raise SystemExit(f"error: no such file: {img}")
    if args.output and len(args.images) > 1:
        raise SystemExit("error: -o works with a single image; use --outdir for batches")

    # device
    device = args.device
    if not device and not all(is_mesh_file(p) for p in args.images):
        import torch
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda:0" if torch.cuda.is_available() else "cpu")

    # bed size
    fit_bed = None
    if args.fit_bed:
        try:
            fit_bed = tuple(float(x) for x in args.fit_bed.lower().split("x"))
            if len(fit_bed) != 3:
                raise ValueError
        except ValueError:
            raise SystemExit("error: --fit-bed wants WxDxH, e.g. 220x220x250")

    os.makedirs(args.outdir, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="triposr_")
    made, reports = [], []

    try:
        for idx, img in enumerate(args.images):
            scanned = is_mesh_file(img)
            if scanned:
                src = img
            else:
                # One image per TripoSR process: two images at once gets
                # OOM-killed on an 18-24 GB Mac.
                sub = os.path.join(workdir, str(idx))
                os.makedirs(sub, exist_ok=True)
                run_triposr([img], sub, device, args.mc_resolution, args.texture,
                            args.texture_resolution, args.no_remove_bg,
                            args.foreground_ratio, args.chunk_size)

                src = find_mesh(sub, 0)
                if not src:
                    print(f"warning: TripoSR produced no mesh for {img}", file=sys.stderr)
                    continue

            mesh, report = prepare_mesh(
                src,
                height=args.height,
                size=args.size,
                fit_bed=fit_bed,
                base_plate=not args.no_base,
                base_thickness=args.base_thickness,
                target_format=args.format,
                units_to_mm=UNITS_TO_MM[args.units] if scanned else None,
                up_axis=args.up if scanned else "z",
            )

            if args.output:
                out_path = args.output
            else:
                stem = os.path.splitext(os.path.basename(img))[0]
                out_path = os.path.join(args.outdir, f"{stem}.{args.format}")
                if os.path.abspath(out_path) == os.path.abspath(img):
                    out_path = os.path.join(args.outdir, f"{stem}_print.{args.format}")

            mesh.export(out_path)

            if args.keep_obj and not scanned:
                keep = os.path.splitext(out_path)[0] + "_raw.obj"
                shutil.copy(src, keep)
                report["raw_copy"] = keep

            made.append(out_path)
            reports.append(report)
            if not args.quiet:
                print_report(report, out_path, len(made))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if args.quiet:
        for p in made:
            print(p)
    else:
        print()
        print(f"  ✓ done — {len(made)} model(s): {', '.join(made)}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
