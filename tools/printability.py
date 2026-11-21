"""Is this mesh safe to send to an FDM printer? `check(mesh)` -> {"printable", "errors", "warnings", ...}.

Errors mean it can't be printed at all (not a closed solid, or bigger than the bed); everything
else that could spoil a print is a warning.
Units are mm, Z up, the bed at the mesh's lowest point.
    python tools/printability.py model.stl [--bed 220x220x250]
"""

import sys

import numpy as np

COMMON_BED = (220.0, 220.0, 250.0)
MIN_WALL_MM = 0.8  # two perimeters of a 0.4 mm nozzle
OVERHANG_DEG = 45.0
BED_TOL_MM = 0.5
CONTACT_TOL_MM = 0.05
PLA_G_CM3 = 1.24


def check(m, bed=None, wall_samples=1500, seed=0):
    import trimesh

    errors, warnings = [], []
    ex = [float(v) for v in m.extents]
    z0 = float(m.bounds[0, 2])
    area = float(m.area) or 1.0

    # ---- solid: the slicer needs a closed, consistently wound, outward-facing surface ----
    edges = m.edges_sorted
    _, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges, nonmanifold = int((counts == 1).sum()), int((counts > 2).sum())
    if not m.is_watertight:
        errors.append(f"not watertight ({open_edges} open edges)")
    if nonmanifold:
        warnings.append(f"{nonmanifold} non-manifold edges")
    if not m.is_winding_consistent:
        warnings.append("inconsistent face winding")
    volume = float(m.volume) if m.is_watertight else None
    if volume is not None and volume <= 0:
        warnings.append("normals point inward (negative volume)")
    degenerate = int((m.area_faces < 1e-10).sum())

    # ---- shells: every part must stand on the bed ----
    parts = m.split(only_watertight=False)
    floating = [p for p in parts if p.bounds[0, 2] - z0 > BED_TOL_MM]
    if floating:
        warnings.append(f"{len(floating)} part(s) float above the bed: print with supports")
    elif len(parts) > 1:
        warnings.append(f"{len(parts)} separate parts")

    # ---- size ----
    if bed is not None:
        if any(e > b + 1e-6 for e, b in zip(sorted(ex[:2]) + [ex[2]], sorted(bed[:2]) + [bed[2]])):
            errors.append(f"{ex[0]:.0f}×{ex[1]:.0f}×{ex[2]:.0f} mm does not fit the "
                          f"{bed[0]:g}×{bed[1]:g}×{bed[2]:g} bed")
    elif any(e > b for e, b in zip(sorted(ex[:2]) + [ex[2]], COMMON_BED)):
        warnings.append(f"{ex[0]:.0f}×{ex[1]:.0f}×{ex[2]:.0f} mm is larger than a common "
                        "220×220×250 mm bed")
    if min(ex) < 2.0:
        warnings.append(f"very small: thinnest dimension {min(ex):.1f} mm")

    # ---- the base: contact with the bed and whether it tips over ----
    nz = m.face_normals[:, 2]
    fz = m.triangles[:, :, 2]
    on_bed = (nz < -0.999) & (fz.max(axis=1) - z0 < CONTACT_TOL_MM)
    contact = float(m.area_faces[on_bed].sum())
    if contact < 1.0:
        warnings.append("nothing flat touches the bed: print with supports or a raft")
    else:
        tips = _com_outside(m, on_bed) if volume else False
        if tips:
            warnings.append("centre of mass is outside the base: it may tip over")
        footprint = ex[0] * ex[1] or 1.0
        if contact / footprint < 0.05:
            warnings.append(f"small base ({contact:.0f} mm²) for its size: add a brim")

    # ---- overhangs that need supports (the bottom face itself is fine) ----
    steep = (nz < -np.cos(np.radians(90 - OVERHANG_DEG))) & ~on_bed
    overhang = float(m.area_faces[steep].sum()) / area
    if overhang > 0.15:
        warnings.append(f"{overhang * 100:.0f}% of the surface overhangs more than "
                        f"{OVERHANG_DEG:g}°: print with supports")

    # ---- wall thickness: from each probe, the nearest opposite-facing surface behind it ----
    thin, min_wall = None, None
    if m.is_watertight and len(m.faces):
        t = _wall_thickness(m, wall_samples, seed)
        if len(t):
            thin = float((t < MIN_WALL_MM).mean())
            min_wall = float(np.percentile(t, 2))
            if thin > 0.02:
                warnings.append(f"{thin * 100:.0f}% of the walls are thinner than "
                                f"{MIN_WALL_MM} mm and may not print")

    return {
        "printable": not errors,
        "repairable": bool(open_edges or nonmanifold or floating or not m.is_winding_consistent
                           or (volume is not None and volume <= 0)),
        "errors": errors,
        "warnings": warnings,
        "open_edges": open_edges,
        "nonmanifold_edges": nonmanifold,
        "degenerate_faces": degenerate,
        "parts": len(parts),
        "base_contact_mm2": round(contact, 1),
        "overhang_pct": round(overhang * 100, 1),
        "thin_wall_pct": None if thin is None else round(thin * 100, 1),
        "min_wall_mm": None if min_wall is None else round(min_wall, 2),
        "solid_mass_g": None if not volume else round(volume / 1000 * PLA_G_CM3, 1),
    }


def fix(m):
    """One repair pass for what `check` calls errors: drop floating parts, close holes, fix normals."""
    import trimesh

    z0 = m.bounds[0, 2]
    parts = [p for p in m.split(only_watertight=False) if p.bounds[0, 2] - z0 <= BED_TOL_MM]
    if parts:
        m = trimesh.util.concatenate(parts)
    if not m.is_watertight or not m.is_winding_consistent:
        try:
            import pymeshfix

            mf = pymeshfix.MeshFix(m.vertices, m.faces)
            mf.repair()
            m = trimesh.Trimesh(vertices=mf.points, faces=mf.faces, process=True)
        except Exception:
            pass
    trimesh.repair.fix_normals(m)
    return m


def _wall_thickness(m, probes, seed, reach=3.0):
    """Wall thickness (mm, capped at `reach`) at `probes` random surface points.

    A dense surface sample stands in for the mesh: the wall behind a probe is the closest
    sample that faces the other way and lies inward along the probe's normal.
    """
    import trimesh
    from scipy.spatial import cKDTree

    n = int(np.clip(m.area / 0.25 ** 2, 20_000, 400_000))
    pts, fi = trimesh.sample.sample_surface(m, n, seed=seed)
    nrm = m.face_normals[fi]
    tree = cKDTree(pts)
    k = min(probes, n)
    out = np.full(k, reach)
    for i, nb in enumerate(tree.query_ball_point(pts[:k], reach)):
        nb = np.asarray(nb)
        if len(nb) == 0:
            continue
        depth = -(pts[nb] - pts[i]) @ nrm[i]  # how far behind the probe's surface
        ok = (nrm[nb] @ nrm[i] < -0.5) & (depth > 1e-3)
        if ok.any():
            out[i] = depth[ok].min()
    return out


def _com_outside(m, on_bed):
    from scipy.spatial import Delaunay

    pts = m.triangles[on_bed][:, :, :2].reshape(-1, 2)
    try:
        return bool(Delaunay(pts).find_simplex(m.center_mass[:2]) < 0)
    except Exception:  # collinear contact points
        return True


def main():
    import argparse
    import json

    import trimesh

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mesh")
    ap.add_argument("--bed", help="WxDxH mm")
    a = ap.parse_args()
    bed = tuple(float(v) for v in a.bed.lower().split("x")) if a.bed else None
    r = check(trimesh.load(a.mesh, force="mesh"), bed)
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["printable"] else 1)


if __name__ == "__main__":
    main()
