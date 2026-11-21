"""Level Z-up TripoSR meshes before scaling or adding a print base."""

import json

import numpy as np
import trimesh

import depth_fusion


def photo_up(camera_transform, image_orientation=1):
    """Return sensor-frame gravity and the same upright turn used by fusion."""
    if image_orientation not in (1, 3, 6, 8):
        raise ValueError("image_orientation must be 1, 3, 6 or 8")
    try:
        vals = np.asarray(json.loads(camera_transform), dtype=float)
        if vals.size != 16 or not np.isfinite(vals).all():
            raise ValueError
        rotation = vals.reshape(4, 4).T[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=0.01) or np.linalg.det(rotation) < 0:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("camera_transform wants 16 finite floats with a camera rotation")
    up = depth_fusion.CV_FROM_ARKIT @ (rotation.T @ [0., 1., 0.])
    return up, depth_fusion.upright_turns(up)


def base_normal(mesh):
    """Largest supporting planar hull facet within 35 degrees of down."""
    if len(mesh.vertices) < 4 or len(mesh.faces) == 0:
        return None
    try:
        hull = mesh.convex_hull
    except (ValueError, RuntimeError):
        return None
    candidates = []
    for faces, normal, area in zip(hull.facets, hull.facets_normal, hull.facets_area):
        if normal[2] < -np.cos(np.deg2rad(35)) and area > hull.area * 0.01:
            candidates.append((area, normal))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def level_mesh(mesh, gravity=None):
    """Mutate a Z-up mesh; report the applied tilt, not the remaining tilt."""
    if gravity is not None:
        up, turns = gravity
        x, y, z = up
        for _ in range(turns):
            x, y = y, -x
        # tsr/utils.py:get_spherical_cameras: view zero is at +X,
        # looking toward -X, image right +Y and image up +Z.
        normal = np.array([-z, x, -y])
        target = np.array([0., 0., 1.])
        source = "gravity"
    else:
        normal = base_normal(mesh)
        target = np.array([0., 0., -1.])
        source = "base plane"
    if normal is None:
        return {"source": "none", "tilt_deg": 0.0}
    tilt = np.degrees(np.arccos(np.clip(np.dot(normal, target) / np.linalg.norm(normal), -1, 1)))
    mesh.apply_transform(trimesh.geometry.align_vectors(normal, target))
    return {"source": source, "tilt_deg": round(float(tilt), 2)}
