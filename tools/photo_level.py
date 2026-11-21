"""Level Z-up TripoSR meshes before scaling or adding a print base."""

import json

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import depth_fusion


def pitch_search(mesh):
    """Minimise yaw-optimised AABB volume for downward camera pitch, 0..75°."""
    if len(mesh.vertices) < 4 or len(mesh.faces) == 0:
        return None, np.array([])
    try:
        vertices = mesh.convex_hull.vertices
    except (ValueError, RuntimeError):
        return None, np.array([])
    yaw = np.deg2rad(np.arange(90))
    cosine, sine = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    scores = []
    for degrees in range(76):
        angle = np.deg2rad(degrees)
        # Camera horizontal is +Y: undo downward pitch with a negative Y turn.
        x = np.cos(angle) * vertices[:, 0] - np.sin(angle) * vertices[:, 2]
        y = vertices[:, 1]
        z = np.sin(angle) * vertices[:, 0] + np.cos(angle) * vertices[:, 2]
        widths = np.ptp(cosine * x - sine * y, axis=1)
        depths = np.ptp(sine * x + cosine * y, axis=1)
        scores.append(float(np.min(widths * depths) * np.ptp(z)))
    scores = np.asarray(scores)
    best = int(scores.argmin())
    if scores[0] <= 0 or scores[best] >= 0.95 * scores[0]:
        return None, scores
    angle = np.deg2rad(best)
    return np.array([np.sin(angle), 0., np.cos(angle)]), scores


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


def shape_up(mesh):
    """Fit a box-like up direction from dominant parallel/orthogonal planes."""
    areas = mesh.area_faces
    if not len(areas) or areas.sum() <= 0:
        return None
    normals = mesh.face_normals
    cosine = np.cos(np.deg2rad(10))
    sine = np.sin(np.deg2rad(10))
    remaining = areas > 0
    groups, weights = [], []
    _, seeds = np.unique(np.round(normals / 0.08).astype(int), axis=0, return_index=True)
    tree = cKDTree(np.vstack([normals, -normals]))
    neighborhoods = [np.unique(np.asarray(indices) % len(normals)) for indices in
                     tree.query_ball_point(normals[seeds], np.sqrt(2 - 2 * cosine))]
    masses = np.array([areas[indices].sum() for indices in neighborhoods])
    while len(masses) and masses.max() >= areas.sum() * 0.025:
        best = masses.argmax()
        members = neighborhoods[best]
        members = members[remaining[members]]
        weight = areas[members].sum()
        if weight < areas.sum() * 0.025:
            break
        dots = normals[members] @ normals[seeds[best]]
        normal = (normals[members] * (areas[members] * np.sign(dots))[:, None]).sum(axis=0)
        normal /= np.linalg.norm(normal)
        groups.append(normal)
        weights.append(weight)
        remaining[members] = False
        masses = np.array([areas[indices[remaining[indices]]].sum() for indices in neighborhoods])
    if not groups:
        return None
    groups, weights = np.array(groups), np.array(weights)

    def score(up):
        dots = np.abs(groups @ up)
        return weights[(dots > cosine) | (dots < sine)].sum()

    identity = np.array([0., 0., 1.])
    candidates = groups * np.where(groups[:, 2] < 0, -1, 1)[:, None]
    candidates = candidates[candidates[:, 2] > np.cos(np.deg2rad(35))]
    if not len(candidates):
        return None
    up = max(candidates, key=lambda candidate: (score(candidate), candidate[2]))
    baseline = score(identity)
    if score(up) <= baseline + max(areas.sum() * 0.05, baseline * 0.1):
        return None
    dots = np.abs(groups @ up)
    parallel, perpendicular = dots > cosine, dots < sine
    matrix = np.zeros((3, 3))
    for normal, weight, para, perp in zip(groups, weights, parallel, perpendicular):
        outer = np.outer(normal, normal)
        if para:
            matrix += weight * (np.eye(3) - outer)
        elif perp:
            matrix += weight * outer
    _, vectors = np.linalg.eigh(matrix)
    refined = vectors[:, 0]
    refined *= 1 if refined[2] >= 0 else -1
    if refined[2] < np.cos(np.deg2rad(35)) or score(refined) <= baseline + max(areas.sum() * 0.05, baseline * 0.1):
        return None
    return refined


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
        normal = shape_up(mesh)
        if normal is None:
            normal, _ = pitch_search(mesh)
        target = np.array([0., 0., 1.])
        source = "shape"
    if normal is None:
        return {"source": "none", "tilt_deg": 0.0}
    tilt = np.degrees(np.arccos(np.clip(np.dot(normal, target) / np.linalg.norm(normal), -1, 1)))
    mesh.apply_transform(trimesh.geometry.align_vectors(normal, target))
    return {"source": source, "tilt_deg": round(float(tilt), 2)}
