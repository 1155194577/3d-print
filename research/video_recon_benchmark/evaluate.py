import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tum import load_sequence


def similarity(source, target):
    """Least-squares, positive-scale, no-reflection Umeyama alignment."""
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    variance = np.sum(source_centered**2) / len(source)
    if np.linalg.matrix_rank(source_centered) < 2 or np.linalg.matrix_rank(target_centered) < 2:
        raise ValueError("camera trajectory is degenerate for Sim(3) alignment")
    u, singular, vt = np.linalg.svd(target_centered.T @ source_centered / len(source))
    sign = np.array([1, 1, np.sign(np.linalg.det(u @ vt))])
    rotation = (u * sign) @ vt
    scale = (singular @ sign) / variance
    offset = target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))
    return scale, rotation, offset


def rmse(values):
    return float(np.sqrt(np.mean(values**2)))


def pose_metrics(predicted, reference, transform):
    scale, rotation, offset = transform
    aligned = predicted.copy().astype(float)
    aligned[:, :3, :3] = rotation @ predicted[:, :3, :3]
    aligned[:, :3, 3] = scale * (predicted[:, :3, 3] @ rotation.T) + offset
    ate = np.linalg.norm(aligned[:, :3, 3] - reference[:, :3, 3], axis=1)
    relative_predicted = np.linalg.solve(aligned[:-1], aligned[1:])
    relative_reference = np.linalg.solve(reference[:-1], reference[1:])
    error = np.linalg.solve(relative_reference, relative_predicted)
    translation = np.linalg.norm(error[:, :3, 3], axis=1)
    angle = Rotation.from_matrix(error[:, :3, :3]).magnitude()
    return dict(ate_rmse_m=rmse(ate), rpe_translation_rmse_m=rmse(translation),
                rpe_rotation_rmse_deg=rmse(np.degrees(angle)))


def depth_metrics(predicted, reference):
    metrics, pixel_count, valid_count = [], 0, 0
    for pred, gt in zip(predicted, reference):
        valid_gt = np.isfinite(gt) & (gt > 0)
        keep = valid_gt & np.isfinite(pred) & (pred > 0)
        pixel_count += np.count_nonzero(valid_gt)
        valid_count += np.count_nonzero(keep)
        if not keep.any():
            raise ValueError("a frame has no jointly valid predicted and reference depth")
        pred, gt = pred[keep], gt[keep]
        aligned = pred * (np.median(gt) / np.median(pred))
        metrics.append([np.mean(np.abs(aligned - gt) / gt),
                        np.mean(np.maximum(aligned / gt, gt / aligned) < 1.25)])
    abs_rel, delta = np.mean(metrics, axis=0)
    return dict(depth_abs_rel=float(abs_rel), depth_delta_125=float(delta),
                depth_valid_fraction=valid_count / pixel_count)


def point_cloud(rgb, depth, poses, intrinsics=None, local_points=None, stride=4):
    y, x = np.mgrid[0:depth.shape[1]:stride, 0:depth.shape[2]:stride]
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
    points, colors = [], []
    for i, pose in enumerate(poses):
        z = depth[i, ::stride, ::stride]
        if local_points is None:
            k = intrinsics if intrinsics.ndim == 2 else intrinsics[i]
            xyz = (pixels @ np.linalg.inv(k).T) * z[..., None]
        else:
            xyz = local_points[i, ::stride, ::stride]
        keep = np.isfinite(z) & (z > 0) & np.isfinite(xyz).all(axis=-1)
        points.append(xyz[keep] @ pose[:3, :3].T + pose[:3, 3])
        colors.append(rgb[i, ::stride, ::stride][keep])
    return np.concatenate(points), np.concatenate(colors)


def voxel_sample(points, colors, side=0.01):
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("point cloud is empty or contains non-finite coordinates")
    _, ids = np.unique(np.floor(points / side).astype(np.int64), axis=0, return_index=True)
    return points[ids], colors[ids]


def cloud_metrics(predicted, reference, reference_tree):
    accuracy = reference_tree.query(predicted, workers=4)[0].mean()
    completeness = cKDTree(predicted).query(reference, workers=4)[0].mean()
    return dict(accuracy_m=float(accuracy), completeness_m=float(completeness),
                chamfer_l1_m=float((accuracy + completeness) / 2),
                points=len(predicted))


def write_ply(path, points, colors):
    data = np.empty(len(points), dtype=[(name, "<f4") for name in ("x", "y", "z")]
                    + [(name, "u1") for name in ("red", "green", "blue")])
    for name, values in zip(data.dtype.names, (*points.T, *colors.T)):
        data[name] = values
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        data.tofile(stream)


def render(path, clouds):
    fig, axes = plt.subplots(1, len(clouds), squeeze=False, figsize=(4 * len(clouds), 4.5),
                             subplot_kw={"projection": "3d"}, layout="constrained")
    bounds = np.quantile(clouds[0][1], [0.01, 0.99], axis=0)
    center = bounds.mean(axis=0)
    half = np.max(bounds[1] - bounds[0]) * 0.57
    rng = np.random.default_rng(0)
    for ax, (name, points, colors) in zip(axes.flat, clouds):
        ids = np.flatnonzero((np.abs(points - center) <= half).all(axis=1))
        if len(ids) > 65_000:
            ids = rng.choice(ids, 65_000, replace=False)
        ax.scatter(*points[ids].T, c=colors[ids] / 255, s=0.6, linewidths=0, depthshade=False)
        ax.set(xlim=(center[0] - half, center[0] + half), ylim=(center[1] - half, center[1] + half),
               zlim=(center[2] - half, center[2] + half), title=name)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=46, azim=-105)
        ax.set_axis_off()
    fig.suptitle("TUM fr1_desk · RGB point clouds in the same reference frame and bounds")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(archives, dataset, output, figure, seconds):
    with np.load(archives[0]) as first:
        count = len(first["timestamps"])
    # Reference sampling grid is identical for every model, independent of its resize.
    ref = load_sequence(dataset, count, seconds, 640, 480)
    reference, colors = voxel_sample(*point_cloud(ref["rgb"], ref["gt_depth"], ref["gt_depth_pose"],
                                                  ref["gt_intrinsics"], stride=5))
    tree = cKDTree(reference)
    clouds = [("RGB-D reference", reference, colors)]
    output.mkdir(parents=True, exist_ok=True)
    write_ply(output / "reference.ply", reference, colors)
    metrics = {}
    for path in archives:
        with np.load(path) as data:
            if not np.array_equal(data["timestamps"], ref["timestamps"]):
                raise ValueError(f"{path}: frame timestamps differ from the reference clip")
            if not np.array_equal(data["depth_offsets"], ref["depth_offsets"]):
                raise ValueError(f"{path}: depth associations differ from the reference clip")
            predicted = data["poses_c2w"]
            if predicted.shape != ref["gt_pose"].shape or not np.isfinite(predicted).all():
                raise ValueError(f"{path}: invalid pose outputs")
            transform = similarity(predicted[:, :3, 3], ref["gt_pose"][:, :3, 3])
            depth = data["depth"][..., 0]
            summary = pose_metrics(predicted, ref["gt_pose"], transform)
            summary.update(depth_metrics(depth, data["gt_depth"]))
            raw, color = point_cloud(data["rgb"], depth, predicted,
                                     intrinsics=data.get("intrinsics"),
                                     local_points=data.get("local_points"))
            scale, rotation, offset = transform
            points, color = voxel_sample(scale * (raw @ rotation.T) + offset, color)
            summary.update(cloud_metrics(points, reference, tree))
            summary.update(wall_seconds=float(data["wall_seconds"]),
                           peak_rss_gib=float(data["peak_rss_gib"]), scale=float(scale))
        metrics[path.stem] = summary
        write_ply(output / f"{path.stem}.ply", points, color)
        clouds.append((path.stem, points, color))
    figure.parent.mkdir(parents=True, exist_ok=True)
    render(figure, clouds)
    report = dict(frames=count, seconds=float(np.ptp(ref["timestamps"])),
                  max_depth_offset_s=float(np.max(np.abs(ref["depth_offsets"]))),
                  reference_points=len(reference), models=metrics)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Score predicted and ground-truth desk trajectories and geometry")
    parser.add_argument("archives", type=Path, nargs="+")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=20)
    args = parser.parse_args()
    report = evaluate(args.archives, args.dataset, args.output, args.figure, args.seconds)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
