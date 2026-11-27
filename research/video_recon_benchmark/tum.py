from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def records(path):
    return [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def interpolate_poses(trajectory, timestamps):
    poses = np.broadcast_to(np.eye(4), (len(timestamps), 4, 4)).copy()
    gt = trajectory[:, 0]
    poses[:, :3, :3] = Slerp(gt, Rotation.from_quat(trajectory[:, 4:8]))(timestamps).as_matrix()
    poses[:, :3, 3] = np.column_stack([np.interp(timestamps, gt, v) for v in trajectory[:, 1:4].T])
    return poses


def load_sequence(root: Path, count: int, seconds: float, width: int, height: int):
    rgb = records(root / "rgb.txt")
    depth = records(root / "depth.txt")
    trajectory = np.array(records(root / "groundtruth.txt"), dtype=float)
    rt = np.array([float(row[0]) for row in rgb])
    dt = np.array([float(row[0]) for row in depth])
    gt = trajectory[:, 0]
    depth_ids = np.abs(rt[:, None] - dt).argmin(axis=1)
    paired_depth = dt[depth_ids]
    valid = np.flatnonzero((rt >= gt[0]) & (rt <= gt[-1])
                           & (paired_depth >= gt[0]) & (paired_depth <= gt[-1])
                           & (np.abs(rt - paired_depth) <= 0.02))
    if not valid.size or count < 3 or seconds <= 0:
        raise ValueError("need at least three time-associated frames and positive duration")
    start = rt[valid[0]]
    end = min(start + seconds, rt[valid[-1]])
    valid = valid[rt[valid] <= end]
    target = np.linspace(start, end, count)
    ids = valid[np.abs(rt[valid, None] - target).argmin(axis=0)]
    if np.unique(ids).size != count or np.max(np.abs(rt[ids] - target)) > 0.15:
        raise ValueError("not enough synchronized RGB/depth/pose timestamps for the requested clip")
    timestamps = rt[ids]
    depth_timestamps = paired_depth[ids]

    size = (width, height)
    rgb_frames, depths = [], []
    for i in ids:
        image = cv2.imread(str(root / rgb[i][1]))
        raw_depth = cv2.imread(str(root / depth[depth_ids[i]][1]), cv2.IMREAD_UNCHANGED)
        if image is None or raw_depth is None:
            raise ValueError(f"cannot decode RGB/depth at timestamp {rt[i]}")
        if image.shape[:2] != (480, 640) or raw_depth.shape != (480, 640):
            raise ValueError("expected native 640×480 Freiburg 1 images")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_frames.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
        depths.append(cv2.resize(raw_depth, size, interpolation=cv2.INTER_NEAREST_EXACT))
    # Freiburg 1 RGB calibration; TUM distributes depth registered to RGB.
    sx, sy = width / 640, height / 480
    intrinsics = np.array([[517.3 * sx, 0, (318.6 + 0.5) * sx - 0.5],
                           [0, 516.5 * sy, (255.3 + 0.5) * sy - 0.5], [0, 0, 1]])
    return dict(rgb=np.stack(rgb_frames), gt_depth=np.stack(depths).astype(np.float32) / 5000,
                gt_pose=interpolate_poses(trajectory, timestamps),
                gt_depth_pose=interpolate_poses(trajectory, depth_timestamps),
                gt_intrinsics=intrinsics, timestamps=timestamps,
                depth_offsets=depth_timestamps - timestamps)
