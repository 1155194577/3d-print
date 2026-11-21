"""Fuse a TripoSR mesh with the LiDAR depth map of the same ARFrame.

TripoSR guesses the whole shape from one photo but not its size or pose; the
depth map measures the visible side exactly. So:

1. back-project the depth pixels on the object (photo mask, eroded) to 3D;
2. find the floor (camera gravity, else RANSAC on the background points) and
   work in a frame with z up and the floor at z=0;
3. align the TripoSR mesh to those points with a trimmed similarity ICP from
   several yaw starts, which gives true size and pose;
4. pull the TripoSR vertices the camera saw onto the LiDAR surface, fading the
   correction into the hidden side so there is no seam.

Output: a mesh in metres, z up, floor at z=0, ready for img2mesh.prepare_mesh.
"""

import json

import numpy as np

CV_FROM_ARKIT = np.diag([1.0, -1.0, -1.0])
MIN_POINTS = 200


class FusionError(Exception):
    """The depth can't be used (too few points, nothing to align)."""


class Frame:
    """Depth, intrinsics (scaled to the depth map) and gravity of one ARFrame."""

    def __init__(self, depth, K, confidence=None, camera_to_world=None,
                 image_orientation=1, photo_size=None, taps=None):
        self.depth = depth
        self.K = K
        self.confidence = confidence
        self.camera_to_world = camera_to_world
        self.image_orientation = image_orientation
        self.photo_size = photo_size
        self.taps = taps  # depth pixels (u, v)

    @classmethod
    def load(cls, depth_path, width=None, height=None, confidence_path=None,
             intrinsics=None, camera_transform=None, image_orientation=1,
             object_points=None, depth_intrinsics=None, depth_extrinsics=None):
        if depth_path.lower().endswith(".png"):
            from PIL import Image
            try:
                depth = np.asarray(Image.open(depth_path)).astype(np.float32) / 1000.0
            except OSError as exc:
                raise ValueError(f"depth PNG could not be read: {exc}")
            if depth.ndim != 2:
                raise ValueError("depth PNG must be single-channel (16-bit mm)")
            depth[depth <= 0] = np.nan
        else:
            if not width or not height:
                raise ValueError("depth_width and depth_height are required for raw depth")
            raw = np.fromfile(depth_path, dtype="<f4")
            if raw.size != width * height:
                raise ValueError(f"depth has {raw.size} floats, expected "
                                 f"{width}x{height}={width * height}")
            depth = raw.reshape(height, width).astype(np.float32)
        h, w = depth.shape

        conf = None
        if confidence_path:
            conf = np.fromfile(confidence_path, dtype=np.uint8)
            if conf.size != w * h:
                raise ValueError(f"depth_confidence has {conf.size} bytes, expected {w * h}")
            conf = conf.reshape(h, w)

        if not intrinsics:
            raise ValueError("intrinsics are required with depth")
        try:
            k = json.loads(intrinsics)
            fx, fy, cx, cy = (float(k[n]) for n in ("fx", "fy", "cx", "cy"))
            pw, ph = float(k["width"]), float(k["height"])
        except (ValueError, KeyError, TypeError):
            raise ValueError('intrinsics wants JSON {"fx","fy","cx","cy","width","height"}')
        if min(fx, fy, pw, ph) <= 0:
            raise ValueError("intrinsics must be positive")
        if not np.isfinite([fx, fy, cx, cy, pw, ph]).all() or max(pw, ph) > 20000:
            raise ValueError("intrinsics must be finite with width/height <= 20000")
        sx, sy = w / pw, h / ph
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1.0]])
        photo_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        E = np.eye(4)
        if depth_extrinsics is not None:
            try:
                vals = np.asarray(json.loads(depth_extrinsics), dtype=float)
                if vals.shape != (16,) or not np.isfinite(vals).all():
                    raise ValueError
                E = vals.reshape(4, 4).T
            except (ValueError, TypeError):
                raise ValueError("depth_extrinsics wants a JSON array of 16 finite floats")
        if depth_intrinsics is not None:
            try:
                dk = json.loads(depth_intrinsics)
                df, dg, dc, dd, dw, dh = (float(dk[n]) for n in
                                          ("fx", "fy", "cx", "cy", "width", "height"))
                if not np.isfinite([df, dg, dc, dd, dw, dh]).all() or min(df, dg, dw, dh) <= 0:
                    raise ValueError
                if max(dw, dh) > 20000:
                    raise ValueError
                K = np.array([[df * w / dw, 0, dc * w / dw],
                              [0, dg * h / dh, dd * h / dh], [0, 0, 1.]])
            except (ValueError, KeyError, TypeError):
                raise ValueError('depth_intrinsics wants JSON {"fx","fy","cx","cy","width","height"}')

        T = None
        if camera_transform:
            try:
                vals = np.array(json.loads(camera_transform), dtype=float).ravel()
                if vals.size != 16:
                    raise ValueError
            except (ValueError, TypeError):
                raise ValueError("camera_transform wants a JSON array of 16 floats")
            T = vals.reshape(4, 4).T  # column-major in, row-major here

        if image_orientation not in (1, 3, 6, 8):
            raise ValueError("image_orientation must be 1, 3, 6 or 8")

        taps = None
        if object_points:
            try:
                taps = np.array(json.loads(object_points), dtype=float).reshape(-1, 2)
            except (ValueError, TypeError):
                raise ValueError("object_points wants JSON [[u, v], ...] in photo pixels")
            if depth_intrinsics is None:
                taps = taps * [sx, sy]
        frame = cls(depth, K, conf, T, image_orientation, (pw, ph), taps)
        frame.depth_to_photo = E
        frame.photo_K = photo_K if depth_intrinsics is not None else None
        return frame

    def up_camera(self):
        """World up in OpenCV depth camera coordinates (x right, y down, z forward)."""
        if self.camera_to_world is not None:
            up = CV_FROM_ARKIT @ (self.camera_to_world[:3, :3].T @ [0.0, 1.0, 0.0])
        else:  # the upright direction of the photo, as EXIF would rotate it
            up = {1: [0, -1, 0], 3: [0, 1, 0], 6: [-1, 0, 0], 8: [1, 0, 0]}[self.image_orientation]
        up = np.asarray(up, dtype=float)
        up = getattr(self, "depth_to_photo", np.eye(4))[:3, :3].T @ up
        return up / np.linalg.norm(up)


def upright_turns(up_cv):
    """Quarter turns counter-clockwise (PIL rotate(90*k)) that put `up` at the top."""
    best, best_y, (x, y) = 0, None, up_cv[:2]
    for k in range(4):
        if best_y is None or y < best_y:
            best, best_y = k, y
        x, y = y, -x
    return best


def backproject(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return np.stack([(u - K[0, 2]) / K[0, 0] * depth,
                     (v - K[1, 2]) / K[1, 1] * depth, depth], axis=-1)


def _fit_floor(points, up_guess, iters=300, tol=0.008, rng=None):
    """RANSAC a plane roughly perpendicular to up_guess; returns (normal, point) or None."""
    if len(points) < 300:
        return None
    rng = rng or np.random.default_rng(0)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ up_guess < 0:
            n = -n
        if n @ up_guess < 0.8:
            continue
        count = np.count_nonzero(np.abs((points - a) @ n) < tol)
        if count > best_n:
            best, best_n = (n, a), count
    if best is None or best_n < 0.3 * len(points):
        return None
    n, a = best
    inl = points[np.abs((points - a) @ n) < tol]
    c = inl.mean(axis=0)
    n = np.linalg.svd(inl - c, full_matrices=False)[2][-1]
    return (n if n @ up_guess > 0 else -n), c


def _object_mask(mask, taps):
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    if taps is not None and len(taps):
        h, w = mask.shape
        hit = {labels[int(np.clip(v, 0, h - 1)), int(np.clip(u, 0, w - 1))] for u, v in taps}
        hit.discard(0)
        if hit:
            return np.isin(labels, list(hit))
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


def _depth_mask(frame, mask):
    from PIL import Image

    h, w = frame.depth.shape
    mask = np.asarray(mask)
    mask = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask.astype(np.uint8)
    if getattr(frame, "photo_K", None) is None:
        return _object_mask(np.asarray(Image.fromarray(mask).resize((w, h), Image.NEAREST)) > 127,
                            frame.taps)
    pw, ph = frame.photo_size
    mask = mask > 127
    mh, mw = mask.shape
    points = backproject(np.nan_to_num(frame.depth), frame.K)
    E = frame.depth_to_photo
    photo = points @ E[:3, :3].T + E[:3, 3]
    projected = photo @ frame.photo_K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[..., :2] / projected[..., 2:]
    valid = np.isfinite(uv).all(-1) & (photo[..., 2] > 0) & np.isfinite(frame.depth)
    ix = np.floor(np.clip(np.nan_to_num(uv[..., 0]) * (mw / pw), -1, mw)).astype(int)
    iy = np.floor(np.clip(np.nan_to_num(uv[..., 1]) * (mh / ph), -1, mh)).astype(int)
    valid &= (ix >= 0) & (ix < mw) & (iy >= 0) & (iy < mh)
    registered = np.zeros((h, w), bool)
    registered[valid] = mask[iy[valid], ix[valid]]
    taps = None
    if frame.taps is not None and valid.any():
        vu = np.argwhere(valid)
        taps = [vu[np.argmin(np.sum((uv[valid] - tap) ** 2, axis=1))][::-1]
                for tap in frame.taps]
    return _object_mask(registered, taps)


def _umeyama(src, dst):
    """Similarity (s, R, t) with dst ≈ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _icp(points, target_tree, target, init, iters=40, keep=0.85):
    """Trimmed similarity ICP of `points` onto `target`; returns (s, R, t, rmse)."""
    s, R, t = init
    for _ in range(iters):
        moved = s * points @ R.T + t
        d, idx = target_tree.query(moved)
        sel = d <= np.quantile(d, keep)
        s, R, t = _umeyama(points[sel], target[idx[sel]])
    d, _ = target_tree.query(s * points @ R.T + t)
    d = np.sort(d)[: int(len(d) * keep)]
    return s, R, t, float(np.sqrt((d ** 2).mean()) / s)


def fuse(mesh, frame, mask, yaw_starts=8, max_points=4000):
    """Align a TripoSR `mesh` (z up, arbitrary scale) to the frame's depth.

    `mask` is the photo's foreground (any resolution; resized to the depth map).
    Returns (mesh in metres with z up and the floor at z=0, report dict).
    """
    return _fuse(mesh, frame, mask, yaw_starts, max_points)


def fuse_face(mesh, frame, mask, face_to_world=None):
    """Fuse a bust in a gravity frame whose origin is the horizontal neck cut."""
    return _fuse(mesh, frame, mask, 8, 4000, face=True, face_to_world=face_to_world)


def cut_bust(mesh, height):
    import trimesh

    try:
        cut = trimesh.intersections.slice_mesh_plane(
            mesh, [0, 0, 1], [0, 0, height], cap=True)
    except (ValueError, IndexError) as exc:
        raise FusionError(f"could not cap bust: {exc}") from exc
    if len(cut.vertices) == 0:
        raise FusionError("no mesh above bust cut")
    cut.apply_translation([0, 0, -height])
    return cut


def _fuse(mesh, frame, mask, yaw_starts, max_points, face=False, face_to_world=None):
    import trimesh
    from scipy import ndimage, sparse
    from scipy.spatial import cKDTree

    depth, K = frame.depth, frame.K
    h, w = depth.shape
    mask = _depth_mask(frame, mask)

    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 5.0)
    if frame.confidence is not None:
        valid &= frame.confidence >= 1
    P = backproject(np.nan_to_num(depth), K)

    up = frame.up_camera()
    floor = None if face else _fit_floor(P[valid & ~ndimage.binary_dilation(mask, iterations=3)], up)
    if floor is not None:
        up, origin = floor
    obj = valid & ndimage.binary_erosion(mask, iterations=2)
    pts = P[obj]
    if len(pts) < MIN_POINTS:
        raise FusionError(f"only {len(pts)} depth points on the object")
    med = np.median(pts[:, 2])
    mad = np.median(np.abs(pts[:, 2] - med))
    pts = pts[np.abs(pts[:, 2] - med) < max(0.1, 6 * mad)]
    if face:
        pts = pts[pts[:, 2] <= pts[:, 2].min() + 0.25]
        if face_to_world is not None:
            if frame.camera_to_world is None:
                raise FusionError("face_transform requires camera_transform")
            head = np.linalg.inv(frame.camera_to_world) @ face_to_world[:, 3]
            head_cv = CV_FROM_ARKIT @ head[:3]
            E = getattr(frame, "depth_to_photo", np.eye(4))
            origin = E[:3, :3].T @ (head_cv - E[:3, 3]) - 0.12 * up
        else:
            cut = max((pts @ up).max() - 0.24, (pts @ up).min())
            origin = up * cut
    elif floor is None:
        origin = pts[np.argmin(pts @ up)]
    pts = pts[(pts - origin) @ up > (0 if face else 0.004)]
    if len(pts) < MIN_POINTS:
        raise FusionError(f"only {len(pts)} depth points above the {'cut' if face else 'floor'}")

    # gravity frame G: z up, floor/cut at z=0, x along the camera's horizontal right
    # (a portrait frame has the camera's x vertical; use its y then)
    ref = np.eye(3)[0 if abs(up[0]) < 0.9 else 1]
    x = ref - (ref @ up) * up
    x /= np.linalg.norm(x)
    RG = np.stack([x, np.cross(up, x), up])
    to_g = lambda p: (p - origin) @ RG.T  # noqa: E731
    lidar = to_g(pts)
    if len(lidar) > max_points:
        lidar = lidar[np.random.default_rng(0).choice(len(lidar), max_points, replace=False)]

    mesh = mesh.copy()
    target = np.vstack([mesh.vertices, trimesh.sample.sample_surface(mesh, 20000, seed=0)[0]])
    tree = cKDTree(target)
    lo, hi = mesh.bounds
    s0 = (lidar[:, 2].max() - 0) / max(hi[2] - lo[2], 1e-9)
    fwd = RG @ [0, 0, 1.0]
    fwd[2] = 0
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    centre = lidar.mean(0)
    centre[2] = lidar[:, 2].max() if face else 0
    best = None
    for k in range(yaw_starts):
        Rm = _rot_z(2 * np.pi * k / yaw_starts)
        # mesh -> G guess: scale, yaw, bottom on floor or top aligned, behind visible side
        ext = (Rm @ (hi - lo) * s0)
        t = centre + fwd * 0.25 * np.abs(ext[:2]).mean() - s0 * Rm @ [(lo[0] + hi[0]) / 2,
                                                                     (lo[1] + hi[1]) / 2, hi[2] if face else lo[2]]
        for factor in ([1.0, 1.5, 2.0] if face else [1.0]):
            scale = s0 * factor
            translation = t if factor == 1 else centre + (t - centre) * factor
            # ICP moves the LiDAR points into mesh space: invert the guess
            init = (1 / scale, Rm.T, -(Rm.T @ translation) / scale)
            s, R, tt, rmse = _icp(lidar, tree, target, init)
            if best is None or rmse < best[3]:
                best = (s, R, tt, rmse)
    s, R, tt, rmse = best
    # mesh -> G is the inverse of G -> mesh
    v_g = (mesh.vertices - tt) @ R / s

    # pull the seen side onto the LiDAR surface
    v_c = v_g @ RG + origin
    z = v_c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(K[0, 0] * v_c[:, 0] / z + K[0, 2]).astype(np.int64)
        v = np.floor(K[1, 1] * v_c[:, 1] / z + K[1, 2]).astype(np.int64)
    inb = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    zbuf = np.full((h, w), np.inf)
    np.minimum.at(zbuf, (v[inb], u[inb]), z[inb])
    size = float(np.ptp(v_g, axis=0).max())
    normals = trimesh.Trimesh(v_c, mesh.faces, process=False).vertex_normals
    seen = np.zeros(len(v_c), bool)
    ui, vi = u[inb], v[inb]
    d_l = np.where(valid & mask, depth, np.nan)[vi, ui]
    seen[inb] = ((z[inb] <= zbuf[vi, ui] + 0.02 * size)
                 & (np.einsum("ij,ij->i", normals[inb], -v_c[inb]) > 0)
                 & np.isfinite(d_l) & (np.abs(d_l - z[inb]) < max(4 * rmse, 0.005)))
    disp = np.zeros_like(v_c)
    idx = np.flatnonzero(inb)[seen[inb]]
    disp[idx] = v_c[idx] / z[idx, None] * depth[v[idx], u[idx], None] - v_c[idx]

    e = mesh.edges_unique
    n = len(v_c)
    A = sparse.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                          shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    W = sparse.diags(1 / deg) @ A
    fixed = disp[seen].copy()
    for _ in range(3):  # smooth LiDAR noise on the seen side
        disp[seen] = 0.5 * fixed + 0.5 * (W @ disp)[seen]
    for _ in range(15):  # fade the correction into the hidden side
        disp[~seen] = 0.85 * (W @ disp)[~seen]
    fused = trimesh.Trimesh(to_g(v_c + disp), mesh.faces, process=False)
    if face:
        fused = cut_bust(fused, 0)

    report = {
        "lidar_points": int(len(pts)),
        "icp_rmse_mm": round(rmse * 1000, 2),
        "scale": round(float(1 / s), 5),
        "seen_vertices": round(float(seen.mean()), 3),
    }
    if face:
        report["cut"] = "face anchor" if face_to_world is not None else "top - 24 cm"
    else:
        report["floor"] = ("camera gravity + plane" if floor is not None and frame.camera_to_world is not None
                           else "plane" if floor is not None else "lowest point")
    return fused, report
