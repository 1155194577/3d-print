"""Synthetic front TrueDepth busts and independently calibrated depth cameras."""

import json

import numpy as np
import pytest
import trimesh

from server import print_api as api
from server.tests.test_fusion import client, photo_bytes, result
from depth_fusion import Frame, _depth_mask, backproject


def head_mesh():
    head = trimesh.creation.icosphere(subdivisions=3)
    head.apply_scale([0.075, 0.085, 0.12])
    neck = trimesh.creation.cylinder(radius=0.035, height=0.15, sections=32)
    neck.apply_translation([0, 0, -0.15])
    shoulders = trimesh.creation.box([0.32, 0.14, 0.08])
    shoulders.apply_translation([0, 0, -0.25])
    return trimesh.boolean.union([head, neck, shoulders], engine="manifold")


def face_frame():
    mesh = head_mesh()
    w, h, f = 160, 192, 230
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(u - w / 2) / f, np.ones_like(u), -(v - h / 2) / f], -1)
    origins = np.tile([0, -0.4, 0], (w * h, 1))
    # Analytic ellipsoid, cylinder and box intersections, in mesh (Z-up) space.
    d = rays.reshape(-1, 3)
    o = origins
    radii = np.array([0.075, 0.085, 0.12])
    a = np.sum((d / radii) ** 2, axis=1)
    b = 2 * np.sum(o * d / radii ** 2, axis=1)
    c = np.sum((o / radii) ** 2, axis=1) - 1
    disc = b * b - 4 * a * c
    t_head = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
    a = np.sum(d[:, :2] ** 2, axis=1)
    b = 2 * np.sum(o[:, :2] * d[:, :2], axis=1)
    c = np.sum(o[:, :2] ** 2, axis=1) - 0.035 ** 2
    disc = b * b - 4 * a * c
    t_neck = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    z = t_neck * d[:, 2]
    t_neck = np.where((disc >= 0) & (z >= -0.225) & (z <= -0.075), t_neck, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (np.array([-0.16, -0.07, -0.29]) - o) / d
        t2 = (np.array([0.16, 0.07, -0.21]) - o) / d
    near, far = np.min(np.maximum(t1, t2), axis=1), np.max(np.minimum(t1, t2), axis=1)
    t_box = np.where((far > 0) & (far <= near), far, np.inf)
    depth = np.minimum.reduce([t_head, t_neck, t_box]).reshape(h, w).astype("<f4")
    mask = np.isfinite(depth)
    depth[~mask] = np.nan
    intr = dict(fx=f, fy=f, cx=w / 2, cy=h / 2, width=w, height=h)
    return mesh, depth, mask, intr


class FakeFace:
    def __init__(self, mesh, mask):
        self.mesh, self.mask = mesh, mask

    def __call__(self, image_path, out_path, quarter_turns=None):
        mesh = self.mesh.copy()
        mesh.apply_scale(3.7)
        mesh.apply_translation([0.1, -0.2, 0.3])
        mesh.export(out_path)

    def foreground_mask(self, image_path):
        return self.mask.astype(np.uint8) * 255


@pytest.mark.parametrize("anchor", [False, True])
def test_face_depth_bust(client, monkeypatch, anchor):
    mesh, depth, mask, intr = face_frame()
    monkeypatch.setattr(api, "reconstruct_photo", FakeFace(mesh, mask))
    fused_meshes = []
    fuse_face = api.depth_fusion.fuse_face

    def record_fusion(*args, **kwargs):
        fused, report = fuse_face(*args, **kwargs)
        fused_meshes.append(fused)
        return fused, report

    monkeypatch.setattr(api.depth_fusion, "fuse_face", record_fusion)
    camera = np.eye(4)
    camera[:3, 3] = [0.1, 0.2, 0.3]
    data = dict(mode="face", depth_width=depth.shape[1], depth_height=depth.shape[0],
                intrinsics=json.dumps(intr), camera_transform=json.dumps(camera.T.ravel().tolist()))
    if anchor:
        transform = np.eye(4)
        transform[:3, 3] = camera[:3, 3] + [0, 0, -0.4]
        data["face_transform"] = json.dumps(transform.T.ravel().tolist())
    stl, report = result(client.post("/print", data=data, files={
        "photo": ("face.png", photo_bytes()), "depth": ("face.f32", depth.tobytes())}))
    assert report["mode"] == "face"
    assert report["fusion"]["cut"] == ("face anchor" if anchor else "top - 24 cm")
    fused = fused_meshes[0]
    assert fused.bounds[0, 2] == pytest.approx(0, abs=1e-7)
    assert fused.bounds[1, 2] == pytest.approx(0.24, abs=0.025)
    assert fused.extents[0] < 0.18
    assert stl.is_watertight
    assert stl.extents[2] == pytest.approx(102.5, abs=0.5)
    assert stl.extents[0] < 95  # shoulders would be wider than the bust is tall
    bottom = stl.triangles[np.all(np.isclose(stl.triangles[:, :, 2], stl.bounds[0, 2]), axis=1)]
    assert len(bottom) >= 2


@pytest.mark.parametrize("bad_depth", [False, True])
@pytest.mark.parametrize("height", [None, 75])
def test_face_photo_fallback(client, monkeypatch, bad_depth, height):
    mesh, depth, mask, intr = face_frame()
    monkeypatch.setattr(api, "reconstruct_photo", FakeFace(mesh, mask))
    data = dict(mode="face")
    if height is not None:
        data["height_mm"] = str(height)
    files = {"photo": ("face.png", photo_bytes())}
    if bad_depth:
        files["depth"] = ("face.f32", np.full_like(depth, np.nan).tobytes())
        data.update(depth_width=depth.shape[1], depth_height=depth.shape[0], intrinsics=json.dumps(intr))
    stl, report = result(client.post("/print", data=data, files=files))
    assert stl.is_watertight
    assert stl.extents[2] == pytest.approx((height or 100) + 2.5, abs=0.5)
    assert stl.extents[0] < (height or 100) * 0.85
    if bad_depth:
        assert "skipped" in report["fusion"]


@pytest.mark.parametrize("transform", ["oops", "[1,2]", "{}", "[null]", "[NaN]"])
def test_bad_face_transform(client, transform):
    response = client.post("/print", data={"mode": "face", "face_transform": transform},
                           files={"photo": ("a.png", photo_bytes())})
    assert response.status_code == 400


def test_face_requires_photo(client):
    response = client.post("/print", data={"mode": "face"}, files={"mesh": ("a.obj", b"v 0 0 0")})
    assert response.status_code == 400


@pytest.mark.parametrize("shift", [0, 0.025])
def test_depth_calibration_registration(tmp_path, shift):
    depth = np.full((48, 64), 0.4, dtype="<f4")
    path = tmp_path / "depth.f32"
    path.write_bytes(depth.tobytes())
    photo_k = dict(fx=95, fy=95, cx=64, cy=36, width=128, height=72)
    depth_k = dict(fx=55, fy=55, cx=32, cy=24, width=64, height=48)
    E = np.eye(4)
    E[:3, :3] = trimesh.transformations.rotation_matrix(shift, [0, 0, 1])[:3, :3]
    E[0, 3] = shift
    frame = Frame.load(str(path), 64, 48, intrinsics=json.dumps(photo_k),
                       depth_intrinsics=json.dumps(depth_k), depth_extrinsics=json.dumps(E.T.ravel().tolist()),
                       object_points=json.dumps([[90, 30]]))
    mask = np.zeros((72, 128), bool)
    mask[18:43, 78:105] = True
    actual = _depth_mask(frame, mask)
    p = backproject(depth, frame.K) @ E[:3, :3].T + E[:3, 3]
    x = np.floor(95 * p[..., 0] / p[..., 2] + 64).astype(int)
    y = np.floor(95 * p[..., 1] / p[..., 2] + 36).astype(int)
    expected = (x >= 78) & (x < 105) & (y >= 18) & (y < 43)
    assert np.array_equal(actual, expected)
    assert np.allclose(frame.up_camera(), E[:3, :3].T @ [0, -1, 0])
    legacy = Frame.load(str(path), 64, 48, intrinsics=json.dumps(photo_k))
    assert not np.array_equal(actual, _depth_mask(legacy, mask))


@pytest.mark.parametrize("field,value", [
    ("depth_intrinsics", "{}"), ("depth_intrinsics", '{"fx":-1}'),
    ("depth_extrinsics", "[1,2]"), ("depth_extrinsics", "null"),
])
def test_bad_depth_calibration(client, field, value):
    response = client.post("/print", data={
        "intrinsics": json.dumps(dict(fx=1, fy=1, cx=0, cy=0, width=1, height=1)),
        "depth_width": 1, "depth_height": 1, field: value,
    }, files={"photo": ("a.png", photo_bytes()),
              "depth": ("a.f32", np.ones(1, dtype="<f4").tobytes())})
    assert response.status_code == 400


def test_invalid_mode(client):
    response = client.post("/print", data={"mode": "portrait"}, files={"photo": ("a.png", photo_bytes())})
    assert response.status_code == 422
