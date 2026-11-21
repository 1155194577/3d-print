"""Synthetic front TrueDepth busts and independently calibrated depth cameras."""

import json
import io

import numpy as np
import pytest
import trimesh

from server import print_api as api
from server.tests.test_fusion import client, photo_bytes, result
from depth_fusion import Frame, _depth_mask, backproject
from PIL import Image


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
        self.turns = "unset"

    def __call__(self, image_path, out_path, quarter_turns=None, progress=None):
        self.turns = quarter_turns
        mesh = self.mesh.copy()
        mesh.apply_scale(3.7)
        mesh.apply_translation([0.1, -0.2, 0.3])
        mesh.export(out_path)

    def foreground_mask(self, image_path):
        return self.mask.astype(np.uint8) * 255


def turned_head(yaw):
    truth = trimesh.creation.icosphere(subdivisions=4)
    truth.apply_scale([0.075, 0.085, 0.12])
    p = truth.vertices
    p[:, 1] -= 0.016 * np.exp(-((p[:, 0] / 0.022) ** 2 + (p[:, 2] / 0.035) ** 2)) * (p[:, 1] < 0)
    anchor = trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 1, 0])
    anchor[:3, 3] = [0, 0, -0.4]
    basis = np.eye(4)
    basis[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    cv = np.diag([1., -1., -1., 1.]) @ anchor @ basis
    rendered = truth.copy()
    rendered.apply_transform(cv)
    w, h, f = 128, 160, 200
    u, v = np.meshgrid(np.arange(w) + .5, np.arange(h) + .5)
    rays = np.stack([(u-w/2)/f, (v-h/2)/f, np.ones_like(u)], -1).reshape(-1, 3)
    locations, indices, _ = rendered.ray.intersects_location(np.zeros_like(rays), rays, multiple_hits=False)
    depth = np.full(w*h, np.nan)
    depth[indices] = locations[:, 2]
    frame = Frame(depth.reshape(h, w), np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]),
                  camera_to_world=np.eye(4))
    return truth, frame, np.isfinite(frame.depth), anchor


def test_multiview_turning_head_improves_sides():
    truth, front, mask, anchor = turned_head(0)
    guess = truth.copy()
    guess.vertices[:, 0] *= .91
    views = []
    for yaw in [35, -35, 65, -65]:
        _, frame, other_mask, other_anchor = turned_head(yaw)
        views.append(dict(frame=frame, mask=other_mask, face_to_world=other_anchor))
    single, one = api.depth_fusion.fuse_face(guess, front, mask, anchor)
    multi, many = api.depth_fusion.fuse_face(guess, front, mask, anchor, views)
    assert many["views_used"] == 5
    assert many["seen_vertices"] > one["seen_vertices"]
    # Compare side widths against the measured 150 mm ellipsoid width.
    assert abs(multi.extents[0] - .15) < abs(single.extents[0] - .15)


def test_multiview_skips_sparse_and_missing_pose():
    truth, frame, mask, anchor = turned_head(0)
    sparse = Frame(np.full_like(frame.depth, np.nan), frame.K, camera_to_world=np.eye(4))
    _, report = api.depth_fusion.fuse_face(truth, frame, mask, anchor, [
        dict(frame=sparse, mask=mask, face_to_world=anchor),
        dict(frame=frame, mask=mask, face_to_world=None)])
    assert report["views_used"] == 1
    assert "0 valid head points" in report["views_skipped"][0]["reason"]
    assert "missing" in report["views_skipped"][1]["reason"]


@pytest.mark.parametrize("count", ["6", "0", "bad", "1.5"])
def test_invalid_view_count(client, count):
    response = client.post("/print", data={"view_count": count},
                           files={"photo": ("face.jpg", photo_bytes())})
    assert response.status_code == 400
    assert "view_count" in response.json()["detail"]


def test_multiview_request_validation(client, monkeypatch):
    mesh, depth, mask, intr = face_frame()
    monkeypatch.setattr(api, "reconstruct_photo", FakeFace(mesh, mask))
    fields = dict(mode="face", view_count="2", depth_width=str(depth.shape[1]),
                  depth_height=str(depth.shape[0]), intrinsics=json.dumps(intr))
    files = {"photo": ("face.jpg", photo_bytes()), "depth": ("depth.f32", depth.tobytes()),
             "view_1_photo": ("face.jpg", photo_bytes()), "view_1_depth": ("depth.f32", depth.tobytes())}
    fields.update({"view_1_" + key: fields[key] for key in ("depth_width", "depth_height", "intrinsics")})
    fields["view_1_face_transform"] = "{"
    assert client.post("/print", data=fields, files=files).status_code == 400
    del fields["view_1_face_transform"]
    _, report = result(client.post("/print", data=fields, files=files))
    assert report["views_used"] == 1
    assert "missing" in report["fusion"]["views_skipped"][0]["reason"]
    fields["mode"] = "object"
    assert client.post("/print", data=fields, files=files).status_code == 400


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
    assert "level" not in report
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


@pytest.mark.parametrize("separate_depth", [False, True])
@pytest.mark.parametrize("field", ["intrinsics", "depth_intrinsics"])
def test_calibration_dimensions_bounded(tmp_path, field, separate_depth):
    path = tmp_path / "depth.f32"
    path.write_bytes(np.ones((8, 8), dtype="<f4").tobytes())
    calibration = dict(fx=8, fy=8, cx=4, cy=4, width=8, height=8)
    kwargs = {"intrinsics": json.dumps(calibration)}
    if separate_depth:
        kwargs["depth_intrinsics"] = json.dumps(calibration)
    kwargs[field] = json.dumps({**calibration, "width": 1000000000})
    with pytest.raises(ValueError, match=field):
        Frame.load(str(path), 8, 8, **kwargs)


def test_calibrated_mask_uses_native_resolution(tmp_path, monkeypatch):
    path = tmp_path / "depth.f32"
    path.write_bytes(np.ones((8, 8), dtype="<f4").tobytes())
    photo = dict(fx=16000, fy=16000, cx=8000, cy=8000, width=16000, height=16000)
    depth = dict(fx=8, fy=8, cx=4, cy=4, width=8, height=8)
    frame = Frame.load(str(path), 8, 8, intrinsics=json.dumps(photo),
                       depth_intrinsics=json.dumps(depth))
    mask = np.zeros((16, 16), bool)
    mask[4:12, 4:12] = True

    def reject_resize(*args, **kwargs):
        pytest.fail("calibrated registration must not resize the mask")

    monkeypatch.setattr(Image.Image, "resize", reject_resize)
    expected = np.zeros((8, 8), bool)
    expected[2:6, 2:6] = True
    assert np.array_equal(_depth_mask(frame, mask), expected)


@pytest.mark.parametrize("tap", [None, [20, 24], [44, 24], [32, 24]])
def test_calibrated_mask_bridges_invalid_depth_stripe(tmp_path, tap):
    depth = np.full((48, 64), 0.4, dtype="<f4")
    depth[:, 31:33] = np.nan
    path = tmp_path / "depth.f32"
    path.write_bytes(depth.tobytes())
    calibration = json.dumps(dict(fx=55, fy=55, cx=32, cy=24, width=64, height=48))
    frame = Frame.load(str(path), 64, 48, intrinsics=calibration,
                       depth_intrinsics=calibration,
                       object_points=json.dumps([tap]) if tap is not None else None)
    mask = np.zeros((48, 64), bool)
    mask[8:40, 8:56] = True
    expected = mask & np.isfinite(depth)
    mask[1:4, 1:4] = True
    actual = _depth_mask(frame, mask)
    assert np.array_equal(actual, expected)
    assert actual[:, :31].any() and actual[:, 33:].any()
    assert not actual[:, 31:33].any()


def test_calibrated_mask_without_valid_depth(tmp_path):
    path = tmp_path / "depth.f32"
    path.write_bytes(np.full((8, 8), np.nan, dtype="<f4").tobytes())
    calibration = json.dumps(dict(fx=8, fy=8, cx=4, cy=4, width=8, height=8))
    frame = Frame.load(str(path), 8, 8, intrinsics=calibration,
                       depth_intrinsics=calibration, object_points="[[4, 4]]")
    assert not _depth_mask(frame, np.ones((8, 8), bool)).any()


def test_calibrated_mask_keeps_nearby_objects_separate(tmp_path):
    path = tmp_path / "depth.f32"
    path.write_bytes(np.full((48, 64), 0.4, dtype="<f4").tobytes())
    calibration = json.dumps(dict(fx=55, fy=55, cx=32, cy=24, width=64, height=48))
    frame = Frame.load(str(path), 64, 48, intrinsics=calibration,
                       depth_intrinsics=calibration, object_points="[[20, 24]]")
    expected = np.zeros((48, 64), bool)
    expected[8:40, 8:31] = True
    mask = expected.copy()
    mask[8:40, 33:56] = True
    assert np.array_equal(_depth_mask(frame, mask), expected)


@pytest.mark.parametrize("with_depth", [False, True])
@pytest.mark.parametrize("vertices_as_file", [False, True])
def test_ios_face_capture_payload(client, monkeypatch, with_depth, vertices_as_file):
    mesh, depth, mask, intr = face_frame()
    depth, mask = np.rot90(depth).copy(), np.rot90(mask).copy()
    intr.update(cx=96, cy=80, width=192, height=160)
    fake = FakeFace(mesh, mask)
    monkeypatch.setattr(api, "reconstruct_photo", fake)
    camera = np.eye(4)
    camera[:3, :3] = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])[:3, :3]
    anchor = np.eye(4)
    anchor[:3, 3] = camera[:3, :3] @ [0, 0, -0.4]
    data = dict(mode="face", image_orientation="6", intrinsics=json.dumps(intr),
                camera_transform=json.dumps(camera.T.ravel().tolist()),
                face_transform=json.dumps(anchor.T.ravel().tolist()), height_mm="90")
    photo = io.BytesIO()
    Image.new("RGB", (intr["width"], intr["height"])).save(photo, format="JPEG")
    files = {"photo": ("face.jpg", photo.getvalue())}
    vertices = np.zeros((4, 3), dtype="<f4").tobytes()
    if vertices_as_file:
        files["face_vertices"] = ("face.f32", vertices)
        files["face_vertex_count"] = ("count.txt", b"4")
    else:
        data.update(face_vertices=vertices.decode("ascii"), face_vertex_count="4")
    if with_depth:
        files["depth"] = ("face.f32", depth.tobytes())
        data.update(depth_width=depth.shape[1], depth_height=depth.shape[0],
                    depth_intrinsics=json.dumps(intr),
                    depth_extrinsics=json.dumps(np.eye(4).T.ravel().tolist()))
    stl, report = result(client.post("/print", data=data, files=files))
    assert report["mode"] == "face"
    assert stl.is_watertight
    assert stl.extents[2] == pytest.approx(92.5, abs=0.5)
    assert fake.turns == 3


@pytest.mark.parametrize("orientation,turns", [(None, None), ("1", 0), ("6", 3)])
def test_photo_only_orientation_preserves_exif_default(client, monkeypatch, orientation, turns):
    mesh, _, mask, _ = face_frame()
    fake = FakeFace(mesh, mask)
    monkeypatch.setattr(api, "reconstruct_photo", fake)
    data = {} if orientation is None else {"image_orientation": orientation}
    response = client.post("/print", data=data, files={"photo": ("photo.png", photo_bytes())})
    assert response.status_code == 200
    assert fake.turns == turns
