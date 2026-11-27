"""Photo + LiDAR depth: a synthetic ARFrame of a box on a floor."""

import io
import json

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from server import print_api as api
from server.app import Settings, create_app

W, H = 256, 192
PW, PH = 1920, 1440
INTR = {"fx": 1500.0, "fy": 1500.0, "cx": 960.0, "cy": 720.0, "width": PW, "height": PH}
BOX = np.array([[-0.05, 0.0, -0.03], [0.05, 0.15, 0.03]])  # ARKit world, Y up, metres


def camera_to_world(eye=(0.12, 0.33, 0.42), target=(0.0, 0.07, 0.0)):
    eye, target = np.array(eye), np.array(target)
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(f, [0, 1.0, 0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4)
    M[:3, :3] = np.stack([r, u, -f], axis=1)
    M[:3, 3] = eye
    return M


def render(M):
    """Depth (cv z, metres) and object mask by ray-casting the box and the floor."""
    s = W / PW
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d_cv = np.stack([(u - INTR["cx"] * s) / (INTR["fx"] * s),
                     (v - INTR["cy"] * s) / (INTR["fy"] * s), np.ones_like(u)], -1)
    d = (d_cv * [1, -1, -1]) @ M[:3, :3].T
    o = M[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(d[..., 1] < 0, -o[1] / d[..., 1], np.inf)
        t1, t2 = (BOX[0] - o) / d, (BOX[1] - o) / d
    tn = np.nanmax(np.minimum(t1, t2), -1)
    tf = np.nanmin(np.maximum(t1, t2), -1)
    hit = (tn <= tf) & (tn > 0)
    t = np.where(hit, tn, t_floor)
    return t.astype(np.float32), hit


class FakeTripoSR:
    """Returns the true box, but at TripoSR's arbitrary scale, yaw and offset (Z up)."""

    def __init__(self, mask):
        self.mask = mask
        self.turns = "unset"

    def __call__(self, image_path, out_path, quarter_turns=None, progress=None, input_path=None):
        if input_path is not None:
            from PIL import Image
            Image.new("RGB", (128, 128), "grey").save(input_path)
        self.turns = quarter_turns
        ext = BOX[1] - BOX[0]
        box = trimesh.creation.box(extents=[ext[0], ext[2], ext[1]]).subdivide().subdivide()
        box = box.subdivide()
        box.apply_scale(1 / 0.37)
        box.apply_transform(trimesh.transformations.rotation_matrix(1.2, [0, 0, 1]))
        box.apply_translation([0.1, -0.2, 0.05])
        box.export(out_path)

    def foreground_mask(self, image_path):
        return self.mask.astype(np.uint8) * 255


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv("PRINT_API_TOKEN", raising=False)
    with TestClient(create_app(Settings(jobs_dir=tmp_path / "jobs"))) as c:
        yield c


def photo_bytes():
    buf = io.BytesIO()
    from PIL import Image
    Image.new("RGB", (PW, PH), (120, 120, 120)).save(buf, "JPEG")
    return buf.getvalue()


def frame_request(depth, M=None, **extra):
    files = {"photo": ("frame.jpg", photo_bytes()),
             "depth": ("depth.f32", depth.astype("<f4").tobytes(), "application/octet-stream")}
    data = {"depth_width": str(W), "depth_height": str(H), "intrinsics": json.dumps(INTR),
            "base_plate": "false", **extra}
    if M is not None:
        data["camera_transform"] = json.dumps(M.T.ravel().tolist())
    return files, data


def footprint(stl):
    """Width and depth along the box's own axes (it stands yawed to the camera)."""
    xy = stl.vertices[:, :2]
    best = None
    for a in np.radians(np.arange(0, 90, 0.5)):
        c, s = np.cos(a), np.sin(a)
        ext = np.ptp(xy @ [[c, -s], [s, c]], axis=0)
        if best is None or ext.prod() < best.prod():
            best = ext
    return sorted(best)


def result(resp):
    assert resp.status_code == 200, resp.text
    stl = trimesh.load(io.BytesIO(resp.content), file_type="stl")
    return stl, json.loads(resp.headers["x-print-report"])


@pytest.mark.parametrize("with_transform", [True, False])
def test_depth_gives_true_size(client, monkeypatch, with_transform):
    M = camera_to_world()
    depth, mask = render(M)
    fake = FakeTripoSR(mask)
    monkeypatch.setattr(api, "reconstruct_photo", fake)
    files, data = frame_request(depth, M if with_transform else None, height_mm="999")
    stl, rep = result(client.post("/print", files=files, data=data))
    assert rep["fusion"]["lidar_points"] > 1000
    assert rep["fusion"]["icp_rmse_mm"] < 3
    # 100 x 60 x 150 mm box: height_mm is ignored when the depth is used
    assert stl.extents[2] == pytest.approx(150, abs=4)
    assert footprint(stl) == pytest.approx([60, 100], abs=4)
    assert stl.is_watertight
    assert fake.turns == 0


def test_portrait_frame_is_turned_upright(client, monkeypatch):
    M = camera_to_world()
    roll = np.eye(4)
    roll[:3, :3] = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])[:3, :3]
    M = M @ roll  # phone held in portrait: sensor x points down in the world
    depth, mask = render(M)
    fake = FakeTripoSR(mask)
    monkeypatch.setattr(api, "reconstruct_photo", fake)
    files, data = frame_request(depth, M)
    stl, rep = result(client.post("/print", files=files, data=data))
    assert fake.turns in (1, 3)
    assert stl.extents[2] == pytest.approx(150, abs=4)


def test_unusable_depth_falls_back_to_height(client, monkeypatch):
    M = camera_to_world()
    _, mask = render(M)
    monkeypatch.setattr(api, "reconstruct_photo", FakeTripoSR(mask))
    files, data = frame_request(np.full((H, W), np.nan), M, height_mm="80")
    stl, rep = result(client.post("/print", files=files, data=data))
    assert "skipped" in rep["fusion"]
    assert rep["level"]["source"] == "gravity"
    assert stl.extents[2] == pytest.approx(80, abs=0.5)


def test_skipped_fusion_preserves_sensor_gravity(client, monkeypatch):
    M = camera_to_world()
    roll = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    M = M @ roll
    depth, mask = render(M)
    fake = FakeTripoSR(mask)
    monkeypatch.setattr(api, "reconstruct_photo", fake)

    def skip(*args):
        raise api.depth_fusion.FusionError("forced skip")

    monkeypatch.setattr(api.depth_fusion, "fuse", skip)
    files, data = frame_request(depth, M, height_mm="80")
    _, rep = result(client.post("/print", files=files, data=data))
    assert fake.turns == api.depth_fusion.upright_turns(
        api.depth_fusion.CV_FROM_ARKIT @ (M[:3, :3].T @ [0., 1., 0.]))
    assert rep["fusion"] == {"skipped": "forced skip"}
    assert rep["level"]["source"] == "gravity"


@pytest.mark.parametrize("change, msg", [
    ({"depth_width": "100"}, "expected"),
    ({"intrinsics": "{}"}, "intrinsics"),
    ({"camera_transform": "[1, 2]"}, "16 floats"),
    ({"image_orientation": "5"}, "image_orientation"),
])
def test_bad_frame_fields(client, monkeypatch, change, msg):
    monkeypatch.setattr(api, "reconstruct_photo", FakeTripoSR(np.zeros((H, W), bool)))
    files, data = frame_request(np.ones((H, W)), camera_to_world())
    data.update(change)
    resp = client.post("/print", files=files, data=data)
    assert resp.status_code == 400 and msg in resp.json()["detail"]


def test_depth_needs_photo(client):
    resp = client.post("/print", files={"mesh": ("a.obj", b"v 0 0 0"),
                                        "depth": ("d.f32", b"\0\0\0\0")})
    assert resp.status_code == 400


def test_object_extra_view_fusion_and_cameras(client, monkeypatch):
    from server import model_check

    M = camera_to_world()
    depth, mask = render(M)
    other = camera_to_world(eye=(-0.12, 0.33, 0.42))
    other_depth, other_mask = render(other)
    fake = FakeTripoSR(mask)
    fake.foreground_mask = lambda path: (other_mask if "view_1" in path else mask).astype(np.uint8) * 255
    monkeypatch.setattr(api, "reconstruct_photo", fake)
    checked = []

    def check(photo, mesh, camera, views, can_retake):
        assert can_retake and len(views) == 1 and views[0]["name"] == "view_1"
        assert views[0]["side"] == "left"
        K, R, t = camera
        extra_K, extra_R, extra_t = views[0]["camera"]
        np.testing.assert_allclose(K, [[1500, 0, 960], [0, 1500, 720], [0, 0, 1]])
        np.testing.assert_allclose(extra_K, K)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.linalg.norm(extra_t - t) > .01
        assert not np.allclose(extra_R, R)
        checked.append(True)
        return {"ok": True}

    monkeypatch.setattr(model_check, "check", check)
    files, data = frame_request(depth, M, view_count="2", model_check="true")
    files.update(view_1_photo=("extra.jpg", photo_bytes()),
                 view_1_depth=("extra.f32", other_depth.astype("<f4").tobytes()))
    data.update(view_1_depth_width=str(W), view_1_depth_height=str(H),
                view_1_intrinsics=json.dumps(INTR), view_1_camera_transform=json.dumps(other.T.ravel().tolist()),
                view_1_image_orientation="1", view_1_side="left")
    _, rep = result(client.post("/print", files=files, data=data))
    assert rep["fusion"]["views_used"] == 2 and rep["fusion"]["views_skipped"] == []
    assert rep["model_check"] == {"ok": True} and checked
    assert "front" not in rep["fusion"]


def test_extra_views_require_primary_depth(client):
    response = client.post("/print", files={"photo": ("front.jpg", photo_bytes())}, data={"view_count": "2"})
    assert response.status_code == 400
    assert "primary photo with depth" in response.json()["detail"]


def test_photo_camera_uses_depth_extrinsics():
    frame = api.depth_fusion.Frame(np.ones((20, 30)), np.eye(3), photo_size=(60, 40))
    E = trimesh.transformations.rotation_matrix(.2, [0, 1, 0])
    E[:3, 3] = [.01, .02, .03]
    frame.depth_to_photo = E
    frame.photo_K = np.diag([100., 110., 1.])
    G = trimesh.transformations.rotation_matrix(.3, [1, 0, 0])[:3, :3]
    origin = np.array([.1, .2, .3])
    K, R, t = api.depth_fusion.photo_camera(frame, G, origin)
    np.testing.assert_allclose(K, frame.photo_K)
    np.testing.assert_allclose(R, E[:3, :3] @ G.T)
    np.testing.assert_allclose(t, E[:3, :3] @ origin + E[:3, 3])


def test_portrait_without_transform_or_floor(client, monkeypatch):
    roll = np.eye(4)
    roll[:3, :3] = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])[:3, :3]
    M = camera_to_world() @ roll
    depth, mask = render(M)
    depth[~mask] = np.nan  # no floor in view
    monkeypatch.setattr(api, "reconstruct_photo", FakeTripoSR(mask))
    files, data = frame_request(depth, image_orientation="6")
    stl, rep = result(client.post("/print", files=files, data=data))
    assert rep["fusion"]["floor"] == "lowest point" and stl.is_watertight


def test_corrupt_depth_png(client, monkeypatch):
    monkeypatch.setattr(api, "reconstruct_photo", FakeTripoSR(np.zeros((H, W), bool)))
    files, data = frame_request(np.ones((H, W)))
    files["depth"] = ("depth.png", b"not a png")
    resp = client.post("/print", files=files, data=data)
    assert resp.status_code == 400 and "PNG" in resp.json()["detail"]
