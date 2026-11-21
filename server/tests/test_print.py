"""POST /print tests. Mesh paths run the real cleanup; the photo path swaps
TripoSR for a fixed cylinder, so no GPU or weights are needed."""

import io
import json

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from server import print_api as api
from server.app import Settings, create_app

# A 10 x 20 x 30 cm box as ARKit exports it: metres, Y-up (30 cm tall on Y).
W, H, D = 0.10, 0.30, 0.20


def scan_box():
    m = trimesh.creation.box(extents=[W, H, D])
    m.apply_translation([0.5, 1.2, -0.3])  # scans are not centred
    return m


def usdz_bytes(tmp_path, metres_per_unit=1.0):
    from pxr import Usd, UsdGeom, UsdUtils

    m = scan_box()
    usdc = tmp_path / "scan.usdc"
    stage = Usd.Stage.CreateNew(str(usdc))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, metres_per_unit)
    geo = UsdGeom.Mesh.Define(stage, "/Scan/Mesh")
    geo.GetPointsAttr().Set([tuple(v / metres_per_unit) for v in m.vertices])
    geo.GetFaceVertexCountsAttr().Set([3] * len(m.faces))
    geo.GetFaceVertexIndicesAttr().Set([int(i) for i in m.faces.ravel()])
    stage.GetRootLayer().Save()
    usdz = tmp_path / "scan.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(usdc), str(usdz))
    return usdz.read_bytes()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv("PRINT_API_TOKEN", raising=False)
    with TestClient(create_app(Settings(jobs_dir=tmp_path / "jobs"))) as c:
        yield c


def post(client, files, data=None, headers=None):
    return client.post("/print", files=files, data=data or {}, headers=headers or {})


def report(resp):
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "model/stl"
    stl = trimesh.load(io.BytesIO(resp.content), file_type="stl")
    return stl, json.loads(resp.headers["x-print-report"])


@pytest.mark.parametrize("ext", ["obj", "ply"])
def test_mesh_keeps_real_size_and_stands_up(client, ext):
    data = scan_box().export(file_type=ext)
    data = data.encode() if isinstance(data, str) else data
    stl, rep = report(post(client, {"mesh": (f"scan.{ext}", data)}))
    x, y, z = stl.extents
    # Y-up 300 mm becomes Z; the plate adds 2.5 mm below and a 9 mm margin around.
    assert z == pytest.approx(300 + 2.5, abs=0.5)
    assert x == pytest.approx(100 + 18, abs=0.5)
    assert y == pytest.approx(200 + 18, abs=0.5)
    assert rep["watertight"] and stl.is_watertight
    assert rep["printability"]["printable"] and rep["printability"]["errors"] == []


@pytest.mark.parametrize("units", ["m", "cm", "mm"])
def test_usdz_uses_its_own_units(client, tmp_path, units):
    stl, _ = report(post(client, {"mesh": ("scan.usdz", usdz_bytes(tmp_path, 0.01))},
                         {"base_plate": "false", "units": units}))
    assert sorted(np.round(stl.extents)) == [100, 200, 300]
    assert stl.extents[2] == pytest.approx(300, abs=0.5)


def test_mesh_height_override_and_fit_bed(client):
    data = scan_box().export(file_type="obj")
    stl, _ = report(post(client, {"mesh": ("scan.obj", data)},
                         {"height_mm": "150", "base_plate": "false"}))
    assert stl.extents[2] == pytest.approx(150, abs=0.5)
    stl, rep = report(post(client, {"mesh": ("scan.obj", data)},
                           {"fit_bed": "220x220x200", "base_plate": "false"}))
    assert stl.extents[2] <= 200.01 and "shrunk" in rep["scale"]


def test_photo_uses_measured_height(client, monkeypatch):
    def fake_triposr(image_path, out_path, **_):
        trimesh.creation.cylinder(radius=0.3, height=0.8).export(out_path)
    monkeypatch.setattr(api, "reconstruct_photo", fake_triposr)
    stl, rep = report(post(client, {"photo": ("can.jpg", b"jpeg")}, {"height_mm": "123"}))
    # height_mm is the object; the plate adds 2.5 mm below it.
    assert stl.extents[2] == pytest.approx(125.5, abs=0.5)
    assert rep["extents_mm"][2] == pytest.approx(125.5, abs=0.5)
    stl, _ = report(post(client, {"photo": ("can.jpg", b"jpeg")}))
    assert stl.extents[2] == pytest.approx(200 + 2.5, abs=1)


@pytest.mark.parametrize("files,data,code", [
    ({}, {}, 400),
    ({"mesh": ("a.obj", b"x"), "photo": ("a.jpg", b"x")}, {}, 400),
    ({"mesh": ("a.fbx", b"x")}, {}, 415),
    ({"photo": ("a.heic", b"x")}, {}, 415),
    ({"mesh": ("a.obj", b"")}, {}, 400),
    ({"mesh": ("a.obj", b"x")}, {"height_mm": "-5"}, 422),
    ({"mesh": ("a.obj", b"x")}, {"fit_bed": "220x220"}, 400),
    ({"mesh": ("a.obj", b"not a mesh")}, {}, 422),
])
def test_bad_requests(client, files, data, code):
    resp = post(client, files, data)
    assert resp.status_code == code, resp.text
    assert "detail" in resp.json()


def test_bearer_token(client, monkeypatch):
    monkeypatch.setenv("PRINT_API_TOKEN", "s3cret")
    data = scan_box().export(file_type="obj")
    assert post(client, {"mesh": ("scan.obj", data)}).status_code == 401
    ok = post(client, {"mesh": ("scan.obj", data)},
              headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_bad_token_rejected_before_upload_is_read(client, monkeypatch):
    monkeypatch.setenv("PRINT_API_TOKEN", "s3cret")
    monkeypatch.setattr(api, "_save_upload", lambda *a: pytest.fail("upload was read"))
    data = scan_box().export(file_type="obj")
    resp = post(client, {"mesh": ("scan.obj", data)}, headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401 and "detail" in resp.json()


def test_print_only_app(monkeypatch):
    monkeypatch.setenv("PRINT_API_TOKEN", "s3cret")
    with TestClient(api.create_print_app()) as c:
        assert c.get("/api/health").json() == {"status": "ok"}
        assert c.post("/api/jobs").status_code in (404, 405)
        data = scan_box().export(file_type="obj")
        assert post(c, {"mesh": ("scan.obj", data)}).status_code == 401
        ok = post(c, {"mesh": ("scan.obj", data)}, headers={"Authorization": "Bearer s3cret"})
        report(ok)
