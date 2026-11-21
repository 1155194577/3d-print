"""tools/printability.py: what stops a print (errors) vs what is worth a warning."""

import pytest
import trimesh
from fastapi import HTTPException

from server import print_api as api  # puts tools/ on sys.path

import printability as p  # noqa: E402


def on_bed(m):
    m.apply_translation([0, 0, -m.bounds[0, 2]])
    return m


def box(*ex):
    return on_bed(trimesh.creation.box(extents=ex))


def test_solid_box_prints_cleanly():
    r = p.check(box(20, 20, 20))
    assert r["printable"] and not r["errors"] and not r["warnings"] and not r["repairable"]
    assert r["base_contact_mm2"] == pytest.approx(400, rel=0.01)
    assert r["solid_mass_g"] == pytest.approx(8 * 1.24, rel=0.01)


def test_open_mesh_is_an_error():
    m = box(20, 20, 20)
    m.update_faces(m.face_normals[:, 2] < 0.9)  # cut the lid off
    r = p.check(m)
    assert not r["printable"] and "watertight" in r["errors"][0] and r["repairable"]


def test_bed_too_small_is_an_error_otherwise_a_warning():
    m = box(250, 100, 50)
    assert not p.check(m, bed=(220, 220, 250))["printable"]
    assert p.check(m, bed=(100, 260, 250))["printable"]  # turned on the bed
    r = p.check(m)
    assert r["printable"] and any("220" in w for w in r["warnings"])


def test_thin_walls_floating_parts_and_overhangs_are_warnings():
    r = p.check(box(40, 40, 0.4))
    assert r["printable"] and any("thinner" in w for w in r["warnings"])
    assert r["min_wall_mm"] == pytest.approx(0.4, abs=0.05)

    floating = trimesh.util.concatenate([box(10, 10, 10),
                                         box(5, 5, 5).apply_translation([20, 0, 20])])
    r = p.check(floating)
    assert r["printable"] and r["repairable"] and any("float" in w for w in r["warnings"])

    cone = trimesh.creation.cone(radius=30, height=10, sections=64)
    cone.apply_transform(trimesh.transformations.rotation_matrix(3.14159, [1, 0, 0]))
    r = p.check(on_bed(cone))
    assert r["printable"] and r["overhang_pct"] > 50
    assert any("supports" in w for w in r["warnings"])


def test_server_repairs_or_refuses():
    floating = trimesh.util.concatenate([box(10, 10, 10),
                                         box(5, 5, 5).apply_translation([20, 0, 20])])
    body, check = api._check_printable(floating.export(file_type="stl"), None)
    assert check["printable"] and check["parts"] == 1 and check["repaired"]
    assert trimesh.load(trimesh.util.wrap_as_stream(body), file_type="stl").is_watertight

    with pytest.raises(HTTPException) as exc:
        api._check_printable(box(300, 50, 50).export(file_type="stl"), (220, 220, 250))
    assert exc.value.status_code == 422 and "bed" in exc.value.detail


def test_sealed_cavity_is_not_a_floating_part():
    inner = box(10, 10, 10).apply_translation([0, 0, 5])
    inner.invert()
    hollow = trimesh.util.concatenate([box(30, 30, 30), inner])
    r = p.check(hollow)
    assert r["parts"] == 2 and not any("float" in w for w in r["warnings"])
    assert p.fix(hollow).volume == pytest.approx(30 ** 3 - 10 ** 3, rel=1e-6)


def test_fit_bed_allows_stl_rounding():
    assert p.check(box(220.00002, 100, 50), bed=(220, 220, 250))["printable"]


def test_tiny_model_wall_check_is_cheap():
    import tracemalloc

    tracemalloc.start()
    r = p.check(box(1, 1, 1))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert r["min_wall_mm"] is not None and peak < 200e6
