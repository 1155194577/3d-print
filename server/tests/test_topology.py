import numpy as np
import pytest
import trimesh

from tools import topology


@pytest.mark.parametrize("shape", ["box", "cylinder"])
def test_optimise_preserves_exterior_and_encloses_cavities(shape):
    mesh = (trimesh.creation.box(extents=[40, 40, 60]) if shape == "box"
            else trimesh.creation.cylinder(radius=20, height=60, sections=48))
    mesh.apply_translation([13, -7, 30])
    result, report = topology.optimise(mesh)
    assert "skipped" not in report, report
    assert result.is_volume and result.is_watertight
    np.testing.assert_allclose(result.bounds, mesh.bounds, atol=1e-5)
    assert result.volume < mesh.volume * .95
    assert report["volume_saved_pct"] >= 5
    assert report["volume_cm3_after"] == round(result.volume / 1000, 1)
    assert report["volume_cm3_before"] == round(mesh.volume / 1000, 1)
    surfaces = result.split(only_watertight=False)
    outer = max(surfaces, key=lambda m: abs(m.volume))
    assert len(surfaces) > 1
    assert abs(outer.volume) == pytest.approx(mesh.volume, rel=1e-5)
    for cavity in surfaces:
        if cavity is outer:
            continue
        points = np.vstack([cavity.vertices, cavity.triangles_center])
        # Analytic distances avoid ray/proximity engines (no Embree on the box).
        z_clearance = np.minimum(points[:, 2], 60 - points[:, 2])
        xy = points[:, :2] - [13, -7]
        side = (20 - np.abs(xy).max(axis=1) if shape == "box"
                else 20 * np.cos(np.pi / 48) - np.linalg.norm(xy, axis=1))
        assert np.min(np.minimum(z_clearance, side)) >= 2 - 1e-4
        assert points[:, 2].min() >= 3 - 1e-4


def test_sparse_bracket_with_large_padded_grid_still_optimises(monkeypatch):
    arms = []
    for axis in range(3):
        extents = np.full(3, 40.)
        extents[axis] = 120
        arm = trimesh.creation.box(extents=extents)
        arm.apply_translation(extents / 2)
        arms.append(arm)
    bracket = trimesh.boolean.union(arms, engine="manifold")
    original = topology._densities
    grid_sizes = []
    def tracked(solid, *args):
        grid_sizes.append(solid.size)
        assert solid.sum() <= 6000
        return original(solid, *args)
    monkeypatch.setattr(topology, "_densities", tracked)
    result, report = topology.optimise(bracket)
    assert grid_sizes and grid_sizes[0] > 12000
    assert "skipped" not in report, report
    assert result.is_volume and result.volume < bracket.volume * .95
    np.testing.assert_allclose(result.bounds, bracket.bounds)


@pytest.mark.parametrize("mesh", [trimesh.Trimesh(), trimesh.creation.box(extents=[1, 1, 1])])
def test_empty_or_no_interior_is_unchanged(mesh):
    result, report = topology.optimise(mesh)
    assert result is mesh
    assert "skipped" in report


def test_solver_failure_is_unchanged(monkeypatch):
    mesh = trimesh.creation.box(extents=[40, 40, 60])
    def fail(*args, **kwargs):
        raise RuntimeError("solver unavailable")
    monkeypatch.setattr(topology, "_densities", fail)
    result, report = topology.optimise(mesh)
    assert result is mesh
    assert report == {"skipped": "optimisation failed: solver unavailable"}


def test_nonconverged_solver_with_large_residual_is_unchanged(monkeypatch):
    mesh = trimesh.creation.box(extents=[40, 40, 60])
    monkeypatch.setattr(topology, "cg", lambda matrix, force, **kwargs: (np.zeros_like(force), 250))
    result, report = topology.optimise(mesh)
    assert result is mesh
    assert report == {"skipped": "optimisation failed: elasticity solve residual 1 exceeds tolerance"}


def test_large_sample_accepts_small_residual_despite_iteration_limit(monkeypatch):
    original = topology.cg
    residuals = []
    def limited(matrix, force, **kwargs):
        displacement, _ = original(matrix, force, **kwargs)
        residuals.append(np.linalg.norm(matrix @ displacement - force) / np.linalg.norm(force))
        return displacement, 250
    monkeypatch.setattr(topology, "cg", limited)
    mesh = trimesh.creation.box(extents=[120, 120, 180])
    result, report = topology.optimise(mesh)
    assert "skipped" not in report, report
    assert len(residuals) == 30 and max(residuals) <= 1e-2
    assert result.is_volume and report["volume_saved_pct"] >= 5


@pytest.mark.parametrize("force_void", [False, True])
def test_subvoxel_bore_preserves_original_surface_clearance(monkeypatch, force_void):
    block = trimesh.creation.box(extents=[40, 40, 60])
    bore = trimesh.creation.cylinder(radius=1.25, height=70, sections=48)
    mesh = trimesh.boolean.difference([block, bore], engine="manifold")
    if force_void:
        monkeypatch.setattr(topology, "_densities", lambda solid, *args: np.zeros(solid.shape))
    result, report = topology.optimise(mesh)
    if "skipped" in report:
        assert result is mesh
    else:
        cavities = [p for p in result.split() if p.volume < 0]
        assert cavities
        for cavity in cavities:
            points = topology._surface_samples(cavity, .5)
            outer_distance = np.min(np.array([20, 20, 30]) - np.abs(points), axis=1)
            bore_distance = np.linalg.norm(points[:, :2], axis=1) - 1.25
            assert np.minimum(outer_distance, bore_distance).min() >= 2
    if force_void:
        assert report == {"skipped": "no cavities preserve clearance from the original surface"}


def test_insignificant_saving_is_unchanged(monkeypatch):
    mesh = trimesh.creation.box(extents=[40, 40, 60])
    def tiny_void(solid, *args):
        density = np.ones(solid.shape)
        middle = np.array(solid.shape) // 2
        density[middle[0]:middle[0]+2, middle[1]:middle[1]+2, middle[2]:middle[2]+2] = 0
        return density
    monkeypatch.setattr(topology, "_densities", tiny_void)
    result, report = topology.optimise(mesh)
    assert result is mesh
    assert report == {"skipped": "less than 5% volume saving"}
