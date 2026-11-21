import json

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from server import print_api
import photo_level


def tilted_box():
    mesh = trimesh.creation.box(extents=[2, 3, 5])
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(20), [0, 1, 0]))
    return mesh


def test_base_plane():
    mesh = tilted_box()
    report = photo_level.level_mesh(mesh)
    assert report == {"source": "shape", "tilt_deg": 20.0}
    assert np.max(mesh.face_normals[:, 2]) > np.cos(np.deg2rad(1))


@pytest.mark.parametrize("mesh", [trimesh.creation.icosphere(), trimesh.Trimesh()])
def test_no_base_untouched(mesh):
    before = mesh.vertices.copy()
    assert photo_level.level_mesh(mesh)["source"] == "none"
    np.testing.assert_array_equal(mesh.vertices, before)


@pytest.mark.parametrize("roll", [0, 90, 180, 270])
def test_gravity_pitch_down(roll):
    # ARKit looks down -Z: negative X camera rotation pitches the phone down.
    pitch = trimesh.transformations.rotation_matrix(np.deg2rad(-20), [1, 0, 0])
    rotation = pitch @ trimesh.transformations.rotation_matrix(np.deg2rad(roll), [0, 0, 1])
    gravity = photo_level.photo_up(json.dumps(rotation.T.ravel().tolist()))
    mesh = tilted_box()
    report = photo_level.level_mesh(mesh, gravity)
    assert report == {"source": "gravity", "tilt_deg": 20.0}
    assert np.max(mesh.face_normals[:, 2]) > np.cos(np.deg2rad(1))


@pytest.mark.parametrize("cut", ["missing", "horizontal"])
def test_damaged_bottom(cut):
    mesh = tilted_box()
    if cut == "missing":
        mesh.update_faces(mesh.face_normals[:, 2] > -0.9)
    else:
        vertices, faces, _ = trimesh.intersections.slice_faces_plane(
            mesh.vertices, mesh.faces, [0, 0, 1], [0, 0, -1])
        boundary = vertices[np.isclose(vertices[:, 2], -1)]
        centre = boundary.mean(axis=0)
        angles = np.arctan2(boundary[:, 1] - centre[1], boundary[:, 0] - centre[0])
        boundary = boundary[np.argsort(angles)]
        cap = trimesh.Trimesh(vertices=np.vstack([centre, boundary]),
                              faces=[[0, (i + 1) % len(boundary) + 1, i + 1]
                                     for i in range(len(boundary))])
        mesh = trimesh.util.concatenate([trimesh.Trimesh(vertices, faces), cap])
    report = photo_level.level_mesh(mesh)
    assert report == {"source": "shape", "tilt_deg": 20.0}
    wall = mesh.face_normals[np.abs(mesh.face_normals[:, 0]) > 0.9]
    assert len(wall) and np.max(np.abs(wall[:, 2])) < np.sin(np.deg2rad(1))


def test_already_level_untouched():
    mesh = trimesh.creation.box()
    before = mesh.vertices.copy()
    assert photo_level.level_mesh(mesh)["source"] == "none"
    np.testing.assert_array_equal(mesh.vertices, before)


def test_steep_camera_pitch():
    mesh = trimesh.creation.box(extents=[2, 3, 5])
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(40), [0, 1, 0]))
    assert photo_level.level_mesh(mesh) == {"source": "shape", "tilt_deg": 40.0}
    assert np.max(mesh.face_normals[:, 2]) > np.cos(np.deg2rad(1))


def test_pitch_search_55():
    mesh = trimesh.creation.box(extents=[2, 3, 5])
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(55), [0, 1, 0]))
    up, scores = photo_level.pitch_search(mesh)
    assert up is not None
    assert np.argmin(scores) == pytest.approx(55, abs=2)
    gravity_mesh = mesh.copy()
    camera = trimesh.transformations.rotation_matrix(np.deg2rad(-55), [1, 0, 0])
    gravity = photo_level.photo_up(json.dumps(camera.T.ravel().tolist()))
    assert photo_level.level_mesh(gravity_mesh, gravity) == {"source": "gravity", "tilt_deg": 55.0}
    assert photo_level.level_mesh(mesh) == {"source": "shape", "tilt_deg": 55.0}
    assert np.max(mesh.face_normals[:, 2]) > np.cos(np.deg2rad(3))
    np.testing.assert_allclose(mesh.vertices, gravity_mesh.vertices, atol=1e-10)


@pytest.mark.parametrize("value", ["[]", "null", "not json", json.dumps([0] * 16)])
def test_invalid_transform(value):
    with pytest.raises(ValueError, match="camera_transform"):
        photo_level.photo_up(value)


@pytest.mark.parametrize("pitch", [20, 50])
def test_photo_api_gravity(monkeypatch, pitch):
    monkeypatch.delenv("PRINT_API_TOKEN", raising=False)
    calls = []

    def reconstruct(image, out, **kwargs):
        calls.append(kwargs)
        mesh = trimesh.creation.box(extents=[2, 3, 5])
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(pitch), [0, 1, 0]))
        mesh.export(out)

    monkeypatch.setattr(print_api, "reconstruct_photo", reconstruct)
    rotation = trimesh.transformations.rotation_matrix(np.deg2rad(-pitch), [1, 0, 0])
    with TestClient(print_api.create_print_app()) as client:
        response = client.post("/print", files={"photo": ("box.jpg", b"photo")},
                               data={"camera_transform": json.dumps(rotation.T.ravel().tolist()),
                                     "height_mm": "100", "base_plate": "false",
                                     "image_orientation": "1"})
        assert response.status_code == 200, response.text
        assert response.headers["x-print-report"]
        assert json.loads(response.headers["x-print-report"])["level"] == {
            "source": "gravity", "tilt_deg": float(pitch)}
        np.testing.assert_allclose(json.loads(response.headers["x-print-report"])["extents_mm"],
                                   [40, 60, 100], atol=0.2)
        assert calls == [{"quarter_turns": 0}]
        response = client.post("/print", files={"photo": ("box.jpg", b"photo")},
                               data={"camera_transform": "[]"})
        assert response.status_code == 400
