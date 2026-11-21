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
        mesh = mesh.slice_plane([0, 0, -1], [0, 0, 1])
        # Close the horizontal cut without changing the tilted walls.
        boundary = mesh.vertices[np.isclose(mesh.vertices[:, 2], -1)]
        centre = boundary.mean(axis=0)
        angles = np.arctan2(boundary[:, 1] - centre[1], boundary[:, 0] - centre[0])
        boundary = boundary[np.argsort(angles)]
        cap = trimesh.Trimesh(vertices=np.vstack([centre, boundary]),
                              faces=[[0, (i + 1) % len(boundary) + 1, i + 1]
                                     for i in range(len(boundary))])
        mesh = trimesh.util.concatenate([mesh, cap])
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


@pytest.mark.parametrize("value", ["[]", "null", "not json", json.dumps([0] * 16)])
def test_invalid_transform(value):
    with pytest.raises(ValueError, match="camera_transform"):
        photo_level.photo_up(value)


def test_photo_api_gravity(monkeypatch):
    monkeypatch.delenv("PRINT_API_TOKEN", raising=False)
    calls = []

    def reconstruct(image, out, **kwargs):
        calls.append(kwargs)
        tilted_box().export(out)

    monkeypatch.setattr(print_api, "reconstruct_photo", reconstruct)
    rotation = trimesh.transformations.rotation_matrix(np.deg2rad(-20), [1, 0, 0])
    with TestClient(print_api.create_print_app()) as client:
        response = client.post("/print", files={"photo": ("box.jpg", b"photo")},
                               data={"camera_transform": json.dumps(rotation.T.ravel().tolist()),
                                     "height_mm": "100", "base_plate": "false"})
        assert response.status_code == 200, response.text
        assert response.headers["x-print-report"]
        assert json.loads(response.headers["x-print-report"])["level"] == {
            "source": "gravity", "tilt_deg": 20.0}
        assert calls == [{"quarter_turns": 0}]
        response = client.post("/print", files={"photo": ("box.jpg", b"photo")},
                               data={"camera_transform": "[]"})
        assert response.status_code == 400
