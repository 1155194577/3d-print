import numpy as np
import pytest
import trimesh

from tools.model_render import render


K = np.array([[100., 0, 64], [0, 100., 64], [0, 0, 1.]])


def test_cube_fills_square_with_calibrated_size():
    cube = trimesh.creation.box()
    image = np.asarray(render(cube.vertices, cube.faces, K, np.eye(3), [0, 0, 3], 128, 128))
    filled = (image < 255).any(axis=2)
    y, x = np.where(filled)
    assert (x.min(), x.max(), y.min(), y.max()) == (44, 83, 44, 83)
    assert filled[44:84, 44:84].all()
    assert (image[filled] > 0).all()


def test_depth_buffer_is_independent_of_face_order():
    cube = trimesh.creation.box()
    first = render(cube.vertices, cube.faces, K, np.eye(3), [0, 0, 3], 128, 128)
    second = render(cube.vertices, cube.faces[::-1], K, np.eye(3), [0, 0, 3], 128, 128)
    np.testing.assert_array_equal(first, second)


def test_empty_behind_and_degenerate_meshes_are_white():
    assert np.asarray(render([], [], K, np.eye(3), [0, 0, 0], 32, 32)).min() == 255
    cube = trimesh.creation.box()
    assert np.asarray(render(cube.vertices, cube.faces, K, np.eye(3), [0, 0, -3], 32, 32)).min() == 255
    assert np.asarray(render([[0, 0, 1]] * 3, [[0, 1, 2]], K, np.eye(3), [0, 0, 0], 32, 32)).min() == 255


def test_near_plane_crossing_is_clipped():
    image = render([[-.1, -.1, -.1], [.1, -.1, .2], [0, .1, .2]], [[0, 1, 2]],
                   K, np.eye(3), [0, 0, 0], 128, 128)
    assert (np.asarray(image) < 255).any()


def test_invalid_geometry_raises():
    with pytest.raises(ValueError):
        render([[np.nan, 0, 1]], [], K, np.eye(3), [0, 0, 0], 32, 32)
