import subprocess
import sys

import numpy as np
import trimesh

from turbine import design


def test_archived_hk_stations_produce_distinct_bed_size_blades(tmp_path):
    tip_chords = {}
    for station, expected_speed in (("King's Park", "5.04"), ("Cheung Chau", "10.77")):
        output = tmp_path / station
        result = subprocess.run(
            [sys.executable, "-m", "turbine.design", "--data", str(design.SAMPLE),
             "--station", station, "--output-dir", str(output)],
            capture_output=True, text=True, check=True,
        )
        assert f"{station}: 96 samples" in result.stdout
        assert f"Energy-mode design wind: {expected_speed} m/s" in result.stdout
        inner, outer = (
            trimesh.load(output / f"blade-{segment}.stl", force="mesh")
            for segment in ("inner", "outer")
        )
        for part in (inner, outer):
            assert part.is_volume
            assert np.all(part.extents <= 256)
            assert np.all(part.area_faces > 0)
        split_mm = design.SPLIT_M * 1000
        inner_edge = inner.vertices[np.isclose(inner.vertices[:, 1], split_mm)]
        outer_edge = outer.vertices[np.isclose(outer.vertices[:, 1], split_mm)]
        assert len(inner_edge) == len(outer_edge) > 0
        np.testing.assert_array_equal(np.unique(inner_edge, axis=0),
                                      np.unique(outer_edge, axis=0))
        tip_vertices = outer.vertices[np.isclose(outer.vertices[:, 1], design.RADIUS_M * 1000)]
        tip_chords[station] = np.ptp(tip_vertices[:, 0])
    assert tip_chords["King's Park"] < tip_chords["Cheung Chau"]
