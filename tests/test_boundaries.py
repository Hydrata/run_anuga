"""Component tests for create_boundary_polygon_from_boundaries().

Requires shapely (marked requires_geo).
"""

import json
import os

import pytest

from run_anuga.run_utils import create_boundary_polygon_from_boundaries

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "data", "minimal_package")


def _make_boundary_geojson(features, crs_name="EPSG:28355"):
    """Helper to build a boundary GeoJSON FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": crs_name}
        },
        "features": features
    }


def _make_boundary_feature(feature_id, coords, boundary_name, location="External"):
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": {
            "type": "LineString",
            "coordinates": coords
        },
        "properties": {
            "boundary": boundary_name,
            "location": location
        }
    }


@pytest.mark.requires_geo
class TestCreateBoundaryPolygon:
    def test_simple_rectangle(self):
        """4 boundary segments forming a rectangle."""
        geojson = _make_boundary_geojson([
            _make_boundary_feature("b1", [[0, 0], [100, 0]], "south"),
            _make_boundary_feature("b2", [[100, 0], [100, 100]], "east"),
            _make_boundary_feature("b3", [[100, 100], [0, 100]], "north"),
            _make_boundary_feature("b4", [[0, 100], [0, 0]], "west"),
        ])
        polygon, tags = create_boundary_polygon_from_boundaries(geojson)
        assert len(polygon) >= 4
        assert len(tags) > 0
        # All boundary names should be present
        assert set(tags.keys()) == {"south", "east", "north", "west"}

    def test_boundary_tags_have_indices(self):
        """Each boundary tag should map to a list of point indices."""
        geojson = _make_boundary_geojson([
            _make_boundary_feature("b1", [[0, 0], [100, 0]], "south"),
            _make_boundary_feature("b2", [[100, 0], [100, 100]], "east"),
            _make_boundary_feature("b3", [[100, 100], [0, 100]], "north"),
            _make_boundary_feature("b4", [[0, 100], [0, 0]], "west"),
        ])
        polygon, tags = create_boundary_polygon_from_boundaries(geojson)
        total_indices = sum(len(v) for v in tags.values())
        assert total_indices == len(polygon)

    def test_internal_boundaries_excluded(self):
        """Internal boundaries are filtered out."""
        geojson = _make_boundary_geojson([
            _make_boundary_feature("b1", [[0, 0], [100, 0]], "south"),
            _make_boundary_feature("b2", [[100, 0], [100, 100]], "east"),
            _make_boundary_feature("b3", [[100, 100], [0, 100]], "north"),
            _make_boundary_feature("b4", [[0, 100], [0, 0]], "west"),
            _make_boundary_feature("wall", [[50, 50], [60, 60]], "wall", location="Internal"),
        ])
        polygon, tags = create_boundary_polygon_from_boundaries(geojson)
        assert "wall" not in tags

    def test_no_crs_returns_empty(self):
        """Missing CRS returns empty polygon and tags."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                _make_boundary_feature("b1", [[0, 0], [100, 0]], "south"),
            ]
        }
        polygon, tags = create_boundary_polygon_from_boundaries(geojson)
        assert polygon == []
        assert tags == {}

    def test_duplicate_boundary_names(self):
        """Two segments with the same boundary name are merged."""
        geojson = _make_boundary_geojson([
            _make_boundary_feature("b1", [[0, 0], [50, 0]], "Transmissive"),
            _make_boundary_feature("b2", [[50, 0], [100, 0]], "Transmissive"),
            _make_boundary_feature("b3", [[100, 0], [100, 100]], "Reflective"),
            _make_boundary_feature("b4", [[100, 100], [0, 100]], "Transmissive"),
            _make_boundary_feature("b5", [[0, 100], [0, 0]], "Reflective"),
        ])
        polygon, tags = create_boundary_polygon_from_boundaries(geojson)
        assert "Transmissive" in tags
        assert "Reflective" in tags
        # Transmissive has more segments
        assert len(tags["Transmissive"]) > len(tags["Reflective"])

    def test_real_fixture(self):
        """Test with the actual minimal_package boundary.geojson data."""
        path = os.path.join(FIXTURE_DIR, "inputs", "boundary.geojson")
        with open(path) as f:
            geojson = json.load(f)
        polygon, tags = create_boundary_polygon_from_boundaries(geojson)
        assert len(polygon) == 8  # 4 segments x 2 points each
        assert "Transmissive" in tags
        assert "Reflective" in tags

    def test_polygon_coordinates_are_numeric(self):
        """All polygon coordinates should be numbers."""
        geojson = _make_boundary_geojson([
            _make_boundary_feature("b1", [[0, 0], [100, 0]], "south"),
            _make_boundary_feature("b2", [[100, 0], [100, 100]], "east"),
            _make_boundary_feature("b3", [[100, 100], [0, 100]], "north"),
            _make_boundary_feature("b4", [[0, 100], [0, 0]], "west"),
        ])
        polygon, tags = create_boundary_polygon_from_boundaries(geojson)
        for point in polygon:
            assert isinstance(point[0], (int, float))
            assert isinstance(point[1], (int, float))
