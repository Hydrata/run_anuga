"""Component tests for mesh region and friction functions.

Tests make_interior_regions, make_frictions, make_interior_holes_and_tags
with realistic GeoJSON-like input data. These functions are pure data
transforms that don't require ANUGA but operate on geo data structures.

Marked requires_geo because the parent module imports shapely lazily.
"""

import pytest

from run_anuga import defaults
from run_anuga.run_utils import (
    make_frictions,
    make_interior_holes_and_tags,
    make_interior_regions,
)


# These tests duplicate some from test_data_transforms but with
# more realistic GeoJSON structures matching real scenarios.


def _polygon_coords(x0, y0, size=100):
    """Create a simple square polygon coordinate ring."""
    return [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]]


@pytest.mark.requires_geo
class TestMakeInteriorRegionsGeo:
    def test_single_region_with_resolution(self):
        input_data = {
            "mesh_region": {
                "features": [
                    {
                        "geometry": {"coordinates": [_polygon_coords(321000, 5812000, 50)]},
                        "properties": {"resolution": 2.5}
                    }
                ]
            }
        }
        regions = make_interior_regions(input_data)
        assert len(regions) == 1
        poly, res = regions[0]
        assert res == 2.5
        assert len(poly) == 5  # closed polygon

    def test_multiple_regions_different_resolutions(self):
        input_data = {
            "mesh_region": {
                "features": [
                    {
                        "geometry": {"coordinates": [_polygon_coords(321000, 5812000, 50)]},
                        "properties": {"resolution": 5.0}
                    },
                    {
                        "geometry": {"coordinates": [_polygon_coords(321100, 5812100, 30)]},
                        "properties": {"resolution": 1.0}
                    },
                ]
            }
        }
        regions = make_interior_regions(input_data)
        assert len(regions) == 2
        resolutions = [r[1] for r in regions]
        assert 5.0 in resolutions
        assert 1.0 in resolutions


@pytest.mark.requires_geo
class TestMakeFrictionsGeo:
    def test_building_friction_value(self):
        input_data = {
            "structure": {
                "features": [
                    {
                        "geometry": {"coordinates": [_polygon_coords(321040, 5812040, 20)]},
                        "properties": {"method": "Mannings"}
                    }
                ]
            }
        }
        frictions = make_frictions(input_data)
        building_friction = frictions[0]
        assert building_friction[1] == defaults.BUILDING_MANNINGS_N

    def test_custom_friction_value(self):
        input_data = {
            "friction": {
                "features": [
                    {
                        "geometry": {"coordinates": [_polygon_coords(321000, 5812000, 100)]},
                        "properties": {"mannings": 0.035}
                    }
                ]
            }
        }
        frictions = make_frictions(input_data)
        assert frictions[0][1] == 0.035

    def test_all_default_always_present(self):
        """The 'All' default friction is always the last entry."""
        for input_data in [{}, {"structure": {"features": []}}, {"friction": {"features": []}}]:
            frictions = make_frictions(input_data)
            assert frictions[-1] == ["All", defaults.DEFAULT_MANNINGS_N]


@pytest.mark.requires_geo
class TestMakeInteriorHolesGeo:
    def test_reflective_hole_has_indices(self):
        coords = _polygon_coords(321040, 5812040, 20)
        input_data = {
            "structure": {
                "features": [
                    {
                        "geometry": {"coordinates": [coords]},
                        "properties": {"method": "Reflective"}
                    }
                ]
            }
        }
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes is not None
        assert len(holes) == 1
        assert "reflective" in tags[0]
        # Reflective tag should have indices for each vertex
        assert len(tags[0]["reflective"]) == len(coords)

    def test_empty_features_returns_none(self):
        input_data = {"structure": {"features": []}}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes is None
        assert tags is None
