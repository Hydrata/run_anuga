"""Tests for pure-logic data transformation functions from run_utils.py."""

import argparse

import pytest

from run_anuga.run_utils import (
    RunContext,
    is_dir_check,
    lookup_boundary_tag,
    make_new_inflow,
    make_frictions,
    make_interior_regions,
    make_interior_holes_and_tags,
)
from run_anuga import defaults


class TestRunContext:
    def test_defaults(self):
        ctx = RunContext(package_dir="/some/path")
        assert ctx.package_dir == "/some/path"
        assert ctx.username is None
        assert ctx.password is None

    def test_with_credentials(self):
        ctx = RunContext("/pkg", "user@example.com", "secret")
        assert ctx.username == "user@example.com"
        assert ctx.password == "secret"


class TestIsDirCheck:
    def test_valid_directory(self, tmp_path):
        assert is_dir_check(str(tmp_path)) == str(tmp_path)

    def test_file_raises(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(argparse.ArgumentTypeError):
            is_dir_check(str(f))

    def test_nonexistent_raises(self):
        with pytest.raises(argparse.ArgumentTypeError):
            is_dir_check("/nonexistent/path/xyz")


class TestMakeNewInflow:
    def test_structure(self):
        result = make_new_inflow("test_1", [[0, 0], [1, 1]], 0.5)
        assert result["type"] == "Feature"
        assert result["geometry"]["type"] == "LineString"
        assert result["geometry"]["coordinates"] == [[0, 0], [1, 1]]

    def test_flow_in_properties(self):
        result = make_new_inflow("test_1", [[0, 0], [1, 1]], 0.5)
        assert result["properties"]["data"] == "0.5"
        assert result["properties"]["type"] == "Surface"

    def test_id_format(self):
        result = make_new_inflow("inflow_42", [[0, 0], [1, 1]], 1.0)
        assert result["id"] == "inf_16_inflow_01.inflow_42"

    def test_zero_flow(self):
        result = make_new_inflow("zero", [[0, 0], [1, 1]], 0)
        assert result["properties"]["data"] == "0"


class TestLookupBoundaryTag:
    def test_found(self):
        tags = {"ocean": [0, 1, 2], "river": [3, 4]}
        assert lookup_boundary_tag(1, tags) == "ocean"
        assert lookup_boundary_tag(4, tags) == "river"

    def test_not_found(self):
        tags = {"ocean": [0, 1]}
        assert lookup_boundary_tag(99, tags) is None

    def test_first_match(self):
        tags = {"ocean": [0, 1, 2], "river": [3, 4, 5]}
        assert lookup_boundary_tag(0, tags) == "ocean"
        assert lookup_boundary_tag(5, tags) == "river"

    def test_empty_tags(self):
        assert lookup_boundary_tag(0, {}) is None


class TestMakeInteriorRegions:
    def test_with_mesh_region(self):
        input_data = {
            "mesh_region": {
                "features": [
                    {
                        "geometry": {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
                        "properties": {"resolution": 5.0}
                    }
                ]
            }
        }
        regions = make_interior_regions(input_data)
        assert len(regions) == 1
        polygon, resolution = regions[0]
        assert resolution == 5.0
        assert len(polygon) == 5

    def test_multiple_regions(self):
        input_data = {
            "mesh_region": {
                "features": [
                    {
                        "geometry": {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                        "properties": {"resolution": 5.0}
                    },
                    {
                        "geometry": {"coordinates": [[[2, 2], [3, 2], [3, 3], [2, 2]]]},
                        "properties": {"resolution": 10.0}
                    },
                ]
            }
        }
        regions = make_interior_regions(input_data)
        assert len(regions) == 2

    def test_no_mesh_region(self):
        assert make_interior_regions({}) == []

    def test_empty_features(self):
        input_data = {"mesh_region": {"features": []}}
        assert make_interior_regions(input_data) == []


class TestMakeFrictions:
    def test_with_buildings(self):
        input_data = {
            "structure": {
                "features": [
                    {
                        "geometry": {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
                        "properties": {"method": "Mannings"}
                    }
                ]
            }
        }
        frictions = make_frictions(input_data)
        # Building friction + 'All' default
        assert len(frictions) == 2
        assert frictions[0][1] == defaults.BUILDING_MANNINGS_N
        assert frictions[1] == ["All", defaults.DEFAULT_MANNINGS_N]

    def test_with_friction_polygons(self):
        input_data = {
            "friction": {
                "features": [
                    {
                        "geometry": {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                        "properties": {"mannings": 0.08}
                    }
                ]
            }
        }
        frictions = make_frictions(input_data)
        assert len(frictions) == 2
        assert frictions[0][1] == 0.08
        assert frictions[1] == ["All", defaults.DEFAULT_MANNINGS_N]

    def test_no_friction_or_structure(self):
        frictions = make_frictions({})
        assert len(frictions) == 1
        assert frictions[0] == ["All", defaults.DEFAULT_MANNINGS_N]

    def test_structure_non_mannings_excluded(self):
        input_data = {
            "structure": {
                "features": [
                    {
                        "geometry": {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                        "properties": {"method": "Holes"}
                    }
                ]
            }
        }
        frictions = make_frictions(input_data)
        # Only the 'All' default, Holes structures aren't friction
        assert len(frictions) == 1

    def test_combined_buildings_and_friction(self):
        input_data = {
            "structure": {
                "features": [
                    {
                        "geometry": {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                        "properties": {"method": "Mannings"}
                    }
                ]
            },
            "friction": {
                "features": [
                    {
                        "geometry": {"coordinates": [[[2, 2], [3, 2], [3, 3], [2, 2]]]},
                        "properties": {"mannings": 0.06}
                    }
                ]
            }
        }
        frictions = make_frictions(input_data)
        assert len(frictions) == 3  # building + friction polygon + All
        # Verify ordering and exact values
        assert frictions[0][1] == defaults.BUILDING_MANNINGS_N  # 10.0
        assert frictions[1][1] == 0.06                           # custom friction
        assert frictions[2] == ["All", defaults.DEFAULT_MANNINGS_N]


class TestMakeInteriorHolesAndTags:
    def test_no_structures(self):
        holes, tags = make_interior_holes_and_tags({})
        assert holes is None
        assert tags is None

    def test_mannings_structures_skipped(self):
        input_data = {
            "structure": {
                "features": [
                    {
                        "geometry": {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                        "properties": {"method": "Mannings"}
                    }
                ]
            }
        }
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes is None
        assert tags is None

    def test_holes_structure(self):
        input_data = {
            "structure": {
                "features": [
                    {
                        "geometry": {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
                        "properties": {"method": "Holes"}
                    }
                ]
            }
        }
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes is not None
        assert len(holes) == 1
        # Holes → reflective boundary (water cannot enter the void)
        assert tags[0] == {"reflective": [0, 1, 2, 3, 4]}

    def test_reflective_structure(self):
        input_data = {
            "structure": {
                "features": [
                    {
                        "geometry": {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                        "properties": {"method": "Reflective"}
                    }
                ]
            }
        }
        holes, tags = make_interior_holes_and_tags(input_data)
        # Reflective → DEM-burned elevation, NOT a mesh hole
        assert holes is None
        assert tags is None

    def test_mixed_structures(self):
        input_data = {
            "structure": {
                "features": [
                    {
                        "geometry": {"coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                        "properties": {"method": "Mannings"}
                    },
                    {
                        "geometry": {"coordinates": [[[2, 2], [3, 2], [3, 3], [2, 2]]]},
                        "properties": {"method": "Holes"}
                    },
                    {
                        "geometry": {"coordinates": [[[4, 4], [5, 4], [5, 5], [4, 4]]]},
                        "properties": {"method": "Reflective"}
                    },
                ]
            }
        }
        holes, tags = make_interior_holes_and_tags(input_data)
        # Only Holes → 1 mesh hole (Mannings and Reflective are not mesh holes)
        assert len(holes) == 1
        assert len(tags) == 1
        assert tags[0] == {"reflective": [0, 1, 2, 3]}  # Holes → reflective tag
