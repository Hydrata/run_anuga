"""Assert ANUGA-side polygon consumers accept Polygon and MultiPolygon shapes."""

import logging

import pytest

from run_anuga.run_utils import (
    _extract_polygon_outer_ring,
    _flatten_line_coordinates,
    check_coordinates_are_in_polygon,
    make_frictions,
    make_interior_holes_and_tags,
    make_interior_regions,
)


OUTER_RING = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
OUTER_RING_2 = [[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0], [10.0, 10.0]]


def _polygon(ring=None):
    return {'type': 'Polygon', 'coordinates': [ring or OUTER_RING]}


def _multipolygon(*rings):
    return {'type': 'MultiPolygon', 'coordinates': [[r] for r in (rings or (OUTER_RING,))]}


class TestExtractPolygonOuterRing:
    def test_polygon_returns_outer_ring(self):
        assert _extract_polygon_outer_ring(_polygon()) == OUTER_RING

    def test_multipolygon_returns_first_subpolygon_outer_ring(self):
        assert _extract_polygon_outer_ring(_multipolygon()) == OUTER_RING

    def test_multipolygon_with_multiple_subpolygons_warns_and_takes_first(self, caplog):
        geometry = _multipolygon(OUTER_RING, OUTER_RING_2)
        with caplog.at_level(logging.WARNING, logger='run_anuga.run_utils'):
            result = _extract_polygon_outer_ring(geometry)
        assert result == OUTER_RING
        assert any('2 sub-polygons' in rec.message for rec in caplog.records)

    def test_empty_coordinates_returns_empty_list(self):
        assert _extract_polygon_outer_ring({'type': 'Polygon', 'coordinates': []}) == []
        assert _extract_polygon_outer_ring({'type': 'MultiPolygon', 'coordinates': []}) == []

    def test_none_coordinates_returns_empty_list(self):
        assert _extract_polygon_outer_ring({'type': 'Polygon', 'coordinates': None}) == []

    def test_multipolygon_with_empty_subpolygon_returns_empty_list(self):
        # Defensive: pathological "MultiPolygon with [[]]" doesn't crash.
        assert _extract_polygon_outer_ring({'type': 'MultiPolygon', 'coordinates': [[]]}) == []

    def test_empty_coordinates_emits_debug_log(self, caplog):
        with caplog.at_level(logging.DEBUG, logger='run_anuga.run_utils'):
            result = _extract_polygon_outer_ring({'type': 'Polygon', 'coordinates': []})
        assert result == []
        assert any(
            rec.levelno == logging.DEBUG
            and '_extract_polygon_outer_ring' in rec.message
            and 'empty/missing coordinates' in rec.message
            for rec in caplog.records
        )


class TestMakeFrictions:
    def test_polygon_friction_feature_returns_2d_ring(self):
        input_data = {'friction': {'features': [{
            'geometry': _polygon(),
            'properties': {'mannings': 0.03},
        }]}}
        result = make_frictions(input_data)
        # First friction tuple: (outer_ring, value) — outer ring must be 2-D
        assert result[0][0] == OUTER_RING
        assert result[0][1] == 0.03

    def test_multipolygon_friction_feature_returns_2d_ring(self):
        # Canary 17 failing shape: PostGIS-normalised MultiPolygon
        input_data = {'friction': {'features': [{
            'geometry': _multipolygon(),
            'properties': {'mannings': 0.03},
        }]}}
        result = make_frictions(input_data)
        assert result[0][0] == OUTER_RING
        assert result[0][1] == 0.03

    def test_mannings_structure_multipolygon_returns_2d_ring(self):
        input_data = {'structure': {'features': [{
            'geometry': _multipolygon(),
            'properties': {'method': 'Mannings'},
        }]}}
        result = make_frictions(input_data)
        assert result[0][0] == OUTER_RING

    def test_friction_raster_short_circuits_polygon_path(self):
        # Raster precedence (TASK-830) is unchanged by the helper introduction.
        input_data = {'friction_raster_filename': '/tmp/x.tif', 'friction': {'features': [
            {'geometry': _polygon(), 'properties': {'mannings': 0.03}},
        ]}}
        assert make_frictions(input_data) == [['Extent', '/tmp/x.tif']]


class TestMakeInteriorRegions:
    def test_polygon_returns_2d_ring(self):
        input_data = {'mesh_region': {'features': [{
            'geometry': _polygon(),
            'properties': {'resolution': 5.0},
        }]}}
        assert make_interior_regions(input_data) == [(OUTER_RING, 5.0)]

    def test_multipolygon_returns_2d_ring(self):
        input_data = {'mesh_region': {'features': [{
            'geometry': _multipolygon(),
            'properties': {'resolution': 5.0},
        }]}}
        assert make_interior_regions(input_data) == [(OUTER_RING, 5.0)]


class TestMakeInteriorHolesAndTags:
    def test_polygon_holes_method_returns_2d_ring(self):
        input_data = {'structure': {'features': [{
            'geometry': _polygon(),
            'properties': {'method': 'Holes'},
        }]}}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes == [OUTER_RING]
        assert tags == [None]

    def test_multipolygon_holes_method_returns_2d_ring(self):
        input_data = {'structure': {'features': [{
            'geometry': _multipolygon(),
            'properties': {'method': 'Holes'},
        }]}}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes == [OUTER_RING]
        assert tags == [None]


LINESTRING_COORDS = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
LINESTRING_COORDS_2 = [[2.0, 2.0], [3.0, 2.0]]


class TestFlattenLineCoordinates:
    def test_linestring_returns_coords_as_is(self):
        geometry = {'type': 'LineString', 'coordinates': LINESTRING_COORDS}
        assert _flatten_line_coordinates(geometry) == LINESTRING_COORDS

    def test_multilinestring_single_subline_is_flattened_one_level(self):
        geometry = {'type': 'MultiLineString', 'coordinates': [LINESTRING_COORDS]}
        assert _flatten_line_coordinates(geometry) == LINESTRING_COORDS

    def test_multilinestring_multiple_sublines_are_concatenated(self):
        geometry = {'type': 'MultiLineString',
                    'coordinates': [LINESTRING_COORDS, LINESTRING_COORDS_2]}
        assert _flatten_line_coordinates(geometry) == LINESTRING_COORDS + LINESTRING_COORDS_2

    def test_none_coordinates_returns_empty_list(self):
        assert _flatten_line_coordinates({'type': 'LineString', 'coordinates': None}) == []
        assert _flatten_line_coordinates({'type': 'MultiLineString', 'coordinates': None}) == []

    def test_empty_coordinates_returns_empty_list(self):
        assert _flatten_line_coordinates({'type': 'LineString', 'coordinates': []}) == []
        assert _flatten_line_coordinates({'type': 'MultiLineString', 'coordinates': []}) == []

    def test_empty_coordinates_emits_debug_log(self, caplog):
        with caplog.at_level(logging.DEBUG, logger='run_anuga.run_utils'):
            result = _flatten_line_coordinates({'type': 'LineString', 'coordinates': []})
        assert result == []
        assert any(
            rec.levelno == logging.DEBUG
            and '_flatten_line_coordinates' in rec.message
            and 'empty/missing coordinates' in rec.message
            for rec in caplog.records
        )


# Square polygon spanning (0,0) -> (10,10); used for in/out membership checks.
CONTAINER_POLYGON = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


class TestCheckCoordinatesAreInPolygon:
    def test_flat_linestring_inside_polygon_returns_true(self):
        coords = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        assert check_coordinates_are_in_polygon(coords, CONTAINER_POLYGON) is True

    def test_flat_linestring_outside_polygon_returns_false(self):
        coords = [[100.0, 100.0], [101.0, 101.0]]
        assert check_coordinates_are_in_polygon(coords, CONTAINER_POLYGON) is False

    def test_flat_linestring_partially_outside_polygon_returns_false(self):
        coords = [[1.0, 1.0], [100.0, 100.0]]
        assert check_coordinates_are_in_polygon(coords, CONTAINER_POLYGON) is False

    def test_single_point_as_xy_inside_polygon_returns_true(self):
        assert check_coordinates_are_in_polygon([5.0, 5.0], CONTAINER_POLYGON) is True

    def test_single_point_as_xy_outside_polygon_returns_false(self):
        assert check_coordinates_are_in_polygon([50.0, 50.0], CONTAINER_POLYGON) is False

    def test_empty_coordinates_list_returns_false(self):
        assert check_coordinates_are_in_polygon([], CONTAINER_POLYGON) is False


class TestApplyInflowsCoordinateHandling:
    BOUNDARY_POLYGON = [
        [382000.0, 6354000.0],
        [383000.0, 6354000.0],
        [383000.0, 6355000.0],
        [382000.0, 6355000.0],
    ]

    def test_multilinestring_inflow_shape_does_not_raise(self):
        inflow_geometry = {
            'type': 'MultiLineString',
            'coordinates': [[[382260.0, 6354275.0], [382270.0, 6354285.0]]],
        }
        flat = _flatten_line_coordinates(inflow_geometry)
        assert check_coordinates_are_in_polygon(flat, self.BOUNDARY_POLYGON) is True

    def test_linestring_equivalent_shape_returns_true(self):
        inflow_geometry = {
            'type': 'LineString',
            'coordinates': [[382260.0, 6354275.0], [382270.0, 6354285.0]],
        }
        flat = _flatten_line_coordinates(inflow_geometry)
        assert check_coordinates_are_in_polygon(flat, self.BOUNDARY_POLYGON) is True

    def test_multilinestring_outside_polygon_returns_false(self):
        inflow_geometry = {
            'type': 'MultiLineString',
            'coordinates': [[[500000.0, 7000000.0], [500010.0, 7000010.0]]],
        }
        flat = _flatten_line_coordinates(inflow_geometry)
        assert check_coordinates_are_in_polygon(flat, self.BOUNDARY_POLYGON) is False
