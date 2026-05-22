"""Assert ANUGA-side polygon consumers accept Polygon and MultiPolygon shapes."""

import logging

import pytest

from run_anuga.run_utils import (
    _extract_polygon_outer_ring,
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
