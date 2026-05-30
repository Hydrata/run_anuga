"""Assert ANUGA-side polygon consumers accept Polygon and MultiPolygon shapes."""

import importlib.util
import logging

import pytest

from run_anuga.run_utils import (
    _extract_polygon_outer_ring,
    _flatten_line_coordinates,
    assert_raster_has_no_nodata_inside_boundary,
    check_coordinates_are_in_polygon,
    make_frictions,
    make_interior_holes_and_tags,
    make_interior_regions,
)

_HAS_RASTERIO = importlib.util.find_spec("rasterio") is not None


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
    # ADR-4 (TASK-1269/1270): 'Holes' method removed; 'Reflective' is now the
    # interior-hole method (with sliver-merge via shapely). After shapely
    # simplify the coords come back as tuples, so we compare via tuple-conversion.

    def test_reflective_polygon_yields_hole_with_tag(self):
        input_data = {'structure': {'features': [{
            'geometry': _polygon(),
            'properties': {'method': 'Reflective'},
        }]}}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes is not None
        assert len(holes) == 1
        assert tags is not None
        assert 'reflective' in tags[0]

    def test_reflective_multipolygon_yields_hole_with_tag(self):
        input_data = {'structure': {'features': [{
            'geometry': _multipolygon(),
            'properties': {'method': 'Reflective'},
        }]}}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes is not None
        assert len(holes) >= 1
        assert tags is not None

    def test_mannings_yields_none(self):
        input_data = {'structure': {'features': [{
            'geometry': _polygon(),
            'properties': {'method': 'Mannings'},
        }]}}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes is None
        assert tags is None


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


# --- assert_raster_has_no_nodata_inside_boundary (TASK-1138) ----------------
#
# These tests write tiny real Float32 GeoTIFFs (no live ANUGA) and assert the
# pre-flight nodata guard raises a clear error when a nodata cell falls INSIDE
# the model boundary, and passes otherwise. Applies equally to elevation and
# friction rasters (the guard is quantity-agnostic; quantity_name only labels
# the error message).

# 10x10 grid, 1 m pixels, top-left origin at (0, 10). The raster therefore
# spans x in [0, 10], y in [0, 10]; pixel (row, col) has centre
# (col + 0.5, 10 - row - 0.5).
_RASTER_WIDTH = 10
_RASTER_HEIGHT = 10
# UTM zone 56S (used elsewhere in these tests) — a real projected CRS.
_PROJECTED_EPSG = 32756
# A square boundary well inside the raster extent: x,y in [2, 7].
_INNER_BOUNDARY = [[2.0, 2.0], [7.0, 2.0], [7.0, 7.0], [2.0, 7.0]]
_NODATA_TAG = -9999.0


def _write_geotiff(path, *, nodata_rowcol=None, declared_nodata=_NODATA_TAG,
                   fill_value=5.0, nodata_is_nan=False):
    """Write a tiny Float32 GeoTIFF in a projected CRS.

    nodata_rowcol: (row, col) to stamp with the nodata sentinel (or NaN), or
        None for an all-valid raster.
    declared_nodata: value written to the GeoTIFF nodata tag (None = no tag).
    nodata_is_nan: if True, stamp the cell with NaN instead of the sentinel.
    """
    import numpy
    import rasterio
    from rasterio.transform import from_origin

    data = numpy.full((_RASTER_HEIGHT, _RASTER_WIDTH), fill_value, dtype='float32')
    if nodata_rowcol is not None:
        row, col = nodata_rowcol
        data[row, col] = numpy.nan if nodata_is_nan else _NODATA_TAG
    transform = from_origin(0.0, float(_RASTER_HEIGHT), 1.0, 1.0)
    profile = {
        'driver': 'GTiff',
        'height': _RASTER_HEIGHT,
        'width': _RASTER_WIDTH,
        'count': 1,
        'dtype': 'float32',
        'crs': rasterio.crs.CRS.from_epsg(_PROJECTED_EPSG),
        'transform': transform,
    }
    if declared_nodata is not None:
        profile['nodata'] = declared_nodata
    with rasterio.open(path, 'w', **profile) as dataset:
        dataset.write(data, 1)
    return str(path)


@pytest.mark.skipif(not _HAS_RASTERIO, reason="rasterio is a [sim] extra, not installed in light CI")
class TestAssertRasterHasNoNodataInsideBoundary:
    def test_elevation_nodata_inside_boundary_raises(self, tmp_path):
        # Cell (row 5, col 4) -> centre (4.5, 4.5), inside the [2,7] square.
        raster = _write_geotiff(tmp_path / 'elev.tif', nodata_rowcol=(5, 4))
        with pytest.raises(ValueError) as excinfo:
            assert_raster_has_no_nodata_inside_boundary(
                raster, _INNER_BOUNDARY, quantity_name='elevation'
            )
        message = str(excinfo.value)
        assert 'elevation' in message
        assert 'nodata' in message
        assert 'inside the model boundary' in message
        assert 'gdal_fillnodata' in message

    def test_elevation_nodata_outside_boundary_ok(self, tmp_path):
        # Cell (row 0, col 0) -> centre (0.5, 9.5), outside the [2,7] square.
        raster = _write_geotiff(tmp_path / 'elev_outside.tif', nodata_rowcol=(0, 0))
        # Should NOT raise.
        assert assert_raster_has_no_nodata_inside_boundary(
            raster, _INNER_BOUNDARY, quantity_name='elevation'
        ) is None

    def test_elevation_no_nodata_tag_ok(self, tmp_path):
        # All-valid raster with NO declared nodata tag -> nothing to check.
        raster = _write_geotiff(
            tmp_path / 'elev_no_tag.tif', nodata_rowcol=None, declared_nodata=None
        )
        assert assert_raster_has_no_nodata_inside_boundary(
            raster, _INNER_BOUNDARY, quantity_name='elevation'
        ) is None

    def test_elevation_nan_nodata_inside_boundary_raises(self, tmp_path):
        # NaN sentinel inside the boundary is treated as nodata even though the
        # cell value is NaN rather than the finite -9999 tag.
        raster = _write_geotiff(
            tmp_path / 'elev_nan.tif', nodata_rowcol=(5, 4), nodata_is_nan=True
        )
        with pytest.raises(ValueError) as excinfo:
            assert_raster_has_no_nodata_inside_boundary(
                raster, _INNER_BOUNDARY, quantity_name='elevation'
            )
        assert 'elevation' in str(excinfo.value)

    def test_elevation_empty_boundary_is_noop(self, tmp_path):
        # No boundary polygon -> nothing the guard can assert, even with nodata.
        raster = _write_geotiff(tmp_path / 'elev_nb.tif', nodata_rowcol=(5, 4))
        assert assert_raster_has_no_nodata_inside_boundary(
            raster, [], quantity_name='elevation'
        ) is None

    def test_friction_nodata_inside_boundary_raises(self, tmp_path):
        # Same guard, friction raster: a friction nodata gap inside the boundary
        # must also fail-fast with a clear message.
        raster = _write_geotiff(tmp_path / 'friction.tif', nodata_rowcol=(5, 4))
        with pytest.raises(ValueError) as excinfo:
            assert_raster_has_no_nodata_inside_boundary(
                raster, _INNER_BOUNDARY, quantity_name='friction'
            )
        message = str(excinfo.value)
        assert 'friction' in message
        assert 'nodata' in message

    def test_friction_nodata_outside_boundary_ok(self, tmp_path):
        raster = _write_geotiff(tmp_path / 'friction_outside.tif', nodata_rowcol=(0, 0))
        assert assert_raster_has_no_nodata_inside_boundary(
            raster, _INNER_BOUNDARY, quantity_name='friction'
        ) is None
