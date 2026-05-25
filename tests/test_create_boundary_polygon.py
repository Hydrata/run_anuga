"""Defence-in-depth tests for ``create_boundary_polygon_from_boundaries``.

Guards the ``max()/min()`` calls (L537-540 pre-fix) against the 2026-04-30
prod ``ValueError: max() arg is an empty sequence`` when scenarios reach
this code path with boundaries that have empty / malformed coordinate
lists. See TASK-976.
"""

import pytest

# osgeo.ogr is a [sim] extra not present in light CI; skip the whole module
# rather than error at collection.
pytest.importorskip("osgeo.ogr")

from run_anuga.run_utils import create_boundary_polygon_from_boundaries


CRS_EPSG_32616 = {
    'type': 'name',
    'properties': {'name': 'urn:ogc:def:crs:EPSG::32616'},
}


def _external_feature(fid='b.1', coords=None, boundary='north'):
    if coords is None:
        coords = [[0.0, 0.0], [100.0, 0.0]]
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {'type': 'LineString', 'coordinates': coords},
        'properties': {'location': 'External', 'boundary': boundary},
    }


def _internal_feature(fid='i.1', coords=None):
    if coords is None:
        coords = [[10.0, 10.0], [20.0, 10.0]]
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {'type': 'LineString', 'coordinates': coords},
        'properties': {'location': 'Internal', 'boundary': 'i'},
    }


def test_happy_path_two_external_boundaries_returns_polygon():
    geojson = {
        'crs': CRS_EPSG_32616,
        'features': [
            _external_feature(fid='b.1', coords=[[0.0, 0.0], [100.0, 0.0]], boundary='south'),
            _external_feature(fid='b.2', coords=[[100.0, 0.0], [100.0, 100.0]], boundary='east'),
            _external_feature(fid='b.3', coords=[[100.0, 100.0], [0.0, 100.0]], boundary='north'),
            _external_feature(fid='b.4', coords=[[0.0, 100.0], [0.0, 0.0]], boundary='west'),
        ],
    }
    boundary_polygon, boundary_tags = create_boundary_polygon_from_boundaries(geojson)
    assert len(boundary_polygon) == 8
    assert set(boundary_tags.keys()) == {'south', 'east', 'north', 'west'}


def test_missing_crs_returns_empty():
    geojson = {'features': [_external_feature()]}
    boundary_polygon, boundary_tags = create_boundary_polygon_from_boundaries(geojson)
    assert boundary_polygon == []
    assert boundary_tags == {}


def test_empty_features_list_raises_clear_value_error():
    geojson = {'crs': CRS_EPSG_32616, 'features': []}
    with pytest.raises(ValueError, match='no valid External-location boundary coordinates'):
        create_boundary_polygon_from_boundaries(geojson)


def test_all_internal_boundaries_raises_clear_value_error():
    geojson = {
        'crs': CRS_EPSG_32616,
        'features': [
            _internal_feature(fid='i.1'),
            _internal_feature(fid='i.2', coords=[[30.0, 30.0], [40.0, 30.0]]),
        ],
    }
    with pytest.raises(ValueError, match='no valid External-location boundary coordinates'):
        create_boundary_polygon_from_boundaries(geojson)


def test_external_boundary_with_empty_coordinates_raises_clear_value_error():
    geojson = {
        'crs': CRS_EPSG_32616,
        'features': [_external_feature(fid='b.empty', coords=[])],
    }
    with pytest.raises(ValueError, match='no valid External-location boundary coordinates'):
        create_boundary_polygon_from_boundaries(geojson)


def _external_mls_feature(fid='b.1', coords=None, boundary='north'):
    """MultiLineString variant — PostGIS / GeoServer normalises every boundary
    feature to MultiLineString on the round-trip from the standard upload
    pipeline, so this is what real prod scenarios see when the BE reads the
    boundary GeoJSON back from PG via WFS."""
    if coords is None:
        coords = [[[0.0, 0.0], [100.0, 0.0]]]
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {'type': 'MultiLineString', 'coordinates': coords},
        'properties': {'location': 'External', 'boundary': boundary},
    }


def test_multilinestring_boundary_features_handled():
    """Regression for TASK-1048 prod canary: Merewether boundary features
    came back from PG as MultiLineString with one ring each. The pre-fix
    coordinate loop yielded [x, y] lists into all_x_coordinates, then
    `max([list, list, ...]) - min(...)` raised TypeError on line 540."""
    geojson = {
        'crs': CRS_EPSG_32616,
        'features': [
            _external_mls_feature(fid='b.1', coords=[[[0.0, 0.0], [100.0, 0.0]]], boundary='south'),
            _external_mls_feature(fid='b.2', coords=[[[100.0, 0.0], [100.0, 100.0]]], boundary='east'),
            _external_mls_feature(fid='b.3', coords=[[[100.0, 100.0], [0.0, 100.0]]], boundary='north'),
            _external_mls_feature(fid='b.4', coords=[[[0.0, 100.0], [0.0, 0.0]]], boundary='west'),
        ],
    }
    boundary_polygon, boundary_tags = create_boundary_polygon_from_boundaries(geojson)
    assert len(boundary_polygon) == 8
    assert set(boundary_tags.keys()) == {'south', 'east', 'north', 'west'}


def test_multilinestring_with_multiple_rings_per_feature():
    """A MultiLineString feature with more than one ring is rare in the
    Hydrata FE but valid GeoJSON. Each ring's points contribute to the
    bounding-box calc and the boundary_polygon ring."""
    geojson = {
        'crs': CRS_EPSG_32616,
        'features': [
            _external_mls_feature(
                fid='b.1',
                coords=[[[0.0, 0.0], [50.0, 0.0]], [[50.0, 0.0], [100.0, 0.0]]],
                boundary='south',
            ),
            _external_feature(fid='b.2', coords=[[100.0, 0.0], [100.0, 100.0]], boundary='east'),
            _external_feature(fid='b.3', coords=[[100.0, 100.0], [0.0, 0.0]], boundary='diag'),
        ],
    }
    boundary_polygon, boundary_tags = create_boundary_polygon_from_boundaries(geojson)
    # 4 points from the two-ring MultiLineString + 2 + 2 = 8 total
    assert len(boundary_polygon) == 8
    assert set(boundary_tags.keys()) == {'south', 'east', 'diag'}
