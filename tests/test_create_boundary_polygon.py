"""Defence-in-depth tests for ``create_boundary_polygon_from_boundaries``.

Guards the ``max()/min()`` calls (L537-540 pre-fix) against the 2026-04-30
prod ``ValueError: max() arg is an empty sequence`` when scenarios reach
this code path with boundaries that have empty / malformed coordinate
lists. See TASK-976.
"""

import pytest

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
