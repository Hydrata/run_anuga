"""Defence-in-depth tests for ``create_boundary_polygon_from_boundaries``.

Guards the ``max()/min()`` calls (L537-540 pre-fix) against the 2026-04-30
prod ``ValueError: max() arg is an empty sequence`` when scenarios reach
this code path with boundaries that have empty / malformed coordinate
lists. See TASK-976.
"""

import pytest

# TASK-3457: the outline is built with shapely only (the ogr/srs code was
# vestigial); skip the whole module where shapely is absent.
pytest.importorskip("shapely")

from run_anuga.run_utils import create_boundary_polygon_from_boundaries


CRS_EPSG_32616 = {
    'type': 'name',
    'properties': {'name': 'urn:ogc:def:crs:EPSG::32616'},
}


# The happy-path square (drawn anticlockwise) as the chained, clockwise ring.
SQUARE_RING = [[0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0]]
SQUARE_TAGS = {'west': [0], 'north': [1], 'east': [2], 'south': [3]}


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
    # TASK-3457: chained (shared corners merged) and clockwise.
    assert boundary_polygon == SQUARE_RING
    assert boundary_tags == SQUARE_TAGS


def test_missing_crs_equals_with_crs():
    """TASK-3457 (D8): crs is optional; without it the outline is unchanged."""
    features = [
        _external_feature(fid='b.1', coords=[[0.0, 0.0], [100.0, 0.0]], boundary='south'),
        _external_feature(fid='b.2', coords=[[100.0, 0.0], [100.0, 100.0]], boundary='east'),
        _external_feature(fid='b.3', coords=[[100.0, 100.0], [0.0, 100.0]], boundary='north'),
        _external_feature(fid='b.4', coords=[[0.0, 100.0], [0.0, 0.0]], boundary='west'),
    ]
    with_crs = create_boundary_polygon_from_boundaries({'crs': CRS_EPSG_32616, 'features': features})
    without_crs = create_boundary_polygon_from_boundaries({'features': features})
    assert without_crs == with_crs
    assert without_crs == (SQUARE_RING, SQUARE_TAGS)


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
    assert boundary_polygon == SQUARE_RING
    assert boundary_tags == SQUARE_TAGS


def test_multilinestring_with_multiple_rings_per_feature():
    """A MultiLineString feature with more than one ring is rare in the
    Hydrata FE but valid GeoJSON. Each ring is a separate part of the
    chained outline (TASK-3457)."""
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
    # The triangle (0,0) -> (100,100) -> (100,0) -> (50,0), clockwise; the two
    # south parts keep their shared (50, 0) vertex.
    assert boundary_polygon == [[0.0, 0.0], [100.0, 100.0], [100.0, 0.0], [50.0, 0.0]]
    assert boundary_tags == {'diag': [0], 'east': [1], 'south': [2, 3]}
