"""Tests for the W2.3 inflow/rainfall split in ``apply_inflows_to_domain``.

Polygonal_rate_operator (rainfall) is registered for every feature under
``input_data['rainfall']['features']``. Inlet_operator (surface inflow) is
registered for every feature under ``input_data['inflow']['features']``.
No more ``properties.type`` branching. Geometry IS the discriminator.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from run_anuga.run_utils import apply_inflows_to_domain


BOUNDARY_POLYGON = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def _rainfall_feature(fid='rain.1', data=1.0):
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[
                [10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0], [10.0, 10.0],
            ]],
        },
        'properties': {'data': data},
    }


def _surface_feature(fid='surf.1', data=0.5):
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {
            'type': 'LineString',
            'coordinates': [[15.0, 15.0], [25.0, 15.0]],
        },
        'properties': {'data': data},
    }


def _input_data(rainfall_features=None, inflow_features=None):
    return {
        'rainfall': {'features': list(rainfall_features or [])},
        'inflow': {'features': list(inflow_features or [])},
        'catchment': {'features': []},
        'boundary_polygon': BOUNDARY_POLYGON,
    }


@pytest.fixture
def mocks():
    return {
        'domain': MagicMock(name='domain'),
        'Polygonal_rate_operator': MagicMock(name='Polygonal_rate_operator'),
        'Inlet_operator': MagicMock(name='Inlet_operator'),
    }


@pytest.fixture
def start():
    return datetime(2020, 1, 1, tzinfo=timezone.utc)


def test_rainfall_only_routes_to_polygonal_rate_operator(mocks, start):
    input_data = _input_data(rainfall_features=[_rainfall_feature()])
    apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    assert mocks['Polygonal_rate_operator'].call_count == 1
    mocks['Inlet_operator'].assert_not_called()


def test_inflow_only_routes_to_inlet_operator(mocks, start):
    input_data = _input_data(inflow_features=[_surface_feature()])
    apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    mocks['Polygonal_rate_operator'].assert_not_called()
    assert mocks['Inlet_operator'].call_count == 1


def test_rainfall_and_inflow_both_route_independently(mocks, start):
    input_data = _input_data(
        rainfall_features=[_rainfall_feature(fid='r1'), _rainfall_feature(fid='r2', data=2.0)],
        inflow_features=[_surface_feature(fid='s1'), _surface_feature(fid='s2', data=1.5)],
    )
    apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    assert mocks['Polygonal_rate_operator'].call_count == 2
    assert mocks['Inlet_operator'].call_count == 2


def test_missing_keys_treated_as_empty(mocks, start):
    apply_inflows_to_domain(
        input_data={'boundary_polygon': BOUNDARY_POLYGON, 'catchment': {'features': []}},
        domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    mocks['Polygonal_rate_operator'].assert_not_called()
    mocks['Inlet_operator'].assert_not_called()


def test_no_properties_type_filter_required(mocks, start):
    rainfall = _rainfall_feature()
    rainfall['properties'].pop('type', None)
    surface = _surface_feature()
    surface['properties'].pop('type', None)
    input_data = _input_data(rainfall_features=[rainfall], inflow_features=[surface])
    apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    assert mocks['Polygonal_rate_operator'].call_count == 1
    assert mocks['Inlet_operator'].call_count == 1


# --------------------------------------------------------------------------- #
# TASK-3457 (epic 3456, D4/D10): a water feature with NO part inside the       #
# outline BLOCKS the run (named error, nothing half-applied); anything that    #
# reaches inside goes to ANUGA unchanged (ANUGA uses only in-domain triangles).#
# --------------------------------------------------------------------------- #
from run_anuga.run_utils import (  # noqa: E402
    FEATURE_OUTSIDE_ERROR,
    check_water_features_inside_outline,
)


def _line_feature(fid, coords, data=0.5):
    return {
        'type': 'Feature', 'id': fid,
        'geometry': {'type': 'LineString', 'coordinates': coords},
        'properties': {'data': data},
    }


def _catchment_feature(fid, ring):
    # Catchment has no FeatureDataMixin: it never carries ``data``.
    return {
        'type': 'Feature', 'id': fid,
        'geometry': {'type': 'Polygon', 'coordinates': [ring]},
        'properties': {},
    }


def _apply(mocks, start, input_data):
    return apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )


OUTSIDE_LINE = [[150.0, 50.0], [180.0, 50.0]]
CROSSING_LINE = [[50.0, 50.0], [150.0, 50.0]]
EDGE_LINE = [[0.0, 20.0], [0.0, 80.0]]  # lies ONLY on the west edge
OUTSIDE_RING = [[150.0, 10.0], [190.0, 10.0], [190.0, 40.0], [150.0, 10.0]]
HALF_OUT_RING = [[80.0, 10.0], [140.0, 10.0], [140.0, 40.0], [80.0, 40.0], [80.0, 10.0]]


def test_inflow_wholly_outside_raises_and_registers_nothing(mocks, start):
    input_data = _input_data(
        rainfall_features=[_rainfall_feature()],
        inflow_features=[_surface_feature(fid='ok.1'), _line_feature('inf.out', OUTSIDE_LINE)],
    )
    with pytest.raises(ValueError) as exc:
        _apply(mocks, start, input_data)
    assert str(exc.value).startswith(FEATURE_OUTSIDE_ERROR)
    assert 'surface inflow inf.out' in str(exc.value)
    assert 'ok.1' not in str(exc.value)
    mocks['Polygonal_rate_operator'].assert_not_called()
    mocks['Inlet_operator'].assert_not_called()


def test_inflow_crossing_the_outline_reaches_inlet_operator_unchanged(mocks, start):
    _apply(mocks, start, _input_data(inflow_features=[_line_feature('inf.x', CROSSING_LINE)]))
    assert mocks['Inlet_operator'].call_count == 1
    assert mocks['Inlet_operator'].call_args.args[1] == CROSSING_LINE


def test_inflow_only_on_the_outline_edge_raises(mocks, start):
    with pytest.raises(ValueError, match='surface inflow inf.edge has no part inside'):
        _apply(mocks, start, _input_data(inflow_features=[_line_feature('inf.edge', EDGE_LINE)]))
    mocks['Inlet_operator'].assert_not_called()


def test_catchment_wholly_outside_raises(mocks, start):
    input_data = _input_data(rainfall_features=[_rainfall_feature()])
    input_data['catchment'] = {'features': [_catchment_feature('cat.out', OUTSIDE_RING)]}
    with pytest.raises(ValueError) as exc:
        _apply(mocks, start, input_data)
    assert str(exc.value).startswith(FEATURE_OUTSIDE_ERROR)
    assert 'catchment cat.out' in str(exc.value)
    mocks['Polygonal_rate_operator'].assert_not_called()


def test_catchment_half_outside_reaches_polygonal_rate_operator(mocks, start):
    input_data = _input_data(rainfall_features=[_rainfall_feature()])
    input_data['catchment'] = {'features': [_catchment_feature('cat.half', HALF_OUT_RING)]}
    _apply(mocks, start, input_data)
    polygons = [c.kwargs['polygon'] for c in mocks['Polygonal_rate_operator'].call_args_list]
    assert HALF_OUT_RING in polygons


def test_d10_catchment_outside_with_no_rainfall_does_not_raise(mocks, start):
    input_data = _input_data()
    input_data['catchment'] = {'features': [_catchment_feature('cat.out', OUTSIDE_RING)]}
    _apply(mocks, start, input_data)
    mocks['Polygonal_rate_operator'].assert_not_called()


def test_d10_inflow_with_no_data_wholly_outside_is_skipped(mocks, start):
    _apply(mocks, start, _input_data(inflow_features=[_line_feature('inf.nodata', OUTSIDE_LINE, data=None)]))
    mocks['Inlet_operator'].assert_not_called()


def test_check_water_features_inside_outline_direct_call():
    outline = [list(p) for p in BOUNDARY_POLYGON]
    inflows = {'features': [_line_feature('b.2', OUTSIDE_LINE), _line_feature('a.1', OUTSIDE_LINE),
                            _line_feature('in', CROSSING_LINE)]}
    catchments = {'features': [_catchment_feature('cat.out', OUTSIDE_RING)]}
    rainfall = {'features': [_rainfall_feature()]}
    # inside / crossing only: no raise
    check_water_features_inside_outline(outline, inflow_geojson={'features': [_line_feature('in', CROSSING_LINE)]})
    with pytest.raises(ValueError) as exc:
        check_water_features_inside_outline(outline, inflow_geojson=inflows, catchment_geojson=catchments,
                                            rainfall_geojson=rainfall)
    assert str(exc.value) == (
        'Feature is outside the model area: surface inflow a.1, surface inflow b.2, '
        'catchment cat.out have no part inside the boundary outline.'
    )
    # D10: without rainfall the catchment is not checked
    check_water_features_inside_outline(outline, catchment_geojson=catchments)
