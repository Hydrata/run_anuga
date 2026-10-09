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


# ── TASK-3547 (epic 3483 W5): a rainfall hyetograph stops after its last block ──
#
# Operator decision 08-10-26 (option B): after a RAINFALL series' last row ends
# (last_t + last_dt, the series' own final interval) the engine applies 0, not
# the last value held to the end of the run. hydrology/design_storm.py writes
# one row per block (value = block depth / block hours), so the old ffill turned
# a "50 mm over 2 h" storm into 50 mm + 20 mm/h x 4 h inside a 6 h run. Inflow
# hydrographs keep their ffill (a held base flow is the intended semantics).

HOUR = 3600


def _storm_rows(start, offsets_h, values):
    from datetime import timedelta
    return [
        {'timestamp': (start + timedelta(hours=h)).isoformat(), 'value': v}
        for h, v in zip(offsets_h, values)
    ]


def _apply_series(mocks, start, duration, rainfall=None, inflow=None):
    input_data = _input_data(
        rainfall_features=[_rainfall_feature('rain.storm', rainfall)] if rainfall is not None else None,
        inflow_features=[_surface_feature('surf.hydro', inflow)] if inflow is not None else None,
    )
    return apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=duration,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )


def _depth(fn, duration):
    """Integral of a per-second rate (units/h) over the run, in units."""
    return sum(float(fn(t)) for t in range(duration)) / HOUR


def test_rainfall_storm_2h_inside_6h_run_applies_exactly_its_designed_depth(mocks, start):
    fns = _apply_series(mocks, start, 6 * HOUR, rainfall=_storm_rows(start, [0, 1], [30.0, 20.0]))
    rain = fns['rain.storm']
    assert _depth(rain, 6 * HOUR) == pytest.approx(50.0)
    assert float(rain(2 * HOUR - 1)) == 20.0, 'the last block still rains to its end'
    assert float(rain(2 * HOUR)) == 0.0, 'rain is 0 once the last block ends'
    assert float(rain(6 * HOUR)) == 0.0


def test_rainfall_storm_starting_mid_run_is_zero_before_and_after(mocks, start):
    fns = _apply_series(mocks, start, 6 * HOUR, rainfall=_storm_rows(start, [1, 2, 3], [10.0, 40.0, 10.0]))
    rain = fns['rain.storm']
    assert float(rain(HOUR - 1)) == 0.0
    assert float(rain(4 * HOUR)) == 0.0
    assert _depth(rain, 6 * HOUR) == pytest.approx(60.0)


def test_rainfall_rows_out_of_order_still_end_after_the_last_block(mocks, start):
    rows = _storm_rows(start, [1, 0], [20.0, 30.0])
    fns = _apply_series(mocks, start, 6 * HOUR, rainfall=rows)
    assert _depth(fns['rain.storm'], 6 * HOUR) == pytest.approx(50.0)


def test_rainfall_series_longer_than_the_run_is_unchanged(mocks, start):
    fns = _apply_series(mocks, start, 2 * HOUR, rainfall=_storm_rows(start, [0, 1, 2, 3], [5.0, 5.0, 5.0, 5.0]))
    assert _depth(fns['rain.storm'], 2 * HOUR) == pytest.approx(10.0)


def test_rainfall_single_row_series_keeps_its_held_value(mocks, start):
    """One row has no interval to end on: it stays a held rate (as before)."""
    fns = _apply_series(mocks, start, 2 * HOUR, rainfall=_storm_rows(start, [0], [7.0]))
    assert float(fns['rain.storm'](2 * HOUR - 1)) == 7.0


def test_inflow_hydrograph_still_holds_its_last_value(mocks, start):
    """AC2 — inflow (Inlet_operator) series keep ffill semantics."""
    fns = _apply_series(mocks, start, 6 * HOUR, inflow=_storm_rows(start, [0, 1], [3.0, 2.0]))
    q = fns['surf.hydro']
    assert float(q(2 * HOUR)) == 2.0
    assert float(q(6 * HOUR)) == 2.0


def test_rainfall_and_inflow_same_rows_differ_only_after_the_last_block(mocks, start):
    rows = _storm_rows(start, [0, 1], [30.0, 20.0])
    fns = _apply_series(mocks, start, 4 * HOUR, rainfall=rows, inflow=rows)
    for t in (0, HOUR - 1, HOUR, 2 * HOUR - 1):
        assert float(fns['rain.storm'](t)) == float(fns['surf.hydro'](t))
    assert float(fns['rain.storm'](3 * HOUR)) == 0.0
    assert float(fns['surf.hydro'](3 * HOUR)) == 20.0
