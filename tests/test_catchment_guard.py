"""Tests for the catchment + rainfall combo branch of
``run_anuga.run_utils.apply_inflows_to_domain`` (TASK-882 / W3.0 followup).

Covers two pre-existing bugs:

* (a) The "multiple rainfall + catchment" guard must fail-fast BEFORE any
  ``Polygonal_rate_operator`` side effects are registered on the catchment.
* (b) If the first rainfall's ``data`` is ``None`` (e.g. only a free-text
  legacy row with no ``data_constant`` / ``data_timeseries_id``), the
  catchment branch must raise a clear ``NotImplementedError`` instead of
  crashing inside ``float(None)``.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from run_anuga.run_utils import apply_inflows_to_domain


# A 100x100 square boundary, big enough to wholly contain the tiny catchments
# / inflows we construct below.
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
        'properties': {'type': 'Rainfall', 'data': data},
    }


def _catchment_feature(fid='catch.1'):
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[
                [30.0, 30.0], [40.0, 30.0], [40.0, 40.0], [30.0, 40.0], [30.0, 30.0],
            ]],
        },
        'properties': {'type': 'Catchment'},
    }


def _input_data(rainfall_features=None, catchment_features=None):
    return {
        'inflow': {'features': list(rainfall_features or [])},
        'catchment': {'features': list(catchment_features or [])},
        'boundary_polygon': BOUNDARY_POLYGON,
    }


@pytest.fixture
def mocks():
    """Mocks for the Anuga operator constructors + a stub domain."""
    return {
        'domain': MagicMock(name='domain'),
        'Polygonal_rate_operator': MagicMock(name='Polygonal_rate_operator'),
        'Inlet_operator': MagicMock(name='Inlet_operator'),
    }


@pytest.fixture
def start():
    return datetime(2020, 1, 1, tzinfo=timezone.utc)


def test_multi_rainfall_plus_catchment_raises_before_any_side_effect(mocks, start):
    """Bug (a): the guard must fire BEFORE the catchment loop registers any
    ``Polygonal_rate_operator`` for a configuration we cannot honour."""
    input_data = _input_data(
        rainfall_features=[
            _rainfall_feature(fid='rain.1', data=1.0),
            _rainfall_feature(fid='rain.2', data=2.0),
        ],
        catchment_features=[_catchment_feature()],
    )

    with pytest.raises(NotImplementedError, match='multiple rainfall polygons'):
        apply_inflows_to_domain(
            input_data,
            mocks['domain'],
            start,
            duration=10,
            Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
            Inlet_operator=mocks['Inlet_operator'],
        )

    # No catchment-side Polygonal_rate_operator should have been registered.
    # (The two rainfall ones registered in the rainfall loop above are a
    # separate code path; this assertion is the load-bearing one.)
    call_polygons = [
        kwargs.get('polygon')
        for _args, kwargs in mocks['Polygonal_rate_operator'].call_args_list
    ]
    catchment_geom = [
        [30.0, 30.0], [40.0, 30.0], [40.0, 40.0], [30.0, 40.0], [30.0, 30.0],
    ]
    assert catchment_geom not in call_polygons, (
        'Catchment Polygonal_rate_operator was registered before the '
        'multi-rainfall guard fired; guard must hoist above the loop.'
    )


def test_first_rainfall_data_none_raises_clearly(mocks, start):
    """Bug (b): a ``None`` first-rainfall data value must raise a clear
    NotImplementedError instead of crashing inside ``float(None)``."""
    input_data = _input_data(
        rainfall_features=[_rainfall_feature(fid='rain.1', data=None)],
        catchment_features=[_catchment_feature()],
    )

    with pytest.raises(NotImplementedError, match='resolved data value'):
        apply_inflows_to_domain(
            input_data,
            mocks['domain'],
            start,
            duration=10,
            Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
            Inlet_operator=mocks['Inlet_operator'],
        )

    # The None row should have been skipped in the rainfall loop (warning +
    # continue) so no Polygonal_rate_operator was registered there either.
    assert mocks['Polygonal_rate_operator'].call_count == 0


def test_single_constant_rainfall_plus_catchment_happy_path(mocks, start):
    """A single constant rainfall + one catchment should call
    ``Polygonal_rate_operator`` twice (once for rainfall +RAINFALL_FACTOR,
    once for the catchment -RAINFALL_FACTOR) without raising."""
    input_data = _input_data(
        rainfall_features=[_rainfall_feature(fid='rain.1', data=2.5)],
        catchment_features=[_catchment_feature(fid='catch.1')],
    )

    result = apply_inflows_to_domain(
        input_data,
        mocks['domain'],
        start,
        duration=10,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )

    assert mocks['Polygonal_rate_operator'].call_count == 2
    factors = [
        kwargs.get('factor')
        for _args, kwargs in mocks['Polygonal_rate_operator'].call_args_list
    ]
    # One positive (rainfall), one negative (catchment absorption).
    assert any(f > 0 for f in factors), factors
    assert any(f < 0 for f in factors), factors
    # The rainfall inflow callable is returned for test introspection.
    assert 'rain.1' in result
