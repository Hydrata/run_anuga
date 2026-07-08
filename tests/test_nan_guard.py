"""TASK-2155 (epic 2147 W2) — NaN guard on the rain/inflow merge.

``_merge_timeseries`` (``run_utils.py`` ~1242) builds a per-second index
from ``start`` and LEFT-merges a timeseries onto it by absolute timestamp,
then ffills. If the attached series' timestamps don't overlap the model
window AT ALL (e.g. a 30-year model_start mismatch), the left-merge yields
every row NaN and ffill can't invent values before the first real sample —
so the merged column is entirely NaN and the operator silently applies
ZERO rain/inflow for the whole simulation, with no error or log.

Operator decision (2147 W2): ERROR the run, NEVER warn-and-continue.
"""
from datetime import datetime, timezone

import pytest

from run_anuga.run_utils import apply_inflows_to_domain


BOUNDARY_POLYGON = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def _timeseries_rainfall_feature(fid, rows):
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[
                [10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0], [10.0, 10.0],
            ]],
        },
        'properties': {'data': rows},
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
    from unittest.mock import MagicMock
    return {
        'domain': MagicMock(name='domain'),
        'Polygonal_rate_operator': MagicMock(name='Polygonal_rate_operator'),
        'Inlet_operator': MagicMock(name='Inlet_operator'),
    }


def test_entirely_nan_rain_window_raises_naming_series(mocks):
    """A rain series whose timestamps sit ~30 years outside the model window
    merges to all-NaN -> must raise, naming the series, instead of silently
    applying zero rain."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    stray_rows = [
        {'timestamp': '1994-01-01T00:00:00Z', 'value': 10.0},
        {'timestamp': '1994-01-01T00:00:30Z', 'value': 5.0},
    ]
    input_data = _input_data(
        rainfall_features=[_timeseries_rainfall_feature('rain.stray', stray_rows)],
    )
    with pytest.raises(ValueError) as excinfo:
        apply_inflows_to_domain(
            input_data=input_data, domain=mocks['domain'], start=start, duration=60,
            Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
            Inlet_operator=mocks['Inlet_operator'],
        )
    assert 'rain.stray' in str(excinfo.value)
    # No operator side effect must be registered on a window we can't honour.
    mocks['Polygonal_rate_operator'].assert_not_called()


def test_aligned_rain_window_unaffected(mocks):
    """A rain series whose timestamps overlap the model window merges fine
    and is NOT affected by the guard."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    aligned_rows = [
        {'timestamp': '2024-01-01T00:00:00Z', 'value': 10.0},
        {'timestamp': '2024-01-01T00:00:30Z', 'value': 5.0},
    ]
    input_data = _input_data(
        rainfall_features=[_timeseries_rainfall_feature('rain.aligned', aligned_rows)],
    )
    apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    assert mocks['Polygonal_rate_operator'].call_count == 1


def test_partial_overlap_lead_in_ffill_gap_does_not_raise(mocks):
    """A series that starts PARTWAY through the model window has legitimate
    leading NaNs (nothing to ffill from) — that is NOT the all-NaN failure
    mode and must not raise."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    partial_rows = [
        {'timestamp': '2024-01-01T00:00:30Z', 'value': 10.0},
        {'timestamp': '2024-01-01T00:01:00Z', 'value': 5.0},
    ]
    input_data = _input_data(
        rainfall_features=[_timeseries_rainfall_feature('rain.partial', partial_rows)],
    )
    apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    assert mocks['Polygonal_rate_operator'].call_count == 1
