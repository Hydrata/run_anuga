"""TASK-2825 (epic 2815 W1.6) — merge-path audit: lead-in gap semantics.

``_merge_timeseries`` (``run_utils.py`` ~1313) LEFT-merges a series onto the
model's per-second index and ``ffill``s. A series that starts PARTWAY through
the model window has nothing to forward-fill from before its first sample,
so every lead-in second stays NaN — and ``create_inflow_function`` hands that
NaN straight to ANUGA. TASK-2155 deliberately does NOT raise on a partial
lead-in (only an ALL-NaN column raises), so the run proceeds.

PROVEN on a real ``rectangular_cross_domain`` (probe, 2026-09-04): a
``Polygonal_rate_operator`` whose rate is NaN for ``t < 30`` writes NaN into
``stage`` on the polygon's triangles at the first step and it never recovers
(``rate >= 0.0`` is False for NaN, so the "negative rate" branch runs
``num.maximum(NaN, -heights)`` -> NaN). An ``Inlet_operator`` skips a NaN
volume only by accident of ``volume >= 0.0`` being False, and its
``total_applied_volume`` still goes NaN.

Contract fixed here: before a series' first sample the operator gets ZERO
rain / ZERO inflow (the same "nothing yet" the ``default_rate=0.00`` already
expresses for out-of-range times). The fill is applied PER MERGE, inside
``_merge_timeseries``, because each closure holds the frame object as bound
at ITS creation — a single end-of-function fill on the final frame would
fix only the last series (the stale-closure shape the audit documents), so
this test asserts BOTH an earlier-merged rainfall closure and a later-merged
inflow closure.

Harness rides tests/test_nan_guard.py (TASK-2155) fixture shapes.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from run_anuga.run_utils import apply_inflows_to_domain


BOUNDARY_POLYGON = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def _rows(pairs):
    """``[(seconds_after_midnight, value), ...]`` -> rowData for the
    2024-01-01T00:00:00Z model window."""
    return [
        {'timestamp': f'2024-01-01T00:{sec // 60:02d}:{sec % 60:02d}Z', 'value': value}
        for sec, value in pairs
    ]


def _rainfall_feature(fid, rows):
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


def _inflow_feature(fid, rows):
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {
            'type': 'LineString',
            'coordinates': [[30.0, 30.0], [40.0, 30.0]],
        },
        'properties': {'data': rows},
    }


@pytest.fixture
def mocks():
    return {
        'domain': MagicMock(name='domain'),
        'Polygonal_rate_operator': MagicMock(name='Polygonal_rate_operator'),
        'Inlet_operator': MagicMock(name='Inlet_operator'),
    }


def test_partial_lead_in_yields_zero_not_nan_for_every_closure(mocks):
    """Rain starts at t=30 and the inflow hydrograph at t=45 of a 60 s window.
    Every lead-in second must read 0.0 (NOT NaN) from BOTH closures — the
    rainfall one was created before the inflow merge rebound the frame."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    input_data = {
        'rainfall': {'features': [
            _rainfall_feature('rain.partial', _rows([(30, 10.0), (60, 5.0)])),
        ]},
        'inflow': {'features': [
            _inflow_feature('inflow.partial', _rows([(45, 2.0)])),
        ]},
        'catchment': {'features': []},
        'boundary_polygon': BOUNDARY_POLYGON,
    }

    inflow_functions = apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )
    rain = inflow_functions['rain.partial']
    flow = inflow_functions['inflow.partial']

    # Lead-in: zero, never NaN (``nan == 0.0`` is False, so a NaN fails here).
    for t in (0, 15.5, 29.999):
        assert rain(t) == 0.0, f'rain lead-in at t={t} -> {rain(t)!r}'
    for t in (0, 30, 44.9):
        assert flow(t) == 0.0, f'inflow lead-in at t={t} -> {flow(t)!r}'

    # From the first sample on, the existing ffill hold is unchanged.
    assert rain(30) == 10.0
    assert rain(59.9) == 10.0
    assert rain(60) == 5.0
    assert flow(45) == 2.0
    assert flow(60) == 2.0

    # Both operators still registered — a partial lead-in must not raise (2155).
    assert mocks['Polygonal_rate_operator'].call_count == 1
    assert mocks['Inlet_operator'].call_count == 1
