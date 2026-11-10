"""TASK-2947 (epic 3754 W3) — the residual edges of ``_merge_timeseries``.

The TASK-2825 merge-path audit (epic 2815 W1) proved four ways a user's
rainfall or inflow series can be silently corrupted on its way onto the
model's per-second index (``run_utils.apply_inflows_to_domain``):

(a) a DUPLICATE timestamp inflates the per-second frame by one row, and
    every later positional lookup (``create_inflow_function`` indexes the
    frame by whole second) reads the value one second early;
(b) FRACTIONAL-second stamps never equal a whole-second index row, so the
    exact-equality join drops them; (b2) a list MIXING whole and fractional
    stamps crashes ``pd.to_datetime`` ("unconverted data remains .5");
(c) an all-None series trips the TASK-2155 guard with the misleading
    "ZERO overlap" wording;
(d) the timezone contract (TASK-3802, epic 3801 D4): hydrata emits naive
    UTC stamps and the engine reads a naive stamp as UTC. Pinned, not
    changed.

Harness copied from tests/test_merge_lead_in_zero_fill.py (mocked operators,
no real ANUGA domain).
"""
import inspect
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest
from run_anuga.run_utils import apply_inflows_to_domain, describe_empty_merge

BOUNDARY_POLYGON = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _stamp(sec, suffix='Z'):
    """Whole or fractional seconds after 2024-01-01T00:00:00 as ISO 8601."""
    whole = int(sec)
    frac = sec - whole
    base = f'2024-01-01T00:{whole // 60:02d}:{whole % 60:02d}'
    if frac:
        base += f'{frac:.1f}'[1:]  # '.5'
    return base + suffix


def _rows(pairs, suffix='Z'):
    return [{'timestamp': _stamp(sec, suffix), 'value': value} for sec, value in pairs]


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
        'geometry': {'type': 'LineString', 'coordinates': [[30.0, 30.0], [40.0, 30.0]]},
        'properties': {'data': rows},
    }


@pytest.fixture
def mocks():
    return {
        'domain': MagicMock(name='domain'),
        'Polygonal_rate_operator': MagicMock(name='Polygonal_rate_operator'),
        'Inlet_operator': MagicMock(name='Inlet_operator'),
    }


def _apply(mocks, rainfall=(), inflow=(), duration=60):
    return apply_inflows_to_domain(
        input_data={
            'rainfall': {'features': list(rainfall)},
            'inflow': {'features': list(inflow)},
            'catchment': {'features': []},
            'boundary_polygon': BOUNDARY_POLYGON,
        },
        domain=mocks['domain'], start=START, duration=duration,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )


def _frame_of(closure):
    """The per-second frame a create_inflow_function closure indexes."""
    return inspect.getclosurevars(closure).nonlocals['dataframe']


# --- (a) duplicate timestamps ----------------------------------------------

def test_duplicate_timestamp_does_not_shift_lookups(mocks, caplog):
    """A stamp repeated at t=10 must not add a row: the 60 s window keeps 61
    rows and every second reads its own value (the LAST duplicate wins)."""
    pairs = [(0, 1.0), (10, 2.0), (10, 3.0), (20, 4.0), (30, 5.0), (40, 6.0)]
    with caplog.at_level(logging.WARNING, logger='run_anuga.run_utils'):
        flow = _apply(mocks, inflow=[_inflow_feature('inflow.dup', _rows(pairs))])['inflow.dup']

    assert len(_frame_of(flow)) == 61
    expected = {**{t: 1.0 for t in range(0, 10)}, **{t: 3.0 for t in range(10, 20)},
                **{t: 4.0 for t in range(20, 30)}, **{t: 5.0 for t in range(30, 40)},
                **{t: 6.0 for t in range(40, 61)}}
    for t, value in expected.items():
        assert flow(t) == value, f't={t}: {flow(t)!r} != {value}'
    assert any('inflow.dup' in r.getMessage() and 'duplicate' in r.getMessage()
               for r in caplog.records)


# --- (b) fractional seconds -------------------------------------------------

def test_fractional_seconds_rounded_not_dropped(mocks, caplog):
    """Every stamp at .5 s: the samples must survive (rounded to whole
    seconds), and a warning names the series."""
    pairs = [(10.5, 2.0), (20.5, 4.0), (30.5, 6.0)]
    with caplog.at_level(logging.WARNING, logger='run_anuga.run_utils'):
        flow = _apply(mocks, inflow=[_inflow_feature('inflow.frac', _rows(pairs))])['inflow.frac']

    for stamp_s, value in pairs:
        sec = pd.Timestamp(_stamp(stamp_s)).round('s').second
        assert flow(sec) == value, f'sample at {sec} s dropped: {flow(sec)!r}'
    assert flow(59) == 6.0
    assert any('inflow.frac' in r.getMessage() and 'fractional' in r.getMessage()
               for r in caplog.records)


def test_mixed_whole_and_fractional_stamps_merge(mocks):
    """'T00:00:00' beside 'T00:00:30.5' used to raise 'unconverted data
    remains'; both rows must be present after rounding."""
    rows = [{'timestamp': '2024-01-01T00:00:00Z', 'value': 1.0},
            {'timestamp': '2024-01-01T00:00:30.5Z', 'value': 7.0}]
    flow = _apply(mocks, inflow=[_inflow_feature('inflow.mixed', rows)])['inflow.mixed']
    assert flow(0) == 1.0
    assert flow(29) == 1.0
    second = pd.Timestamp('2024-01-01T00:00:30.5Z').round('s').second
    assert flow(second) == 7.0
    assert flow(60) == 7.0


def test_all_fractional_series_merges(mocks):
    """An all-fractional rainfall merges (it used to trip the guard)."""
    rows = _rows([(0.5, 12.0), (30.5, 6.0)])
    rain = _apply(mocks, rainfall=[_rainfall_feature('rain.frac', rows)])['rain.frac']
    assert rain(5) == 12.0
    assert rain(45) == 6.0
    mocks['Polygonal_rate_operator'].assert_called_once()


# --- (c) the guard names the real cause ------------------------------------

def test_all_none_window_message_names_cause(mocks):
    rows = [{'timestamp': _stamp(s), 'value': None} for s in (0, 30, 60)]
    with pytest.raises(ValueError) as excinfo:
        _apply(mocks, rainfall=[_rainfall_feature('rain.none', rows)])
    message = str(excinfo.value)
    assert 'rain.none' in message
    assert 'no values' in message
    assert 'ZERO overlap' not in message
    mocks['Polygonal_rate_operator'].assert_not_called()


def test_all_fractional_message_names_cause():
    """Without the (b) rounding path an all-fractional series lands on no
    whole-second row; the cause helper names fractional stamps, not overlap."""
    stamps = pd.Series(pd.to_datetime([_stamp(0.5), _stamp(30.5)], format='ISO8601'))
    message = describe_empty_merge(
        'rain.frac', stamps, pd.Series([1.0, 2.0]),
        pd.Timestamp('2024-01-01T00:00:00Z'), pd.Timestamp('2024-01-01T00:01:00Z'),
    )
    assert 'rain.frac' in message
    assert 'fractional' in message
    assert 'ZERO overlap' not in message


def test_no_overlap_message_still_names_the_spans():
    stamps = pd.Series(pd.to_datetime([_stamp(0).replace('2024', '1994')], format='ISO8601'))
    message = describe_empty_merge(
        'rain.stray', stamps, pd.Series([1.0]),
        pd.Timestamp('2024-01-01T00:00:00Z'), pd.Timestamp('2024-01-01T00:01:00Z'),
    )
    assert 'rain.stray' in message
    assert 'no overlap' in message
    assert '1994' in message


# --- (d) the timezone contract (TASK-3802), pinned --------------------------

def test_naive_z_and_utc_offset_stamps_are_utc(mocks):
    """Hydrata emits naive UTC; the engine reads naive as UTC. A naive stamp,
    the same stamp with 'Z' and with '+00:00' (what TASK-4295 stores) merge
    identically."""
    pairs = [(0, 1.0), (20, 2.0), (40, 3.0)]
    results = []
    for suffix in ('', 'Z', '+00:00'):
        fid = f'inflow.tz{len(results)}'
        flow = _apply(mocks, inflow=[_inflow_feature(fid, _rows(pairs, suffix))])[fid]
        results.append([flow(t) for t in range(61)])
    assert results[0] == results[1] == results[2]
    assert results[0][0] == 1.0 and results[0][20] == 2.0 and results[0][60] == 3.0
