"""TASK-2816 (epic 2815, P0/H1) — 4-timeseries merge regression.

``_merge_timeseries`` (``run_utils.py`` ~1313) builds each per-series frame
as ``{timestamp, value, <name>}`` and left-merges it onto the accumulating
``inflow_dataframe`` WITHOUT dropping the raw ``value`` column. The leaked
``value`` column collides across successive merges: merge 2 suffixes it to
``value_x``/``value_y``, merge 3 re-adds a bare ``value``, and merge 4
raises ``pandas.errors.MergeError`` ("Passing 'suffixes' which cause
duplicate columns {'value_x', 'value_y'} is not allowed").

3 inflow hydrographs + 1 design-storm rainfall = 4 series = the canonical
Hydrata model shape (hit live on prod 2026-08-17: project 770, scenario
412, run 1341). Fix: narrow each per-series frame to
``[['timestamp', name]]`` before the merge.

Harness rides tests/test_nan_guard.py (TASK-2155) fixture shapes.
"""
from datetime import datetime, timezone

import pytest

from run_anuga.run_utils import apply_inflows_to_domain


BOUNDARY_POLYGON = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def _rows(values):
    """Per-30s timeseries rows aligned to the 2024-01-01 model window."""
    return [
        {'timestamp': f'2024-01-01T00:00:{sec:02d}Z', 'value': value}
        for sec, value in zip((0, 30), values)
    ]


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


def _timeseries_inflow_feature(fid, rows):
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
    from unittest.mock import MagicMock
    return {
        'domain': MagicMock(name='domain'),
        'Polygonal_rate_operator': MagicMock(name='Polygonal_rate_operator'),
        'Inlet_operator': MagicMock(name='Inlet_operator'),
    }


def test_three_inflows_plus_rainfall_merges_all_four_series(mocks):
    """The canonical 3-hydrograph + 1-design-storm model must build all four
    timeseries columns — the leaked raw 'value' column made the FOURTH merge
    raise pandas.errors.MergeError before TASK-2816."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    series_values = {
        'rainfall.storm': (10.0, 5.0),
        'inflow.c1_upper': (1.0, 2.0),
        'inflow.c2_middle': (3.0, 4.0),
        'inflow.c3_lower': (5.0, 6.0),
    }
    input_data = {
        'rainfall': {'features': [
            _timeseries_rainfall_feature('rainfall.storm', _rows(series_values['rainfall.storm'])),
        ]},
        'inflow': {'features': [
            _timeseries_inflow_feature(fid, _rows(series_values[fid]))
            for fid in ('inflow.c1_upper', 'inflow.c2_middle', 'inflow.c3_lower')
        ]},
        'catchment': {'features': []},
        'boundary_polygon': BOUNDARY_POLYGON,
    }

    inflow_functions = apply_inflows_to_domain(
        input_data=input_data, domain=mocks['domain'], start=start, duration=60,
        Polygonal_rate_operator=mocks['Polygonal_rate_operator'],
        Inlet_operator=mocks['Inlet_operator'],
    )

    # All four series registered their operators...
    assert mocks['Polygonal_rate_operator'].call_count == 1
    assert mocks['Inlet_operator'].call_count == 3
    assert set(inflow_functions) == set(series_values)
    # ...and each series' function reads ITS OWN values (first sample at t=0,
    # ffilled second sample at t=45), proving the narrowing didn't cross-wire
    # or corrupt any column.
    for name, (first, second) in series_values.items():
        assert inflow_functions[name](0) == first, name
        assert inflow_functions[name](45.0) == second, name
