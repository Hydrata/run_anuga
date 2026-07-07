"""Tests for run_anuga.run_utils.build_time_boundary_function (TASK-795).

These exercise the pure function that builds the callable passed to
anuga.Time_boundary(domain, function=...). They don't need a real Anuga
domain — only the callable contract (f(t_seconds) -> [stage, xmom, ymom]).
"""

import pytest

from run_anuga.run_utils import build_time_boundary_function


def _feature(boundary_kind='Time', data=None, location='External', fid='bdy.1'):
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {'type': 'LineString', 'coordinates': [[0, 0], [1, 1]]},
        'properties': {
            'boundary': boundary_kind,
            'location': location,
            'data': data,
        },
    }


class TestConstantCase:
    def test_numeric_constant_returns_constant_stage(self):
        fn = build_time_boundary_function([_feature(data=2.5)])
        assert fn(0) == [2.5, 0.0, 0.0]
        assert fn(100) == [2.5, 0.0, 0.0]
        assert fn(99999) == [2.5, 0.0, 0.0]

    def test_int_constant_coerced_to_float(self):
        fn = build_time_boundary_function([_feature(data=3)])
        result = fn(50)
        assert result == [3.0, 0.0, 0.0]
        assert isinstance(result[0], float)

    def test_zero_constant_is_valid(self):
        fn = build_time_boundary_function([_feature(data=0)])
        assert fn(0) == [0.0, 0.0, 0.0]

    def test_numeric_string_constant_coerced(self):
        # Defensive — Boundary.make_file should have coerced this already,
        # but we tolerate a numeric string at the engine boundary.
        fn = build_time_boundary_function([_feature(data='4.4')])
        assert fn(0) == [4.4, 0.0, 0.0]

    def test_non_numeric_string_raises(self):
        with pytest.raises(ValueError):
            build_time_boundary_function([_feature(data='not_a_number')])


class TestTimeSeriesCase:
    def test_two_point_linear_interpolation(self):
        rowdata = [
            {'timestamp': '2020-01-01T00:00:00+00:00', 'value': 0.0},
            {'timestamp': '2020-01-01T00:01:00+00:00', 'value': 6.0},  # +60s, +6m
        ]
        fn = build_time_boundary_function([_feature(data=rowdata)])
        # At t=0, stage=0; at t=60, stage=6; at t=30, stage=3 (linear)
        assert fn(0) == [0.0, 0.0, 0.0]
        assert fn(60) == [6.0, 0.0, 0.0]
        assert abs(fn(30)[0] - 3.0) < 1e-9

    def test_clamps_below_first_sample(self):
        rowdata = [
            {'timestamp': '2020-01-01T00:00:00+00:00', 'value': 1.0},
            {'timestamp': '2020-01-01T00:01:00+00:00', 'value': 5.0},
        ]
        fn = build_time_boundary_function([_feature(data=rowdata)])
        # t<0 should return the first value (numpy.interp clamps).
        assert fn(-100)[0] == 1.0

    def test_clamps_above_last_sample(self):
        rowdata = [
            {'timestamp': '2020-01-01T00:00:00+00:00', 'value': 1.0},
            {'timestamp': '2020-01-01T00:01:00+00:00', 'value': 5.0},
        ]
        fn = build_time_boundary_function([_feature(data=rowdata)])
        # t > last → last value.
        assert fn(99999)[0] == 5.0

    def test_three_point_series(self):
        rowdata = [
            {'timestamp': '2020-01-01T00:00:00+00:00', 'value': 0.0},
            {'timestamp': '2020-01-01T00:00:30+00:00', 'value': 3.0},
            {'timestamp': '2020-01-01T00:01:00+00:00', 'value': 0.0},
        ]
        fn = build_time_boundary_function([_feature(data=rowdata)])
        assert fn(0)[0] == 0.0
        assert fn(30)[0] == 3.0
        assert fn(60)[0] == 0.0
        assert abs(fn(45)[0] - 1.5) < 1e-9  # halfway between (30,3) and (60,0)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            build_time_boundary_function([_feature(data=[])])

    def test_row_missing_timestamp_raises(self):
        with pytest.raises(ValueError):
            build_time_boundary_function([_feature(data=[{'value': 1.0}])])

    def test_row_missing_value_raises(self):
        with pytest.raises(ValueError):
            build_time_boundary_function([_feature(data=[
                {'timestamp': '2020-01-01T00:00:00+00:00'}
            ])])


class TestEdgeCases:
    def test_no_features_raises(self):
        with pytest.raises(ValueError):
            build_time_boundary_function([])

    def test_multiple_features_uses_first_and_warns(self, caplog):
        import logging
        feat_a = _feature(data=1.0, fid='bdy.1')
        feat_b = _feature(data=99.0, fid='bdy.2')
        with caplog.at_level(logging.WARNING):
            fn = build_time_boundary_function([feat_a, feat_b])
        assert fn(10) == [1.0, 0.0, 0.0]
        assert any('Multiple Time boundary features' in r.message for r in caplog.records)

    def test_none_data_returns_zero_stage_function(self, caplog):
        import logging
        with caplog.at_level(logging.ERROR):
            fn = build_time_boundary_function([_feature(data=None)])
        assert fn(0) == [0.0, 0.0, 0.0]
        assert any('no resolved data' in r.message for r in caplog.records)
