"""TASK-3775: rain mass check in run_summary (expected vs applied rain).

Negative controls are built in: a halved applied volume, a dropped rain
polygon (on a REAL ANUGA CPU sim) and an untagged operator each turn
``rain.ok`` false or leave it out of the sum.
"""

import json
import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from run_anuga import defaults
from run_anuga.rain_check import (
    RAIN_OPERATOR_ROLE,
    RAIN_OPERATOR_ROLE_ATTR,
    RAIN_TOLERANCE_PCT,
    build_rain_check,
    collect_applied_rain_m3,
    expected_rain_volume_m3,
    rate_time_integral,
    run_rain_check,
    tag_rain_operator,
)
from run_anuga.run_utils import apply_inflows_to_domain

START = "2020-01-01T00:00:00+00:00"
OUTLINE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def _square(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def _rain(fid, geom, data):
    return {"type": "Feature", "id": fid, "geometry": geom, "properties": {"data": data}}


def _series(*rows):
    """rows = (seconds after START, mm/h)."""
    return [{"timestamp": f"2020-01-01T{int(s) // 3600:02d}:{int(s) % 3600 // 60:02d}:{int(s) % 60:02d}",
             "value": v} for s, v in rows]


# --- expected: rate integral ---------------------------------------------------

def test_constant_rate_is_held_for_the_whole_run():
    assert rate_time_integral(36.0, START, 600) == pytest.approx(36.0 * 600)


def test_no_data_integrates_to_zero():
    assert rate_time_integral(None, START, 600) == 0.0


def test_series_is_a_step_function_with_zero_before_first_sample_and_a_last_block():
    # 0 before t=60; 10 mm/h on [60,120); 20 on [120,180); last block = one interval
    data = _series((60, 10.0), (120, 20.0))
    assert rate_time_integral(data, START, 600) == pytest.approx(10 * 60 + 20 * 60)


def test_series_is_clipped_to_the_model_window():
    data = _series((0, 10.0), (300, 20.0), (900, 5.0))
    # window 0..600: 10 x 300 + 20 x 300 (the 20 block runs 300..900)
    assert rate_time_integral(data, START, 600) == pytest.approx(10 * 300 + 20 * 300)


def test_one_row_series_is_a_held_rate():
    assert rate_time_integral(_series((100, 6.0)), START, 600) == pytest.approx(6.0 * 500)


def test_naive_timestamps_are_utc_like_the_runtime():
    data = [{"timestamp": "2020-01-01T00:01:00", "value": 1.0},
            {"timestamp": "2020-01-01T00:02:00", "value": 1.0}]
    assert rate_time_integral(data, START, 600) == pytest.approx(120.0)


def test_integral_agrees_with_the_rate_function_the_engine_is_fed():
    """Independent implementations: the step integral must equal the sum of
    apply_inflows_to_domain's per-second rate function over the run."""
    duration = 900
    data = _series((30, 12.0), (90, 0.0), (150, 48.0), (390, 6.0), (600, 3.0))
    funcs = apply_inflows_to_domain(
        input_data={"rainfall": {"features": [_rain("r1", _square(10, 10, 20, 20), data)]},
                    "inflow": {"features": []}, "catchment": {"features": []},
                    "boundary_polygon": OUTLINE},
        domain=MagicMock(), start=START, duration=duration,
        Polygonal_rate_operator=MagicMock(), Inlet_operator=MagicMock(),
    )
    engine_fed = sum(float(funcs["r1"](t)) for t in range(duration))
    assert rate_time_integral(data, START, duration) == pytest.approx(engine_fed)


# --- expected: areas -------------------------------------------------------------

def test_expected_volume_clips_polygons_to_the_outline_and_skips_no_data():
    feats = [
        _rain("in", _square(10, 10, 30, 30), 3600.0),       # 400 m2
        _rain("half", _square(90, 0, 110, 10), 3600.0),     # 100 of 200 m2 inside
        _rain("none", _square(40, 40, 50, 50), None),        # skipped (engine skips too)
    ]
    out = expected_rain_volume_m3(feats, OUTLINE, START, 100, defaults.RAINFALL_FACTOR, holes=[])
    # 3600 mm/h = 1 mm/s = 1e-3 m/s; 100 s -> 0.1 m depth
    assert out["expected_m3"] == pytest.approx(0.1 * 500)
    assert out["n_polygons"] == 2 and out["n_polygons_skipped"] == 1
    assert out["rain_area_m2"] == pytest.approx(500)


def test_expected_volume_excludes_reflective_building_holes():
    feats = [_rain("r", _square(0, 0, 100, 100), 3600.0)]
    hole = [(10, 10), (20, 10), (20, 20), (10, 20)]
    out = expected_rain_volume_m3(feats, OUTLINE, START, 100, defaults.RAINFALL_FACTOR, holes=[hole])
    assert out["rain_area_m2"] == pytest.approx(10000 - 100)


# --- verdict ----------------------------------------------------------------------

def _expected(m3):
    return {"expected_m3": m3, "n_polygons": 1, "n_polygons_skipped": 0, "rain_area_m2": 1.0}


def test_rain_block_has_the_acceptance_keys():
    rain = build_rain_check(_expected(1000.0), 1005.0)
    assert {"expected_m3", "applied_m3", "diff_pct", "ok"} <= set(rain)
    assert rain["diff_pct"] == pytest.approx(0.5)
    assert rain["ok"] is True
    json.dumps(rain)


def test_negative_control_halved_applied_rain_is_not_ok():
    rain = build_rain_check(_expected(1000.0), 500.0)
    assert rain["ok"] is False
    assert rain["diff_pct"] == pytest.approx(-50.0)


def test_just_outside_the_tolerance_is_not_ok():
    over = 1000.0 * (1 + (RAIN_TOLERANCE_PCT + 0.01) / 100)
    assert build_rain_check(_expected(1000.0), over)["ok"] is False


def test_no_rain_expected_and_none_applied_is_ok():
    rain = build_rain_check(_expected(0.0), 0.0)
    assert rain["ok"] is True and rain["diff_pct"] is None
    assert build_rain_check(_expected(0.0), 5.0)["ok"] is False


# --- applied: operator tagging ------------------------------------------------------

def _op(influx, tagged=True):
    op = SimpleNamespace(cumulative_influx=influx)
    return tag_rain_operator(op) if tagged else op


def test_applied_sums_only_tagged_rain_operators():
    domain = SimpleNamespace(fractional_step_operators=[_op(10.0), _op(5.0), _op(-99.0, tagged=False)])
    assert collect_applied_rain_m3(domain) == pytest.approx(15.0)


def test_apply_inflows_tags_rain_but_not_catchment_operators():
    made = []

    def factory(domain, **kw):
        op = SimpleNamespace(factor=kw["factor"])
        made.append(op)
        return op

    apply_inflows_to_domain(
        input_data={"rainfall": {"features": [_rain("r1", _square(10, 10, 20, 20), 5.0)]},
                    "inflow": {"features": []},
                    "catchment": {"features": [{"id": "c1", "geometry": _square(30, 30, 40, 40),
                                                "properties": {}}]},
                    "boundary_polygon": OUTLINE},
        domain=MagicMock(), start=START, duration=60,
        Polygonal_rate_operator=factory, Inlet_operator=MagicMock(),
    )
    rain = [op for op in made if op.factor > 0]
    catchment = [op for op in made if op.factor < 0]
    assert len(rain) == 1 and len(catchment) == 1
    assert getattr(rain[0], RAIN_OPERATOR_ROLE_ATTR) == RAIN_OPERATOR_ROLE
    assert not hasattr(catchment[0], RAIN_OPERATOR_ROLE_ATTR)


def test_run_rain_check_never_raises_and_flags_a_broken_input(caplog):
    domain = SimpleNamespace(fractional_step_operators=[_op(1.0)])
    rain = run_rain_check(domain, {"rainfall": {"features": [{"geometry": None,
                                                              "properties": {"data": 1.0}}]},
                                   "boundary_polygon": OUTLINE},
                          model_start=START, duration_s=60, factor=defaults.RAINFALL_FACTOR,
                          is_rank0=True, holes=[])
    assert rain["ok"] is False and "error" in rain


def test_run_rain_check_logs_a_warn_line_to_the_run_logger_when_rain_is_missing(caplog):
    import logging

    run_log = logging.getLogger("test.run_log")
    # 10 m2 x 3600 mm/h (1e-3 m/s) x 100 s = 1 m3 expected; operators applied half
    domain = SimpleNamespace(fractional_step_operators=[_op(0.5)])
    feats = [_rain("r", _square(0, 0, 10, 1), 3600.0)]
    with caplog.at_level(logging.INFO, logger="test.run_log"):
        rain = run_rain_check(domain, {"rainfall": {"features": feats}, "boundary_polygon": OUTLINE},
                              model_start=START, duration_s=100, factor=defaults.RAINFALL_FACTOR,
                              is_rank0=True, holes=[], log=run_log)
    assert rain["ok"] is False
    assert any(r.name == "test.run_log" and "rain check WARN" in r.getMessage() for r in caplog.records)


def test_run_rain_check_returns_none_off_rank0():
    domain = SimpleNamespace(fractional_step_operators=[_op(1.0)])
    assert run_rain_check(domain, {}, model_start=START, duration_s=60,
                          factor=defaults.RAINFALL_FACTOR, is_rank0=False) is None


# --- real ANUGA CPU sim -------------------------------------------------------------

def _real_sim(drop_feature=None, duration=120):
    anuga = pytest.importorskip("anuga")
    from anuga.operators.rate_operators import Polygonal_rate_operator

    domain = anuga.rectangular_cross_domain(20, 20, len1=100.0, len2=100.0)
    domain.set_name("rain_check_test")
    domain.set_store(False)
    domain.set_quantity("elevation", lambda x, y: -0.001 * x)
    domain.set_quantity("friction", 0.03)
    domain.set_quantity("stage", expression="elevation")
    domain.set_boundary({k: anuga.Reflective_boundary(domain) for k in ("left", "right", "top", "bottom")})
    feats = [
        _rain("a", _square(0, 0, 50, 100), _series((0, 36.0), (40, 72.0), (80, 18.0))),
        _rain("b", _square(50, 0, 100, 100), 54.0),
    ]
    input_data = {"rainfall": {"features": feats}, "inflow": {"features": []},
                  "catchment": {"features": []}, "boundary_polygon": OUTLINE}
    apply_inflows_to_domain(input_data=input_data, domain=domain, start=START, duration=duration,
                            Polygonal_rate_operator=Polygonal_rate_operator, Inlet_operator=MagicMock())
    if drop_feature is not None:
        # Negative control: the engine loses one rain polygon after the inputs were read.
        from run_anuga.rain_check import rain_operators
        domain.fractional_step_operators.remove(rain_operators(domain)[drop_feature])
    for _ in domain.evolve(yieldstep=30.0, finaltime=float(duration)):
        pass
    return run_rain_check(domain, input_data, model_start=START, duration_s=duration,
                          factor=defaults.RAINFALL_FACTOR, is_rank0=True, holes=[])


def test_real_cpu_sim_applied_rain_matches_expected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rain = _real_sim()
    # a: 36x40 + 72x40 + 18x40 mm/h.s over 5000 m2; b: 54 x 120 over 5000 m2
    expected = (36 * 40 + 72 * 40 + 18 * 40 + 54 * 120) * defaults.RAINFALL_FACTOR * 5000
    assert rain["expected_m3"] == pytest.approx(expected, rel=1e-6)
    assert rain["ok"] is True, rain
    assert abs(rain["diff_pct"]) < 0.5


def test_negative_control_real_sim_dropping_one_rain_polygon_is_not_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rain = _real_sim(drop_feature=0)
    assert rain["ok"] is False, rain
    assert rain["diff_pct"] < -20


# --- run_summary integration ----------------------------------------------------------

def _monitor(tmp_path):
    from tests.test_diagnostics import _make_mock_domain
    from run_anuga.diagnostics import SimulationMonitor

    return SimulationMonitor(_make_mock_domain(), str(tmp_path), 1, 10.0, duration_s=10.0,
                             run_label="r", scenario_config={})


def test_run_summary_carries_the_rain_block(tmp_path):
    mon = _monitor(tmp_path)
    mon.record(10.0, wall_time_s=1.0)
    mon.set_rain_check(build_rain_check(_expected(100.0), 50.0))
    mon.finalize()
    summary = json.load(open(tmp_path / "run_summary_1.json"))
    assert summary["rain"]["ok"] is False
    assert summary["rain"]["expected_m3"] == 100.0 and summary["rain"]["applied_m3"] == 50.0
    assert not math.isnan(summary["rain"]["diff_pct"])


def test_run_summary_has_no_rain_block_when_no_check_ran(tmp_path):
    mon = _monitor(tmp_path)
    mon.record(10.0, wall_time_s=1.0)
    mon.finalize()
    assert "rain" not in json.load(open(tmp_path / "run_summary_1.json"))


def test_run_sim_runs_the_rain_check_after_evolve_on_every_rank():
    import inspect
    import run_anuga.run as run_module

    src = inspect.getsource(run_module.run_sim)
    i_evolve = src.index("domain.evolve(")
    i_check = src.index("run_rain_check(")
    i_finalize = src.index("finalize_monitor_safely(monitor)")
    assert i_evolve < i_check < i_finalize
    # not inside a rank-0 gate: the reduction is collective
    line = src[src.rfind("\n", 0, i_check) + 1:i_check]
    assert line.startswith("        _rain = ")
