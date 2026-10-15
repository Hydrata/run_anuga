"""Per-culvert results written to the StructureResult contract (TASK-3642, D16).

After the evolve loop :func:`write_culvert_outputs` turns the coupler's recorded
series into

* ``<output>/culverts/timeseries_<feature_id>.csv`` — columns
  ``time_s,q_m3s,hw_level_m,tw_level_m,control,q_requested_m3s`` (hydrata reads the first
  four BY NAME), and
* the hydraulic keys of ``<output>/culverts/metrics.json`` (schema_version 1),
  replacing the nulls the build step wrote.

``q_m3s`` is the flow the coupler ACTUALLY MOVED between the exchange regions (TASK-3700), so
every D16 key (peak Q, velocity, regime at peak) describes water that moved; SWMM's request
is kept only as the diagnostic ``q_requested_m3s`` column (it exceeds ``q_m3s`` when the
losing end held less water than SWMM asked for).

Sign convention: Q > 0 flows Inlet -> Outlet. ``hw_level_m`` / ``tw_level_m`` are
water-surface levels at the Inlet / Outlet exchange regions (m, model datum).
:func:`culvert_hydraulics` is pure numpy and unit-tested without anuga.
"""

from __future__ import annotations

import csv
import math
import os

from run_anuga._imports import import_optional
from run_anuga.culverts import constants
from run_anuga.culverts.metrics import write_culvert_metrics


def _number(value, digits=4):
    """A JSON-safe float (None for NaN / inf)."""
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def barrel_area_m2(culvert) -> float:
    """Flow area of all barrels after reduced-area blockage (D11)."""
    keep = 1.0 - culvert.blockage_pct / 100.0
    if culvert.shape == "circular":
        one = math.pi * culvert.diameter_m ** 2 / 4.0 * keep
    else:
        one = culvert.width_m * keep * culvert.height_m
    return one * culvert.barrels


def culvert_hydraulics(culvert, upstream_invert_m, series, control_names, downstream_invert_m=None) -> dict:
    """The D16 hydraulic keys from one recorded series.

    ``series``: dict of arrays time_s, q_m3s, hw_level_m, tw_level_m, control (int index
    into ``control_names``). Definitions: peak_q_m3s is the largest |Q| (signed);
    barrel_velocity_ms is |peak Q| / blocked flow area; regime_at_peak is the governing
    control at the peak-Q sample; regime_fractions are the shares of samples per control.

    Per-end keys (unchanged meaning, drawn at each end by the Profile tool):
    peak_hw_level_m / peak_hw_depth_m / peak_hw_time_s are the INLET end's peak level,
    depth over the upstream invert and time; peak_tw_level_m is the OUTLET end's peak
    level. peak_hw_inlet_m / peak_hw_outlet_m name the two ends' peak levels explicitly.

    Governing direction (TASK-3695, additive to schema_version 1): the headwater end is
    the end upstream of the peak-|Q| direction (``hw_end`` = "inlet" for Q >= 0, "outlet"
    for a reverse-dominant culvert, Q < 0). peak_hw_governing_level_m / _depth_m / _time_s
    are that end's peak level, depth over THAT end's invert (``downstream_invert_m`` for the
    Outlet; the upstream invert when it is not given) and time. hw_over_d (over the
    UNBLOCKED rise) and surcharge_s (time the headwater depth exceeded the rise) follow the
    governing end, so a culvert surcharged in reverse no longer reads near-dry. For a
    forward-dominant series every pre-existing key is unchanged.
    """
    numpy = import_optional("numpy")
    t = numpy.asarray(series["time_s"], dtype=float)
    if t.size == 0:
        return {}
    q = numpy.asarray(series["q_m3s"], dtype=float)
    hw = numpy.asarray(series["hw_level_m"], dtype=float)
    tw = numpy.asarray(series["tw_level_m"], dtype=float)
    control = numpy.asarray(series["control"], dtype=int)
    upstream_invert_m = float(upstream_invert_m)
    downstream_invert_m = upstream_invert_m if downstream_invert_m is None else float(downstream_invert_m)

    # peak_tw_level_m was a nanmax (an all-NaN Outlet series gave null); keep that, never raise.
    i_tw = 0 if numpy.all(numpy.isnan(tw)) else int(numpy.nanargmax(tw))
    i_hw, i_q = int(numpy.nanargmax(hw)), int(numpy.nanargmax(numpy.abs(q)))
    reverse = bool(q[i_q] < 0)
    gov_level, gov_invert, i_gov = (tw, downstream_invert_m, i_tw) if reverse else (hw, upstream_invert_m, i_hw)
    inlet_depth = max(0.0, float(hw[i_hw]) - upstream_invert_m)
    gov_depth = max(0.0, float(gov_level[i_gov]) - gov_invert)
    area = barrel_area_m2(culvert)
    step = numpy.diff(t)
    dt = numpy.append(step, step[-1] if step.size else 0.0)
    depth = gov_level - gov_invert
    fractions = {name: _number(float(numpy.mean(control == k))) for k, name in enumerate(control_names)}
    return {
        "peak_hw_level_m": _number(hw[i_hw]),
        "peak_hw_depth_m": _number(inlet_depth),
        "peak_hw_time_s": _number(t[i_hw], 1),
        "hw_over_d": _number(gov_depth / culvert.rise_m),
        "peak_tw_level_m": _number(tw[i_tw]),
        "peak_q_m3s": _number(q[i_q]),
        "peak_q_time_s": _number(t[i_q], 1),
        "barrel_velocity_ms": _number(abs(q[i_q]) / area) if area > 0 else None,
        "regime_at_peak": control_names[int(control[i_q])],
        "regime_fractions": {name: share for name, share in fractions.items() if share},
        "surcharge_s": _number(float(numpy.sum(dt[depth > culvert.rise_m])), 1),
        "peak_hw_inlet_m": _number(hw[i_hw]),
        "peak_hw_outlet_m": _number(tw[i_tw]),
        "hw_end": "outlet" if reverse else "inlet",
        "peak_hw_governing_level_m": _number(gov_level[i_gov]),
        "peak_hw_governing_depth_m": _number(gov_depth),
        "peak_hw_governing_time_s": _number(t[i_gov], 1),
    }


def timeseries_relative_path(feature_id) -> str:
    return os.path.join(constants.METRICS_DIRNAME, f"timeseries_{feature_id}.csv")


def write_timeseries_csv(output_directory, feature_id, series, control_names) -> str:
    """Write one culvert's series (thinned evenly to TIMESERIES_MAX_ROWS); returns the path relative to the output dir."""
    relative = timeseries_relative_path(feature_id)
    path = os.path.join(output_directory, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = [series[name] for name in constants.TIMESERIES_HEADER]
    requested = series.get(constants.TIMESERIES_DIAGNOSTIC_COLUMN)
    count = len(columns[0])
    limit = constants.TIMESERIES_MAX_ROWS
    rows = range(count) if count <= limit else [round(i * (count - 1) / (limit - 1)) for i in range(limit)]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(constants.TIMESERIES_HEADER + ("control",)
                        + ((constants.TIMESERIES_DIAGNOSTIC_COLUMN,) if requested is not None else ()))
        for i in rows:
            row = [f"{column[i]:.6g}" for column in columns] + [control_names[int(series["control"][i])]]
            if requested is not None:
                row.append(f"{requested[i]:.6g}")
            writer.writerow(row)
    return relative


def write_culvert_outputs(output_directory, results, coupler, log=None) -> str:
    """Finalise the coupler, write the time series and rewrite metrics.json with the hydraulic keys.

    ``results`` and ``coupler.culverts`` are in the same order. Returns the metrics path.
    """
    from anuga_drainage.culverts.inlet_control import CONTROL_NAMES

    engine_summary = coupler.finalize() or {}
    hydraulics, paths = [], []
    for index, result in enumerate(results):
        series = coupler.timeseries(index)
        built = coupler.culverts[index]
        hydraulics.append(culvert_hydraulics(
            result.culvert, built.upstream_invert_m, series, CONTROL_NAMES,
            downstream_invert_m=built.downstream_invert_m))
        paths.append(write_timeseries_csv(output_directory, result.culvert.feature_id, series, CONTROL_NAMES))
    path = write_culvert_metrics(
        output_directory, results, hydraulics=hydraulics, timeseries_paths=paths, run_fields=engine_summary)
    if log is not None:
        log.info("culvert results written: %s (%s)", path, ", ".join(f"{k}={v}" for k, v in engine_summary.items()))
    return path
