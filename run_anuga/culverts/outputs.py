"""Per-culvert results written to the StructureResult contract (TASK-3642, D16).

After the evolve loop :func:`write_culvert_outputs` turns the coupler's recorded
series into

* ``<output>/culverts/timeseries_<feature_id>.csv`` — columns
  ``time_s,q_m3s,hw_level_m,tw_level_m,control`` (hydrata reads the first four), and
* the hydraulic keys of ``<output>/culverts/metrics.json`` (schema_version 1),
  replacing the nulls the build step wrote.

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


def culvert_hydraulics(culvert, upstream_invert_m, series, control_names) -> dict:
    """The D16 hydraulic keys from one recorded series.

    ``series``: dict of arrays time_s, q_m3s, hw_level_m, tw_level_m, control (int index
    into ``control_names``). Definitions: peak_q_m3s is the largest |Q| (signed);
    hw_over_d is peak headwater depth over the UNBLOCKED rise; barrel_velocity_ms is
    |peak Q| / blocked flow area; regime_at_peak is the governing control at the peak-Q
    sample; regime_fractions are the shares of samples per control; surcharge_s is the
    time the headwater depth exceeded the rise.
    """
    numpy = import_optional("numpy")
    t = numpy.asarray(series["time_s"], dtype=float)
    if t.size == 0:
        return {}
    q = numpy.asarray(series["q_m3s"], dtype=float)
    hw = numpy.asarray(series["hw_level_m"], dtype=float)
    tw = numpy.asarray(series["tw_level_m"], dtype=float)
    control = numpy.asarray(series["control"], dtype=int)

    i_hw, i_q = int(numpy.nanargmax(hw)), int(numpy.nanargmax(numpy.abs(q)))
    peak_depth = max(0.0, float(hw[i_hw]) - float(upstream_invert_m))
    area = barrel_area_m2(culvert)
    step = numpy.diff(t)
    dt = numpy.append(step, step[-1] if step.size else 0.0)
    depth = hw - float(upstream_invert_m)
    fractions = {name: _number(float(numpy.mean(control == k))) for k, name in enumerate(control_names)}
    return {
        "peak_hw_level_m": _number(hw[i_hw]),
        "peak_hw_depth_m": _number(peak_depth),
        "peak_hw_time_s": _number(t[i_hw], 1),
        "hw_over_d": _number(peak_depth / culvert.rise_m),
        "peak_tw_level_m": _number(numpy.nanmax(tw)),
        "peak_q_m3s": _number(q[i_q]),
        "peak_q_time_s": _number(t[i_q], 1),
        "barrel_velocity_ms": _number(abs(q[i_q]) / area) if area > 0 else None,
        "regime_at_peak": control_names[int(control[i_q])],
        "regime_fractions": {name: share for name, share in fractions.items() if share},
        "surcharge_s": _number(float(numpy.sum(dt[depth > culvert.rise_m])), 1),
    }


def timeseries_relative_path(feature_id) -> str:
    return os.path.join(constants.METRICS_DIRNAME, f"timeseries_{feature_id}.csv")


def write_timeseries_csv(output_directory, feature_id, series, control_names) -> str:
    """Write one culvert's series (thinned evenly to TIMESERIES_MAX_ROWS); returns the path relative to the output dir."""
    relative = timeseries_relative_path(feature_id)
    path = os.path.join(output_directory, relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = [series[name] for name in constants.TIMESERIES_HEADER]
    count = len(columns[0])
    limit = constants.TIMESERIES_MAX_ROWS
    rows = range(count) if count <= limit else [round(i * (count - 1) / (limit - 1)) for i in range(limit)]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(constants.TIMESERIES_HEADER + ("control",))
        for i in rows:
            writer.writerow([f"{column[i]:.6g}" for column in columns] + [control_names[int(series["control"][i])]])
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
        hydraulics.append(culvert_hydraulics(
            result.culvert, coupler.culverts[index].upstream_invert_m, series, CONTROL_NAMES))
        paths.append(write_timeseries_csv(output_directory, result.culvert.feature_id, series, CONTROL_NAMES))
    path = write_culvert_metrics(
        output_directory, results, hydraulics=hydraulics, timeseries_paths=paths, run_fields=engine_summary)
    if log is not None:
        log.info("culvert results written: %s (%s)", path, ", ".join(f"{k}={v}" for k, v in engine_summary.items()))
    return path
