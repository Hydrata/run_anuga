"""Rain mass check for the run summary (TASK-3775, epic 3585).

Answers one question per run: did the engine put in the rain the scenario
asked for? ``run_summary_N.json`` gains a ``rain`` block::

    "rain": {
        "expected_m3": ...,   # from the scenario's rainfall inputs
        "applied_m3": ...,    # what the rain operators actually added
        "diff_pct": ...,      # 100 * (applied - expected) / expected
        "ok": true,           # |diff_pct| <= tolerance_pct
        "tolerance_pct": 2.0,
        ...
    }

Expected volume
    Worked out from the rainfall features alone, independently of the
    dataframe/operator code that feeds the engine: for each rainfall polygon,
    the time integral of its rate over the model window x the polygon's area
    inside the model outline (minus Reflective-building mesh holes, where no
    mesh exists to rain on) x ``RAINFALL_FACTOR`` (mm/h -> m/s). The rate is
    read the way ``apply_inflows_to_domain`` feeds it: a constant is held for
    the whole run; a timeseries is a step function (each sample holds until
    the next one), zero before its first sample, and its last block ends one
    interval after the last sample (TASK-3547). Features with no resolved data
    are skipped, as the engine skips them.

Applied volume
    The sum of ``cumulative_influx`` over the rain operators (the
    ``Polygonal_rate_operator`` instances ``apply_inflows_to_domain`` tags as
    rainfall), reduced across MPI ranks. ANUGA's ``Rate_operator`` adds each
    step's applied volume to ``cumulative_influx`` on BOTH paths: the CPU path
    sums ``rate * factor * dt * area`` over owned triangles, and the GPU path
    adds the value the device kernel returns (``gpu_rate_operator_apply``: an
    OpenMP-target reduction of the same product over owned, ghost-masked
    triangles, anuga/shallow_water/gpu/gpu_rate_operator.c). Rate operators
    stay Python-called every step in GPU mode (apply_fractional_steps), so the
    accumulator is updated in both modes. Catchment operators (negative
    factor) are not rain and are not counted.

The difference between the two is the mesh's discretisation of the rain
polygons (triangles are assigned by centroid) plus anything the engine
dropped (a polygon outside the mesh, a MultiPolygon's later parts or a
polygon's holes, which ``_extract_polygon_outer_ring`` discards, a NaN rate).
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

#: Attribute set on each rain operator so it can be found again after evolve,
#: including on a domain restored from a checkpoint pickle.
RAIN_OPERATOR_ROLE_ATTR = "run_anuga_role"
RAIN_OPERATOR_ROLE = "rainfall"

#: |applied - expected| / expected above this (in %) flags the run.
RAIN_TOLERANCE_PCT = 2.0

APPLIED_METHOD = "rate_operator_cumulative_influx"


# --- applied (engine side) ----------------------------------------------------

def tag_rain_operator(operator):
    """Mark a rate operator as rainfall; returns it."""
    try:
        setattr(operator, RAIN_OPERATOR_ROLE_ATTR, RAIN_OPERATOR_ROLE)
    except Exception:
        logger.debug("could not tag rain operator %r", operator, exc_info=True)
    return operator


def rain_operators(domain) -> list:
    """The domain's fractional-step operators tagged as rainfall."""
    ops = getattr(domain, "fractional_step_operators", None) or []
    return [op for op in ops if getattr(op, RAIN_OPERATOR_ROLE_ATTR, None) == RAIN_OPERATOR_ROLE]


def local_applied_rain_m3(domain) -> float:
    """Rain volume this rank's rain operators have applied so far (m3)."""
    return float(sum(float(getattr(op, "cumulative_influx", 0.0) or 0.0) for op in rain_operators(domain)))


def collect_applied_rain_m3(domain) -> float:
    """Applied rain summed over all MPI ranks. COLLECTIVE: call on every rank.

    Serial runs (no launcher env) never import mpi4py. Each rank's
    ``cumulative_influx`` counts only the triangles it owns, so SUM is exact.
    """
    from run_anuga.diagnostics import _mpi_world_size_from_env

    local = local_applied_rain_m3(domain)
    if _mpi_world_size_from_env() <= 1:
        return local
    try:
        from mpi4py import MPI
    except ImportError:
        logger.warning("rain check: MPI env present but mpi4py missing; applied rain is RANK-LOCAL")
        return local
    return float(MPI.COMM_WORLD.allreduce(local, op=MPI.SUM))


# --- expected (input side) ----------------------------------------------------

def _utc(ts):
    import pandas as pd

    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def rate_time_integral(data, model_start, duration_s) -> float:
    """Integral of a feature's rate over [0, duration_s] (rate unit x s).

    ``data`` is a constant, ``None`` (0) or a list of ``{timestamp, value}``.
    """
    duration_s = float(duration_s)
    if data is None:
        return 0.0
    if not isinstance(data, list):
        return float(data) * duration_s
    if not data:
        return 0.0
    t0 = _utc(model_start)
    rows = sorted(
        (((_utc(r["timestamp"]) - t0).total_seconds(), float(r["value"])) for r in data),
        key=lambda r: r[0],
    )
    starts = [r[0] for r in rows]
    total = 0.0
    for i, (begin, value) in enumerate(rows):
        if i + 1 < len(rows):
            end = starts[i + 1]
        elif len(rows) >= 2:
            end = begin + (begin - starts[i - 1])  # last block lasts the last interval
        else:
            end = math.inf  # a one-row series is a held rate
        overlap = min(end, duration_s) - max(begin, 0.0)
        if overlap > 0 and math.isfinite(value):
            total += value * overlap
    return total


def _rain_surface(boundary_polygon, holes):
    """The modelled surface rain can land on: outline minus mesh holes."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    surface = Polygon(boundary_polygon)
    if not surface.is_valid:
        surface = surface.buffer(0)
    if holes:
        surface = surface.difference(unary_union([Polygon(h).buffer(0) for h in holes if len(h) >= 3]))
    return surface


def expected_rain_volume_m3(rainfall_features, boundary_polygon, model_start, duration_s, factor,
                            holes=None) -> dict:
    """Expected rain volume from the rainfall inputs (see module docstring)."""
    from shapely.geometry import shape

    surface = _rain_surface(boundary_polygon, holes)
    total = 0.0
    area_total = 0.0
    n_used = n_skipped = 0
    for feature in rainfall_features or []:
        data = (feature.get("properties") or {}).get("data")
        if data is None:
            n_skipped += 1
            continue
        geom = shape(feature.get("geometry"))
        if not geom.is_valid:
            geom = geom.buffer(0)
        area = geom.intersection(surface).area
        total += rate_time_integral(data, model_start, duration_s) * float(factor) * area
        area_total += area
        n_used += 1
    return {
        "expected_m3": total,
        "rain_area_m2": area_total,
        "n_polygons": n_used,
        "n_polygons_skipped": n_skipped,
    }


# --- verdict --------------------------------------------------------------------

def build_rain_check(expected: dict, applied_m3: float, tolerance_pct: float = RAIN_TOLERANCE_PCT) -> dict:
    """The ``rain`` block of the run summary."""
    expected_m3 = float(expected["expected_m3"])
    applied_m3 = float(applied_m3)
    if expected_m3 > 0:
        diff_pct = 100.0 * (applied_m3 - expected_m3) / expected_m3
        ok = bool(math.isfinite(diff_pct) and abs(diff_pct) <= tolerance_pct)
        diff_pct = round(diff_pct, 4) if math.isfinite(diff_pct) else None
    else:
        diff_pct = None
        ok = bool(abs(applied_m3) < 1e-6)  # no rain asked for: none may be applied
    return {
        "expected_m3": round(expected_m3, 3),
        "applied_m3": round(applied_m3, 3),
        "diff_pct": diff_pct,
        "ok": ok,
        "tolerance_pct": tolerance_pct,
        "n_polygons": expected.get("n_polygons", 0),
        "n_polygons_skipped": expected.get("n_polygons_skipped", 0),
        "rain_area_m2": round(float(expected.get("rain_area_m2", 0.0)), 1),
        "applied_method": APPLIED_METHOD,
    }


def run_rain_check(domain, input_data, *, model_start, duration_s, factor, is_rank0, holes="auto",
                   log=None):
    """Glue for run_sim: call on EVERY rank after evolve; returns the block on
    rank 0 (``None`` elsewhere). ``holes="auto"`` uses the Reflective-building
    mesh holes ``create_anuga_mesh`` cut. Never raises: a telemetry failure
    must not kill a completed run (TASK-2033). ``log`` = the run's logger
    (run_anuga_N.log), so the verdict lands in the run log."""
    log = log or logger
    try:
        applied = collect_applied_rain_m3(domain)
    except Exception:
        log.warning("rain check: applied volume unavailable", exc_info=True)
        applied = None
    if not is_rank0:
        return None
    try:
        features = (input_data.get("rainfall") or {}).get("features") or []
        if holes == "auto":
            from run_anuga.run_utils import make_interior_holes_and_tags
            holes = make_interior_holes_and_tags(input_data)[0]
        expected = expected_rain_volume_m3(
            features, input_data["boundary_polygon"], model_start, duration_s, factor, holes=holes,
        )
        if applied is None:
            return {"expected_m3": round(expected["expected_m3"], 3), "applied_m3": None,
                    "diff_pct": None, "ok": False, "error": "applied volume unavailable"}
        rain = build_rain_check(expected, applied)
    except Exception as exc:
        log.warning("rain check failed", exc_info=True)
        return {"expected_m3": None, "applied_m3": applied, "diff_pct": None, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}
    if rain["ok"]:
        log.info("rain check OK: applied %.1f m3 vs expected %.1f m3 (%s%%)",
                 rain["applied_m3"], rain["expected_m3"], rain["diff_pct"])
    else:
        log.warning(
            "rain check WARN: applied %.1f m3 vs expected %.1f m3 (diff %s%%, tolerance %s%%)",
            rain["applied_m3"], rain["expected_m3"], rain["diff_pct"], rain["tolerance_pct"],
        )
    return rain
