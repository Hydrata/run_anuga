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


#: The .rpt "Flow Routing Continuity" lines (label prefix) that add to the inflow / outflow.
_RPT_INFLOWS = ("Dry Weather Inflow", "Wet Weather Inflow", "Groundwater Inflow", "RDII Inflow", "External Inflow")
_RPT_OUTFLOWS = ("External Outflow", "Flooding Loss", "Evaporation Loss", "Exfiltration Loss")
#: m3 per unit of the block's second volume column (CMS models write 10^6 ltr).
_RPT_VOLUME_UNITS = {"10^6 ltr": 1000.0, "10^6 gal": 3785.411784}
#: Shared by every note: the coupler moves exactly what the losing exchange region gives up
#: (anuga_drainage coupler ``_exchange``), so SWMM's barrel storage never enters ANUGA's water.
_EXACT = ("ANUGA moved exactly the water a culvert took in at one end out at its other end, so the "
          "flood model's water balance does not depend on this figure.")


#: The ``swmm_continuity`` key the D9 "< 1 %" continuity check reads (TASK-3741, operator ruling
#: D36 on gate 637: continuity is judged by mass, never by SWMM's raw %).
GATE_FIGURE = "mass_error_pct"


def _continuity_note(error, inflow, stored, share, mass=None, losses=None, residual=None, entered=None):
    """The plain-English ``swmm_continuity.note`` (TASK-3717; TASK-3741 leads with the mass figure)."""
    figure = "SWMM's own continuity report on the culvert barrels"
    if abs(error) < constants.SWMM_CONTINUITY_OK_PCT:
        raw = f"{figure} is within 1% ({error:.1f}%). {_EXACT}"
    elif error < 0 and share is not None and share >= -error:
        raw = (
            f"{figure} reads {error:.1f}%. This is not water lost or created in the flood model: {_EXACT} "
            f"When the run stopped the barrels still held {stored:.0f} m3 of water, {share:.1f}% of the "
            f"{inflow:.0f} m3 that entered them, most of it standing in the outlet end of a barrel at the "
            f"outlet's water level. SWMM counts that water as stored without an inflow that delivered it, "
            f"so it shows as a continuity error; it leaves again the same way as the outlet water level falls."
        )
    else:
        raw = f"{figure} reads {error:.1f}%. It checks SWMM's barrel model only: {_EXACT}"
    if mass is None:
        return raw
    if abs(mass) < constants.SWMM_CONTINUITY_OK_PCT:
        lead = (
            f"By mass the culvert barrels balance to within 1% ({mass:.2f}%), and the continuity check uses "
            f"this figure: water in, less water out, less the change in the water the barrels hold, counting "
            f"the water standing in a barrel's outlet end as water that came in through the outlet."
        )
    else:
        way = "more water entered them than left or stayed in them" if mass > 0 else (
            "more water left or stayed in them than entered")
        size = f" The imbalance is {abs(residual):.3g} m3 of the {entered:.3g} m3 that entered." if (
            residual is not None and entered) else ""
        lost = f" SWMM recorded {losses:.0f} m3 leaving the barrels other than through their ends." if losses else ""
        lead = (
            f"By mass the culvert barrels do not balance: {mass:.1f}% ({way}), outside the 1% continuity "
            f"check, which uses this figure.{size}{lost}"
        )
    return f"{lead} {raw}"


def swmm_continuity(rpt_path, mass_balance=None):
    """SWMM's flow-routing continuity for the culvert barrels, from the run's .rpt (TASK-3717),
    with the engine's mass-based figure (TASK-3741).

    Returns ``{error_pct, inflow_m3, outflow_m3, initial_stored_m3, final_stored_m3,
    stored_share_pct, mass_error_pct, outlet_end_stored_m3, gate_figure, note}`` or None (no .rpt
    or no "Flow Routing Continuity" block). The volumes are None when the block's units are
    unknown. ``stored_share_pct`` is the barrels' storage gain over the run as a share of the
    inflow. The engine imposes the 2D stages on two fixed-stage outfalls per culvert, and SWMM
    books the water the imposed outlet stage stands in the barrel's outlet end as storage with no
    inflow, so a run that stops with a risen outlet pond reads a negative error about that size
    (run 1590: -22.2 %, 25.1 % stored).

    ``mass_balance``: the fork engine's ledger (anuga_drainage ``SwmmHeadwallEngine.mass_balance``,
    ``swmm_engine.mass_continuity``): ``mass_error_pct`` = 100 (V_in + (W_1 - W_0) - V_out -
    (S_1 - S_0)) / (V_in + S_0), i.e. SWMM's own volumes with the water standing in the conduits'
    outfall ends (W, geometry at SWMM's node depths) counted as having come in through the outfall,
    and every loss SWMM books (flooding, evaporation, seepage) counted as an error. That is the
    figure the D9 "< 1 %" check reads (``gate_figure``); run 1590 replayed: +0.01 % against SWMM's
    -22.2 %. ``outlet_end_stored_m3`` is the outfall-end water gained over the run (W_1 - W_0).
    Without a ledger (an engine before TASK-3741) both are None.
    """
    if not rpt_path:
        return None
    try:
        with open(rpt_path) as f:
            text = f.read()
    except OSError:
        return None
    parts = text.split("Flow Routing Continuity", 1)
    if len(parts) < 2:
        return None
    header, _, body = parts[1].partition("\n")
    scale = next((m3 for unit, m3 in _RPT_VOLUME_UNITS.items() if unit in header), None)
    values, error = {}, None
    try:  # a garbled .rpt must not fail a finished run's outputs: no figure, no note
        for line in body.splitlines():
            label = line.strip().split(" ..", 1)[0].strip()
            if label == "Continuity Error (%)":
                error = float(line.split()[-1])
                break
            if " .." in line:
                values[label] = float(line.split()[-1])
    except (ValueError, IndexError):
        return None
    if error is None:
        return None
    volume = (lambda *labels: sum(values.get(k, 0.0) for k in labels) * scale) if scale else (lambda *labels: None)
    inflow, outflow = volume(*_RPT_INFLOWS), volume(*_RPT_OUTFLOWS)
    initial, final = volume("Initial Stored Volume"), volume("Final Stored Volume")
    share = 100.0 * (final - initial) / inflow if scale and inflow > 0 else None
    mass, outlet_end, losses, residual, entered = _mass_fields(mass_balance)
    return {
        "error_pct": error,
        "inflow_m3": _number(inflow, 1) if scale else None,
        "outflow_m3": _number(outflow, 1) if scale else None,
        "initial_stored_m3": _number(initial, 1) if scale else None,
        "final_stored_m3": _number(final, 1) if scale else None,
        "stored_share_pct": _number(share, 1) if share is not None else None,
        "mass_error_pct": mass,
        "mass_residual_m3": residual,
        "outlet_end_stored_m3": outlet_end,
        "gate_figure": GATE_FIGURE,
        "note": _continuity_note(error, inflow, final - initial if scale else None, share, mass, losses,
                                 residual, entered),
    }


def _mass_fields(mass_balance):
    """(mass_error_pct, outlet-end gain m3, booked losses m3, mass residual m3, denominator m3)
    from the engine's ledger, rounded; Nones when there is no ledger or no figure (nothing
    entered the barrels). The residual is mass_error_pct x denominator / 100, i.e. the m3
    behind the %, so a large % of a tiny inflow reads as the small volume it is."""
    nothing = (None,) * 5
    try:
        mass = mass_balance["mass_error_pct"]
        if mass is None:
            return nothing
        gain = mass_balance["final_outfall_end_m3"] - mass_balance["initial_outfall_end_m3"]
        den = float(mass_balance["denominator_m3"])
        return (_number(mass, 3), _number(gain, 1), _number(mass_balance.get("losses_m3", 0.0), 1),
                _number(mass * den / 100.0, 3), _number(den, 3))
    except (TypeError, KeyError, ValueError):  # no ledger / an unexpected shape: no figure, no claim
        return nothing


def write_culvert_outputs(output_directory, results, coupler, log=None) -> str:
    """Finalise the coupler, write the time series and rewrite metrics.json with the hydraulic keys.

    ``results`` and ``coupler.culverts`` are in the same order. Returns the metrics path.
    """
    from anuga_drainage.culverts.inlet_control import CONTROL_NAMES

    engine_summary = dict(coupler.finalize() or {})
    # TASK-3717: the bare swmm_routing_error_pct stays (unchanged key); swmm_continuity carries
    # the numbers behind it and the plain-English note. Read after finalize (SWMM writes the .rpt).
    # TASK-3741: the fork's mass ledger folds into swmm_continuity; its figure is ALSO top level
    # (swmm_mass_error_pct, the D9 gate figure), never the raw ledger dict.
    mass_balance = engine_summary.pop("swmm_mass_balance", None)
    continuity = swmm_continuity(getattr(getattr(coupler, "engine_swmm", None), "rpt_path", None), mass_balance)
    if "swmm_mass_error_pct" in engine_summary or continuity is not None:
        engine_summary["swmm_mass_error_pct"] = _mass_fields(mass_balance)[0]
    if continuity is not None:
        engine_summary["swmm_continuity"] = continuity
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
        scalars = ", ".join(f"{k}={v}" for k, v in engine_summary.items() if not isinstance(v, dict))
        log.info("culvert results written: %s (%s)", path, scalars)
        if continuity is not None:
            log.info("culvert SWMM continuity: %s", continuity["note"])
    return path
