"""
Simulation diagnostics — per-yieldstep metrics collection and run summary.

Produces two output files:

``run_diagnostics_N.csv``
    Per-yieldstep timeseries — one row per yieldstep (``CSV_FIELDS``).
    Useful for plotting timestep evolution and spotting instability onset
    during or after a single run.

``run_summary_N.json``
    Single-record structured summary of the complete run.
    Designed for bulk storage across many runs so you can query:
    "which scenarios stalled?", "does resolution affect throughput?",
    "how does this CPU compare to that one?"

Stability diagnostics
---------------------
The key instability signal is ``implied_max_speed_ms``:

    implied = CFL * min_inradius_wet / cfl_dt

where ``cfl_dt`` is the engine's CFL step BEFORE it is clipped to land on the
yield time (TASK-4316; recorded as ``flux_timestep_ms``). Dividing by the
clipped last step (``last_dt_ms``) made every short remainder read as a
blow-up: run 1613's "407.9 m/s" was 3.26 m / 8 ms.

What it tells you:

* ``implied ≈ actual max_speed`` — normal CFL behaviour.
* ``implied >> actual max_speed`` — a dry/thin-water triangle with small
  inradius is constraining the timestep (mesh quality issue; consider
  ``optimise_dry_cells``).
* ``implied`` growing exponentially — numerical blowup (e.g. Manning's
  n=10 building instability forcing supercritical jets through gaps).

The run summary grades it (``classify_stability``, TASK-4316): clip_artefact,
local_transient, localised_persistent, propagating, or boundary_fed (water
entering through an open boundary), from the per-record honest speed, the
timestep trend and the engine's exact volume accounting (V, boundary flux and
fractional-step integrals, recorded per yield).

Usage (inside run.py)::

    monitor = SimulationMonitor(
        domain, output_dir, batch_number, yieldstep,
        duration_s=duration,
        run_label=input_data['run_label'],
        scenario_config=input_data['scenario_config'],
    )
    for t in domain.evolve(...):
        if anuga.myid == 0:
            diag = monitor.record(t, wall_time_s=elapsed, mem_mb=mem)
            logger.info('... | %s', monitor.format_log_suffix(diag))
    if anuga.myid == 0:
        monitor.finalize()
"""

import csv
import importlib.metadata
import json
import logging
import os
import platform
import re
import time
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


# --- anuga_core sha corroboration (epic 2280 W3.2 / TASK-2287) --------------
def anuga_core_sha_from_version(version):
    """Extract the anuga_core git sha from a PEP440 local version, or ``None``.

    ANUGA's setuptools-scm version carries the installed commit as a ``+g``
    local segment, e.g. ``3.3.1.dev123+g57a64ab`` -> ``57a64ab`` (the ``+g``
    suffix IS the anuga_core sha). Used to CORROBORATE the baked manifest's
    anuga_core sha against what is actually importable in-container. Tolerant:
    a version with no ``+g`` segment (a plain release, or ``"unknown"``) or a
    non-string returns ``None``. Never raises.
    """
    if not isinstance(version, str) or "+g" not in version:
        return None
    tail = version.split("+g", 1)[1]
    match = re.match(r"[0-9a-fA-F]+", tail)
    return match.group(0) if match else None


def installed_anuga_core_sha():
    """The anuga_core sha from the INSTALLED anuga package version, or ``None``.

    Reads ``importlib.metadata.version('anuga')`` (does NOT ``import anuga`` —
    metadata is available even where the compiled fork won't import, e.g. the
    workstation) and parses its ``+g`` suffix. Never raises: any lookup failure
    (package absent / no local segment) yields ``None`` so corroboration simply
    degrades to "cannot corroborate".
    """
    try:
        return anuga_core_sha_from_version(importlib.metadata.version("anuga"))
    except Exception:
        return None


# --- run_anuga code version in the run summary (TASK-3775) ------------------
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
#: The bake tooling (deploy scripts/rebuild-batch-image.sh and
#: scripts/gpu-dev/build-gpu-batch-image.sh) stamps ``git rev-parse --short``,
#: so the manifest normally carries a 7-char abbreviation. Accept 7-40 hex.
_SHA_ABBREV = re.compile(r"^[0-9a-f]{7,40}$")


def _git_head_sha(package_dir):
    """HEAD sha of the git checkout that holds ``package_dir``, or ``None``.

    Only trusted when the checkout's top level is the package's parent (the
    run_anuga repo itself), so a run_anuga installed into site-packages that
    happens to sit inside some other repo never reports that repo's sha.
    """
    import subprocess

    try:
        top = subprocess.run(
            ["git", "-C", package_dir, "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    if top.returncode != 0:
        return None
    lines = top.stdout.split()
    if len(lines) != 2:
        return None
    toplevel, sha = lines
    if os.path.realpath(toplevel) != os.path.realpath(os.path.dirname(package_dir)):
        return None
    sha = sha.strip().lower()
    return sha if _SHA40.match(sha) else None


def resolve_run_anuga_sha(provenance_path=None, package_dir=None):
    """``(sha, source)`` of the run_anuga code running this sim.

    Order: the baked BUILD_PROVENANCE.json (``run_anuga.sha``, via
    ``_handoff.load_build_provenance``) -> ``git`` HEAD of the checkout the
    package was imported from -> ``("unknown", "unknown")``. Never raises.

    The manifest sha is recorded AS GIVEN when it is 7-40 lowercase hex: the
    bake tooling writes ``git rev-parse --short`` (e.g. ``a1374f2``), so a
    prod run reports the abbreviation. Anything else in the manifest (empty,
    ``nogit``, non-hex, shorter than 7) is ignored and the git fallback runs.
    The git fallback always reports the full 40-hex HEAD.

    CAVEAT: the manifest describes what was BAKED into the image. A dev run
    that bind-mounts different run_anuga code over a baked image still reports
    the image's baked sha, not the mounted code's. (The summary's
    ``anuga_core_sha`` is read from the installed package and is unaffected.)
    """
    try:
        from run_anuga._handoff import load_build_provenance

        prov = load_build_provenance(provenance_path)
        sha = ((prov or {}).get("run_anuga") or {}).get("sha")
        sha = str(sha).strip().lower() if sha else ""
        if _SHA_ABBREV.match(sha):
            return sha, "build_provenance"
    except Exception:
        logger.debug("run_anuga sha: build provenance unreadable", exc_info=True)
    if package_dir is None:
        package_dir = os.path.dirname(os.path.abspath(__file__))
    sha = _git_head_sha(package_dir)
    if sha:
        return sha, "git"
    return "unknown", "unknown"


#: Depth below which a cell is treated as dry for velocity/volume calculations.
WET_THRESHOLD = 1e-3  # 1 mm

#: Launcher-provided world-size env vars (mirrors run_anuga._handoff's rank
#: detection): authoritative, readable without importing mpi4py (AC2 of
#: TASK-2675 — a serial run must pay ZERO MPI import cost).
_MPI_SIZE_ENV_VARS = ("OMPI_COMM_WORLD_SIZE", "PMIX_SIZE", "PMI_SIZE")


def _mpi_world_size_from_env() -> int:
    """World size from the MPI launcher environment; 1 when not launched."""
    for var in _MPI_SIZE_ENV_VARS:
        val = os.environ.get(var)
        if val is not None and val.strip():
            try:
                return int(val.strip())
            except ValueError:
                continue
    return 1


def compute_local_flow_scalars(domain) -> dict:
    """This rank's LOCAL flow-state scalars from its sub-domain partition.

    Pure local read (no collectives): stage/elevation/momentum centroid
    values filtered to full (non-ghost) triangles. ``min_r_wet`` is ``inf``
    when the partition is dry so a global MIN reduction ignores dry ranks.
    Never raises — any failure degrades to the all-zero dict (matching the
    monitor's historical fallback).
    """
    try:
        stage = np.asarray(domain.quantities["stage"].centroid_values, dtype=float)
        elev = np.asarray(domain.quantities["elevation"].centroid_values, dtype=float)
        xmom = np.asarray(domain.quantities["xmomentum"].centroid_values, dtype=float)
        ymom = np.asarray(domain.quantities["ymomentum"].centroid_values, dtype=float)
        areas = np.asarray(domain.mesh.areas, dtype=float)
        radii = np.asarray(domain.mesh.radii, dtype=float)
        centroids = np.asarray(domain.mesh.centroid_coordinates, dtype=float)

        # Ghost-cell filter (1 = full triangle, 0 = ghost halo)
        try:
            full = np.asarray(domain.tri_full_flag, dtype=bool)
        except AttributeError:
            full = np.ones(len(stage), dtype=bool)

        depth = stage - elev
        wet = (depth > WET_THRESHOLD) & full
        n_wet = int(wet.sum())
        n_full = int(full.sum())

        volume_m3 = float(np.sum(depth[wet] * areas[wet])) if n_wet > 0 else 0.0
        max_depth_m = float(depth[full].max()) if n_full > 0 else 0.0

        if n_wet > 0:
            d_safe = np.maximum(depth[wet], WET_THRESHOLD)
            speed_wet = np.sqrt((xmom[wet] / d_safe) ** 2 + (ymom[wet] / d_safe) ** 2)
            peak_idx = int(np.argmax(speed_wet))
            max_speed_ms = float(speed_wet[peak_idx])
            wet_centroids = centroids[wet]
            peak_speed_x = float(wet_centroids[peak_idx, 0])
            peak_speed_y = float(wet_centroids[peak_idx, 1])
            min_r_wet = float(radii[wet].min())
        else:
            max_speed_ms = 0.0
            peak_speed_x = peak_speed_y = 0.0
            min_r_wet = float("inf")

        out = {
            "n_wet": n_wet,
            "n_full": n_full,
            "volume_m3": volume_m3,
            "max_depth_m": max_depth_m,
            "max_speed_ms": max_speed_ms,
            "peak_speed_x": peak_speed_x,
            "peak_speed_y": peak_speed_y,
            "min_r_wet": min_r_wet,
        }
    except Exception as exc:
        logger.debug("Diagnostics: could not compute flow quantities: %s", exc)
        out = {
            "n_wet": 0, "n_full": 0, "volume_m3": 0.0, "max_depth_m": 0.0,
            "max_speed_ms": 0.0, "peak_speed_x": 0.0, "peak_speed_y": 0.0,
            "min_r_wet": float("inf"),
        }
    out.update(_local_cfl_peak(domain))
    return out


#: The CFL-peak location fields (TASK-4316 (e)); None when not known.
CFL_PEAK_FIELDS = ("cfl_peak_x", "cfl_peak_y", "cfl_peak_depth_m", "cfl_peak_boundary_dist_m")

#: Boundary tags that let no water through (everything else is open).
CLOSED_BOUNDARY_TAGS = frozenset({"Reflective", "reflective", "ghost"})


def _cfl_location_mode(domain) -> str:
    """'gpu' (mode 2: the host max_speed is never synced, so no location),
    'none' (no max_speed array yet, e.g. the t=0 yield) or 'cpu'. Decided from
    state that is the same on every rank, because it gates a collective."""
    if getattr(domain, "multiprocessor_mode", None) == 2:
        return "gpu"
    if getattr(domain, "max_speed", None) is None:
        return "none"
    return "cpu"


def _local_cfl_peak(domain) -> dict:
    """This rank's fastest triangle by the engine's per-triangle CFL speed
    (``domain.max_speed``, the wave speed the timestep is limited by): its
    centroid, depth and distance to the nearest boundary edge midpoint.

    TASK-4316 (e). CPU path only: in GPU mode 2 the device never copies
    max_speed back (fork comment, shallow_water_domain.py:2756-2765), so the
    location is None there, never a guess (the device-to-host copy is on the
    epic 3754 YAGNI ledger). ``cfl_peak_speed`` is -inf when unknown so a MAXLOC
    reduction ignores this rank.
    """
    mode = _cfl_location_mode(domain)
    unknown = {"cfl_location_mode": mode, "cfl_peak_speed": float("-inf"),
               **{k: None for k in CFL_PEAK_FIELDS}}
    if mode != "cpu":
        return unknown
    try:
        speeds = np.asarray(domain.max_speed, dtype=float)
        try:
            full = np.asarray(domain.tri_full_flag, dtype=bool)
        except AttributeError:
            full = np.ones(len(speeds), dtype=bool)
        if not full.any():
            return unknown
        k = int(np.argmax(np.where(full, speeds, -np.inf)))
        centroids = np.asarray(domain.mesh.centroid_coordinates, dtype=float)
        stage = np.asarray(domain.quantities["stage"].centroid_values, dtype=float)
        elev = np.asarray(domain.quantities["elevation"].centroid_values, dtype=float)
        x, y = float(centroids[k, 0]), float(centroids[k, 1])
        return {
            "cfl_location_mode": mode,
            "cfl_peak_speed": float(speeds[k]),
            "cfl_peak_x": x,
            "cfl_peak_y": y,
            "cfl_peak_depth_m": float(stage[k] - elev[k]),
            "cfl_peak_boundary_dist_m": _distance_to_boundary(domain, x, y),
        }
    except Exception as exc:
        logger.debug("Diagnostics: could not locate the CFL peak: %s", exc)
        return unknown


def _distance_to_boundary(domain, x: float, y: float):
    """Distance from (x, y) to the nearest non-ghost boundary edge midpoint
    (within half an edge length of the edge itself); None if unknown."""
    edges = [key for key, tag in getattr(domain, "boundary", {}).items() if tag != "ghost"]
    if not edges:
        return None
    mids = getattr(domain.mesh, "edge_midpoint_coordinates", None)
    if mids is None:
        return None
    mids = np.asarray(mids, dtype=float)
    idx = np.array([3 * vol + edge for vol, edge in edges], dtype=int)
    return float(np.min(np.hypot(mids[idx, 0] - x, mids[idx, 1] - y)))


def allreduce_flow_scalars(local: dict) -> dict:
    """Reduce the per-rank flow scalars to GLOBAL values (TASK-2675).

    The rank-0-local trap this kills (proven on prod run 1314): the monitor
    reads only rank 0's partition, so a 32-rank run whose water lives on
    other ranks logged ``wet=0 vol=0`` for 14+ model-% while the GPU twin
    showed ~300 m3/s inflow — the status line LIED. Reductions:

    * ``n_wet`` / ``n_full`` / ``volume_m3`` — SUM
    * ``max_depth_m`` — MAX
    * ``max_speed_ms`` — MAXLOC (value, rank); the winning rank broadcasts
      its peak coordinates so vmax and its location stay a consistent pair
    * ``min_r_wet`` — MIN (dry ranks contribute ``inf``)

    COLLECTIVE: every rank must call this at the same point (run.py calls it
    at the yieldstep boundary OUTSIDE the ``myid == 0`` gate — reducing
    inside the rank-0 block would deadlock against the other ranks' evolve
    collectives). Serial path (AC2): when no MPI launcher env is present the
    input is returned unchanged and ``mpi4py`` is NEVER imported.
    """
    if _mpi_world_size_from_env() <= 1:
        return dict(local)
    try:
        from mpi4py import MPI
    except ImportError:
        logger.warning(
            "allreduce_flow_scalars: MPI launcher env present but mpi4py is "
            "not importable — status scalars stay RANK-LOCAL (dishonest on "
            "multi-rank runs)",
        )
        return dict(local)
    comm = MPI.COMM_WORLD
    out = dict(local)
    out["n_wet"] = comm.allreduce(local["n_wet"], op=MPI.SUM)
    out["n_full"] = comm.allreduce(local["n_full"], op=MPI.SUM)
    out["volume_m3"] = comm.allreduce(local["volume_m3"], op=MPI.SUM)
    out["max_depth_m"] = comm.allreduce(local["max_depth_m"], op=MPI.MAX)
    out["min_r_wet"] = comm.allreduce(local["min_r_wet"], op=MPI.MIN)
    max_speed, winner = comm.allreduce(
        (local["max_speed_ms"], comm.Get_rank()), op=MPI.MAXLOC,
    )
    out["max_speed_ms"] = max_speed
    out["peak_speed_x"], out["peak_speed_y"] = comm.bcast(
        (local["peak_speed_x"], local["peak_speed_y"]), root=winner,
    )
    # TASK-4316 (e): the fastest CFL triangle, same MAXLOC + bcast pattern.
    # The mode is the same on every rank, so all ranks take or skip this.
    if local.get("cfl_location_mode") == "cpu":
        cfl_speed, cfl_winner = comm.allreduce(
            (local["cfl_peak_speed"], comm.Get_rank()), op=MPI.MAXLOC,
        )
        out["cfl_peak_speed"] = cfl_speed
        location = comm.bcast(tuple(local[k] for k in CFL_PEAK_FIELDS), root=cfl_winner)
        out.update(zip(CFL_PEAK_FIELDS, location))
    return out


def collect_volume_accounting(domain) -> dict:
    """The engine's exact volume accounting (TASK-4316 (b)): the water volume
    V, the boundary flux integral BF (+ = net inflow since the start) and the
    fractional-step volume integral FS (rain, inflow and catchment operators),
    so V - BF - FS - V0 closes to round-off on a healthy run.

    COLLECTIVE: each getter allreduces across ranks inside anuga_core, so this
    must run on EVERY rank at the same point (it rides collect_flow_scalars,
    the TASK-2675 pattern). A getter that raises (get_boundary_flux_integral
    on a non-DE flux method) raises before its reduction on every rank alike;
    its value is None.
    """
    out = {}
    for key, getter in (
        ("water_volume_m3", "get_water_volume"),
        ("boundary_flux_integral_m3", "get_boundary_flux_integral"),
        ("fractional_step_volume_m3", "get_fractional_step_volume_integral"),
    ):
        try:
            out[key] = float(getattr(domain, getter)())
        except Exception as exc:
            logger.debug("Diagnostics: %s unavailable: %s", getter, exc)
            out[key] = None
    return out


def collect_flow_scalars(domain) -> dict:
    """Local flow scalars + global reduction + the engine's volume accounting
    — call on EVERY rank, then hand the result to ``SimulationMonitor.record``
    on rank 0 (TASK-2675, TASK-4316)."""
    out = allreduce_flow_scalars(compute_local_flow_scalars(domain))
    out.update(collect_volume_accounting(domain))
    return out


def unclipped_cfl_step_s(domain):
    """The engine's CFL-limited step BEFORE the yield clip, in seconds; None
    when the domain does not expose it (legacy mocks).

    TASK-4316 (a). generic_domain.update_timestep sets timestep = min(CFL *
    flux_timestep, evolve_max_timestep), records it in recorded_min_timestep,
    THEN clips it to land on the yield time and stores that in
    domain.timestep. flux_timestep is the step before the CFL factor.
    GPU mode 2's C step loops never write the host flux_timestep; they keep the
    CFL step (already CFL-scaled and capped, before the clip) in
    gpu_dom.recorded_flux_timestep. The Python-orchestrated GPU fallback loops
    write the host flux_timestep instead, so that is read when the device
    value is not set. MPI: both are global minima already.
    """
    if getattr(domain, "multiprocessor_mode", None) == 2:
        gpu_dom = getattr(getattr(domain, "gpu_interface", None), "gpu_dom", None)
        device_step = _positive(getattr(gpu_dom, "recorded_flux_timestep", None))
        if device_step is not None:
            return device_step
    fixed = _positive(getattr(domain, "fixed_flux_timestep", None))
    if fixed is not None:
        return fixed
    flux = _positive(getattr(domain, "flux_timestep", None))
    cfl = _positive(getattr(domain, "CFL", None))
    if flux is None or cfl is None:
        return None
    step = cfl * flux
    cap = _positive(getattr(domain, "evolve_max_timestep", None))
    return min(step, cap) if cap is not None else step


def _positive(value):
    """float(value) when it is a finite number > 0, else None."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None

#: Implied max speed above this (m/s) flags a run as numerically unstable.
#: Urban floods peak at ~5 m/s; 20 m/s is physically impossible for shallow water.
INSTABILITY_SPEED_THRESHOLD_MS = 20.0

#: Schema version for the run_summary JSON — bump when fields are added/removed.
#: "2" (TASK-4316): honest implied speed, graded stability class, volume closure.
SUMMARY_SCHEMA_VERSION = "2"

# ---------------------------------------------------------------------------
# TASK-4316 graded stability classes. Every threshold below is PROVISIONAL: an
# INFERRED proposal from the research note (docs/reports/2026-11-09-task-4103-
# instability-classification.html, section 4: TUFLOW/HEC-RAS/ARR practice),
# to be calibrated on the CSVs of the last 50 production runs. Not facts.
# ---------------------------------------------------------------------------

#: Net boundary inflow above this fraction of rain plus inflow means the model
#: is being filled from outside (gn_anuga's OPEN_BOUNDARY_NOISE, TASK-4290).
OPEN_BOUNDARY_NOISE = 0.05
#: PROVISIONAL. Clip screen: a last step under this fraction of the window's
#: own minimum CFL step was shortened to land on the yield time.
CLIP_SCREEN_RATIO = 0.5
#: PROVISIONAL. ...and the window's step count within this factor of its
#: neighbours' median (the flow did not change; only the remainder did).
CLIP_SCREEN_STEPS_RATIO = 1.5
#: PROVISIONAL. Local transient: at most this many consecutive records over
#: the threshold, and under TRANSIENT_MAX_FRACTION of the stepped records.
TRANSIENT_MAX_CONSECUTIVE = 2
TRANSIENT_MAX_FRACTION = 0.05
#: PROVISIONAL. Propagating: after onset the honest min step falls under this
#: fraction of the pre-onset median, or the step count exceeds this multiple.
COLLAPSE_RATIO_PROPAGATING = 0.1
STEPS_RATIO_PROPAGATING = 3.0
#: PROVISIONAL. Propagating: |closure| over this % of the inputs (TUFLOW's
#: "values exceeding 3% usually indicate there are significant problems").
CLOSURE_FAIL_PCT = 3.0
#: Records forming the collapse-ratio baseline (median of the first stepped).
COLLAPSE_BASELINE_RECORDS = 3
#: Inputs have ended once FS reaches this fraction of its final value.
INPUTS_END_FRACTION = 0.999

#: Most severe first: the summary's ``class`` is the first that applies.
STABILITY_CLASSES = (
    "boundary_fed", "propagating", "localised_persistent", "local_transient",
    "clip_artefact",
)


class SimulationMonitor:
    """
    Per-yieldstep metrics collection for an ANUGA domain.

    Parameters
    ----------
    domain : anuga.Domain
        The ANUGA domain (after ``distribute()``).  In parallel mode this
        is process 0's sub-domain; metrics will reflect only that partition.
    output_dir : str
        Directory for output files.
    batch_number : int
        Used to name output files (e.g. ``run_diagnostics_1.csv``).
    yieldstep : float
        Yieldstep in seconds, used to compute the mean internal timestep.
    cfl : float
        CFL number for the active flow algorithm (DE0 = 0.9, DE1/DE2 = 1.0).
    duration_s : float, optional
        Requested simulation duration in seconds.  Used to determine whether
        the run completed and to compute the sim/wall throughput ratio.
    run_label : str, optional
        Run label (e.g. ``"run_1_1_1"``).  Written to the JSON summary.
    scenario_config : dict, optional
        Scenario configuration dict from ``scenario.json``.  Fields used:
        ``name``, ``project``, ``id``, ``run_id``, ``epsg``, ``resolution``.
    boundary_tags : iterable of str, optional
        The scenario's boundary tags (TASK-4316: the open ones are named when
        the run is boundary_fed). Defaults to the domain's own tags.
    """

    CSV_FIELDS = [
        "sim_time_s",
        "wall_time_s",
        "n_steps",
        "mean_dt_ms",
        "last_dt_ms",
        "min_dt_ms",
        "implied_max_speed_ms",
        "wet_cells",
        "wet_fraction",
        "volume_m3",
        "max_depth_m",
        "max_speed_ms",
        "peak_speed_x",
        "peak_speed_y",
        "mem_mb",
        # TASK-4316: appended, so the first 15 columns keep their positions.
        "flux_timestep_ms",
        "boundary_flux_integral_m3",
        "fractional_step_volume_m3",
        "closure_m3",
        "cfl_peak_x",
        "cfl_peak_y",
        "cfl_peak_depth_m",
        "cfl_peak_boundary_dist_m",
    ]

    def __init__(
        self,
        domain,
        output_dir,
        batch_number,
        yieldstep,
        cfl=0.9,
        duration_s=None,
        run_label=None,
        scenario_config=None,
        boundary_tags=None,
    ):
        self.domain = domain
        self.output_dir = output_dir
        self.batch_number = int(batch_number)
        self.yieldstep = float(yieldstep)
        self.cfl = float(cfl)
        self.duration_s = float(duration_s) if duration_s is not None else None
        self.run_label = run_label or ""
        self.scenario_config = scenario_config or {}

        self._csv_path = os.path.join(output_dir, f"run_diagnostics_{batch_number}.csv")
        self._json_path = os.path.join(output_dir, f"run_summary_{batch_number}.json")
        self._records = []
        self._rain = None  # TASK-3775: set_rain_check() after evolve
        # TASK-4316: V0 for the closure column (set at the first full record)
        self._closure_base = None
        # The scenario's boundary tags (all ranks' view); rank 0's partition
        # is the fallback, which may miss a tag on a multi-rank run.
        if boundary_tags is None:
            boundary_tags = set(getattr(domain, "boundary", {}).values())
        self.open_boundary_tags = sorted(set(boundary_tags) - CLOSED_BOUNDARY_TAGS)

        self._start_wall_time = time.time()
        self._started_at = datetime.now(tz=timezone.utc)

        self.mesh_stats = self._compute_mesh_stats()
        logger.info("Mesh diagnostics: %s", self.mesh_stats["summary"])

        self._csv_file = open(self._csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS)
        self._csv_file.write(f"# {self.mesh_stats['summary']}\n")
        self._writer.writeheader()
        self._csv_file.flush()

    # ------------------------------------------------------------------
    # Mesh quality (one-time at startup)
    # ------------------------------------------------------------------

    def _compute_mesh_stats(self) -> dict:
        """Compute one-time mesh quality statistics from the domain mesh."""
        try:
            radii = np.asarray(self.domain.mesh.radii, dtype=float)
            centroids = np.asarray(self.domain.mesh.centroid_coordinates, dtype=float)
            worst_idx = int(np.argmin(radii))
            wx = float(centroids[worst_idx, 0])
            wy = float(centroids[worst_idx, 1])
            min_angle = self._compute_global_min_angle()
            n = len(radii)
            return {
                "n_triangles": n,
                "inradius_min": float(radii.min()),
                "inradius_p5": float(np.percentile(radii, 5)),
                "inradius_median": float(np.median(radii)),
                "min_angle_deg": min_angle,
                "worst_xy": (wx, wy),
                "summary": (
                    f"{n:,} triangles | "
                    f"inradius min={radii.min():.3f}m "
                    f"p5={np.percentile(radii, 5):.3f}m "
                    f"median={np.median(radii):.3f}m | "
                    f"min_angle={min_angle:.1f}° | "
                    f"worst at ({wx:.0f}, {wy:.0f})"
                ),
            }
        except Exception as exc:
            logger.debug("Could not compute mesh stats: %s", exc)
            n = getattr(self.domain, "number_of_triangles", 0)
            return {
                "n_triangles": n,
                "inradius_min": 0.0,
                "inradius_p5": 0.0,
                "inradius_median": 0.0,
                "min_angle_deg": 0.0,
                "worst_xy": (0.0, 0.0),
                "summary": f"{n:,} triangles | mesh quality stats unavailable",
            }

    def _compute_global_min_angle(self) -> float:
        """
        Compute the global minimum triangle angle (degrees) from vertex
        coordinates.

        ``domain.mesh.vertex_coordinates`` is a flattened (N×3, 2) array
        where rows ``[3k, 3k+1, 3k+2]`` are the three vertices of triangle k.
        """
        try:
            vc = np.asarray(self.domain.mesh.vertex_coordinates, dtype=float)
            verts = vc.reshape(-1, 3, 2)  # (N, 3, 2)
            v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]
            a = np.linalg.norm(v1 - v0, axis=1)
            b = np.linalg.norm(v2 - v1, axis=1)
            c = np.linalg.norm(v0 - v2, axis=1)
            eps = 1e-10
            cos_A = (b**2 + c**2 - a**2) / (2 * b * c + eps)
            cos_B = (a**2 + c**2 - b**2) / (2 * a * c + eps)
            cos_C = (a**2 + b**2 - c**2) / (2 * a * b + eps)
            max_cos = np.maximum(cos_A, np.maximum(cos_B, cos_C))
            min_angles = np.degrees(np.arccos(np.clip(max_cos, -1.0, 1.0)))
            return float(min_angles.min())
        except Exception as exc:
            logger.debug("Could not compute min angle: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Per-yieldstep recording
    # ------------------------------------------------------------------

    def record(self, sim_time: float, wall_time_s: float, mem_mb: float = 0.0,
               flow: dict | None = None) -> dict:
        """
        Record metrics at a yieldstep boundary.

        Call this once per yieldstep, inside ``if anuga.myid == 0``.

        Parameters
        ----------
        sim_time : float
            Current simulation time in seconds (the ``t`` from the evolve loop).
        wall_time_s : float
            Wall-clock seconds elapsed for this yieldstep.
        mem_mb : float
            Current process memory in MB (optional; pass 0 to skip).
        flow : dict, optional
            Pre-computed flow scalars from :func:`collect_flow_scalars` —
            REQUIRED for honest values on a multi-rank run (TASK-2675: the
            collective reduction must run on EVERY rank, so it cannot live
            inside this rank-0-only method). When omitted (serial callers,
            legacy tests) the rank-local computation runs here, which is
            exact for a single-process run.

        Returns
        -------
        dict
            The recorded metrics row (also written to the CSV).
        """
        domain = self.domain

        # --- Internal timestep stats ---
        # TASK-3373: the engine zeroes domain.number_of_steps after EVERY yield
        # (generic_domain.evolve), so the value read here IS this window's own
        # step count. The old cumulative delta logged steps=1 for every window
        # after the first. The t=0 yield legitimately has 0 steps; only the dt
        # divisor is floored at 1.
        n_steps = max(0, int(domain.number_of_steps))
        last_dt_ms = float(domain.timestep) * 1000.0
        mean_dt_ms = self.yieldstep / max(1, n_steps) * 1000.0
        # TASK-2622 (W1.1, epic 2618) — the smallest internal substep dt within
        # the just-completed yieldstep. ANUGA already tracks this itself
        # (generic_domain.update_timestep sets domain.recorded_min_timestep on
        # every substep; evolve() resets it to evolve_max_timestep right after
        # each yield — abstract_2d_finite_volumes/generic_domain.py:2055 — so by
        # the time this record() call runs, in the caller's `for t in
        # domain.evolve(...)` loop body, it still holds the just-finished
        # interval's minimum, not yet reset). Falls back to last_dt_ms for
        # domains/mocks that don't expose the attribute.
        min_dt_ms = float(getattr(domain, "recorded_min_timestep", domain.timestep)) * 1000.0

        # --- Flow state (global when a reduced `flow` dict is handed in) ---
        if flow is None:
            # Serial callers only: the engine getters reduce nothing here.
            flow = {**compute_local_flow_scalars(domain), **collect_volume_accounting(domain)}
        n_wet = flow["n_wet"]
        wet_fraction = n_wet / max(1, flow["n_full"])
        volume_m3 = flow["volume_m3"]
        max_depth_m = flow["max_depth_m"]
        max_speed_ms = flow["max_speed_ms"]
        peak_speed_x = flow["peak_speed_x"]
        peak_speed_y = flow["peak_speed_y"]

        # Implied max speed from the CFL condition:
        #   dt = CFL * min_inradius_wet / max_speed
        #   => implied = CFL * min_inradius_wet / dt
        # TASK-4316 (a): dt is the engine's CFL step BEFORE the yield clip
        # (unclipped_cfl_step_s); domain.timestep is the clipped remainder
        # and stays only as the fallback for domains that do not expose it.
        # Both are globally reduced by parallel ANUGA; min_r_wet arrives
        # globally reduced (MIN) in `flow`. Only meaningful when wet
        # somewhere; at t=0 (all dry) dt≈0 causes a spurious divide-near-zero
        # result, so report 0 when n_wet==0.
        cfl_dt = unclipped_cfl_step_s(domain)
        min_r_wet = flow["min_r_wet"]
        if n_wet > 0 and np.isfinite(min_r_wet):
            divisor = cfl_dt if cfl_dt is not None else domain.timestep
            implied_max_speed_ms = self.cfl * min_r_wet / max(divisor, 1e-12)
        else:
            implied_max_speed_ms = 0.0

        # TASK-4316 (b): closure = V - BF - FS - V0. V0 is fixed at the first
        # record carrying all three (V - BF - FS there), so a fresh run's
        # closure starts at 0 and so does a resumed one's (whose BF and FS
        # already hold the earlier part of the run).
        volume = flow.get("water_volume_m3")
        bf = flow.get("boundary_flux_integral_m3")
        fs = flow.get("fractional_step_volume_m3")
        closure_m3 = None
        if None not in (volume, bf, fs):
            if self._closure_base is None:
                self._closure_base = volume - bf - fs
            closure_m3 = volume - bf - fs - self._closure_base

        rec = {
            "sim_time_s": round(sim_time, 1),
            "wall_time_s": round(wall_time_s, 1),
            "n_steps": n_steps,
            "mean_dt_ms": round(mean_dt_ms, 2),
            "last_dt_ms": round(last_dt_ms, 2),
            "min_dt_ms": round(min_dt_ms, 2),
            "implied_max_speed_ms": round(implied_max_speed_ms, 2),
            "wet_cells": n_wet,
            "wet_fraction": round(wet_fraction, 4),
            "volume_m3": round(volume_m3, 1),
            "max_depth_m": round(max_depth_m, 3),
            "max_speed_ms": round(max_speed_ms, 3),
            "peak_speed_x": round(peak_speed_x, 1),
            "peak_speed_y": round(peak_speed_y, 1),
            "mem_mb": round(mem_mb, 1),
            "flux_timestep_ms": _round_or_none(
                cfl_dt * 1000.0 if cfl_dt is not None else None, 2),
            "boundary_flux_integral_m3": _round_or_none(bf, 1),
            "fractional_step_volume_m3": _round_or_none(fs, 1),
            "closure_m3": _round_or_none(closure_m3, 1),
            "cfl_peak_x": _round_or_none(flow.get("cfl_peak_x"), 1),
            "cfl_peak_y": _round_or_none(flow.get("cfl_peak_y"), 1),
            "cfl_peak_depth_m": _round_or_none(flow.get("cfl_peak_depth_m"), 3),
            "cfl_peak_boundary_dist_m": _round_or_none(flow.get("cfl_peak_boundary_dist_m"), 1),
        }
        self._records.append(rec)
        self._writer.writerow(rec)
        self._csv_file.flush()
        self._publish_observed_stats()
        return rec

    # ------------------------------------------------------------------
    # Ledger feed (TASK-3373): the physics that explains GPU cost variance
    # ------------------------------------------------------------------

    #: Max points in the per-yield series handed to the ledger (decimated).
    SERIES_MAX_POINTS = 120

    def observed_stats(self) -> dict:
        """Aggregate the records into the compact ``observed`` run-summary bag.

        Keys: n_steps, mean_dt_ms, min_dt_ms, wet_fraction_final,
        wet_fraction_max, max_depth_m, yieldstep_series
        ``[{t, cum_steps, mean_dt_ms, wet_fraction}]``. ``{}`` before the first
        record. The t=0 yield (0 steps) is excluded from the dt aggregates.
        """
        recs = self._records
        if not recs:
            return {}
        total = sum(r["n_steps"] for r in recs)
        stepped = [r for r in recs if r["n_steps"] > 0]
        mean_dt = (
            sum(r["mean_dt_ms"] * r["n_steps"] for r in stepped) / total
            if total else None
        )
        min_dt = min((r["min_dt_ms"] for r in stepped), default=None)
        series, cum = [], 0
        for r in recs:
            cum += r["n_steps"]
            series.append({
                "t": r["sim_time_s"],
                "cum_steps": cum,
                "mean_dt_ms": r["mean_dt_ms"],
                "wet_fraction": r["wet_fraction"],
            })
        if len(series) > self.SERIES_MAX_POINTS:
            stride = -(-len(series) // self.SERIES_MAX_POINTS)
            kept = series[::stride]
            if kept[-1] is not series[-1]:
                kept.append(series[-1])  # always keep the final point
            series = kept
        return {
            "n_steps": total,
            "mean_dt_ms": round(mean_dt, 3) if mean_dt is not None else None,
            "min_dt_ms": min_dt,
            "wet_fraction_final": recs[-1]["wet_fraction"],
            "wet_fraction_max": max(r["wet_fraction"] for r in recs),
            "max_depth_m": max(r["max_depth_m"] for r in recs),
            "yieldstep_series": series,
        }

    def set_rain_check(self, rain: dict | None) -> None:
        """Attach the rain mass check (run_anuga.rain_check) to the summary."""
        self._rain = rain

    def _publish_observed_stats(self) -> None:
        """Hand the aggregate to the phase tracker for the sampler; never raises."""
        try:
            from run_anuga import phase_tracker
            phase_tracker.set_run_stats(self.observed_stats())
        except Exception:
            logger.debug("observed-stats publish failed", exc_info=True)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def format_log_suffix(self, rec: dict) -> str:
        """
        Return a compact string to append to the main yieldstep log line.

        Example output::

            steps=480 dt=125ms vmax=2.10m/s v_impl=2.2m/s wet=23% vol=1234m³
        """
        return (
            f"steps={rec['n_steps']} "
            f"dt={rec['last_dt_ms']:.0f}ms "
            f"vmax={rec['max_speed_ms']:.2f}m/s "
            f"v_impl={rec['implied_max_speed_ms']:.1f}m/s "
            f"wet={rec['wet_fraction'] * 100:.0f}% "
            f"vol={rec['volume_m3']:.0f}m³"
        )

    # ------------------------------------------------------------------
    # Run summary (JSON)
    # ------------------------------------------------------------------

    def _collect_environment(self) -> dict:
        """Collect hardware and software environment metadata."""
        env = {
            "hostname": platform.node(),
            "os": f"{platform.system()}-{platform.release()}-{platform.machine()}",
            "python_version": platform.python_version(),
            "cpu_model": platform.processor() or "unknown",
            "cpu_count_logical": None,
            "cpu_count_physical": None,
            "cpu_freq_max_mhz": None,
            "total_ram_gb": None,
            "run_anuga_version": "unknown",
            "anuga_version": "unknown",
        }
        try:
            import psutil
            env["cpu_count_logical"] = psutil.cpu_count(logical=True)
            env["cpu_count_physical"] = psutil.cpu_count(logical=False)
            freq = psutil.cpu_freq()
            if freq:
                env["cpu_freq_max_mhz"] = round(freq.max, 1)
            env["total_ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            pass
        for pkg, key in [("run-anuga", "run_anuga_version"), ("anuga", "anuga_version")]:
            try:
                env[key] = importlib.metadata.version(pkg)
            except Exception:
                pass
        # anuga_core sha corroboration (TASK-2287): the '+g<sha>' suffix of
        # anuga.__version__ IS the installed anuga_core commit — surfaced here so
        # the run summary carries the installed-truth sha the container copy of
        # code_provenance is corroborated against. None when the version has no
        # local segment (a plain release / 'unknown').
        env["anuga_core_sha"] = anuga_core_sha_from_version(env["anuga_version"])
        # TASK-3775: run_anuga_version is the static package version ('1.1.0');
        # the sha says which code actually ran.
        env["run_anuga_sha"], env["run_anuga_sha_source"] = resolve_run_anuga_sha()
        return env

    def _build_summary(self, finished_at: datetime) -> dict:
        """Assemble the complete run summary dict from accumulated records."""
        cfg = self.scenario_config
        ms = self.mesh_stats

        total_wall_s = time.time() - self._start_wall_time
        final_sim_s = self._records[-1]["sim_time_s"] if self._records else 0.0
        first_dt_ms = self._records[0]["last_dt_ms"] if self._records else 0.0
        n_yieldsteps = len(self._records)

        # Performance aggregates
        total_internal_steps = sum(r["n_steps"] for r in self._records)
        all_steps = [r["n_steps"] for r in self._records]
        all_dts = [r["last_dt_ms"] for r in self._records]
        mean_dt_overall = (
            sum(r["mean_dt_ms"] * r["n_steps"] for r in self._records) / max(1, total_internal_steps)
        )
        min_dt_ms = min(all_dts) if all_dts else 0.0
        min_dt_at = (
            self._records[all_dts.index(min_dt_ms)]["sim_time_s"] if all_dts else 0.0
        )
        peak_mem_mb = max((r["mem_mb"] for r in self._records), default=0.0)
        sim_per_wall = (
            round(final_sim_s / total_wall_s, 3) if total_wall_s > 0 else 0.0
        )

        # Flow aggregates
        max_speed_ms = max((r["max_speed_ms"] for r in self._records), default=0.0)
        peak_speed_rec = max(self._records, key=lambda r: r["max_speed_ms"], default={})
        final_rec = self._records[-1] if self._records else {}

        # Stability. TASK-3840: only windows that took an internal step count
        # (the t=0 / resume yield has no step and a ~0 dt). TASK-4316: graded
        # from the HONEST per-record speed (old-schema clipped records are
        # screened), plus the volume accounting; see classify_stability.
        rain = self._rain or {}
        has_fs = any(r.get("fractional_step_volume_m3") is not None for r in self._records)
        graded = classify_stability(
            self._records,
            inputs_m3=None if has_fs else rain.get("applied_m3"),
            open_boundary_tags=self.open_boundary_tags,
        )
        max_implied_ms = graded.pop("max_implied_speed_ms")
        max_implied_at = graded.pop("max_implied_speed_at_sim_s")
        # bool() because max_implied_ms may be a numpy scalar (prod domains),
        # making this comparison an np.bool that json.dump refuses (run 1284).
        stable = bool(graded["n_records_over"] == 0)

        # Outcome
        completed = (
            self.duration_s is not None and final_sim_s >= 0.99 * self.duration_s
        ) or (self.duration_s is None and n_yieldsteps > 0)
        if completed:
            outcome = "completed"
        elif not stable:
            outcome = "unstable"
        else:
            outcome = "incomplete"

        # flow_algorithm — try to read from domain, fall back to 'DE0'
        flow_algo = (
            getattr(self.domain, "flow_algorithm", None)
            or getattr(self.domain, "_flow_algorithm", None)
            or "DE0"
        )

        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "run": {
                "run_label": self.run_label,
                "project": cfg.get("project", 0),
                "scenario": cfg.get("id", 0),
                "run_id": cfg.get("run_id", 0),
                "name": cfg.get("name", ""),
                "batch_number": self.batch_number,
                "started_at": self._started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "total_wall_time_s": round(total_wall_s, 1),
                "outcome": outcome,
            },
            "model": {
                "duration_s": self.duration_s,
                "final_sim_time_s": final_sim_s,
                "n_yieldsteps": n_yieldsteps,
                "yieldstep_s": self.yieldstep,
                "epsg": cfg.get("epsg", ""),
                "resolution_m": cfg.get("resolution"),
                "flow_algorithm": str(flow_algo),
                "cfl": self.cfl,
            },
            "mesh": {
                "n_triangles": ms["n_triangles"],
                "inradius_min_m": round(ms["inradius_min"], 4),
                "inradius_p5_m": round(ms["inradius_p5"], 4),
                "inradius_median_m": round(ms["inradius_median"], 4),
                "min_angle_deg": round(ms["min_angle_deg"], 2),
                "worst_triangle_x": round(ms["worst_xy"][0], 1),
                "worst_triangle_y": round(ms["worst_xy"][1], 1),
            },
            "performance": {
                "total_wall_time_s": round(total_wall_s, 1),
                "sim_per_wall_ratio": sim_per_wall,
                "total_internal_steps": total_internal_steps,
                "mean_steps_per_yieldstep": round(
                    total_internal_steps / max(1, n_yieldsteps), 1
                ),
                "max_steps_per_yieldstep": max(all_steps) if all_steps else 0,
                "mean_dt_ms": round(mean_dt_overall, 2),
                "first_dt_ms": round(first_dt_ms, 2),
                "min_dt_ms": round(min_dt_ms, 2),
                "min_dt_at_sim_s": min_dt_at,
                "peak_mem_mb": round(peak_mem_mb, 1),
            },
            "flow": {
                "final_wet_fraction": final_rec.get("wet_fraction", 0.0),
                "final_volume_m3": final_rec.get("volume_m3", 0.0),
                "max_depth_m": max(
                    (r["max_depth_m"] for r in self._records), default=0.0
                ),
                "max_speed_ms": round(max_speed_ms, 3),
                "peak_speed_x": peak_speed_rec.get("peak_speed_x", 0.0),
                "peak_speed_y": peak_speed_rec.get("peak_speed_y", 0.0),
                # TASK-3772: these flow maxima are sampled once per yieldstep;
                # the per-step peaks live in the *_max.tif / playback envelopes.
                "max_basis": "max over saved frames",
            },
            "stability": {
                "stable": stable,
                "max_implied_speed_ms": round(max_implied_ms, 2),
                "max_implied_speed_at_sim_s": max_implied_at,
                "timestep_collapse_ratio": round(graded.pop("timestep_collapse_ratio"), 4),
                "instability_threshold_ms": INSTABILITY_SPEED_THRESHOLD_MS,
                **graded,
            },
            "environment": self._collect_environment(),
        }
        if self._rain is not None:
            summary["rain"] = self._rain  # TASK-3775
        return summary

    def _write_summary(self) -> None:
        """Build and write the JSON run summary."""
        finished_at = datetime.now(tz=timezone.utc)
        summary = self._build_summary(finished_at)
        with open(self._json_path, "w") as f:
            json.dump(summary, f, indent=2, default=_json_default)
        logger.info("Run summary written to: %s", self._json_path)

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Write the JSON run summary, log a brief recap, and close the CSV."""
        if self._records:
            max_speed = max(r["max_speed_ms"] for r in self._records)
            min_dt = min(r["last_dt_ms"] for r in self._records)
            max_steps = max(r["n_steps"] for r in self._records)
            final_wet = self._records[-1]["wet_fraction"] * 100.0
            final_vol = self._records[-1]["volume_m3"]
            logger.info(
                "Diagnostics summary: max_speed=%.2fm/s min_dt=%.1fms "
                "max_steps/yieldstep=%d final_wet=%.0f%% final_vol=%.0fm³",
                max_speed,
                min_dt,
                max_steps,
                final_wet,
                final_vol,
            )
        self._write_summary()
        self._csv_file.close()
        logger.info("Diagnostics written to: %s", self._csv_path)


def _round_or_none(value, ndigits):
    return None if value is None else round(float(value), ndigits)


def _num(value):
    """A CSV cell or record value as float; None when empty or missing."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_diagnostics_csv(path) -> list:
    """Read a run_diagnostics_N.csv (either schema: the 15 pre-TASK-4316
    columns or the current CSV_FIELDS) into numeric record dicts, so a saved
    run can be re-graded with classify_stability. Empty cells are None."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return [{k: _num(v) for k, v in row.items()} for row in rows]


def _honest_implied(rec: dict, neighbour_steps: list) -> tuple:
    """(honest implied speed, screened as a clip?) for one stepped record.

    A record written since TASK-4316 (flux_timestep_ms present) already holds
    the speed from the unclipped step. An older record held CFL * r / last_dt;
    when its last step is far shorter than the window's own minimum CFL step
    (min_dt_ms, which the engine records before the clip) the last step was
    the yield clip, and the speed from the window's step is CFL * r / min_dt
    = recorded * last / min. It is labelled a clip artefact only when the
    step count also sits on its neighbours' (the flow did not change).
    """
    implied = _num(rec.get("implied_max_speed_ms")) or 0.0
    if _num(rec.get("flux_timestep_ms")) is not None:
        return implied, False
    last, min_dt = _num(rec.get("last_dt_ms")), _num(rec.get("min_dt_ms"))
    if not last or not min_dt or last >= CLIP_SCREEN_RATIO * min_dt:
        return implied, False
    honest = implied * last / min_dt
    steps = _num(rec.get("n_steps")) or 0.0
    steady = not neighbour_steps or steps <= CLIP_SCREEN_STEPS_RATIO * float(np.median(neighbour_steps))
    return honest, steady


def _longest_run(indices: list) -> int:
    best = run = 0
    prev = None
    for i in indices:
        run = run + 1 if prev is not None and i == prev + 1 else 1
        best = max(best, run)
        prev = i
    return best


def classify_stability(records, *, threshold=INSTABILITY_SPEED_THRESHOLD_MS,
                       inputs_m3=None, inputs_end_s=None, open_boundary_tags=None) -> dict:
    """Grade a run from its per-yield records (TASK-4316 (c)(d)).

    ``records`` are SimulationMonitor records or load_diagnostics_csv rows of
    either schema. ``inputs_m3`` (rain plus inflow volume) and
    ``inputs_end_s`` (when they stopped) are derived from the
    fractional_step_volume_m3 column when it is present; pass them for an
    older CSV (the rain check's applied_m3, the storm's end). Returns the
    graded stability fields of the run summary plus max_implied_speed_ms,
    max_implied_speed_at_sim_s and timestep_collapse_ratio.

    Classes (STABILITY_CLASSES, most severe first; thresholds PROVISIONAL):

    * clip_artefact — only yield-clip spikes, no honest record over threshold.
    * local_transient — honest speed over threshold for at most
      TRANSIENT_MAX_CONSECUTIVE consecutive records, under
      TRANSIENT_MAX_FRACTION of the stepped records.
    * localised_persistent — over for longer, with no collapse.
    * propagating — after onset the honest step collapses (or the step count
      explodes) against the pre-onset median, or |closure| > CLOSURE_FAIL_PCT.
    * boundary_fed — net boundary inflow above OPEN_BOUNDARY_NOISE of the
      inputs AND the volume rises by more than that after the inputs end.
      From the boundary_flux_integral_m3 column when recorded, otherwise by
      difference (V_end - V_start - inputs). Not graded with no inputs (a
      model driven only by its boundary is meant to be fed by it).

    ``class`` is the most severe that applies, or None; ``speed_class`` is
    the numerical class alone. ``onset_sim_s`` belongs to ``class``.
    """
    records = list(records)
    stepped = [r for r in records if (_num(r.get("n_steps")) or 0) > 0]
    steps = [_num(r.get("n_steps")) or 0.0 for r in stepped]

    honest, artefacts = [], []
    for i, rec in enumerate(stepped):
        neighbours = steps[max(0, i - 2):i] + steps[i + 1:i + 3]
        value, is_clip = _honest_implied(rec, neighbours)
        honest.append(value)
        if is_clip and (_num(rec.get("implied_max_speed_ms")) or 0.0) >= threshold > value:
            artefacts.append(i)
    over = [i for i, v in enumerate(honest) if v >= threshold]
    peak = int(np.argmax(honest)) if honest else None

    # TASK-4024: the collapse ratio's baseline is the median honest min step
    # of the first WET stepped records, never the t=0 record (dt 0 -> ratio
    # 1.0) nor a dry window: a dry domain still steps, with min_dt_ms at the
    # engine's "no CFL step yet" value (1,000 s on runs 1608 and 1613). The
    # ratio is the smallest later step over that baseline, so the wetting
    # transient inside the baseline window (25 ms on run 1613) is not read as
    # a collapse: on the real CSVs the stepped-only rule gave 6.9e-3 and 1.9e-5.
    min_dts = [_num(r.get("min_dt_ms")) or 0.0 for r in stepped]
    wet = [i for i, r in enumerate(stepped) if (_num(r.get("wet_cells")) is None
                                                or _num(r.get("wet_cells")) > 0)]
    base_idx = wet[:COLLAPSE_BASELINE_RECORDS]
    baseline = float(np.median([min_dts[i] for i in base_idx])) if base_idx else 0.0
    later = [min_dts[i] for i in wet[COLLAPSE_BASELINE_RECORDS:]] or [min_dts[i] for i in base_idx]
    collapse_ratio = min(later) / baseline if baseline > 0 else 1.0

    # --- volume accounting ---------------------------------------------
    def series(col):
        vals = [(r, _num(r.get(col))) for r in records]
        return [(r, v) for r, v in vals if v is not None]

    fs = series("fractional_step_volume_m3")
    if inputs_m3 is None and len(fs) >= 2:
        inputs_m3 = fs[-1][1] - fs[0][1]
    if inputs_end_s is None and len(fs) >= 2 and fs[-1][1] - fs[0][1] > 0:
        target = fs[0][1] + INPUTS_END_FRACTION * (fs[-1][1] - fs[0][1])
        inputs_end_s = next(_num(r["sim_time_s"]) for r, v in fs if v >= target)
    closure = series("closure_m3")
    closure_pct = (
        round(100.0 * max(abs(v) for _, v in closure) / inputs_m3, 3)
        if closure and inputs_m3 else None
    )
    bf = series("boundary_flux_integral_m3")
    vol = series("volume_m3")
    if len(bf) >= 2:
        net_in, basis = bf[-1][1] - bf[0][1], "boundary_flux_integral"
    elif len(vol) >= 2 and inputs_m3 is not None:
        net_in, basis = vol[-1][1] - vol[0][1] - inputs_m3, "by_difference"
    else:
        net_in, basis = None, None

    boundary_onset = None
    if net_in is not None and inputs_m3 and inputs_m3 > 0 and inputs_end_s is not None:
        noise = OPEN_BOUNDARY_NOISE * inputs_m3
        before = [v for r, v in vol if _num(r.get("sim_time_s")) <= inputs_end_s]
        if net_in > noise and before and vol[-1][1] - before[-1] > noise:
            boundary_onset = next(
                _num(r.get("sim_time_s")) for r, v in vol
                if _num(r.get("sim_time_s")) > inputs_end_s and v - before[-1] > noise
            )

    # --- numerical class ------------------------------------------------
    speed_class, speed_onset = None, None
    if over:
        first = over[0]
        speed_onset = _num(stepped[first].get("sim_time_s"))
        pre = [i for i in wet if i < first] or base_idx or [first]
        pre_dt = [min_dts[i] for i in pre]
        pre_steps = [steps[i] for i in pre]
        dt_base, steps_base = float(np.median(pre_dt)), float(np.median(pre_steps))
        collapsed = any(
            (dt_base > 0 and min_dts[i] < COLLAPSE_RATIO_PROPAGATING * dt_base)
            or (steps_base > 0 and steps[i] > STEPS_RATIO_PROPAGATING * steps_base)
            for i in range(first, len(stepped))
        )
        if collapsed or (closure_pct is not None and closure_pct > CLOSURE_FAIL_PCT):
            speed_class = "propagating"
        elif (_longest_run(over) <= TRANSIENT_MAX_CONSECUTIVE
              and len(over) < TRANSIENT_MAX_FRACTION * len(stepped)):
            speed_class = "local_transient"
        else:
            speed_class = "localised_persistent"
    elif artefacts:
        speed_class = "clip_artefact"
        speed_onset = _num(stepped[artefacts[0]].get("sim_time_s"))

    if boundary_onset is not None:
        klass, onset = "boundary_fed", boundary_onset
    else:
        klass, onset = speed_class, speed_onset
    location = None
    if peak is not None:
        loc = {k: _num(stepped[peak].get(k)) for k in CFL_PEAK_FIELDS}
        if None not in loc.values():
            location = {
                "x": loc["cfl_peak_x"], "y": loc["cfl_peak_y"],
                "depth_m": loc["cfl_peak_depth_m"],
                "boundary_distance_m": loc["cfl_peak_boundary_dist_m"],
                "at_sim_s": _num(stepped[peak].get("sim_time_s")),
            }
    return {
        "max_implied_speed_ms": honest[peak] if peak is not None else 0.0,
        "max_implied_speed_at_sim_s": _num(stepped[peak].get("sim_time_s")) if peak is not None else 0.0,
        "timestep_collapse_ratio": collapse_ratio,
        "class": klass,
        "speed_class": speed_class,
        "onset_sim_s": onset,
        "n_records_over": len(over),
        "n_clip_artefacts": len(artefacts),
        "closure_pct": closure_pct,
        "boundary_net_in_m3": round(net_in, 1) if net_in is not None else None,
        "boundary_basis": basis,
        "boundary_tag": ",".join(sorted(open_boundary_tags)) if klass == "boundary_fed" and open_boundary_tags else None,
        "location": location,
        "thresholds_provisional": True,
    }


def _json_default(o):
    """Coerce numpy scalars (np.bool/np.integer/np.floating expose .item())
    so the run summary can never crash on domain-derived values (TASK-2033,
    run 1284: np.bool in `stability` killed the container post-evolve)."""
    item = getattr(o, "item", None)
    if callable(item):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)


def finalize_monitor_safely(monitor) -> None:
    """finalize() writes telemetry only — it must never kill a completed run
    (TASK-2033, run 1284: a summary-serialization TypeError after a 100%
    evolve exited the container before the handoff could upload anything)."""
    if monitor is None:
        return
    try:
        monitor.finalize()
    except Exception:
        logger.warning(
            "diagnostics finalize failed; continuing to handoff", exc_info=True
        )
