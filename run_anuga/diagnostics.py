"""
Simulation diagnostics — per-yieldstep metrics collection and run summary.

Produces two output files:

``run_diagnostics_N.csv``
    Per-yieldstep timeseries — 14 columns, one row per yieldstep.
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

    implied = CFL * min_inradius_wet / last_dt

What it tells you:

* ``implied ≈ actual max_speed`` — normal CFL behaviour.
* ``implied >> actual max_speed`` — a dry/thin-water triangle with small
  inradius is constraining the timestep (mesh quality issue; consider
  ``optimise_dry_cells``).
* ``implied`` growing exponentially — numerical blowup (e.g. Manning's
  n=10 building instability forcing supercritical jets through gaps).

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
import time
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

#: Depth below which a cell is treated as dry for velocity/volume calculations.
WET_THRESHOLD = 1e-3  # 1 mm

#: Implied max speed above this (m/s) flags a run as numerically unstable.
#: Urban floods peak at ~5 m/s; 20 m/s is physically impossible for shallow water.
INSTABILITY_SPEED_THRESHOLD_MS = 20.0

#: Schema version for the run_summary JSON — bump when fields are added/removed.
SUMMARY_SCHEMA_VERSION = "1"


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
    """

    CSV_FIELDS = [
        "sim_time_s",
        "wall_time_s",
        "n_steps",
        "mean_dt_ms",
        "last_dt_ms",
        "implied_max_speed_ms",
        "wet_cells",
        "wet_fraction",
        "volume_m3",
        "max_depth_m",
        "max_speed_ms",
        "peak_speed_x",
        "peak_speed_y",
        "mem_mb",
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
        self._prev_steps = int(domain.number_of_steps)
        self._records = []

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

    def record(self, sim_time: float, wall_time_s: float, mem_mb: float = 0.0) -> dict:
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

        Returns
        -------
        dict
            The recorded metrics row (also written to the CSV).
        """
        domain = self.domain

        # --- Internal timestep stats ---
        curr_steps = int(domain.number_of_steps)
        n_steps = max(1, curr_steps - self._prev_steps)
        self._prev_steps = curr_steps
        last_dt_ms = float(domain.timestep) * 1000.0
        mean_dt_ms = self.yieldstep / n_steps * 1000.0

        # --- Flow state ---
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
            wet_fraction = n_wet / max(1, n_full)

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
            else:
                max_speed_ms = 0.0
                peak_speed_x = peak_speed_y = 0.0

            # Implied max speed from the CFL condition:
            #   dt = CFL * min_inradius_wet / max_speed
            #   => implied = CFL * min_inradius_wet / dt
            # Only meaningful when the domain is wet; at t=0 (all dry) domain.timestep≈0
            # causes a spurious divide-near-zero result, so report 0 when n_wet==0.
            if n_wet > 0:
                min_r_wet = float(radii[wet].min())
                implied_max_speed_ms = self.cfl * min_r_wet / max(domain.timestep, 1e-12)
            else:
                implied_max_speed_ms = 0.0

        except Exception as exc:
            logger.debug("Diagnostics: could not compute flow quantities: %s", exc)
            n_wet = 0
            wet_fraction = volume_m3 = max_depth_m = max_speed_ms = 0.0
            peak_speed_x = peak_speed_y = implied_max_speed_ms = 0.0

        rec = {
            "sim_time_s": round(sim_time, 1),
            "wall_time_s": round(wall_time_s, 1),
            "n_steps": n_steps,
            "mean_dt_ms": round(mean_dt_ms, 2),
            "last_dt_ms": round(last_dt_ms, 2),
            "implied_max_speed_ms": round(implied_max_speed_ms, 2),
            "wet_cells": n_wet,
            "wet_fraction": round(wet_fraction, 4),
            "volume_m3": round(volume_m3, 1),
            "max_depth_m": round(max_depth_m, 3),
            "max_speed_ms": round(max_speed_ms, 3),
            "peak_speed_x": round(peak_speed_x, 1),
            "peak_speed_y": round(peak_speed_y, 1),
            "mem_mb": round(mem_mb, 1),
        }
        self._records.append(rec)
        self._writer.writerow(rec)
        self._csv_file.flush()
        return rec

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

        # Stability
        all_implied = [r["implied_max_speed_ms"] for r in self._records]
        max_implied_ms = max(all_implied) if all_implied else 0.0
        max_implied_at = (
            self._records[all_implied.index(max_implied_ms)]["sim_time_s"]
            if all_implied
            else 0.0
        )
        collapse_ratio = (min_dt_ms / first_dt_ms) if first_dt_ms > 0 else 1.0
        stable = max_implied_ms < INSTABILITY_SPEED_THRESHOLD_MS

        # Outcome
        completed = (
            self.duration_s is not None and final_sim_s >= 0.99 * self.duration_s
        ) or (self.duration_s is None and n_yieldsteps > 0)
        if completed:
            outcome = "completed"
        elif max_implied_ms >= INSTABILITY_SPEED_THRESHOLD_MS:
            outcome = "unstable"
        else:
            outcome = "incomplete"

        # flow_algorithm — try to read from domain, fall back to 'DE0'
        flow_algo = (
            getattr(self.domain, "flow_algorithm", None)
            or getattr(self.domain, "_flow_algorithm", None)
            or "DE0"
        )

        return {
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
            },
            "stability": {
                "stable": stable,
                "max_implied_speed_ms": round(max_implied_ms, 2),
                "max_implied_speed_at_sim_s": max_implied_at,
                "timestep_collapse_ratio": round(collapse_ratio, 4),
                "instability_threshold_ms": INSTABILITY_SPEED_THRESHOLD_MS,
            },
            "environment": self._collect_environment(),
        }

    def _write_summary(self) -> None:
        """Build and write the JSON run summary."""
        finished_at = datetime.now(tz=timezone.utc)
        summary = self._build_summary(finished_at)
        with open(self._json_path, "w") as f:
            json.dump(summary, f, indent=2)
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
