"""Tests for run_anuga.diagnostics.SimulationMonitor.

No ANUGA installation required — a lightweight mock domain is used.
"""

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from run_anuga.diagnostics import (
    INSTABILITY_SPEED_THRESHOLD_MS,
    SUMMARY_SCHEMA_VERSION,
    SimulationMonitor,
)

# ---------------------------------------------------------------------------
# Helpers — minimal mock of the ANUGA domain object
# ---------------------------------------------------------------------------

def _make_equilateral_triangle(cx, cy, side=2.0):
    """Return (N*3, 2) vertex_coordinates for one equilateral triangle."""
    h = side * math.sqrt(3) / 2
    return np.array([
        [cx - side / 2, cy - h / 3],
        [cx + side / 2, cy - h / 3],
        [cx,            cy + 2 * h / 3],
    ], dtype=float)


def _make_mock_domain(n=4, timestep=0.125, number_of_steps=480):
    """
    Build a minimal mock domain with n triangles arranged in a row.

    Triangle layout (2m equilateral triangles):
      - triangles 0,1,2 are wet (depth 1.0, 0.5, 0.2 m)
      - triangle 3 is dry (depth -0.1 m, below elev)
    """
    # Vertex coordinates: (n*3, 2)
    all_verts = np.vstack([
        _make_equilateral_triangle(i * 2 + 1, 0.0) for i in range(n)
    ])

    # Centroid coordinates: (n, 2)
    centroids = np.array([[i * 2 + 1, 0.0] for i in range(n)], dtype=float)

    # Equilateral triangle with side=2: area = sqrt(3), inradius = sqrt(3)/3
    area = math.sqrt(3)
    inradius = math.sqrt(3) / 3  # ≈ 0.577 m

    areas = np.full(n, area, dtype=float)
    radii = np.full(n, inradius, dtype=float)

    # Elevation: all 0 except last triangle raised above water
    elev = np.array([0.0, 0.0, 0.0, 5.0], dtype=float)[:n]
    stage = np.array([1.0, 0.5, 0.2, 4.9], dtype=float)[:n]  # last is dry

    # Momentum: non-zero only in wet cells
    xmom = np.array([0.5, 0.2, 0.05, 0.0], dtype=float)[:n]
    ymom = np.array([0.3, 0.1, 0.02, 0.0], dtype=float)[:n]

    mesh = SimpleNamespace(
        radii=radii,
        areas=areas,
        centroid_coordinates=centroids,
        vertex_coordinates=all_verts,
    )

    quantities = {
        "stage":      SimpleNamespace(centroid_values=stage),
        "elevation":  SimpleNamespace(centroid_values=elev),
        "xmomentum":  SimpleNamespace(centroid_values=xmom),
        "ymomentum":  SimpleNamespace(centroid_values=ymom),
    }

    domain = SimpleNamespace(
        mesh=mesh,
        quantities=quantities,
        timestep=timestep,
        number_of_steps=number_of_steps,
        # No tri_full_flag — monitor must fall back to all-full
    )
    return domain


# ---------------------------------------------------------------------------
# Tests: mesh stats
# ---------------------------------------------------------------------------

class TestMeshStats:
    def test_n_triangles(self, tmp_path):
        domain = _make_mock_domain(n=4)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        assert mon.mesh_stats["n_triangles"] == 4

    def test_inradius_min(self, tmp_path):
        domain = _make_mock_domain(n=4)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        expected = math.sqrt(3) / 3
        assert abs(mon.mesh_stats["inradius_min"] - expected) < 1e-6

    def test_inradius_median(self, tmp_path):
        domain = _make_mock_domain(n=4)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        # All triangles identical → median == min
        assert abs(mon.mesh_stats["inradius_median"] - mon.mesh_stats["inradius_min"]) < 1e-6

    def test_min_angle_equilateral(self, tmp_path):
        domain = _make_mock_domain(n=4)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        # Equilateral triangle → all angles = 60°
        assert abs(mon.mesh_stats["min_angle_deg"] - 60.0) < 0.1

    def test_summary_contains_key_info(self, tmp_path):
        domain = _make_mock_domain(n=4)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        s = mon.mesh_stats["summary"]
        assert "4 triangles" in s
        assert "inradius" in s
        assert "min_angle" in s

    def test_csv_header_comment(self, tmp_path):
        domain = _make_mock_domain(n=4)
        SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        csv_path = tmp_path / "run_diagnostics_1.csv"
        first_line = csv_path.read_text().splitlines()[0]
        assert first_line.startswith("#")
        assert "triangles" in first_line


# ---------------------------------------------------------------------------
# Tests: record()
# ---------------------------------------------------------------------------

class TestRecord:
    def _make_monitor(self, tmp_path, timestep=0.125, initial_steps=0):
        domain = _make_mock_domain(timestep=timestep, number_of_steps=initial_steps)
        return SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60, cfl=0.9)

    def test_returns_dict_with_all_fields(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480  # simulate 480 steps in first yieldstep
        rec = mon.record(60.0, wall_time_s=16.0, mem_mb=512.0)
        for field in SimulationMonitor.CSV_FIELDS:
            assert field in rec, f"Missing field: {field}"

    def test_sim_time(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        assert rec["sim_time_s"] == 60.0

    def test_wall_time(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.3)
        assert rec["wall_time_s"] == 16.3

    def test_n_steps(self, tmp_path):
        mon = self._make_monitor(tmp_path, initial_steps=0)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        assert rec["n_steps"] == 480

    def test_n_steps_is_per_window_not_cumulative(self, tmp_path):
        # TASK-3373: the engine zeroes number_of_steps after every yield, so
        # each window's reading IS its own count (no delta against the prior).
        mon = self._make_monitor(tmp_path, initial_steps=0)
        mon.domain.number_of_steps = 480
        mon.record(60.0, wall_time_s=16.0)
        mon.domain.number_of_steps = 470  # engine reset to 0, then 470 steps
        rec2 = mon.record(120.0, wall_time_s=15.5)
        assert rec2["n_steps"] == 470

    @pytest.mark.requires_anuga
    def test_n_steps_sums_across_yields(self, tmp_path):
        """Oracle on a REAL anuga domain: the monitor's summed per-window
        n_steps over 3 yields == an INDEPENDENT count of the engine's steps
        (apply_fractional_steps runs exactly once per evolve step)."""
        import anuga

        d = anuga.rectangular_cross_domain(8, 8, len1=80.0, len2=80.0)
        d.set_quantity("elevation", lambda x, y: -x / 100.0)
        d.set_quantity("friction", 0.0)
        d.set_quantity("stage", lambda x, y: 0.5 * (x < 20))
        refl = anuga.Reflective_boundary(d)
        d.set_boundary({"left": refl, "right": refl, "top": refl, "bottom": refl})
        d.set_name("oracle_diag")
        d.set_datadir(str(tmp_path))

        engine_steps = {"n": 0}
        orig = d.apply_fractional_steps

        def _counted():
            engine_steps["n"] += 1
            return orig()

        d.apply_fractional_steps = _counted

        mon = SimulationMonitor(d, str(tmp_path), 1, yieldstep=2.0)
        yields = 0
        for t in d.evolve(yieldstep=2.0, finaltime=6.0):
            mon.record(t, wall_time_s=0.1)
            yields += 1
        summed = sum(r["n_steps"] for r in mon._records)
        assert yields == 4  # t=0 plus 3 windows
        assert engine_steps["n"] > 3
        assert summed == engine_steps["n"]

    def test_last_dt_ms(self, tmp_path):
        mon = self._make_monitor(tmp_path, timestep=0.125)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        assert abs(rec["last_dt_ms"] - 125.0) < 0.01

    def test_mean_dt_ms(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        # yieldstep=60s, n_steps=480 → mean_dt = 60/480*1000 = 125ms
        assert abs(rec["mean_dt_ms"] - 125.0) < 0.1

    def test_min_dt_ms_reads_domain_recorded_min_timestep(self, tmp_path):
        """TASK-2622 — min_dt_ms reads ANUGA's own domain.recorded_min_timestep
        (reset per-yieldstep by generic_domain.evolve(), abstract_2d_finite_volumes/
        generic_domain.py:2055), NOT last_dt_ms — the smallest internal substep dt
        seen during the just-completed yieldstep, for the playback-store dt_source.
        """
        mon = self._make_monitor(tmp_path, timestep=0.125)
        mon.domain.number_of_steps = 480
        mon.domain.recorded_min_timestep = 0.040  # 40ms, smaller than last_dt_ms=125ms
        rec = mon.record(60.0, wall_time_s=16.0)
        assert abs(rec["min_dt_ms"] - 40.0) < 0.01

    def test_min_dt_ms_falls_back_to_last_dt_ms_when_absent(self, tmp_path):
        """Domains/mocks without recorded_min_timestep (e.g. a stripped test
        double) degrade to last_dt_ms rather than raising."""
        mon = self._make_monitor(tmp_path, timestep=0.125)
        mon.domain.number_of_steps = 480
        assert not hasattr(mon.domain, "recorded_min_timestep")
        rec = mon.record(60.0, wall_time_s=16.0)
        assert abs(rec["min_dt_ms"] - 125.0) < 0.01

    def test_wet_cells_count(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        # 3 of 4 cells are wet (triangle 3 has depth -0.1 → below elev)
        assert rec["wet_cells"] == 3

    def test_wet_fraction(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        assert abs(rec["wet_fraction"] - 0.75) < 1e-3

    def test_volume_positive(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        assert rec["volume_m3"] > 0.0

    def test_max_depth_m(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        # Max depth = 1.0m (triangle 0: stage=1.0, elev=0.0)
        # But triangle 3 has depth = 4.9 - 5.0 = -0.1 → dry, so max is over full triangles
        # depth array: [1.0, 0.5, 0.2, -0.1], max = 1.0
        assert abs(rec["max_depth_m"] - 1.0) < 1e-3

    def test_max_speed_positive_when_wet(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        assert rec["max_speed_ms"] > 0.0

    def test_peak_speed_location_in_domain(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        # Peak speed should be in one of the wet triangle centroids
        assert rec["peak_speed_x"] > 0.0

    def test_implied_max_speed_positive(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        assert rec["implied_max_speed_ms"] > 0.0

    def test_implied_max_speed_formula(self, tmp_path):
        # implied = CFL * min_inradius_wet / dt
        inradius = math.sqrt(3) / 3  # all triangles same
        dt = 0.125
        cfl = 0.9
        expected = cfl * inradius / dt
        mon = self._make_monitor(tmp_path, timestep=dt)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        assert abs(rec["implied_max_speed_ms"] - expected) < 0.1

    def test_mem_mb_recorded(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0, mem_mb=1024.5)
        assert abs(rec["mem_mb"] - 1024.5) < 0.1

    def test_appends_to_records(self, tmp_path):
        mon = self._make_monitor(tmp_path)
        mon.domain.number_of_steps = 480
        mon.record(60.0, wall_time_s=16.0)
        mon.domain.number_of_steps = 960
        mon.record(120.0, wall_time_s=15.5)
        assert len(mon._records) == 2


# ---------------------------------------------------------------------------
# Tests: CSV output
# ---------------------------------------------------------------------------

class TestCSVOutput:
    def test_csv_has_all_columns(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        domain.number_of_steps = 480
        mon.record(60.0, wall_time_s=16.0)
        mon.finalize()

        csv_path = tmp_path / "run_diagnostics_1.csv"
        with csv_path.open() as f:
            lines = f.readlines()
        # Skip comment header
        reader = csv.DictReader(line for line in lines if not line.startswith("#"))
        rows = list(reader)
        assert len(rows) == 1
        for field in SimulationMonitor.CSV_FIELDS:
            assert field in rows[0]

    def test_csv_multiple_rows(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        for step in range(1, 4):
            domain.number_of_steps = step * 480
            mon.record(step * 60.0, wall_time_s=16.0)
        mon.finalize()

        csv_path = tmp_path / "run_diagnostics_1.csv"
        with csv_path.open() as f:
            lines = [line for line in f.readlines() if not line.startswith("#")]
        reader = csv.DictReader(lines)
        rows = list(reader)
        assert len(rows) == 3

    def test_csv_batch_number_in_filename(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 7, yieldstep=60)
        mon.finalize()
        assert (tmp_path / "run_diagnostics_7.csv").exists()

    def test_csv_values_are_numeric(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        domain.number_of_steps = 480
        mon.record(60.0, wall_time_s=16.0, mem_mb=512.0)
        mon.finalize()

        csv_path = tmp_path / "run_diagnostics_1.csv"
        with csv_path.open() as f:
            lines = [line for line in f.readlines() if not line.startswith("#")]
        reader = csv.DictReader(lines)
        row = next(reader)
        # All values should parse as float without raising. TASK-4316's
        # columns are empty when the engine cannot supply them (this mock has
        # no flux_timestep, volume getters or max_speed; a GPU run has no
        # CFL-peak location), never a made-up number.
        nullable = set(SimulationMonitor.CSV_FIELDS[15:])
        for key, val in row.items():
            if key in nullable and val == "":
                continue
            float(val)


# ---------------------------------------------------------------------------
# Tests: format_log_suffix()
# ---------------------------------------------------------------------------

class TestFormatLogSuffix:
    def _make_rec(self, **overrides):
        rec = {
            "sim_time_s": 60.0,
            "wall_time_s": 16.0,
            "n_steps": 480,
            "mean_dt_ms": 125.0,
            "last_dt_ms": 125.0,
            "implied_max_speed_ms": 4.15,
            "wet_cells": 75000,
            "wet_fraction": 0.75,
            "volume_m3": 12345.6,
            "max_depth_m": 1.234,
            "max_speed_ms": 2.10,
            "peak_speed_x": 382400.0,
            "peak_speed_y": 6354400.0,
            "mem_mb": 512.0,
        }
        rec.update(overrides)
        return rec

    def test_contains_steps(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        suffix = mon.format_log_suffix(self._make_rec())
        assert "steps=480" in suffix

    def test_contains_dt(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        suffix = mon.format_log_suffix(self._make_rec())
        assert "dt=125ms" in suffix

    def test_contains_vmax(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        suffix = mon.format_log_suffix(self._make_rec())
        assert "vmax=2.10m/s" in suffix

    def test_contains_implied(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        suffix = mon.format_log_suffix(self._make_rec())
        assert "v_impl=" in suffix

    def test_contains_wet_percent(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        suffix = mon.format_log_suffix(self._make_rec(wet_fraction=0.75))
        assert "wet=75%" in suffix

    def test_contains_volume(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        suffix = mon.format_log_suffix(self._make_rec(volume_m3=12345.6))
        assert "vol=12346m³" in suffix


# ---------------------------------------------------------------------------
# Tests: finalize()
# ---------------------------------------------------------------------------

class TestFinalize:
    def test_closes_csv_file(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        domain.number_of_steps = 480
        mon.record(60.0, wall_time_s=16.0)
        mon.finalize()
        assert mon._csv_file.closed

    def test_finalize_with_no_records(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        mon.finalize()  # should not raise

    def test_csv_still_readable_after_finalize(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        domain.number_of_steps = 480
        mon.record(60.0, wall_time_s=16.0)
        mon.finalize()
        csv_path = tmp_path / "run_diagnostics_1.csv"
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tests: tri_full_flag handling (parallel mode ghost cells)
# ---------------------------------------------------------------------------

class TestGhostCellFiltering:
    def test_full_flag_respected(self, tmp_path):
        """With tri_full_flag masking out 2 of 4 triangles, wet_fraction
        is computed over the 2 full triangles only."""
        domain = _make_mock_domain(n=4)
        # Mark triangles 0 and 1 as full, 2 and 3 as ghost
        domain.tri_full_flag = np.array([1, 1, 0, 0])
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        # Both full triangles have depth>0 → wet_fraction = 2/2 = 1.0
        assert abs(rec["wet_fraction"] - 1.0) < 1e-3

    def test_no_full_flag_attribute_is_ok(self, tmp_path):
        """Domain without tri_full_flag should work (all triangles treated as full)."""
        domain = _make_mock_domain(n=4)
        assert not hasattr(domain, "tri_full_flag")
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        assert rec["wet_cells"] == 3  # 3 of 4 are wet


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_dry_domain(self, tmp_path):
        """Domain where all cells are dry — no division by zero."""
        domain = _make_mock_domain(n=4)
        # Set all elevations above stage
        domain.quantities["elevation"].centroid_values[:] = 10.0
        domain.quantities["stage"].centroid_values[:] = 9.0
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        assert rec["wet_cells"] == 0
        assert rec["wet_fraction"] == 0.0
        assert rec["volume_m3"] == 0.0
        assert rec["max_speed_ms"] == 0.0

    def test_zero_timestep_no_crash(self, tmp_path):
        """domain.timestep=0 should not cause division by zero."""
        domain = _make_mock_domain(timestep=0.0)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        domain.number_of_steps = 480
        rec = mon.record(60.0, wall_time_s=16.0)
        # implied_max_speed should be a large finite number, not inf/nan
        assert math.isfinite(rec["implied_max_speed_ms"])

    def test_instability_signal(self, tmp_path):
        """Simulate the Merewether instability pattern: dt collapses, implied
        speed grows unphysically."""
        domain = _make_mock_domain(n=4)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)

        # Normal step
        domain.number_of_steps = 480
        domain.timestep = 0.125
        rec1 = mon.record(60.0, wall_time_s=16.0)

        # Instability developing
        domain.number_of_steps = 17000
        domain.timestep = 0.001  # dt collapsed to 1ms
        rec2 = mon.record(120.0, wall_time_s=254.0)

        assert rec2["last_dt_ms"] < rec1["last_dt_ms"]
        assert rec2["implied_max_speed_ms"] > rec1["implied_max_speed_ms"]
        assert rec2["n_steps"] > rec1["n_steps"]


# ---------------------------------------------------------------------------
# Tests: run_summary JSON
# ---------------------------------------------------------------------------

def _make_monitor_with_records(tmp_path, n_records=2, duration_s=120.0,
                                scenario_config=None):
    """Helper: create a monitor, record n_records yieldsteps, return it."""
    domain = _make_mock_domain(n=4, number_of_steps=0)
    cfg = scenario_config or {
        "name": "Test Scenario",
        "project": 7, "id": 3, "run_id": 5,
        "epsg": "EPSG:32756", "resolution": 2.0,
    }
    mon = SimulationMonitor(
        domain, str(tmp_path), 1, yieldstep=60,
        duration_s=duration_s, run_label="run_7_3_5", scenario_config=cfg,
    )
    for i in range(1, n_records + 1):
        domain.number_of_steps = 480  # per-window (engine resets each yield)
        domain.timestep = 0.125
        mon.record(i * 60.0, wall_time_s=16.0, mem_mb=512.0)
    return mon


class TestRunSummaryJSON:
    def test_json_file_created(self, tmp_path):
        mon = _make_monitor_with_records(tmp_path)
        mon.finalize()
        assert (tmp_path / "run_summary_1.json").exists()

    def test_json_batch_number_in_filename(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 9, yieldstep=60)
        mon.finalize()
        assert (tmp_path / "run_summary_9.json").exists()

    def test_json_is_valid(self, tmp_path):
        mon = _make_monitor_with_records(tmp_path)
        mon.finalize()
        data = json.loads((tmp_path / "run_summary_1.json").read_text())
        assert isinstance(data, dict)

    def test_schema_version(self, tmp_path):
        mon = _make_monitor_with_records(tmp_path)
        mon.finalize()
        data = json.loads((tmp_path / "run_summary_1.json").read_text())
        assert data["schema_version"] == SUMMARY_SCHEMA_VERSION

    def test_top_level_sections(self, tmp_path):
        mon = _make_monitor_with_records(tmp_path)
        mon.finalize()
        data = json.loads((tmp_path / "run_summary_1.json").read_text())
        for section in ("run", "model", "mesh", "performance", "flow", "stability", "environment"):
            assert section in data, f"Missing top-level key: {section}"

    def test_run_section_fields(self, tmp_path):
        mon = _make_monitor_with_records(tmp_path)
        mon.finalize()
        run = json.loads((tmp_path / "run_summary_1.json").read_text())["run"]
        assert run["run_label"] == "run_7_3_5"
        assert run["project"] == 7
        assert run["scenario"] == 3
        assert run["run_id"] == 5
        assert run["name"] == "Test Scenario"
        assert run["batch_number"] == 1
        assert "started_at" in run
        assert "finished_at" in run
        assert run["total_wall_time_s"] >= 0.0

    def test_model_section_fields(self, tmp_path):
        mon = _make_monitor_with_records(tmp_path, n_records=2, duration_s=120.0)
        mon.finalize()
        model = json.loads((tmp_path / "run_summary_1.json").read_text())["model"]
        assert model["duration_s"] == 120.0
        assert model["final_sim_time_s"] == 120.0
        assert model["n_yieldsteps"] == 2
        assert model["yieldstep_s"] == 60.0
        assert model["epsg"] == "EPSG:32756"
        assert model["resolution_m"] == 2.0
        assert model["cfl"] == 0.9

    def test_mesh_section_fields(self, tmp_path):
        mon = _make_monitor_with_records(tmp_path)
        mon.finalize()
        mesh = json.loads((tmp_path / "run_summary_1.json").read_text())["mesh"]
        assert mesh["n_triangles"] == 4
        assert mesh["inradius_min_m"] > 0.0
        assert mesh["inradius_median_m"] >= mesh["inradius_min_m"]
        assert mesh["min_angle_deg"] > 0.0

    def test_performance_section_fields(self, tmp_path):
        mon = _make_monitor_with_records(tmp_path, n_records=2)
        mon.finalize()
        perf = json.loads((tmp_path / "run_summary_1.json").read_text())["performance"]
        assert perf["total_internal_steps"] == 2 * 480
        assert perf["mean_steps_per_yieldstep"] == 480.0
        assert perf["max_steps_per_yieldstep"] == 480
        assert perf["first_dt_ms"] == 125.0
        assert perf["min_dt_ms"] == 125.0
        assert perf["peak_mem_mb"] == 512.0
        assert perf["sim_per_wall_ratio"] > 0.0

    def test_flow_section_fields(self, tmp_path):
        mon = _make_monitor_with_records(tmp_path)
        mon.finalize()
        flow = json.loads((tmp_path / "run_summary_1.json").read_text())["flow"]
        assert "final_wet_fraction" in flow
        assert "final_volume_m3" in flow
        assert "max_depth_m" in flow
        assert flow["max_speed_ms"] > 0.0

    def test_stability_section_stable(self, tmp_path):
        """Normal run: max implied speed well below threshold → stable=True."""
        mon = _make_monitor_with_records(tmp_path)
        mon.finalize()
        stab = json.loads((tmp_path / "run_summary_1.json").read_text())["stability"]
        assert stab["stable"] is True
        assert stab["max_implied_speed_ms"] < INSTABILITY_SPEED_THRESHOLD_MS
        assert "timestep_collapse_ratio" in stab

    def test_stability_section_unstable(self, tmp_path):
        """Collapsed timestep: implied speed above threshold → stable=False."""
        domain = _make_mock_domain(n=4)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60, duration_s=120.0)
        # First step: normal
        domain.number_of_steps = 480
        domain.timestep = 0.125
        mon.record(60.0, wall_time_s=16.0)
        # Second step: dt collapsed → implied = 0.9 * 0.577 / 0.001 ≈ 520 m/s
        domain.number_of_steps = 60480
        domain.timestep = 0.001
        mon.record(70.0, wall_time_s=600.0)
        mon.finalize()
        stab = json.loads((tmp_path / "run_summary_1.json").read_text())["stability"]
        assert stab["stable"] is False
        assert stab["max_implied_speed_ms"] > INSTABILITY_SPEED_THRESHOLD_MS

    def test_outcome_completed(self, tmp_path):
        """Records up to duration_s → outcome = 'completed'."""
        mon = _make_monitor_with_records(tmp_path, n_records=2, duration_s=120.0)
        mon.finalize()
        run = json.loads((tmp_path / "run_summary_1.json").read_text())["run"]
        assert run["outcome"] == "completed"

    def test_outcome_incomplete(self, tmp_path):
        """Records only to 60s with duration_s=120 and stable physics → 'incomplete'."""
        domain = _make_mock_domain(n=4)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60, duration_s=120.0)
        domain.number_of_steps = 480
        domain.timestep = 0.125
        mon.record(60.0, wall_time_s=16.0)  # only half done, stable
        mon.finalize()
        run = json.loads((tmp_path / "run_summary_1.json").read_text())["run"]
        assert run["outcome"] == "incomplete"

    def test_outcome_unstable(self, tmp_path):
        """Blown-up run that didn't reach duration_s → 'unstable'."""
        domain = _make_mock_domain(n=4)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60, duration_s=1000.0)
        domain.number_of_steps = 480
        domain.timestep = 0.125
        mon.record(60.0, wall_time_s=16.0)
        domain.number_of_steps = 60480
        domain.timestep = 0.001  # blowup
        mon.record(70.0, wall_time_s=600.0)
        mon.finalize()
        run = json.loads((tmp_path / "run_summary_1.json").read_text())["run"]
        assert run["outcome"] == "unstable"

    def test_environment_section_keys(self, tmp_path):
        mon = _make_monitor_with_records(tmp_path)
        mon.finalize()
        env = json.loads((tmp_path / "run_summary_1.json").read_text())["environment"]
        for key in ("hostname", "os", "python_version", "cpu_model",
                    "cpu_count_logical", "total_ram_gb"):
            assert key in env, f"Missing environment key: {key}"

    def test_environment_python_version(self, tmp_path):
        import platform
        mon = _make_monitor_with_records(tmp_path)
        mon.finalize()
        env = json.loads((tmp_path / "run_summary_1.json").read_text())["environment"]
        assert env["python_version"] == platform.python_version()

    def test_no_records_still_writes_json(self, tmp_path):
        """finalize() with zero records should still produce a valid JSON."""
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60, duration_s=60.0)
        mon.finalize()
        data = json.loads((tmp_path / "run_summary_1.json").read_text())
        assert data["run"]["outcome"] == "incomplete"
        assert data["performance"]["total_internal_steps"] == 0

    def test_scenario_config_defaults_when_missing(self, tmp_path):
        """Missing scenario_config fields default gracefully to zero/empty."""
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        mon.finalize()
        run = json.loads((tmp_path / "run_summary_1.json").read_text())["run"]
        assert run["project"] == 0
        assert run["scenario"] == 0
        assert run["name"] == ""


class TestNumpySafetyAndFinalizeGuard:
    """TASK-2033 (run 1284): telemetry must never kill a completed run.

    On the baked x32 image (numpy 2.x) the domain quantities yield numpy
    scalars in the per-yieldstep records, so `stable` (a numpy comparison)
    and the rounded implied speeds reach json.dump as np.bool/np.float64 —
    which crashed _write_summary AFTER a 100%-complete evolve and took the
    whole container (and the handoff) down with exit 1.
    """

    def test_finalize_survives_numpy_scalars_in_records(self, tmp_path):
        domain = _make_mock_domain(n=4)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60, duration_s=60.0)
        mon.record(60.0, wall_time_s=16.0)
        # Reproduce the prod record shape: numpy scalars, not Python floats.
        for rec in mon._records:
            rec["implied_max_speed_ms"] = np.float64(rec["implied_max_speed_ms"])
            rec["sim_time_s"] = np.float64(rec["sim_time_s"])
            rec["max_speed_ms"] = np.float64(rec["max_speed_ms"])
            rec["max_depth_m"] = np.float64(rec["max_depth_m"])
        mon.finalize()
        data = json.loads((tmp_path / "run_summary_1.json").read_text())
        assert isinstance(data["stability"]["stable"], bool)
        assert isinstance(data["stability"]["max_implied_speed_ms"], float)

    def test_finalize_monitor_safely_swallows_exceptions(self, tmp_path):
        from run_anuga.diagnostics import finalize_monitor_safely

        class ExplodingMonitor:
            def finalize(self):
                raise TypeError("Object of type bool is not JSON serializable")

        # Must not raise — a telemetry failure downgrades to a warning.
        finalize_monitor_safely(ExplodingMonitor())
        finalize_monitor_safely(None)

    def test_finalize_monitor_safely_calls_finalize(self, tmp_path):
        from run_anuga.diagnostics import finalize_monitor_safely

        calls = []

        class OkMonitor:
            def finalize(self):
                calls.append(1)

        finalize_monitor_safely(OkMonitor())
        assert calls == [1]


# ---------------------------------------------------------------------------
# TASK-2675 (epic 2662 W2.5) — honest MPI diagnostics: allreduce the status
# scalars. Mocked-MPI ONLY on purpose: anuga_core is unbuilt on the
# workstation, so a proof requiring local mpirun would be invalid — the live
# multi-rank confirmation rides the W2.6 gate's CPU run log.
# ---------------------------------------------------------------------------

import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
from run_anuga.diagnostics import (  # noqa: E402
    _mpi_world_size_from_env,
    allreduce_flow_scalars,
    collect_flow_scalars,
    compute_local_flow_scalars,
)

# Rank 0 is DRY — the exact prod-run-1314 trap: local wet=0/vol=0 while the
# water lives on other ranks.
RANK0_DRY_LOCAL = {
    "n_wet": 0, "n_full": 250, "volume_m3": 0.0, "max_depth_m": 0.0,
    "max_speed_ms": 0.0, "peak_speed_x": 0.0, "peak_speed_y": 0.0,
    "min_r_wet": float("inf"),
}


class ScriptedComm:
    """Mocked MPI communicator: asserts each collective's (method, op,
    sendobj) in the implementation's documented order and returns the
    globally-reduced value a 4-rank world would produce."""

    def __init__(self, script, rank=0):
        self.script = list(script)
        self.rank = rank

    def Get_rank(self):
        return self.rank

    def _next(self, method, op=None, sendobj=None, root=None):
        assert self.script, f"unexpected extra collective: {method}"
        exp = self.script.pop(0)
        assert exp["method"] == method, f"expected {exp['method']}, got {method}"
        if "op" in exp:
            assert exp["op"] == op, f"{method}: expected op {exp['op']}, got {op}"
        if "sendobj" in exp:
            assert exp["sendobj"] == sendobj, (
                f"{method}: expected sendobj {exp['sendobj']}, got {sendobj}")
        if "root" in exp:
            assert exp["root"] == root, f"bcast: expected root {exp['root']}, got {root}"
        return exp["ret"]

    def allreduce(self, sendobj, op=None):
        return self._next("allreduce", op=op, sendobj=sendobj)

    def bcast(self, obj, root=None):
        return self._next("bcast", root=root)


def _install_fake_mpi(monkeypatch, comm):
    fake_mpi = types.SimpleNamespace(
        SUM="SUM", MAX="MAX", MIN="MIN", MAXLOC="MAXLOC",
        COMM_WORLD=comm,
    )
    mod = types.ModuleType("mpi4py")
    mod.MPI = fake_mpi
    monkeypatch.setitem(sys.modules, "mpi4py", mod)
    return fake_mpi


class _ExplodingModule(types.ModuleType):
    """Any attribute access (i.e. `from mpi4py import MPI`) raises."""

    def __getattr__(self, name):
        raise AssertionError(
            f"mpi4py.{name} was touched on the serial path — AC2 violated "
            "(no MPI import cost when serial)"
        )


class TestAllreduceFlowScalars:
    def test_serial_path_unchanged_and_never_touches_mpi(self, monkeypatch):
        """AC2: without an MPI launcher env the input passes through
        unchanged and mpi4py is never imported/touched."""
        for var in ("OMPI_COMM_WORLD_SIZE", "PMIX_SIZE", "PMI_SIZE"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setitem(sys.modules, "mpi4py", _ExplodingModule("mpi4py"))
        local = dict(RANK0_DRY_LOCAL, n_wet=7, volume_m3=1.5)
        out = allreduce_flow_scalars(local)
        assert out == local
        assert out is not local  # a copy, never an aliased mutation surface

    def test_world_size_env_parsing(self, monkeypatch):
        for var in ("OMPI_COMM_WORLD_SIZE", "PMIX_SIZE", "PMI_SIZE"):
            monkeypatch.delenv(var, raising=False)
        assert _mpi_world_size_from_env() == 1
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "32")
        assert _mpi_world_size_from_env() == 32

    def _global_script(self):
        """4-rank world; rank 0 dry; water on ranks 1-3; vmax owned by rank 2."""
        return [
            {"method": "allreduce", "op": "SUM", "sendobj": 0, "ret": 300},
            {"method": "allreduce", "op": "SUM", "sendobj": 250, "ret": 1000},
            {"method": "allreduce", "op": "SUM", "sendobj": 0.0, "ret": 4321.5},
            {"method": "allreduce", "op": "MAX", "sendobj": 0.0, "ret": 2.5},
            {"method": "allreduce", "op": "MIN", "sendobj": float("inf"), "ret": 0.75},
            {"method": "allreduce", "op": "MAXLOC", "sendobj": (0.0, 0), "ret": (3.2, 2)},
            {"method": "bcast", "root": 2, "ret": (512.5, 663.25)},
        ]

    def test_allreduce_returns_global_values_on_a_dry_rank0(self, monkeypatch):
        """AC1: rank 0's DRY partition must report the GLOBAL truth —
        wet/vol/vmax/min-inradius reduced with the right op each, and the
        vmax coordinates broadcast from the rank that owns the max."""
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
        comm = ScriptedComm(self._global_script())
        _install_fake_mpi(monkeypatch, comm)
        out = allreduce_flow_scalars(dict(RANK0_DRY_LOCAL))
        assert out["n_wet"] == 300
        assert out["n_full"] == 1000
        assert out["volume_m3"] == 4321.5
        assert out["max_depth_m"] == 2.5
        assert out["min_r_wet"] == 0.75
        assert out["max_speed_ms"] == 3.2
        assert (out["peak_speed_x"], out["peak_speed_y"]) == (512.5, 663.25)
        assert comm.script == []  # every scripted collective consumed

    def test_allreduce_missing_mpi4py_degrades_loudly(self, monkeypatch, caplog):
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
        monkeypatch.setitem(sys.modules, "mpi4py", None)  # import -> ImportError
        import logging as _logging
        with caplog.at_level(_logging.WARNING, logger="run_anuga.diagnostics"):
            out = allreduce_flow_scalars(dict(RANK0_DRY_LOCAL))
        assert out == RANK0_DRY_LOCAL
        assert "RANK-LOCAL" in caplog.text

    def test_record_reports_global_values_despite_dry_local_domain(
            self, monkeypatch, tmp_path):
        """AC1 end-to-end: a monitor over a DRY rank-0 domain, handed the
        reduced flow dict, logs the global wet/vol/vmax/v_impl."""
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
        domain = _make_mock_domain(timestep=0.125)
        # Dry out rank 0's partition entirely (stage below elevation).
        domain.quantities["stage"].centroid_values = (
            domain.quantities["elevation"].centroid_values - 1.0)
        # Script keyed to THIS domain's locals (4 full triangles, all dry,
        # local max depth -1.0m): the world still reports the global truth.
        comm = ScriptedComm([
            {"method": "allreduce", "op": "SUM", "sendobj": 0, "ret": 300},
            {"method": "allreduce", "op": "SUM", "sendobj": 4, "ret": 1000},
            {"method": "allreduce", "op": "SUM", "sendobj": 0.0, "ret": 4321.5},
            {"method": "allreduce", "op": "MAX", "sendobj": -1.0, "ret": 2.5},
            {"method": "allreduce", "op": "MIN", "sendobj": float("inf"), "ret": 0.75},
            {"method": "allreduce", "op": "MAXLOC", "sendobj": (0.0, 0), "ret": (3.2, 2)},
            {"method": "bcast", "root": 2, "ret": (512.5, 663.25)},
        ])
        _install_fake_mpi(monkeypatch, comm)
        flow = collect_flow_scalars(domain)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60, cfl=0.9)
        rec = mon.record(60.0, wall_time_s=16.0, flow=flow)

        assert rec["wet_cells"] == 300
        assert rec["wet_fraction"] == round(300 / 1000, 4)
        assert rec["volume_m3"] == 4321.5
        assert rec["max_speed_ms"] == 3.2
        # v_impl = cfl * global min_r_wet / dt = 0.9 * 0.75 / 0.125 = 5.4
        assert rec["implied_max_speed_ms"] == pytest.approx(5.4, abs=0.01)
        # the log-suffix line (the one that lied on run 1314) now shows truth
        suffix = mon.format_log_suffix(rec)
        assert "wet=30%" in suffix and "vol=4322" in suffix

    def test_serial_record_unchanged_without_flow_arg(self, tmp_path, monkeypatch):
        """AC2: legacy serial callers (no flow kwarg) keep the exact local
        computation — same values as compute_local_flow_scalars."""
        for var in ("OMPI_COMM_WORLD_SIZE", "PMIX_SIZE", "PMI_SIZE"):
            monkeypatch.delenv(var, raising=False)
        domain = _make_mock_domain(timestep=0.125)
        local = compute_local_flow_scalars(domain)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60, cfl=0.9)
        rec = mon.record(60.0, wall_time_s=16.0)
        assert rec["wet_cells"] == local["n_wet"] == 3
        assert rec["volume_m3"] == round(local["volume_m3"], 1)
        assert rec["max_speed_ms"] == round(local["max_speed_ms"], 3)


class TestObservedStatsFeed:
    """TASK-3373: aggregate handed to the sampler via phase_tracker."""

    def test_observed_stats_and_phase_tracker_publish(self, tmp_path):
        from run_anuga import phase_tracker

        phase_tracker.reset()
        domain = _make_mock_domain(timestep=0.125, number_of_steps=0)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        for i, steps in enumerate([0, 480, 240], start=0):
            domain.number_of_steps = steps
            mon.record(i * 60.0, wall_time_s=1.0)
        stats = phase_tracker.get_run_stats()
        assert stats["n_steps"] == 720
        assert stats["wet_fraction_final"] == stats["wet_fraction_max"] == 0.75
        assert stats["max_depth_m"] > 0
        assert stats["min_dt_ms"] is not None
        assert [p["cum_steps"] for p in stats["yieldstep_series"]] == [0, 480, 720]
        assert set(stats["yieldstep_series"][0]) == {
            "t", "cum_steps", "mean_dt_ms", "wet_fraction"}
        phase_tracker.reset()
        assert phase_tracker.get_run_stats() == {}

    def test_series_is_decimated_but_keeps_final_point(self, tmp_path):
        domain = _make_mock_domain()
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60)
        for i in range(1, 501):
            domain.number_of_steps = 10
            mon.record(i * 60.0, wall_time_s=0.1)
        series = mon.observed_stats()["yieldstep_series"]
        assert len(series) <= SimulationMonitor.SERIES_MAX_POINTS + 1
        assert series[-1]["cum_steps"] == 5000


class TestStabilityIgnoresStartupFrame:
    """TASK-3840: the stability flag must reflect the run, not the t=0 yield.

    ANUGA yields once at the start time before any internal step: n_steps is
    0 and domain.timestep is ~0, so CFL * min_inradius_wet / dt explodes. On
    hydrata.com runs 1591-1594 (run_anuga 0f25d4e) every run_summary read
    stable=False, max_implied_speed_ms=375724339405.99 at sim 0.0 s against a
    20 m/s threshold, although each run completed normally.
    """

    #: The value recorded at sim 0.0 s by prod run 1593 (run_summary_1.json).
    PROD_T0_IMPLIED_MS = 375724339405.99

    def _stab(self, tmp_path, mon):
        mon.finalize()
        data = json.loads((tmp_path / "run_summary_1.json").read_text())
        return data["stability"], data["run"]

    def _prod_1593_like(self, tmp_path, late_blowup_ms=None):
        """Equivalent fixture for the saved run 1593 diagnostics (not on this
        box): a t=0 record carrying the recorded prod implied speed, then
        normal stepped records (and optionally a genuine late blow-up)."""
        domain = _make_mock_domain(n=4, number_of_steps=0)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60,
                                duration_s=300.0)
        base = {
            "wall_time_s": 1.0, "mean_dt_ms": 125.0, "last_dt_ms": 125.0,
            "min_dt_ms": 125.0, "wet_cells": 3, "wet_fraction": 0.75,
            "volume_m3": 10.0, "max_depth_m": 1.0, "max_speed_ms": 1.2,
            "peak_speed_x": 1.0, "peak_speed_y": 0.0, "mem_mb": 100.0,
        }
        mon._records.append({**base, "sim_time_s": 0.0, "n_steps": 0,
                             "mean_dt_ms": 60000.0, "last_dt_ms": 0.0,
                             "min_dt_ms": 0.0,
                             "implied_max_speed_ms": self.PROD_T0_IMPLIED_MS})
        for t, implied in ((60.0, 2.1), (120.0, 3.4), (180.0, 2.7),
                           (240.0, 2.2), (300.0, 1.9)):
            if late_blowup_ms is not None and t == 240.0:
                implied = late_blowup_ms
            mon._records.append({**base, "sim_time_s": t, "n_steps": 480,
                                 "implied_max_speed_ms": implied})
        return mon

    def test_prod_1593_equivalent_reads_stable(self, tmp_path):
        stab, run = self._stab(tmp_path, self._prod_1593_like(tmp_path))
        assert stab["stable"] is True
        assert stab["max_implied_speed_ms"] == 3.4
        assert stab["max_implied_speed_at_sim_s"] == 120.0
        assert stab["max_implied_speed_at_sim_s"] > 0
        assert stab["instability_threshold_ms"] == INSTABILITY_SPEED_THRESHOLD_MS
        assert run["outcome"] == "completed"

    def test_late_blowup_still_reads_unstable(self, tmp_path):
        """Negative control: a genuine blow-up after t=0 is still caught."""
        mon = self._prod_1593_like(tmp_path, late_blowup_ms=520.0)
        stab, _ = self._stab(tmp_path, mon)
        assert stab["stable"] is False
        assert stab["max_implied_speed_ms"] == 520.0
        assert stab["max_implied_speed_at_sim_s"] == 240.0

    def test_recorded_t0_frame_with_near_zero_dt_is_ignored(self, tmp_path):
        """Through record(): wet at t=0 with dt ~1e-12 and no steps taken."""
        domain = _make_mock_domain(n=4, timestep=1.4e-12, number_of_steps=0)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60,
                                duration_s=120.0)
        rec0 = mon.record(0.0, wall_time_s=0.0)
        assert rec0["implied_max_speed_ms"] > 1e11  # the spurious start-up value
        domain.timestep = 0.125
        for t in (60.0, 120.0):
            domain.number_of_steps = 480
            mon.record(t, wall_time_s=16.0)
        stab, _ = self._stab(tmp_path, mon)
        assert stab["stable"] is True
        assert 0 < stab["max_implied_speed_ms"] < INSTABILITY_SPEED_THRESHOLD_MS
        assert stab["max_implied_speed_at_sim_s"] == 60.0

    def test_resume_first_yield_without_steps_is_ignored(self, tmp_path):
        """A checkpoint resume yields first at t>0, again before any step."""
        domain = _make_mock_domain(n=4, timestep=1.4e-12, number_of_steps=0)
        mon = SimulationMonitor(domain, str(tmp_path), 2, yieldstep=60,
                                duration_s=240.0)
        mon.record(120.0, wall_time_s=0.0)
        domain.timestep = 0.125
        domain.number_of_steps = 480
        mon.record(180.0, wall_time_s=16.0)
        mon.finalize()
        stab = json.loads((tmp_path / "run_summary_2.json").read_text())["stability"]
        assert stab["stable"] is True
        assert stab["max_implied_speed_at_sim_s"] == 180.0

    def test_only_startup_frame_reads_zero(self, tmp_path):
        domain = _make_mock_domain(n=4, timestep=1.4e-12, number_of_steps=0)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=60,
                                duration_s=120.0)
        mon.record(0.0, wall_time_s=0.0)
        stab, _ = self._stab(tmp_path, mon)
        assert stab["stable"] is True
        assert stab["max_implied_speed_ms"] == 0.0


# ---------------------------------------------------------------------------
# TASK-4316 (epic 3754 W3, decision D3): an honest stability detector and a
# graded run summary (folds TASK-4024, the collapse ratio pinned at 1.0).
#
# ROOT CAUSE (proven 09-11-26): the implied speed divided by domain.timestep,
# which the engine has already CLIPPED to land on the yield time
# (generic_domain.update_timestep). Run 1613's log: every spike coincides with
# a last step of 8-60 ms while n_steps per 144 s window stays 135/136, and
# v_impl x last_dt = 3.20-3.26 m in every window. The honest step is the
# engine's CFL step before the clip: CFL * domain.flux_timestep on the CPU
# path, gpu_dom.recorded_flux_timestep in GPU mode 2 (whose C step loops never
# update the host flux_timestep).
# ---------------------------------------------------------------------------

from run_anuga.diagnostics import (  # noqa: E402
    OPEN_BOUNDARY_NOISE,
    classify_stability,
    load_diagnostics_csv,
)

INRADIUS = math.sqrt(3) / 3  # every mock triangle; the wet minimum too


def _flux_domain(timestep=0.002, flux_timestep=1.0, cfl=0.9, **extra):
    """A CPU-mode mock whose last step was CLIPPED to `timestep` while the
    engine's own CFL step is CFL * flux_timestep."""
    d = _make_mock_domain(n=4, timestep=timestep, number_of_steps=136)
    d.flux_timestep = flux_timestep
    d.CFL = cfl
    d.evolve_max_timestep = 1000.0
    d.fixed_flux_timestep = None
    d.recorded_min_timestep = cfl * flux_timestep
    d.multiprocessor_mode = 1
    for k, v in extra.items():
        setattr(d, k, v)
    return d


class TestHonestImpliedSpeed:
    def test_implied_speed_uses_flux_timestep_not_clipped_step(self, tmp_path):
        """AC1. Last step clipped to 2 ms, CFL step 0.9 s: honest implied is
        CFL * r / (CFL * flux) = 0.58 m/s. The pre-fix formula, CFL * r /
        domain.timestep, reads 260 m/s (negative control below)."""
        d = _flux_domain(timestep=0.002, flux_timestep=1.0)
        mon = SimulationMonitor(d, str(tmp_path), 1, yieldstep=144, cfl=0.9)
        rec = mon.record(144.0, wall_time_s=1.0)

        assert rec["implied_max_speed_ms"] == pytest.approx(INRADIUS, abs=0.01)
        assert rec["implied_max_speed_ms"] < INSTABILITY_SPEED_THRESHOLD_MS
        assert rec["flux_timestep_ms"] == pytest.approx(900.0)
        assert rec["last_dt_ms"] == pytest.approx(2.0)  # kept, still the clipped step
        pre_fix = 0.9 * INRADIUS / d.timestep
        assert pre_fix > 200.0  # what the clipped divisor reported

    def test_gpu_mode_reads_the_device_cfl_step_not_the_stale_host_value(self, tmp_path):
        """Mode 2's C step loops never write the host domain.flux_timestep
        (it stays 0.0); the CFL step lives in gpu_dom.recorded_flux_timestep
        (already CFL-scaled, before the yield clip)."""
        gpu_dom = SimpleNamespace(recorded_flux_timestep=0.9)
        d = _flux_domain(timestep=0.002, flux_timestep=0.0,
                         multiprocessor_mode=2,
                         gpu_interface=SimpleNamespace(gpu_dom=gpu_dom))
        mon = SimulationMonitor(d, str(tmp_path), 1, yieldstep=144, cfl=0.9)
        rec = mon.record(144.0, wall_time_s=1.0)
        assert rec["flux_timestep_ms"] == pytest.approx(900.0)
        assert rec["implied_max_speed_ms"] == pytest.approx(INRADIUS, abs=0.01)

    def test_gpu_python_fallback_loops_use_the_host_flux_step(self, tmp_path):
        """The Python-orchestrated GPU loops write the host flux_timestep and
        leave the device value unset: the host step is read then."""
        gpu_dom = SimpleNamespace(recorded_flux_timestep=0.0)
        d = _flux_domain(timestep=0.002, flux_timestep=1.0, multiprocessor_mode=2,
                         gpu_interface=SimpleNamespace(gpu_dom=gpu_dom))
        mon = SimulationMonitor(d, str(tmp_path), 1, yieldstep=144, cfl=0.9)
        assert mon.record(144.0, wall_time_s=1.0)["flux_timestep_ms"] == pytest.approx(900.0)

    def test_legacy_domain_without_flux_timestep_keeps_last_step(self, tmp_path):
        """Mocks/engines without flux_timestep fall back to domain.timestep."""
        d = _make_mock_domain(n=4, timestep=0.125, number_of_steps=480)
        mon = SimulationMonitor(d, str(tmp_path), 1, yieldstep=60, cfl=0.9)
        rec = mon.record(60.0, wall_time_s=1.0)
        assert rec["implied_max_speed_ms"] == pytest.approx(0.9 * INRADIUS / 0.125, abs=0.01)
        assert rec["flux_timestep_ms"] is None


class TestCollapseRatio:
    def test_collapse_ratio_ignores_t0_record(self, tmp_path):
        """AC2 (TASK-4024). The t=0 record (no steps, dt 0) used to be the
        baseline, so the ratio read 1.0 on every run. Rebased on stepped
        records, a genuine collapse from 125 ms to 10 ms reads 0.08."""
        d = _make_mock_domain(n=4, timestep=0.0, number_of_steps=0)
        d.recorded_min_timestep = 0.0
        mon = SimulationMonitor(d, str(tmp_path), 1, yieldstep=60, duration_s=240.0)
        mon.record(0.0, wall_time_s=0.0)
        for t, dt in ((60.0, 0.125), (120.0, 0.125), (180.0, 0.125), (240.0, 0.010)):
            d.number_of_steps = 480
            d.timestep = dt
            d.recorded_min_timestep = dt
            mon.record(t, wall_time_s=1.0)
        mon.finalize()
        stab = json.loads((tmp_path / "run_summary_1.json").read_text())["stability"]
        assert stab["timestep_collapse_ratio"] == pytest.approx(0.08)
        assert stab["timestep_collapse_ratio"] < 1.0


class TestVolumeAccounting:
    def test_closure_and_boundary_flux_columns_recorded(self, tmp_path):
        """AC3. V, BF and FS come from the engine getters; closure = V - BF
        - FS - V0 with V0 taken at the first record (V - BF - FS there), so a
        resumed run's closure starts at zero too."""
        state = {"V": 100.0, "BF": 0.0, "FS": 0.0}
        d = _flux_domain(timestep=0.9, flux_timestep=1.0,
                         get_water_volume=lambda: state["V"],
                         get_boundary_flux_integral=lambda: state["BF"],
                         get_fractional_step_volume_integral=lambda: state["FS"])
        mon = SimulationMonitor(d, str(tmp_path), 1, yieldstep=144, cfl=0.9)
        r1 = mon.record(0.0, wall_time_s=0.0)
        state.update(V=150.0, BF=20.0, FS=25.0)
        r2 = mon.record(144.0, wall_time_s=1.0)
        mon.finalize()

        assert (r1["boundary_flux_integral_m3"], r1["fractional_step_volume_m3"], r1["closure_m3"]) == (0.0, 0.0, 0.0)
        assert r2["boundary_flux_integral_m3"] == 20.0
        assert r2["fractional_step_volume_m3"] == 25.0
        assert r2["closure_m3"] == pytest.approx(5.0)
        rows = load_diagnostics_csv(tmp_path / "run_diagnostics_1.csv")
        for col in ("flux_timestep_ms", "boundary_flux_integral_m3",
                    "fractional_step_volume_m3", "closure_m3"):
            assert col in SimulationMonitor.CSV_FIELDS
            assert rows[-1][col] is not None, col
        assert rows[-1]["closure_m3"] == pytest.approx(5.0)

    def test_boundary_flux_getter_failure_leaves_columns_empty(self, tmp_path):
        """Non-DE flux methods raise in get_boundary_flux_integral; the record
        must not die, and closure is not computed from half the terms."""
        def boom():
            raise Exception("Boundary flux integral only supported for DE fluxes")
        d = _flux_domain(timestep=0.9, flux_timestep=1.0,
                         get_water_volume=lambda: 1.0,
                         get_boundary_flux_integral=boom,
                         get_fractional_step_volume_integral=lambda: 0.0)
        mon = SimulationMonitor(d, str(tmp_path), 1, yieldstep=144, cfl=0.9)
        rec = mon.record(144.0, wall_time_s=1.0)
        assert rec["boundary_flux_integral_m3"] is None
        assert rec["closure_m3"] is None


class TestPeakLocation:
    def _with_speeds(self, **extra):
        d = _flux_domain(timestep=0.9, flux_timestep=1.0, **extra)
        d.max_speed = np.array([1.0, 7.5, 2.0, 0.0])
        # one boundary edge: triangle 3 edge 0, midpoint at (7, 0)
        d.boundary = {(3, 0): "Transmissive"}
        mids = np.zeros((12, 2))
        mids[3 * 3 + 0] = (7.0, 0.0)
        d.mesh.edge_midpoint_coordinates = mids
        return d

    def test_cpu_records_the_fastest_triangle_location(self, tmp_path):
        """(e) CPU path: argmax(domain.max_speed) -> centroid, depth and the
        distance to the nearest boundary edge (midpoint)."""
        d = self._with_speeds()
        mon = SimulationMonitor(d, str(tmp_path), 1, yieldstep=144, cfl=0.9)
        rec = mon.record(144.0, wall_time_s=1.0)
        assert (rec["cfl_peak_x"], rec["cfl_peak_y"]) == (3.0, 0.0)  # triangle 1
        assert rec["cfl_peak_depth_m"] == pytest.approx(0.5)
        assert rec["cfl_peak_boundary_dist_m"] == pytest.approx(4.0)

    def test_gpu_location_is_null_never_a_guess(self, tmp_path):
        """AC6. Mode 2 never syncs max_speed to the host: the host array is
        stale, so the location fields are null in the CSV and the summary."""
        gpu_dom = SimpleNamespace(recorded_flux_timestep=0.9)
        d = self._with_speeds(multiprocessor_mode=2,
                              gpu_interface=SimpleNamespace(gpu_dom=gpu_dom))
        mon = SimulationMonitor(d, str(tmp_path), 1, yieldstep=144, cfl=0.9)
        rec = mon.record(144.0, wall_time_s=1.0)
        for k in ("cfl_peak_x", "cfl_peak_y", "cfl_peak_depth_m", "cfl_peak_boundary_dist_m"):
            assert rec[k] is None, k
        mon.finalize()
        stab = json.loads((tmp_path / "run_summary_1.json").read_text())["stability"]
        assert stab["location"] is None


# --- graded classes ---------------------------------------------------------

def _rec(t, n_steps=136, last=1060.0, min_dt=1060.0, implied=3.0, vol=0.0, **cols):
    r = {"sim_time_s": t, "wall_time_s": 1.0, "n_steps": n_steps,
         "mean_dt_ms": 1060.0, "last_dt_ms": last, "min_dt_ms": min_dt,
         "implied_max_speed_ms": implied, "wet_cells": 3, "wet_fraction": 0.75,
         "volume_m3": vol, "max_depth_m": 1.0, "max_speed_ms": 1.2,
         "peak_speed_x": 1.0, "peak_speed_y": 0.0, "mem_mb": 100.0}
    r.update(cols)
    return r


def _t0():
    return _rec(0.0, n_steps=0, last=0.0, min_dt=0.0, implied=0.0)


#: Saved production diagnostics, scrubbed (tests/fixtures/stability/README.md).
STABILITY_FIXTURES = Path(__file__).parent / "fixtures" / "stability"


def _fixture(run_id):
    """(records, summary) of a saved production run: the run's own CSV (the
    15-column schema these runs were written with) and its run summary."""
    recs = load_diagnostics_csv(STABILITY_FIXTURES / f"run_{run_id}_diagnostics_1.csv")
    summary = json.loads((STABILITY_FIXTURES / f"run_{run_id}_summary_1.json").read_text())
    return recs, summary


def _fixture_inputs(summary, storm_end_s=3600.0):
    """Rain applied (the run's own rain check) and the storm end: both runs
    are 1 h design bursts with no inflows."""
    return {"inputs_m3": summary["rain"]["applied_m3"], "inputs_end_s": storm_end_s}


class TestStabilityClasses:
    def test_stability_class_from_fixtures(self):
        """AC4 on the REAL saved runs (tests/fixtures/stability/). Run 1608
        (project 1121 / scenario 502, 6 h, yieldstep 216 s, tags Dirichlet +
        Transmissive) reported 1,122 m/s at sim 15,552 s = yield 72 from a
        2.16 ms clipped last step: every spike is a yield clip and its volume
        falls after the storm. Run 1613 (project 1135 / scenario 508, 4 h,
        yieldstep 144 s, all-Transmissive edge) reported 407.9 m/s at 5,472 s
        from a 7.88 ms clipped step, and its volume rose from 1.41 M m3 at rain
        end to 5.10 M m3 with no inflows: the edge filled the model."""
        recs, summary = _fixture(1608)
        assert summary["stability"]["max_implied_speed_ms"] == 1122.04  # what it said
        s1608 = classify_stability(recs, **_fixture_inputs(summary),
                                   open_boundary_tags=["Dirichlet", "Transmissive"])
        assert s1608["class"] == "clip_artefact"
        assert s1608["speed_class"] == "clip_artefact"
        assert s1608["n_records_over"] == 0
        assert s1608["n_clip_artefacts"] >= 1
        assert s1608["max_implied_speed_ms"] < INSTABILITY_SPEED_THRESHOLD_MS
        assert s1608["boundary_net_in_m3"] < OPEN_BOUNDARY_NOISE * summary["rain"]["applied_m3"]
        assert s1608["boundary_tag"] is None
        spike = next(r for r in recs if r["sim_time_s"] == 15552.0)
        assert spike["implied_max_speed_ms"] == 1122.04 and spike["last_dt_ms"] == 2.16

        recs, summary = _fixture(1613)
        assert summary["stability"]["max_implied_speed_ms"] == 407.9
        s1613 = classify_stability(recs, **_fixture_inputs(summary),
                                   open_boundary_tags=["Transmissive"])
        assert s1613["class"] == "boundary_fed"
        assert s1613["speed_class"] == "clip_artefact"
        assert s1613["n_records_over"] == 0
        assert s1613["max_implied_speed_ms"] < INSTABILITY_SPEED_THRESHOLD_MS
        assert s1613["boundary_tag"] == "Transmissive"
        assert s1613["boundary_basis"] == "by_difference"
        net = summary["flow"]["final_volume_m3"] - summary["rain"]["applied_m3"]
        assert s1613["boundary_net_in_m3"] == pytest.approx(net, abs=0.1)
        assert s1613["boundary_net_in_m3"] > OPEN_BOUNDARY_NOISE * summary["rain"]["applied_m3"]
        assert 3600.0 < s1613["onset_sim_s"] <= 7200.0
        for key in ("class", "onset_sim_s", "n_records_over", "closure_pct",
                    "boundary_net_in_m3", "boundary_tag"):
            assert key in s1613

    def test_collapse_ratio_on_real_runs_skips_dry_and_wetting_records(self):
        """TASK-4024 on the real CSVs. A dry domain still steps (n_steps 1)
        with min_dt_ms at the engine's 1,000 s 'no CFL step yet' value, and
        the first wet window holds the wetting transient (25 ms, 18.93 ms).
        Neither is a collapse: on the stepped-only rule the ratio read
        6.9e-3 (1613) and 1.9e-5 (1608). Both runs' steps shrink only as the
        water deepens, well above the 0.1 propagating line."""
        for run_id, lo, hi in ((1613, 0.15, 0.3), (1608, 0.3, 0.6)):
            recs, summary = _fixture(run_id)
            ratio = classify_stability(recs, **_fixture_inputs(summary))["timestep_collapse_ratio"]
            assert lo < ratio < hi, (run_id, ratio)

    def test_boundary_flux_column_names_the_inflow_exactly(self):
        """With the new columns the boundary volume is the engine's integral,
        and closure_pct is |V - BF - FS - V0| over the inputs."""
        recs = [_t0() | {"boundary_flux_integral_m3": 0.0, "fractional_step_volume_m3": 0.0,
                         "closure_m3": 0.0, "flux_timestep_ms": 0.0}]
        for i in range(1, 11):
            fs = min(i, 4) * 250.0                       # 1,000 m3 of rain by t=4
            bf = 0.0 if i <= 4 else (i - 4) * 400.0       # then the edge feeds
            recs.append(_rec(i * 60.0, vol=fs + bf, flux_timestep_ms=1000.0,
                             boundary_flux_integral_m3=bf, fractional_step_volume_m3=fs,
                             closure_m3=5.0))
        s = classify_stability(recs, open_boundary_tags=["Transmissive", "Time"])
        assert s["class"] == "boundary_fed"
        assert s["boundary_basis"] == "boundary_flux_integral"
        assert s["boundary_net_in_m3"] == pytest.approx(2400.0)
        assert s["closure_pct"] == pytest.approx(0.5)
        assert s["boundary_tag"] == "Time,Transmissive"

    def test_rain_only_run_with_falling_volume_is_not_boundary_fed(self):
        recs = [_t0()] + [_rec(i * 60.0, vol=1000.0 - max(0, i - 4) * 50.0) for i in range(1, 11)]
        s = classify_stability(recs, inputs_m3=1000.0, inputs_end_s=240.0)
        assert s["class"] is None
        assert s["boundary_net_in_m3"] < 0

    def test_single_honest_spike_is_a_local_transient(self):
        recs = [_t0()] + [_rec(i * 60.0) for i in range(1, 41)]
        recs[20] = _rec(1200.0, implied=45.0, min_dt=300.0, last=300.0)
        s = classify_stability(recs)
        assert s["class"] == "local_transient"
        assert s["n_records_over"] == 1
        assert s["onset_sim_s"] == 1200.0

    def test_three_consecutive_honest_records_are_localised_persistent(self):
        recs = [_t0()] + [_rec(i * 60.0) for i in range(1, 41)]
        for i in (20, 21, 22):
            recs[i] = _rec(i * 60.0, implied=30.0, min_dt=200.0, last=200.0)
        assert classify_stability(recs)["class"] == "localised_persistent"

    def test_collapse_after_onset_is_propagating(self):
        recs = [_t0()] + [_rec(i * 60.0) for i in range(1, 41)]
        for i in range(30, 41):
            recs[i] = _rec(i * 60.0, n_steps=136 * 20, implied=300.0, min_dt=10.0, last=10.0)
        s = classify_stability(recs)
        assert s["class"] == "propagating"
        assert s["onset_sim_s"] == 1800.0

    def test_clean_run_has_no_class(self):
        recs = [_t0()] + [_rec(i * 60.0) for i in range(1, 41)]
        s = classify_stability(recs)
        assert s["class"] is None
        assert s["n_records_over"] == 0


class TestSummarySchema:
    def test_summary_schema_bumped_and_new_stability_keys_pinned(self, tmp_path):
        """AC5: SUMMARY_SCHEMA_VERSION moves to "2" and the stability dict
        carries the graded fields beside the old ones."""
        assert SUMMARY_SCHEMA_VERSION == "2"
        mon = _make_monitor_with_records(tmp_path)
        mon.finalize()
        stab = json.loads((tmp_path / "run_summary_1.json").read_text())["stability"]
        for key in ("stable", "max_implied_speed_ms", "max_implied_speed_at_sim_s",
                    "timestep_collapse_ratio", "instability_threshold_ms",
                    "class", "speed_class", "onset_sim_s", "n_records_over",
                    "n_clip_artefacts", "closure_pct", "boundary_net_in_m3",
                    "boundary_basis", "boundary_tag", "location",
                    "thresholds_provisional"):
            assert key in stab, key
        assert stab["class"] is None
        assert stab["thresholds_provisional"] is True

    def test_clipped_spike_no_longer_marks_the_run_unstable(self, tmp_path):
        """Run 1613's real first 5,760 s through the monitor: its old-style
        clipped spikes (234.5 m/s at 3,888 s, 407.9 m/s at 5,472 s from last
        steps of 13.7 and 7.88 ms, window min ~1,066 ms, steps steady) read
        stable."""
        domain = _make_mock_domain(n=4, number_of_steps=0)
        mon = SimulationMonitor(domain, str(tmp_path), 1, yieldstep=144, duration_s=5760.0)
        mon._records.extend(_fixture(1613)[0][:41])
        assert mon._records[38]["implied_max_speed_ms"] == 407.9
        mon.finalize()
        data = json.loads((tmp_path / "run_summary_1.json").read_text())
        assert data["stability"]["stable"] is True
        assert data["stability"]["class"] == "clip_artefact"
        assert data["stability"]["max_implied_speed_ms"] < INSTABILITY_SPEED_THRESHOLD_MS
        assert data["run"]["outcome"] == "completed"


class TestCflPeakAllreduce:
    def test_cfl_peak_location_comes_from_the_winning_rank(self, monkeypatch):
        """TASK-4316 (e) under MPI: the fastest CFL triangle is a MAXLOC over
        ranks and its location is broadcast from the winner, the TASK-2675
        pattern; a 'none'/'gpu' mode (the same on every rank) skips it."""
        monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
        local = dict(RANK0_DRY_LOCAL, cfl_location_mode="cpu", cfl_peak_speed=1.5,
                     cfl_peak_x=1.0, cfl_peak_y=2.0, cfl_peak_depth_m=0.1,
                     cfl_peak_boundary_dist_m=9.0)
        script = TestAllreduceFlowScalars()._global_script() + [
            {"method": "allreduce", "op": "MAXLOC", "sendobj": (1.5, 0), "ret": (8.0, 3)},
            {"method": "bcast", "root": 3, "ret": (500.0, 600.0, 0.02, 1.5)},
        ]
        comm = ScriptedComm(script)
        _install_fake_mpi(monkeypatch, comm)
        out = allreduce_flow_scalars(local)
        assert out["cfl_peak_speed"] == 8.0
        assert (out["cfl_peak_x"], out["cfl_peak_y"], out["cfl_peak_depth_m"],
                out["cfl_peak_boundary_dist_m"]) == (500.0, 600.0, 0.02, 1.5)
        assert comm.script == []

        comm = ScriptedComm(TestAllreduceFlowScalars()._global_script())
        _install_fake_mpi(monkeypatch, comm)
        out = allreduce_flow_scalars(dict(local, cfl_location_mode="gpu", cfl_peak_x=None))
        assert out["cfl_peak_x"] is None
        assert comm.script == []
