"""TASK-3642 (epic 3585 W3): a real ANUGA run with one culvert, in a SUBPROCESS.

run_sim failure ends in MPI.COMM_WORLD.Abort, which kills pytest silently, so
every run here is ``python -m run_anuga.cli run`` and the test asserts on its
output directory and log. Needs anuga, pyswmm and anuga_drainage (marker
requires_anuga: the plain CI legs deselect it, the e2e job runs it).
"""

import csv
import json
import math
import subprocess
import sys

import pytest

from tests.test_culverts import (
    CONTRACT_KEYS,
    HYDRAULIC_KEYS,
    SMALL_TEST_CULVERT,
    _add_culvert_layer,
    _set_scenario,
    building,
    collection,
)

pytestmark = pytest.mark.requires_anuga


def _run_cli(package):
    result = subprocess.run(
        [sys.executable, "-m", "run_anuga.cli", "run", str(package)],
        capture_output=True, text=True, timeout=900,
    )
    return result, result.stdout + result.stderr


class TestCulvertE2E:

    def test_culvert_e2e_run_writes_the_structure_result_contract(self, small_test_copy):
        """AC2: metrics.json has every D16 key with numbers, and the time series is non-empty."""
        _add_culvert_layer(small_test_copy, collection(SMALL_TEST_CULVERT))
        _set_scenario(small_test_copy, duration=120)
        result, output = _run_cli(small_test_copy)
        assert result.returncode == 0, output[-3000:]
        [output_dir] = small_test_copy.glob("outputs_*")
        assert list(output_dir.glob("*.sww"))
        document = json.loads((output_dir / "culverts" / "metrics.json").read_text())
        assert document["schema_version"] == 1
        [entry] = document["culverts"]
        assert set(entry) == CONTRACT_KEYS
        assert entry["feature_id"] == 101 and entry["name"] == "Small test box"
        assert entry["triangles_inlet"] >= 4 and entry["triangles_outlet"] >= 4
        for key in HYDRAULIC_KEYS:
            assert entry[key] is not None, key
        assert math.isfinite(entry["peak_q_m3s"]) and abs(entry["peak_q_m3s"]) > 0.0  # the coupler moved water
        assert entry["regime_at_peak"] in ("barrel", "inlet", "reverse-inlet")
        assert math.isclose(sum(entry["regime_fractions"].values()), 1.0, abs_tol=1e-3)
        assert any("coarse at the" in w for w in entry["warnings"])
        assert "swmm_routing_error_pct" in document

        series_path = output_dir / entry["timeseries_path"]
        with open(series_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 2
        assert list(rows[0]) == ["time_s", "q_m3s", "hw_level_m", "tw_level_m", "control"]
        assert float(rows[-1]["time_s"]) > float(rows[0]["time_s"])
        assert any(float(row["q_m3s"]) != 0.0 for row in rows)
        run_log = "\n".join(p.read_text() for p in output_dir.glob("run_anuga_*.log")) + output
        assert "culvert coupler: Small test box" in run_log

    def test_culvert_e2e_no_culvert_layer_means_no_culvert_outputs(self, small_test_copy):
        """AC3: the same scenario without the layer (and no requires entry) writes no culvert metrics."""
        _set_scenario(small_test_copy, duration=60)
        scenario = json.loads((small_test_copy / "scenario.json").read_text())
        assert not scenario.get("culvert") and not scenario.get("requires")
        result, output = _run_cli(small_test_copy)
        assert result.returncode == 0, output[-3000:]
        [output_dir] = small_test_copy.glob("outputs_*")
        assert list(output_dir.glob("*.sww"))
        assert not (output_dir / "culverts").exists()
        assert "culvert coupler" not in output

    def test_culvert_e2e_end_inside_a_reflective_building_stops_the_run(self, small_test_copy):
        """A placement failure: plain message, non-zero exit, before the mesh."""
        _add_culvert_layer(small_test_copy, collection(SMALL_TEST_CULVERT))
        (small_test_copy / "inputs" / "building.geojson").write_text(json.dumps(collection(building(
            "Reflective", [[251095, 6271895], [251110, 6271895], [251110, 6271910], [251095, 6271910],
                           [251095, 6271895]], fid="shed"))))
        _set_scenario(small_test_copy, building="building.geojson", structure=None, duration=60)
        result, output = _run_cli(small_test_copy)
        assert result.returncode != 0
        assert "is inside Reflective building shed" in output
        assert not list(small_test_copy.glob("outputs_*/*.msh"))
