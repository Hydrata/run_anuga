"""
End-to-end test for the Merewether urban flood benchmark.

Requires ANUGA. Runs the full simulation and validates peak stage at
5 ARR field observation points within ±0.3 m tolerance.

TASK-2186 adds the origin-shift case: the same mesh, rigidly translated 0.5 m
against the 1 m DEM, must still pass ARR and move peak stage at each point by
less than 0.05 m (footprint-integrated bed seating, not a centroid coin flip).

Run with:
    pytest -m requires_anuga -k merewether
"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MEREWETHER_DIR = Path(__file__).parent.parent / "examples" / "merewether"

# TASK-2186 AC4: the mesh origin moves 0.5 m (along the diagonal, so both axes
# cross DEM cell edges) and peak stage at each ARR point may move less than this.
ORIGIN_SHIFT_M = 0.5
ORIGIN_SHIFT_PEAK_STAGE_TOL_M = 0.05


def _copy_merewether(dst):
    shutil.copytree(MEREWETHER_DIR, dst, ignore=shutil.ignore_patterns("outputs_*"))
    return dst


@pytest.fixture
def merewether_copy(tmp_path):
    """Isolated copy of examples/merewether (outputs excluded)."""
    return _copy_merewether(tmp_path / "merewether")


def _prebuild_shifted_mesh(package_dir, shift_m):
    """Build the Merewether mesh, then move its geo_reference origin ``shift_m`` along the diagonal.

    The triangles keep their relative coordinates, so the tiling is identical and every
    triangle moves rigidly against the DEM grid; run_sim finds the .msh and skips meshing.
    """
    import netCDF4 as nc
    from run_anuga.run_utils import create_anuga_mesh, setup_input_data

    input_data = setup_input_data(str(package_dir))
    mesh_filepath, _ = create_anuga_mesh(input_data)
    step = shift_m / math.sqrt(2.0)
    with nc.Dataset(mesh_filepath, "a") as ds:
        ds.xllcorner = float(ds.xllcorner) + step
        ds.yllcorner = float(ds.yllcorner) + step
    return mesh_filepath


def _run_and_read_peak_stages(package_dir):
    """Run the package, assert it completed stably, return {point id: (peak stage, props)}.

    Each run is its own process (``python -m run_anuga.cli run``, as test_culvert_e2e):
    run_sim finalizes MPI on the way out and aborts COMM_WORLD on failure, so a second
    in-process run_sim (the origin-shift pair) kills pytest with no output.
    """
    import netCDF4 as nc

    result = subprocess.run([sys.executable, "-m", "run_anuga.cli", "run", str(package_dir)],
                            capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]

    # ── Diagnostics ──────────────────────────────────────────────────────
    summary_path = package_dir / "outputs_1_1_1" / "run_summary_1.json"
    assert summary_path.exists(), "run_summary_1.json not written"
    summary = json.loads(summary_path.read_text())
    assert summary["run"]["outcome"] == "completed", summary["run"]["outcome"]
    assert summary["stability"]["stable"] is True, (
        f"max_implied={summary['stability']['max_implied_speed_ms']} m/s"
    )

    sww_path = package_dir / "outputs_1_1_1" / "run_1_1_1.sww"
    obs_path = package_dir / "validation" / "observation_points.geojson"
    obs = json.loads(obs_path.read_text())

    ds = nc.Dataset(str(sww_path))
    # SWW x/y are relative to the file's geo_reference; the gauges are absolute UTM.
    xs = np.array(ds.variables["x"][:], dtype=float) + float(getattr(ds, "xllcorner", 0.0))
    ys = np.array(ds.variables["y"][:], dtype=float) + float(getattr(ds, "yllcorner", 0.0))
    stage = np.array(ds.variables["stage"][:], dtype=float)
    ds.close()

    peak_stage = np.max(stage, axis=0)
    peaks = {}
    for feat in obs["features"]:
        props = feat["properties"]
        qx, qy = feat["geometry"]["coordinates"]
        idx = int(np.argmin((xs - qx) ** 2 + (ys - qy) ** 2))
        peaks[props["id"]] = (float(peak_stage[idx]), props)
    return peaks


def _arr_failures(peaks):
    failures = []
    for point_id, (peak, props) in peaks.items():
        diff = abs(peak - props["field_stage_m"])
        if diff > props["tolerance_m"]:
            failures.append(
                f"ID {point_id}: diff={diff:.3f}m > tol={props['tolerance_m']}m"
            )
    return failures


@pytest.mark.requires_anuga
@pytest.mark.slow
def test_merewether_completes_and_validates(merewether_copy):
    # ── ARR Validation ───────────────────────────────────────────────────
    failures = _arr_failures(_run_and_read_peak_stages(merewether_copy))
    assert not failures, "ARR validation failed:\n" + "\n".join(failures)


@pytest.mark.requires_anuga
@pytest.mark.slow
def test_merewether_origin_shift_within_tolerance(tmp_path):
    """TASK-2186 AC4: a 0.5 m mesh-origin shift keeps ARR and moves peak stage < 0.05 m."""
    base = _run_and_read_peak_stages(_copy_merewether(tmp_path / "unshifted"))
    shifted_dir = _copy_merewether(tmp_path / "shifted")
    _prebuild_shifted_mesh(shifted_dir, ORIGIN_SHIFT_M)
    shifted = _run_and_read_peak_stages(shifted_dir)

    deltas = {pid: shifted[pid][0] - base[pid][0] for pid in base}
    table = "\n".join(
        f"ID {pid}: unshifted {base[pid][0]:.3f} m, shifted {shifted[pid][0]:.3f} m "
        f"(field {base[pid][1]['field_stage_m']} m), delta {deltas[pid]:+.3f} m"
        for pid in sorted(base)
    )
    failures = [f"shifted {f}" for f in _arr_failures(shifted)] + [
        f"ID {pid}: peak stage moved {d:+.3f} m (>= {ORIGIN_SHIFT_PEAK_STAGE_TOL_M} m)"
        for pid, d in sorted(deltas.items()) if abs(d) >= ORIGIN_SHIFT_PEAK_STAGE_TOL_M
    ]
    assert not failures, (
        f"{ORIGIN_SHIFT_M} m mesh-origin shift:\n" + "\n".join(failures) + "\n" + table
    )
