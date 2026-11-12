"""
End-to-end test for the Merewether urban flood benchmark.

Requires ANUGA. Runs the full simulation and validates peak stage at
5 ARR field observation points within ±0.3 m tolerance.

TASK-2186 adds the origin-shift case: the same mesh, rigidly translated 0.5 m
against the 1 m DEM, must pass ARR on both runs, with peak stage read at the
containing triangle's engine max_stage_c (operator Decision 776, 12-11-26: a
per-point stage bound is unattainable beside Merewether's 3 m walls for any
seating; the continuity guarantee lives in tests/test_elevation_sampling.py).

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
# cross DEM cell edges); both runs must still pass ARR.
ORIGIN_SHIFT_M = 0.5


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


def _containing_triangles(xs, ys, volumes, points):
    """Index of the triangle containing each (x, y) point (barycentric test)."""
    ax, ay = xs[volumes[:, 0]], ys[volumes[:, 0]]
    bx, by = xs[volumes[:, 1]], ys[volumes[:, 1]]
    cx, cy = xs[volumes[:, 2]], ys[volumes[:, 2]]
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    out = []
    for qx, qy in points:
        l1 = ((by - cy) * (qx - cx) + (cx - bx) * (qy - cy)) / det
        l2 = ((cy - ay) * (qx - cx) + (ax - cx) * (qy - cy)) / det
        inside = np.flatnonzero((l1 >= -1e-9) & (l2 >= -1e-9) & (1.0 - l1 - l2 >= -1e-9))
        assert inside.size, f"point ({qx}, {qy}) lies in no triangle"
        out.append(int(inside[0]))
    return out


def _run_and_read_peak_stages(package_dir, reader="node"):
    """Run the package, assert it completed stably, return {point id: (peak stage, props)}.

    ``reader="node"`` takes the max over frames of stage at the nearest smoothed SWW
    node (the original ARR reader); ``reader="triangle"`` takes the engine's per-step
    ``max_stage_c`` of the triangle containing the point (TASK-2186 AC4, Decision 776).

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
    if reader == "triangle":
        from run_anuga.peaks import engine_peaks_from_dataset

        engine = engine_peaks_from_dataset(ds)
        volumes = np.asarray(ds.variables["volumes"][:], dtype=np.int64)
        ds.close()
        assert engine is not None, "SWW has no engine max_stage_c (true peaks off?)"
        feats = obs["features"]
        tris = _containing_triangles(xs, ys, volumes, [f["geometry"]["coordinates"] for f in feats])
        return {f["properties"]["id"]: (float(engine["max_stage_c"][t]), f["properties"])
                for f, t in zip(feats, tris)}

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
    """TASK-2186 AC4 (Decision 776): ARR passes on both the unshifted and a 0.5 m-shifted run.

    Peak stage is read at the containing triangle's max_stage_c. The deltas are reported,
    not bounded: beside a 3 m wall the bed itself moves with the shift under any seating.
    """
    base = _run_and_read_peak_stages(_copy_merewether(tmp_path / "unshifted"), reader="triangle")
    shifted_dir = _copy_merewether(tmp_path / "shifted")
    _prebuild_shifted_mesh(shifted_dir, ORIGIN_SHIFT_M)
    shifted = _run_and_read_peak_stages(shifted_dir, reader="triangle")

    deltas = {pid: shifted[pid][0] - base[pid][0] for pid in base}
    table = "\n".join(
        f"ID {pid}: unshifted {base[pid][0]:.3f} m, shifted {shifted[pid][0]:.3f} m "
        f"(field {base[pid][1]['field_stage_m']} m), delta {deltas[pid]:+.3f} m"
        for pid in sorted(base)
    )
    failures = ([f"unshifted {f}" for f in _arr_failures(base)]
                + [f"shifted {f}" for f in _arr_failures(shifted)])
    print(f"{ORIGIN_SHIFT_M} m mesh-origin shift (containing-triangle max_stage_c):\n" + table)
    assert not failures, (
        f"{ORIGIN_SHIFT_M} m mesh-origin shift:\n" + "\n".join(failures) + "\n" + table
    )
