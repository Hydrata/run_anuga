"""
End-to-end test for the Merewether urban flood benchmark.

Requires ANUGA. Runs the full simulation and validates peak stage at
5 ARR field observation points within ±0.3 m tolerance.

Run with:
    pytest -m requires_anuga -k merewether
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

MEREWETHER_DIR = Path(__file__).parent.parent / "examples" / "merewether"


@pytest.fixture
def merewether_copy(tmp_path):
    """Isolated copy of examples/merewether (outputs excluded)."""
    dst = tmp_path / "merewether"
    shutil.copytree(MEREWETHER_DIR, dst, ignore=shutil.ignore_patterns("outputs_*"))
    return dst


@pytest.mark.requires_anuga
@pytest.mark.slow
def test_merewether_completes_and_validates(merewether_copy):
    from run_anuga.callbacks import NullCallback
    from run_anuga.run import run_sim

    run_sim(str(merewether_copy), callback=NullCallback())

    # ── Diagnostics ──────────────────────────────────────────────────────
    summary_path = merewether_copy / "outputs_1_1_1" / "run_summary_1.json"
    assert summary_path.exists(), "run_summary_1.json not written"
    summary = json.loads(summary_path.read_text())
    assert summary["run"]["outcome"] == "completed", summary["run"]["outcome"]
    assert summary["stability"]["stable"] is True, (
        f"max_implied={summary['stability']['max_implied_speed_ms']} m/s"
    )

    # ── ARR Validation ───────────────────────────────────────────────────
    import netCDF4 as nc

    sww_path = merewether_copy / "outputs_1_1_1" / "run_1_1_1.sww"
    obs_path = merewether_copy / "validation" / "observation_points.geojson"
    obs = json.loads(obs_path.read_text())

    ds = nc.Dataset(str(sww_path))
    xs = np.array(ds.variables["x"][:], dtype=float)
    ys = np.array(ds.variables["y"][:], dtype=float)
    stage = np.array(ds.variables["stage"][:], dtype=float)
    ds.close()

    peak_stage = np.max(stage, axis=0)

    failures = []
    for feat in obs["features"]:
        props = feat["properties"]
        qx, qy = feat["geometry"]["coordinates"]
        idx = int(np.argmin((xs - qx) ** 2 + (ys - qy) ** 2))
        diff = abs(float(peak_stage[idx]) - props["field_stage_m"])
        if diff > props["tolerance_m"]:
            failures.append(
                f"ID {props['id']}: diff={diff:.3f}m > tol={props['tolerance_m']}m"
            )

    assert not failures, "ARR validation failed:\n" + "\n".join(failures)
