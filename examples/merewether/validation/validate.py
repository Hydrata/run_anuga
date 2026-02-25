#!/usr/bin/env python3
"""
Validate Merewether simulation against ARR benchmark observation points.

Reads the SWW output from outputs_1_1_1/ and compares peak stage at each
of the 5 field observation points against the recorded field data.

Usage:
    python examples/merewether/validation/validate.py

Exit codes:
    0 — all points within tolerance
    1 — one or more points outside tolerance
"""

import json
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent.parent

OBS_FILE = HERE / "observation_points.geojson"
SWW_FILE = REPO_ROOT / "examples/merewether/outputs_1_1_1/run_1_1_1.sww"


def find_nearest_vertex_idx(xs, ys, qx, qy):
    """Return index of vertex nearest to (qx, qy)."""
    dist2 = (xs - qx) ** 2 + (ys - qy) ** 2
    return int(np.argmin(dist2))


def main():
    if not SWW_FILE.exists():
        print(f"ERROR: SWW file not found: {SWW_FILE}")
        sys.exit(1)

    # --- Load observation points ---
    with open(OBS_FILE) as f:
        obs_gj = json.load(f)

    # --- Load SWW file ---
    ds = nc.Dataset(SWW_FILE)
    xs = np.array(ds.variables["x"][:], dtype=float)
    ys = np.array(ds.variables["y"][:], dtype=float)
    stage = np.array(ds.variables["stage"][:], dtype=float)  # (ntimes, nvertices)
    elevation = np.array(ds.variables["elevation"][:], dtype=float)  # (nvertices,)
    times = np.array(ds.variables["time"][:], dtype=float)
    ds.close()

    # Peak stage over all timesteps at each vertex
    peak_stage = np.max(stage, axis=0)

    # --- Validate each point ---
    print(f"\nMerewether validation: {len(obs_gj['features'])} observation points")
    print(f"SWW: {SWW_FILE.relative_to(REPO_ROOT)}  (t=0..{times[-1]:.0f}s, {len(times)} steps)")
    print()
    print(f"{'ID':>3}  {'x':>10}  {'y':>13}  {'field_m':>7}  {'sim_m':>7}  {'diff_m':>7}  {'tol_m':>5}  {'OK?':>4}")
    print("-" * 70)

    all_pass = True
    results = []
    for feat in obs_gj["features"]:
        props = feat["properties"]
        obs_id = props["id"]
        tol = props["tolerance_m"]
        field_stage = props["field_stage_m"]
        qx, qy = feat["geometry"]["coordinates"]

        idx = find_nearest_vertex_idx(xs, ys, qx, qy)
        nearest_dist = float(np.sqrt((xs[idx] - qx) ** 2 + (ys[idx] - qy) ** 2))
        sim_stage = float(peak_stage[idx])
        diff = sim_stage - field_stage
        passed = abs(diff) <= tol

        if not passed:
            all_pass = False

        status = "PASS" if passed else "FAIL"
        print(
            f"{obs_id:>3}  {qx:>10.1f}  {qy:>13.1f}  {field_stage:>7.3f}  "
            f"{sim_stage:>7.3f}  {diff:>+7.3f}  {tol:>5.2f}  {status:>4}"
            f"  (nearest vertex {nearest_dist:.1f}m away)"
        )
        results.append({
            "id": obs_id, "field_m": field_stage, "sim_m": sim_stage,
            "diff_m": diff, "tol_m": tol, "pass": passed,
        })

    print()
    n_pass = sum(1 for r in results if r["pass"])
    print(f"Result: {n_pass}/{len(results)} points within ±tolerance")

    if all_pass:
        print("VALIDATION PASSED")
        sys.exit(0)
    else:
        print("VALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
