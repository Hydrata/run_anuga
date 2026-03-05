#!/usr/bin/env python3
"""Run Merewether mesh library experiments via run-anuga CLI.

Discovers scenarios under meshers/<mesher>/<method>/scenario.json and runs
them with MPI (rank count from experiment.json).

Usage:
    python run_experiments.py                      # all scenarios
    python run_experiments.py --dry-run             # list without running
    python run_experiments.py --mesher triangle     # filter by mesher
    python run_experiments.py --method burn         # filter by method
    python run_experiments.py --timeout 600         # per-run timeout
    python run_experiments.py --clean               # remove outputs first
"""

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MESHERS_DIR = HERE / "meshers"
EXPERIMENT_JSON = HERE / "experiment.json"


def load_experiment_config():
    """Load experiment.json for fixed parameters."""
    with open(EXPERIMENT_JSON) as f:
        return json.load(f)


def find_run_anuga():
    """Find the run-anuga CLI using the current Python interpreter."""
    venv_bin = Path(sys.executable).parent
    exe = venv_bin / "run-anuga"
    if exe.exists():
        return str(exe)
    return None


def discover_scenarios(mesher_filter=None, method_filter=None):
    """Discover scenarios under meshers/<mesher>/<method>/scenario.json."""
    scenarios = []
    for mesher_dir in sorted(MESHERS_DIR.iterdir()):
        if not mesher_dir.is_dir():
            continue
        mesher_name = mesher_dir.name
        if mesher_filter and mesher_name != mesher_filter:
            continue
        for method_dir in sorted(mesher_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method_name = method_dir.name
            if method_filter and method_name != method_filter:
                continue
            scenario_json = method_dir / "scenario.json"
            if scenario_json.exists():
                scenarios.append({
                    "mesher": mesher_name,
                    "method": method_name,
                    "dir": method_dir,
                    "scenario_json": scenario_json,
                })
    return scenarios


def clean_outputs(scenario_dir):
    """Remove output directories from a previous run."""
    for output_dir in scenario_dir.glob("outputs_*"):
        shutil.rmtree(output_dir)
        print(f"    cleaned {output_dir.name}")


def find_run_summary(scenario_dir):
    """Find the run_summary_*.json after a run completes."""
    for output_dir in sorted(scenario_dir.glob("outputs_*")):
        for f in output_dir.glob("run_summary_*.json"):
            return f
    return None


def run_scenario(scenario_dir, mpi_ranks, timeout_s):
    """Run a single scenario, optionally with MPI. Returns result dict."""
    run_anuga = find_run_anuga()
    if run_anuga is None:
        return {"outcome": "error", "error": "run-anuga CLI not found"}

    if mpi_ranks and mpi_ranks > 1:
        cmd = [
            "mpirun", "-np", str(mpi_ranks),
            run_anuga, "run", str(scenario_dir),
        ]
    else:
        cmd = [run_anuga, "run", str(scenario_dir)]

    print(f"    cmd: {' '.join(cmd)}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        wall_time = time.time() - t0
        outcome = "completed" if result.returncode == 0 else "failed"
        return {
            "outcome": outcome,
            "wall_time_s": round(wall_time, 1),
            "returncode": result.returncode,
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        wall_time = time.time() - t0
        return {
            "outcome": "timeout",
            "wall_time_s": round(wall_time, 1),
            "timeout_s": timeout_s,
        }


def load_run_summary_metrics(scenario_dir):
    """Load metrics from run_summary_*.json if present."""
    summary_path = find_run_summary(scenario_dir)
    if summary_path is None:
        return None
    with open(summary_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run Merewether mesh library experiments."
    )
    parser.add_argument(
        "--mesher", help="Run only scenarios for this mesher (e.g. 'triangle')."
    )
    parser.add_argument(
        "--method", help="Run only scenarios for this method (e.g. 'burn', 'holes')."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List scenarios without running them."
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove previous outputs before running."
    )
    parser.add_argument(
        "--timeout", type=int, default=1800,
        help="Per-run timeout in seconds (default: 1800)."
    )
    parser.add_argument(
        "--no-mpi", action="store_true",
        help="Run without MPI (single process)."
    )
    args = parser.parse_args()

    config = load_experiment_config()
    mpi_ranks = 0 if args.no_mpi else config["fixed_params"].get("mpi_ranks", 1)

    scenarios = discover_scenarios(args.mesher, args.method)
    if not scenarios:
        print("No matching scenarios found.")
        sys.exit(1)

    print(f"Experiment: {config['name']}")
    print(f"MPI ranks: {mpi_ranks or 'disabled'}")
    print(f"Timeout: {args.timeout}s per run")
    print(f"\nFound {len(scenarios)} scenario(s):")
    for s in scenarios:
        with open(s["scenario_json"]) as f:
            cfg = json.load(f)
        print(
            f"  {s['mesher']}/{s['method']:10s}"
            f"  res={cfg.get('resolution')}  angle={cfg.get('minimum_triangle_angle')}"
            f"  struct={'yes' if cfg.get('structure') else 'no'}"
        )

    if args.dry_run:
        return

    results = []
    for i, scenario in enumerate(scenarios, 1):
        label = f"{scenario['mesher']}/{scenario['method']}"
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(scenarios)}] {label}")
        print(f"{'=' * 60}")

        if args.clean:
            clean_outputs(scenario["dir"])

        run_result = run_scenario(scenario["dir"], mpi_ranks, args.timeout)

        # Load run_summary metrics if run completed
        summary_data = None
        if run_result["outcome"] == "completed":
            summary_data = load_run_summary_metrics(scenario["dir"])

        entry = {
            "mesher": scenario["mesher"],
            "method": scenario["method"],
            **run_result,
        }

        if summary_data:
            entry["run_summary"] = summary_data
            wall = summary_data.get("run", {}).get("total_wall_time_s")
            tri = summary_data.get("mesh", {}).get("n_triangles")
            stable = summary_data.get("stability", {}).get("stable")
            print(f"    -> {run_result['outcome']}, {wall}s, {tri} triangles, stable={stable}")
        else:
            print(f"    -> {run_result['outcome']}, {run_result.get('wall_time_s', '?')}s")
            if run_result.get("stderr_tail"):
                print(f"    stderr: {run_result['stderr_tail'][-300:]}")

        results.append(entry)

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"{'Mesher':10s}  {'Method':10s}  {'Time':>8s}  {'Triangles':>10s}  {'Outcome':>10s}")
    print("-" * 55)
    n_ok = 0
    for r in results:
        s = r.get("run_summary", {})
        wall = s.get("run", {}).get("total_wall_time_s", r.get("wall_time_s", "?"))
        tri = s.get("mesh", {}).get("n_triangles", "?")
        outcome = r["outcome"]
        if outcome == "completed":
            n_ok += 1
        print(f"{r['mesher']:10s}  {r['method']:10s}  {str(wall):>7s}s  {str(tri):>10s}  {outcome:>10s}")
    print(f"\n{n_ok}/{len(results)} experiments completed successfully")


if __name__ == "__main__":
    main()
