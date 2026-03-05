#!/usr/bin/env python3
"""Collect Merewether experiment results from run_summary_*.json + SWW validation.

Reads metrics from run_summary_*.json (mesh quality, performance, stability)
and validates peak stage at 5 observation points from the SWW file.

Usage:
    python collect_results.py               # all scenarios
    python collect_results.py --mesher triangle --method burn  # filter
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

HERE = Path(__file__).resolve().parent
MESHERS_DIR = HERE / "meshers"
VALIDATION_DIR = HERE / "validation"
OBS_CSV = VALIDATION_DIR / "observed_points.csv"


def load_obs_points():
    """Load observation points from CSV."""
    points = []
    with open(OBS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            points.append({
                "id": int(row["id"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "field_stage_m": float(row["field_stage_m"]),
                "tolerance_m": float(row["tolerance_m"]),
            })
    return points


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
                })
    return scenarios


def find_run_summary(scenario_dir):
    """Find run_summary_*.json in output directory."""
    for output_dir in sorted(scenario_dir.glob("outputs_*")):
        for f in output_dir.glob("run_summary_*.json"):
            return f
    return None


def find_sww(scenario_dir):
    """Find the SWW file in a scenario's output directory."""
    for output_dir in sorted(scenario_dir.glob("outputs_*")):
        for sww_file in output_dir.glob("*.sww"):
            return sww_file
    return None


def validate_sww(sww_path, obs_points):
    """Validate peak stage at observation points. Returns validation dict."""
    ds = nc.Dataset(sww_path)
    xs = np.array(ds.variables["x"][:], dtype=float)
    ys = np.array(ds.variables["y"][:], dtype=float)
    stage = np.array(ds.variables["stage"][:], dtype=float)
    ds.close()

    peak_stage = np.max(stage, axis=0)

    validations = []
    for pt in obs_points:
        dist2 = (xs - pt["x"]) ** 2 + (ys - pt["y"]) ** 2
        idx = int(np.argmin(dist2))
        nearest_dist = float(np.sqrt(dist2[idx]))
        sim_stage = float(peak_stage[idx])
        diff = sim_stage - pt["field_stage_m"]
        passed = abs(diff) <= pt["tolerance_m"]
        validations.append({
            "id": pt["id"],
            "field_m": pt["field_stage_m"],
            "sim_m": round(sim_stage, 3),
            "diff_m": round(diff, 3),
            "tol_m": pt["tolerance_m"],
            "nearest_dist_m": round(nearest_dist, 2),
            "pass": passed,
        })

    n_pass = sum(1 for v in validations if v["pass"])
    errors = [abs(v["diff_m"]) for v in validations]
    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))

    return {
        "n_pass": n_pass,
        "n_total": len(validations),
        "rmse": round(rmse, 4),
        "validations": validations,
    }


def extract_metrics(summary_data):
    """Extract flat metrics from run_summary_*.json."""
    run = summary_data.get("run", {})
    mesh = summary_data.get("mesh", {})
    perf = summary_data.get("performance", {})
    stab = summary_data.get("stability", {})

    return {
        "outcome": run.get("outcome"),
        "wall_time_s": run.get("total_wall_time_s"),
        "n_triangles": mesh.get("n_triangles"),
        "inradius_min_m": mesh.get("inradius_min_m"),
        "inradius_p5_m": mesh.get("inradius_p5_m"),
        "min_angle_deg": mesh.get("min_angle_deg"),
        "total_steps": perf.get("total_internal_steps"),
        "min_dt_ms": perf.get("min_dt_ms"),
        "peak_mem_mb": perf.get("peak_mem_mb"),
        "max_implied_speed": stab.get("max_implied_speed_ms"),
        "stable": stab.get("stable"),
    }


def collect_all(mesher_filter=None, method_filter=None):
    """Collect results from all scenario directories."""
    obs_points = load_obs_points()
    scenarios = discover_scenarios(mesher_filter, method_filter)
    results = []

    for scenario in scenarios:
        label = f"{scenario['mesher']}/{scenario['method']}"
        row = {
            "mesher": scenario["mesher"],
            "method": scenario["method"],
            "label": label,
        }

        # Read run_summary_*.json
        summary_path = find_run_summary(scenario["dir"])
        if summary_path:
            with open(summary_path) as f:
                summary_data = json.load(f)
            row.update(extract_metrics(summary_data))
        else:
            row["outcome"] = "no_output"

        # Validate SWW
        sww_path = find_sww(scenario["dir"])
        if sww_path:
            try:
                val = validate_sww(sww_path, obs_points)
                row.update({
                    "n_pass": val["n_pass"],
                    "n_total": val["n_total"],
                    "rmse": val["rmse"],
                    "validations": val["validations"],
                })
            except Exception as e:
                print(f"    WARNING: SWW validation failed for {label}: {e}")
                row["n_pass"] = None
                row["rmse"] = None
        else:
            row["n_pass"] = None
            row["rmse"] = None

        results.append(row)
        status_str = f"{row.get('n_pass', '?')}/{row.get('n_total', '?')}" if sww_path else row.get("outcome", "?")
        print(f"  {label:25s}  {status_str}  RMSE={row.get('rmse', 'N/A')}  tri={row.get('n_triangles', '?')}  [{row.get('outcome', '?')}]")

    return results


def write_csv(results, path):
    """Write results to CSV."""
    if not results:
        return
    columns = [k for k in results[0].keys() if k != "validations"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in columns})
    print(f"CSV written: {path}")


def write_html(results, path):
    """Write styled HTML comparison report."""
    cols = [
        ("Mesher", "mesher"),
        ("Method", "method"),
        ("Outcome", "outcome"),
        ("Triangles", "n_triangles"),
        ("Min Inradius (m)", "inradius_min_m"),
        ("P5 Inradius (m)", "inradius_p5_m"),
        ("Min Angle (\u00b0)", "min_angle_deg"),
        ("Wall Time (s)", "wall_time_s"),
        ("Total Steps", "total_steps"),
        ("Min dt (ms)", "min_dt_ms"),
        ("Peak Mem (MB)", "peak_mem_mb"),
        ("Implied Speed", "max_implied_speed"),
        ("Stable", "stable"),
        ("Pass", "_pass_str"),
        ("RMSE (m)", "rmse"),
    ]

    for r in results:
        if r.get("n_pass") is not None:
            r["_pass_str"] = f"{r['n_pass']}/{r['n_total']}"
        else:
            r["_pass_str"] = "-"

    rows_html = ""
    for r in results:
        cells = ""
        for _, key in cols:
            val = r.get(key, "")
            if val is None:
                val = "-"

            css = ""
            if key == "outcome":
                css = ' class="pass"' if val == "completed" else ' class="fail"'
            elif key == "stable":
                css = ' class="pass"' if val is True else ' class="fail"' if val is False else ""
                val = "yes" if val is True else "no" if val is False else val
            elif key == "_pass_str" and r.get("n_pass") is not None:
                css = ' class="pass"' if r["n_pass"] == r["n_total"] else ' class="fail"'
            elif key == "rmse" and isinstance(val, (int, float)):
                css = ' class="pass"' if val < 0.3 else ' class="warn"' if val < 0.5 else ' class="fail"'
            elif key == "max_implied_speed" and isinstance(val, (int, float)):
                css = ' class="pass"' if val < 20 else ' class="warn"' if val < 50 else ' class="fail"'

            if isinstance(val, float):
                val = f"{val:.2f}" if val >= 1 else f"{val:.4f}"

            cells += f"      <td{css}>{val}</td>\n"
        rows_html += f"    <tr>\n{cells}    </tr>\n"

    # Per-point validation detail
    detail_html = ""
    for r in results:
        if "validations" not in r:
            continue
        detail_html += f'<h3>{r["label"]}</h3>\n<table class="detail">\n'
        detail_html += "<tr><th>Pt</th><th>Field (m)</th><th>Sim (m)</th><th>Diff (m)</th><th>Tol (m)</th><th>Dist (m)</th><th>Pass</th></tr>\n"
        for v in r["validations"]:
            css = ' class="pass"' if v["pass"] else ' class="fail"'
            detail_html += (
                f'<tr><td>{v["id"]}</td><td>{v["field_m"]}</td><td>{v["sim_m"]}</td>'
                f'<td{css}>{v["diff_m"]:+.3f}</td><td>{v["tol_m"]}</td>'
                f'<td>{v["nearest_dist_m"]}</td><td{css}>{"PASS" if v["pass"] else "FAIL"}</td></tr>\n'
            )
        detail_html += "</table>\n"

    header_cells = "".join(f"<th>{name}</th>" for name, _ in cols)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Merewether Mesh Library Experiment Results</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2em; background: #f5f5f5; }}
  h1 {{ color: #1a365d; border-bottom: 3px solid #2b6cb0; padding-bottom: 0.3em; }}
  h2 {{ color: #2d3748; margin-top: 2em; }}
  h3 {{ color: #4a5568; margin-top: 1.5em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
  th {{ background: #1a365d; color: white; padding: 8px 12px; text-align: left; font-size: 0.85em; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid #e2e8f0; font-size: 0.85em; }}
  tr:hover {{ background: #f7fafc; }}
  .pass {{ color: #276749; font-weight: 600; }}
  .warn {{ color: #c05621; font-weight: 600; }}
  .fail {{ color: #c53030; font-weight: 600; }}
  .detail {{ width: auto; }}
  .detail th {{ font-size: 0.8em; padding: 4px 8px; }}
  .detail td {{ font-size: 0.8em; padding: 3px 8px; }}
  .meta {{ color: #718096; font-size: 0.85em; margin-top: 2em; }}
</style>
</head>
<body>
<h1>Merewether Mesh Library Experiment Results</h1>

<h2>Summary</h2>
<p>Metrics from <code>run_summary_*.json</code>. Validation from SWW peak stage vs 5 observation points (\u00b10.3m tolerance).</p>
<table>
  <tr>{header_cells}</tr>
{rows_html}</table>

<h2>Per-Point Validation Detail</h2>
{detail_html}

<p class="meta">Generated by collect_results.py</p>
</body>
</html>
"""
    path.write_text(html)
    print(f"HTML report written: {path}")


def main():
    parser = argparse.ArgumentParser(description="Collect Merewether experiment results.")
    parser.add_argument("--mesher", help="Filter by mesher name.")
    parser.add_argument("--method", help="Filter by method name.")
    args = parser.parse_args()

    print("Collecting Merewether experiment results...\n")
    results = collect_all(args.mesher, args.method)

    if not results:
        print("No scenarios found.")
        sys.exit(1)

    csv_path = HERE / "results.csv"
    html_path = HERE / "experiment_report.html"
    write_csv(results, csv_path)
    write_html(results, html_path)

    completed = [r for r in results if r.get("outcome") == "completed"]
    all_pass = [r for r in completed if r.get("n_pass") == r.get("n_total")]
    print(f"\n{len(completed)}/{len(results)} completed, {len(all_pass)} with all points passing")


if __name__ == "__main__":
    main()
