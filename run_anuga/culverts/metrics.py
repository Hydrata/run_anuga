"""culverts/metrics.json writer — the PINNED hydrata <-> run_anuga contract (TASK-3638).

Written to ``<run output dir>/culverts/metrics.json`` (next to the SWW). The
mesh fields (triangles per end, coarseness ratio) and the blockage inputs are
filled now; the hydraulic fields are null until the coupling engine lands
(W3, TASK-3642), and so is ``timeseries_path``.
"""

from __future__ import annotations

import json
import os

from run_anuga.culverts import constants

#: Hydraulic result keys, all null from the W2b stub operator.
HYDRAULIC_KEYS = (
    "peak_hw_level_m", "peak_hw_depth_m", "peak_hw_time_s", "hw_over_d", "peak_tw_level_m",
    "peak_q_m3s", "peak_q_time_s", "barrel_velocity_ms", "regime_at_peak", "regime_fractions",
    "surcharge_s",
)


def culvert_metrics_entry(result) -> dict:
    culvert = result.culvert
    entry = {
        "feature_id": culvert.feature_id,
        "name": culvert.name,
        "triangles_inlet": int(result.inlet_region.triangle_count),
        "triangles_outlet": int(result.outlet_region.triangle_count),
        "coarseness_ratio": round(float(result.coarseness_ratio), 4),
        "blockage_pct": float(culvert.blockage_pct),
        "blockage_method": constants.BLOCKAGE_METHOD,
    }
    entry.update({key: None for key in HYDRAULIC_KEYS})
    entry["timeseries_path"] = None
    return entry


def metrics_document(results) -> dict:
    return {
        "schema_version": constants.METRICS_SCHEMA_VERSION,
        "culverts": [culvert_metrics_entry(result) for result in results],
    }


def metrics_path(output_directory) -> str:
    return os.path.join(output_directory, constants.METRICS_DIRNAME, constants.METRICS_FILENAME)


def write_culvert_metrics(output_directory, results) -> str:
    """Write the metrics document; returns its path."""
    path = metrics_path(output_directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(metrics_document(results), f, indent=2)
    os.replace(tmp, path)
    return path
