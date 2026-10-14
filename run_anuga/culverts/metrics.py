"""culverts/metrics.json writer — the PINNED hydrata <-> run_anuga contract (TASK-3638).

Written to ``<run output dir>/culverts/metrics.json`` (next to the SWW). The
mesh fields (triangles per end, coarseness ratio), the blockage inputs and the
warnings are written at build time; the hydraulic fields and ``timeseries_path``
are null until the coupler's results overwrite them after the run
(``outputs.write_culvert_outputs``, TASK-3642).
"""

from __future__ import annotations

import json
import os

from run_anuga.culverts import constants

#: Hydraulic result keys; null in the build-time document, filled after the run.
HYDRAULIC_KEYS = (
    "peak_hw_level_m", "peak_hw_depth_m", "peak_hw_time_s", "hw_over_d", "peak_tw_level_m",
    "peak_q_m3s", "peak_q_time_s", "barrel_velocity_ms", "regime_at_peak", "regime_fractions",
    "surcharge_s",
)


def culvert_metrics_entry(result, hydraulics=None, timeseries_path=None) -> dict:
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
    # Additive to the D16 keys (schema_version stays 1): build-time warnings for this culvert.
    entry["warnings"] = list(result.warnings)
    entry.update({key: None for key in HYDRAULIC_KEYS})
    entry.update({key: value for key, value in (hydraulics or {}).items() if key in HYDRAULIC_KEYS})
    entry["timeseries_path"] = timeseries_path
    return entry


def metrics_document(results, hydraulics=None, timeseries_paths=None, run_fields=None) -> dict:
    """The document. ``hydraulics`` / ``timeseries_paths`` are per-culvert lists aligned with ``results``
    (absent at build time: the hydraulic keys stay null). ``run_fields`` are extra top-level keys."""
    hydraulics = hydraulics or [None] * len(results)
    timeseries_paths = timeseries_paths or [None] * len(results)
    document = {
        "schema_version": constants.METRICS_SCHEMA_VERSION,
        "culverts": [
            culvert_metrics_entry(result, hydraulic, series_path)
            for result, hydraulic, series_path in zip(results, hydraulics, timeseries_paths)
        ],
    }
    document.update(run_fields or {})
    return document


def metrics_path(output_directory) -> str:
    return os.path.join(output_directory, constants.METRICS_DIRNAME, constants.METRICS_FILENAME)


def write_culvert_metrics(output_directory, results, hydraulics=None, timeseries_paths=None, run_fields=None) -> str:
    """Write the metrics document; returns its path."""
    path = metrics_path(output_directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(metrics_document(results, hydraulics, timeseries_paths, run_fields), f, indent=2)
    os.replace(tmp, path)
    return path
