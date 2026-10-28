"""Culverts: Hydraulic Structures support in run_anuga (epic 3585).

TASK-3638 (W2b) is the engine-agnostic plumbing: package parsing, the D15
exchange regions, build-time checks, the coarse-mesh warning, the
culverts/metrics.json contract and a stub operator. Importing this package is
pure Python; numpy/shapely/anuga load lazily inside the functions that use them.
"""

from run_anuga.culverts.checks import (
    analyse_culverts,
    check_culvert_placement,
    culvert_warning_records,
)
from run_anuga.culverts.constants import CULVERT_CAPABILITY
from run_anuga.culverts.features import Culvert, CulvertBuildError, load_culverts
from run_anuga.culverts.metrics import write_culvert_metrics
from run_anuga.culverts.regions import MeshView, write_exchange_regions

__all__ = [
    "CULVERT_CAPABILITY",
    "Culvert",
    "CulvertBuildError",
    "MeshView",
    "analyse_culverts",
    "check_culvert_placement",
    "culvert_warning_records",
    "load_culverts",
    "prepare_culverts",
    "write_culvert_metrics",
]


def prepare_culverts(input_data, mesh, log=None):
    """Analyse the package's culverts on ``mesh``; write culverts/metrics.json + exchange_regions.geojson.

    Returns the per-culvert results ([] when the package has none, and then no
    file is written). Raises CulvertBuildError when a culvert cannot be built.
    """
    culverts = input_data.get("culverts")
    if not culverts:
        return []
    results = analyse_culverts(culverts, mesh, log=log)
    path = write_culvert_metrics(input_data["output_directory"], results)
    # TASK-3646: the exchange regions beside metrics.json, for the map overlay.
    epsg = (input_data.get("scenario_config") or {}).get("epsg")
    regions_path = write_exchange_regions(input_data["output_directory"], results, epsg)
    if log is not None:
        log.info("culvert metrics written: %s, %s (%d culvert(s))", path, regions_path, len(results))
    return results
