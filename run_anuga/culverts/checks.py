"""Build-time culvert checks and the coarse-mesh warning (TASK-3638, D15).

Two stages:

* :func:`check_culvert_placement` — before the mesh is built (cheap): both ends
  inside the model outline and not inside a Reflective building (a mesh hole).
* :func:`analyse_culverts` — once a mesh exists: the local cell edge at each
  end, the D15 exchange regions (>= 4 triangles each), the short-culvert and
  coarse-mesh warnings. It FAILS the build for a culvert that cannot be coupled
  (end off the mesh, < 4 triangles), and WARNS (never fails) when the culvert
  is short for the mesh or the mesh is coarse.

Every failure is collected and raised once, naming each culvert, so the user
fixes the whole layer in one pass.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from run_anuga._imports import import_optional
from run_anuga.culverts import constants
from run_anuga.culverts.features import CulvertBuildError
from run_anuga.culverts.regions import ExchangeRegion, build_exchange_region

logger = logging.getLogger(__name__)

ENDS = ("inlet", "outlet")


def _end_word(end):
    return "Inlet" if end == "inlet" else "Outlet"


def _reflective_buildings(building_geojson):
    """[(feature id, shapely Polygon)] for every Reflective building."""
    shapely_geometry = import_optional("shapely.geometry")
    polygons = []
    for feature in (building_geojson or {}).get("features") or []:
        if (feature.get("properties") or {}).get("method") != "Reflective":
            continue
        try:
            polygon = shapely_geometry.shape(feature.get("geometry"))
        except Exception:
            continue
        polygons.append((feature.get("id"), polygon))
    return polygons


def check_culvert_placement(culverts, boundary_polygon, building_geojson=None):
    """Fail when a culvert end is outside the model outline or inside a Reflective building."""
    if not culverts:
        return
    shapely_geometry = import_optional("shapely.geometry")
    outline = shapely_geometry.Polygon(boundary_polygon) if boundary_polygon and len(boundary_polygon) >= 3 else None
    buildings = _reflective_buildings(building_geojson)
    failures = []
    for culvert in culverts:
        for end in ENDS:
            xy = culvert.inlet_xy if end == "inlet" else culvert.outlet_xy
            point = shapely_geometry.Point(xy)
            where = f"the {_end_word(end)} at ({xy[0]:.2f}, {xy[1]:.2f})"
            if outline is not None and not outline.covers(point):
                failures.append(f"{culvert.label}: {where} is outside the model outline.")
                continue
            for building_id, building in buildings:
                if building.covers(point):
                    failures.append(
                        f"{culvert.label}: {where} is inside Reflective building {building_id}, "
                        "which is a hole in the mesh, so no water can reach it. Move the end "
                        "outside the building or change the building's method."
                    )
                    break
    if failures:
        raise CulvertBuildError("Cannot build the culverts:\n  " + "\n  ".join(failures))


def implied_timestep_s(cell_edge_m, reference_depth_m):
    """Rough ANUGA CFL step for right-isosceles cells of ``cell_edge_m`` in water of the given depth."""
    inradius = cell_edge_m * (2.0 - math.sqrt(2.0)) / 2.0
    depth = max(float(reference_depth_m), constants.MIN_REFERENCE_DEPTH_M)
    return inradius / math.sqrt(constants.GRAVITY_MS2 * depth)


def suggested_max_triangle_area_m2(culvert):
    """The max triangle area that brings the coarseness ratio down to the warning line."""
    target_edge = constants.COARSE_RATIO_WARN * culvert.footprint_width_m
    return target_edge ** 2 / 2.0


def coarse_mesh_warning(culvert, end, cell_edge_m) -> Optional[str]:
    """The warning text when the mesh is coarse at ``end``, else None."""
    ratio = cell_edge_m / culvert.footprint_width_m
    if ratio <= constants.COARSE_RATIO_WARN:
        return None
    severity = (
        "very coarse: this culvert's results are unreliable"
        if ratio > constants.COARSE_RATIO_SEVERE else "coarse"
    )
    target_edge = constants.COARSE_RATIO_WARN * culvert.footprint_width_m
    dt_now = implied_timestep_s(cell_edge_m, culvert.rise_m)
    dt_fine = implied_timestep_s(target_edge, culvert.rise_m)
    return (
        f"{culvert.label}: the mesh is {severity} at the {_end_word(end)}: coarseness ratio "
        f"{ratio:.2f} (cell edge {cell_edge_m:.2f} m against a culvert width of "
        f"{culvert.footprint_width_m:.2f} m; warning above {constants.COARSE_RATIO_WARN:g}, "
        f"unreliable above {constants.COARSE_RATIO_SEVERE:g}). Suggested max triangle area near "
        f"the culvert: {suggested_max_triangle_area_m2(culvert):.2f} m2 (draw a Mesh region). "
        f"ANUGA has no local time stepping, so that would cut the whole run's timestep from about "
        f"{dt_now:.3f} s to {dt_fine:.3f} s."
    )


def short_culvert_warning(culvert, largest_edge_m) -> Optional[str]:
    """The warning text when ``culvert`` is shorter than MIN_LENGTH_CELL_EDGES local cell edges, else None."""
    if culvert.length_m >= constants.MIN_LENGTH_CELL_EDGES * largest_edge_m:
        return None
    target_edge = culvert.length_m / constants.MIN_LENGTH_CELL_EDGES
    return (
        f"{culvert.label}: it is {culvert.length_m:.2f} m long, shorter than "
        f"{constants.MIN_LENGTH_CELL_EDGES:g} local cell edges ({largest_edge_m:.2f} m each), so its Inlet "
        "and Outlet exchange regions may sample the same or neighbouring cells and the result is less "
        f"reliable. Suggested max triangle area near the culvert: {target_edge ** 2 / 2.0:.2f} m2 "
        "(draw a Mesh region)."
    )


@dataclass
class CulvertMeshResult:
    culvert: object
    regions: dict  # end -> ExchangeRegion
    cell_edges_m: dict  # end -> float
    warnings: list = field(default_factory=list)

    @property
    def coarseness_ratio(self) -> float:
        """The worse (larger) of the two ends' cell edge / footprint width."""
        return max(self.cell_edges_m.values()) / self.culvert.footprint_width_m

    @property
    def inlet_region(self) -> ExchangeRegion:
        return self.regions["inlet"]

    @property
    def outlet_region(self) -> ExchangeRegion:
        return self.regions["outlet"]


def analyse_culverts(culverts, mesh, log=None) -> list:
    """Place each culvert on ``mesh`` (a MeshView); raise CulvertBuildError on any failure."""
    log = log or logger
    results, failures = [], []
    for culvert in culverts or []:
        problems = []
        cell_edges = {}
        for end in ENDS:
            xy = culvert.inlet_xy if end == "inlet" else culvert.outlet_xy
            edge = mesh.local_cell_edge(xy)
            if edge is None:
                problems.append(
                    f"the {_end_word(end)} at ({xy[0]:.2f}, {xy[1]:.2f}) is not on the mesh "
                    "(outside the model outline or inside a Reflective building)"
                )
            else:
                cell_edges[end] = edge
        if problems:
            failures.append(f"{culvert.label}: " + "; ".join(problems) + ".")
            continue

        largest_edge = max(cell_edges.values())
        regions = {}
        for end in ENDS:
            xy = culvert.inlet_xy if end == "inlet" else culvert.outlet_xy
            region = build_exchange_region(
                mesh, end, xy, culvert.outward_direction(end), culvert.footprint_width_m, cell_edges[end]
            )
            if region.triangle_count < constants.MIN_REGION_TRIANGLES:
                problems.append(
                    f"the {_end_word(end)} exchange region holds {region.triangle_count} triangle(s) "
                    f"after expanding it {region.expansions} times (at least "
                    f"{constants.MIN_REGION_TRIANGLES} are needed); is the end against the model "
                    "edge or a Reflective building?"
                )
            regions[end] = region
        if problems:
            failures.append(f"{culvert.label}: " + "; ".join(problems) + ".")
            continue

        result = CulvertMeshResult(culvert=culvert, regions=regions, cell_edges_m=cell_edges)
        short_warning = short_culvert_warning(culvert, largest_edge)
        if short_warning:
            result.warnings.append(short_warning)
            log.warning(short_warning)
        for end in ENDS:
            warning = coarse_mesh_warning(culvert, end, cell_edges[end])
            if warning:
                result.warnings.append(warning)
                log.warning(warning)
        log.info(
            "%s: exchange regions inlet=%d triangles, outlet=%d triangles; coarseness ratio %.2f",
            culvert.label, regions["inlet"].triangle_count, regions["outlet"].triangle_count,
            result.coarseness_ratio,
        )
        results.append(result)
    if failures:
        raise CulvertBuildError("Cannot build the culverts:\n  " + "\n  ".join(failures))
    return results
