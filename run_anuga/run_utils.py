import argparse
import datetime
import glob
import json
import logging
import logging.handlers
import math
import os
import re
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from run_anuga._imports import import_optional
from run_anuga import defaults
from run_anuga.breakline_conditioner import condition_breaklines
from run_anuga._logging import install_mname_filter
from run_anuga.config import ScenarioConfig

try:
    from celery.utils.log import get_task_logger
    logger = get_task_logger(__name__)
    from django.conf import settings
except ImportError:
    logger = logging.getLogger(__name__)
    settings = dict()

# Stamp anuga_core's mname/lnum record fields so run_anuga logs format cleanly
# when they propagate to anuga's root %(mname)s formatter (TASK-1276).
install_mname_filter(logger)


@dataclass
class RunContext:
    """Typed replacement for the (package_dir, username, password) tuple."""
    package_dir: str
    username: Optional[str] = None
    password: Optional[str] = None
    # Cached at run_sim startup so _report_run_error reuses the same config
    # the run started with even if scenario.json is overwritten mid-run.
    scenario_config: Optional[dict] = None


def is_dir_check(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def _load_package_data(package_dir):
    """Load and validate scenario config + input files. Pure Python, no geo deps."""
    if not os.path.isfile(os.path.join(package_dir, 'scenario.json')):
        raise FileNotFoundError(f'Could not find "scenario.json" in {package_dir}')

    input_data = dict()
    with open(os.path.join(package_dir, 'scenario.json')) as f:
        input_data['scenario_config'] = json.load(f)
    try:
        ScenarioConfig.model_validate(input_data['scenario_config'])
    except Exception as e:
        logger.warning(f"Scenario validation: {e}")
    project_id = input_data['scenario_config'].get('project')
    scenario_id = input_data['scenario_config'].get('id')
    run_id = input_data['scenario_config'].get('run_id')
    input_data['run_label'] = f"run_{project_id}_{scenario_id}_{run_id}"
    input_data['output_directory'] = os.path.join(package_dir, f'outputs_{project_id}_{scenario_id}_{run_id}')
    input_data['mesh_filepath'] = f"{input_data['output_directory']}/run_{project_id}_{scenario_id}_{run_id}.msh"
    Path(input_data['output_directory']).mkdir(parents=True, exist_ok=True)
    input_data['checkpoint_directory'] = f"{input_data['output_directory']}/checkpoints/"
    Path(input_data['checkpoint_directory']).mkdir(parents=True, exist_ok=True)

    input_data['boundary_filename'] = os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('boundary')}")
    with open(input_data['boundary_filename']) as f:
        input_data['boundary'] = json.load(f)

    data_types = [
        'friction',
        'inflow',
        'rainfall',
        'structure',
        'mesh_region',
        'network',
        'catchment',
        'nodes',
        'links',
        'breakline',  # TASK-1271 W4.3 — breaklines for mesh edge conformance
    ]
    for data_type in data_types:
        filepath = os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get(data_type)}")
        if input_data['scenario_config'].get(data_type) and os.path.isfile(filepath):
            input_data[f'{data_type}_filename'] = filepath
            with open(filepath) as f:
                input_data[data_type] = json.load(f)

    elevation_filepath = os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('elevation')}")
    if input_data['scenario_config'].get('elevation') and os.path.isfile(elevation_filepath):
        input_data['elevation_filename'] = elevation_filepath

    friction_raster_filepath = os.path.join(package_dir, f"inputs/{input_data['scenario_config'].get('friction_raster')}")
    if input_data['scenario_config'].get('friction_raster') and os.path.isfile(friction_raster_filepath):
        input_data['friction_raster_filename'] = friction_raster_filepath

    if input_data['scenario_config'].get('resolution'):
        input_data['resolution'] = input_data['scenario_config'].get('resolution')

    return input_data


def setup_input_data(package_dir):
    """Full setup — requires geo deps for boundary processing."""
    input_data = _load_package_data(package_dir)

    if len(input_data['boundary'].get('features')) == 0:
        raise AttributeError('No boundary features found')
    boundary_polygon, boundary_tags = create_boundary_polygon_from_boundaries(
        input_data['boundary']
    )
    if len(boundary_polygon) == 0 or len(boundary_tags) == 0:
        raise AttributeError('No boundary data found')
    input_data['boundary_polygon'] = boundary_polygon
    input_data['boundary_tags'] = boundary_tags
    return input_data


def update_web_interface(run_args, data, files=None):
    package_dir, username, password = run_args.package_dir, run_args.username, run_args.password
    if username and password:
        requests = import_optional("requests")
        from run_anuga._http import post_to_control_server

        input_data = setup_input_data(package_dir)
        data['project'] = input_data['scenario_config'].get('project')
        data['scenario'] = input_data['scenario_config'].get('id')
        run_id = input_data['scenario_config'].get('run_id')
        control_server = input_data['scenario_config'].get('control_server')
        url = f"{control_server}anuga/api/{data['project']}/{data['scenario']}/run/{run_id}/"
        auth = requests.auth.HTTPBasicAuth(username, password)
        # logger.critical(f"hydrata.com post:{data}")
        post_to_control_server(url, auth=auth, method="PATCH", data=data, files=files)



def get_utm_geo_reference(epsg_str):
    """
    TASK-1260: Derive an anuga.Geo_reference from an EPSG string using pyproj.

    Replaces the fragile ``int(epsg[-2:])`` slice (e.g. "EPSG:32601"[-2:] →
    "01" → zone=1 works, but edge cases like 3-digit zones could fail).
    pyproj reliably extracts the UTM zone number for both pure-UTM (32655/32755)
    and projected CRS whose name includes 'zone NN' (28355 / MGA).

    Parameters
    ----------
    epsg_str : str
        EPSG code, either "EPSG:32755" or bare "32755".

    Returns
    -------
    anuga.Geo_reference
        With ``zone`` set to the correct UTM zone integer.
    """
    anuga = import_optional("anuga")
    pyproj = import_optional("pyproj")
    CRS = pyproj.CRS

    # Normalise to integer EPSG code
    epsg_clean = epsg_str.strip()
    if ':' in epsg_clean:
        epsg_int = int(epsg_clean.split(':')[-1])
    else:
        epsg_int = int(epsg_clean)

    crs = CRS.from_epsg(epsg_int)

    # pyproj.CRS.utm_zone returns e.g. "55N", "55S", "1N", or None for
    # projected CRS that are not pure UTM (e.g. GDA94/MGA).
    utm_zone_str = crs.utm_zone
    if utm_zone_str:
        zone = int(re.search(r'\d+', utm_zone_str).group())
    else:
        # Fall back to parsing the CRS name (e.g. "GDA94 / MGA zone 55")
        m = re.search(r'zone\s+(\d+)', crs.name, re.IGNORECASE)
        if not m:
            raise ValueError(
                f"Cannot determine UTM zone from EPSG:{epsg_int} "
                f"(name={crs.name!r}, utm_zone={utm_zone_str!r}). "
                "Provide a UTM or MGA projected CRS."
            )
        zone = int(m.group(1))

    return anuga.Geo_reference(zone=zone)


def create_anuga_mesh(input_data):
    """Create the ANUGA mesh from boundary, regions, structures and breaklines.

    Structure method routing (ADR-4, TASK-1269/1270):
      Reflective → interior_hole with reflective wall tags (the mesh void path).
                   Sliver-merge applied before passing to Triangle (TASK-1270,
                   ported from run_anuga 5604fc1). NO DEM burn for Reflective.
      Mannings   → friction zone only; not a mesh hole; handled in make_frictions.
      Raised     → post-mesh elevation correction (handled in run.py after meshing).

    The universal gdal_rasterize burn that previously hit EVERY structure is
    REMOVED here. Only the Raised method applies an elevation change, and that
    happens post-mesh as a Domain quantity correction (TASK-1299).

    Mesh geo-reference (TASK-2149): for the NO-HOLE path we pass an absolute-UTM
    mesh_geo_reference (via get_utm_geo_reference); for the WITH-HOLE path we pass
    None and let ANUGA compute a local lower-left offset. Triangle's triangulation
    is float-rounding-sensitive to the coordinate ORIGIN, so the two choices yield
    slightly different tilings. Absolute-UTM is field-validated against the Merewether
    ARR benchmark (local offset mis-placed nodes onto knife-edge DEM walls → failed
    ARR + stability). The local offset is retained for hole-bearing meshes only,
    because absolute coordinates near Reflective hole boundaries produced degenerate
    near-zero-area triangles in the 5604fc1 investigation. (That was NOT a float32
    problem — ANUGA/Triangle coordinates are float64 — the hole-sliver mitigation is
    the unary_union sliver-merge in make_interior_holes_and_tags.)
    """
    anuga = import_optional("anuga")
    mesh_filepath = input_data['mesh_filepath']
    triangle_resolution = (input_data['scenario_config'].get('resolution') ** 2) / 2
    interior_regions = make_interior_regions(input_data)
    # TASK-1271: append breakline grading regions to the interior_regions list.
    breakline_regions = make_breaklines(input_data)
    if breakline_regions:
        logger.critical(f"make_breaklines: {len(breakline_regions)} buffer-ring regions added")
        interior_regions = interior_regions + breakline_regions
    interior_holes, hole_tags = make_interior_holes_and_tags(input_data)
    bounding_polygon = input_data['boundary_polygon']
    boundary_tags = input_data['boundary_tags']
    # TASK-1715: conform mesh edges ALONG each breakline (a Shewchuk Triangle PSLG
    # constraint), not merely grade density near it (the buffer rings above). The
    # build-time conditioner clips to the boundary, nodes crossings, dedupes and
    # drops sub-CFL segments so Triangle always gets a valid PSLG. Coords are
    # absolute (ANUGA offsets them by the boundary lower-left internally, the same
    # transform applied to boundary_polygon/interior_regions). Conform + grade
    # COMPOSE — the grading regions above are kept.
    breaklines = condition_breaklines(
        input_data.get('breakline'),
        bounding_polygon,
        default_near_spacing=input_data.get('scenario_config', {}).get(
            'default_near_spacing', 2.0),
    )
    if breaklines:
        logger.critical(f"condition_breaklines: {len(breaklines)} conforming breaklines passed to Triangle")
    # TASK-2149 — mesh geo-reference, field-validated against the Merewether ARR
    # urban-flood benchmark. Triangle's triangulation is float-rounding-sensitive
    # to the coordinate ORIGIN: the local lower-left offset that ANUGA applies when
    # mesh_geo_reference is None places mesh nodes slightly differently than
    # absolute-UTM coordinates do. On a knife-edge urban DEM (sharp ~3 m building
    # walls beside street-level gauges) that node shift lands nodes on the WALL
    # instead of the STREET, inflating peak stage at the gauges (Merewether ID0
    # +0.37 m, ID3 +0.55 m) — failing ARR validation AND tripping the CFL/implied-
    # speed stability heuristic. Feeding Triangle ABSOLUTE-UTM coordinates (no local
    # offset) reproduces the field-validated result exactly (ARR 5/5, stable).
    #
    # EXCEPTION: when interior holes are present (Reflective structures), KEEP the
    # local offset — absolute coordinates near the hole boundaries produced degenerate
    # near-zero-area triangles in the 5604fc1 investigation. Holes are rare in
    # rain-on-grid; the common no-hole path is what the benchmark validates.
    mesh_geo_reference = None
    if not interior_holes:
        epsg = input_data['scenario_config'].get('epsg')
        if epsg:
            # Robust EPSG -> UTM-zone via pyproj (handles bare/prefixed/MGA codes and
            # raises a CLEAR error on a non-UTM CRS) — NOT the fragile int(epsg[-2:])
            # slice that TASK-1260 already retired.
            mesh_geo_reference = get_utm_geo_reference(epsg)
        else:
            # No epsg -> keep the local offset (pre-fix behaviour). Don't crash a
            # hole-free run that otherwise georeferences fine from the boundary CRS;
            # the absolute-UTM path simply needs a UTM/MGA epsg to engage.
            logger.warning("create_anuga_mesh: no epsg in scenario_config — mesh uses local offset")
    logger.critical(
        f"creating anuga_mesh (mesh_geo_reference="
        f"{'absolute-UTM' if mesh_geo_reference is not None else 'local-offset'})"
    )
    # TASK-1270: universal burn removed. Only Raised structures apply a height
    # change (post-mesh, in run.py). Reflective is a mesh void; Mannings is friction-only.
    def _build_mesh(bl):
        return anuga.pmesh.mesh_interface.create_mesh_from_regions(
            bounding_polygon=bounding_polygon,
            boundary_tags=boundary_tags,
            maximum_triangle_area=triangle_resolution,
            interior_regions=interior_regions,
            interior_holes=interior_holes,
            hole_tags=hole_tags,
            mesh_geo_reference=mesh_geo_reference,
            filename=mesh_filepath,
            breaklines=bl,
            use_cache=False,
            verbose=False,
            fail_if_polygons_outside=False
        )

    # TASK-1715: a conditioned breakline is normally a valid Triangle PSLG, but the
    # conditioner cannot GUARANTEE every hand-drawn line yields a Triangle-acceptable
    # constraint (e.g. a line coincident with a boundary edge, or a crossing the noder
    # could not fully resolve). If Triangle rejects the conformed build, DEGRADE to
    # grading-only (breaklines=None — the pre-1715 behaviour; density is still graded
    # by the make_breaklines buffer rings above) rather than aborting the whole run at
    # mesh-gen. Only the with-breaklines attempt is guarded; a failure of the plain
    # build (OOM, bad boundary, …) still propagates.
    try:
        anuga_mesh = _build_mesh(breaklines if breaklines else None)
    except Exception as exc:
        if not breaklines:
            raise
        logger.critical(
            f"create_anuga_mesh: Triangle rejected the conformed breaklines "
            f"({exc!r}); retrying grading-only (breaklines=None) so the run proceeds "
            f"without edge conformance."
        )
        anuga_mesh = _build_mesh(None)
    logger.critical(f"mesh_triangle_count={len(anuga_mesh.tri_mesh.triangles)}")
    return mesh_filepath, anuga_mesh


def get_sql_triangles_from_anuga_mesh(anuga_mesh):
    vertices = anuga_mesh.tri_mesh.vertices
    triangles = anuga_mesh.tri_mesh.triangles
    output = "MULTIPOLYGON ("
    for triangle in triangles:
        # get coordinates:
        one = str(vertices[triangle[0]])[2:-1]
        two = str(vertices[triangle[1]])[2:-1]
        three = str(vertices[triangle[2]])[2:-1]
        four = str(vertices[triangle[0]])[2:-1]
        triangle_string = f"(({one},{two},{three},{four})),"
        output += str(triangle_string)
    output = output[:-1] + ")"
    return output


def make_interior_regions(input_data):
    interior_regions = list()
    if input_data.get('mesh_region'):
        for mesh_region in input_data['mesh_region']['features']:
            mesh_polygon = _extract_polygon_outer_ring(mesh_region.get('geometry'))
            mesh_resolution = mesh_region.get('properties').get('resolution')
            interior_regions.append((mesh_polygon, mesh_resolution,))
    return interior_regions


def make_breaklines(input_data):
    """Build interior_regions from breaklines for distance-graded mesh sizing.

    TASK-1271 (W4.3): Breaklines force Triangle to align edges along linear
    features (walls, roads, levees). This function synthesises distance-graded
    mesh refinement around each breakline by emitting buffer rings at
    h_near, 2*h_near, 4*h_near … up to the scenario default spacing.

    Each ring is registered as an interior_region with max_area ≈ h²·√3/4
    (the equilateral triangle area for a side length of h). The rings are
    added to the existing interior_regions list so MeshRegion and breakline
    refinement compose cleanly.

    Shewchuk Triangle has no built-in grading — the buffer-ring approach
    synthesises it deterministically with canonical input ordering.

    SCOPE GUARDRAIL (operator 2026-05-29): grading quality is INFORMATIONAL
    only. If Triangle's output is coarser than ideal ("crappy but coarse" is
    fine). The ONE HARD FLOOR is no sub-CFL-area slivers from buffer seams.

    Parameters
    ----------
    input_data : dict
        Must contain 'breakline' (GeoJSON FeatureCollection) and optionally
        'scenario_config' with 'resolution' and 'default_near_spacing'.

    Returns
    -------
    list of (polygon, max_area) tuples — extend existing interior_regions with these.
    """
    if not input_data.get('breakline'):
        return []

    try:
        from shapely.geometry import shape as _shape
    except ImportError:
        logger.warning("shapely not available — breakline grading skipped")
        return []

    import math as _math

    scenario_config = input_data.get('scenario_config', {})
    default_resolution = scenario_config.get('resolution', 10.0)
    default_near_spacing = scenario_config.get('default_near_spacing', 2.0)

    # max_area for an equilateral triangle of side h: h² * sqrt(3) / 4
    def _area_for_spacing(h):
        return (h ** 2) * (_math.sqrt(3) / 4)

    regions = []

    for feature in input_data['breakline']['features']:
        props = feature.get('properties') or {}
        geom = feature.get('geometry')
        if not geom:
            continue

        h_near = props.get('near_spacing') or default_near_spacing
        h_near = float(h_near)
        h_far = float(default_resolution)

        try:
            line = _shape(geom)
        except Exception:
            logger.warning(f"make_breaklines: could not parse geometry for feature {feature.get('id')}")
            continue

        # Build buffer rings at doubling distances: h_near, 2h, 4h, ...
        # Stop when buffer distance exceeds h_far (the scenario resolution).
        # Canonical ordering: process rings from outermost inward so inner
        # rings (smaller area) override outer ones in Triangle's pick.
        ring_distances = []
        d = h_near
        while d < h_far:
            ring_distances.append(d)
            d *= 2

        if not ring_distances:
            continue

        # Sort innermost first (smallest buffer = finest mesh); compute each
        # ring as buffer(d).difference(buffer(d/2)) so we get annular rings.
        # Register each ring with max_area for spacing h=d (the outer edge
        # of the ring). Fine inner rings override coarser outer ones because
        # Triangle uses the SMALLEST max_area constraint for any point.
        ring_distances_sorted = sorted(ring_distances)  # [2, 4, 8, 16]

        inner_poly = None
        for dist in ring_distances_sorted:
            current_poly = line.buffer(dist)
            if current_poly.is_empty:
                continue
            ring_geom = current_poly if inner_poly is None else current_poly.difference(inner_poly)
            inner_poly = current_poly

            # h for this ring = the buffer distance
            h = dist
            max_area = _area_for_spacing(h)
            ring_coords = _ring_to_coords(ring_geom)
            for coords in ring_coords:
                regions.append((coords, max_area))

    return regions


def _ring_to_coords(geom):
    """Extract outer rings from a shapely geometry as lists of [x, y] pairs."""
    if geom.is_empty:
        return []
    if geom.geom_type == 'MultiPolygon':
        polys = list(geom.geoms)
    elif geom.geom_type == 'Polygon':
        polys = [geom]
    else:
        return []
    result = []
    for poly in polys:
        coords = [list(c) for c in poly.exterior.coords]
        if coords:
            result.append(coords)
    return result


def make_interior_holes_and_tags(input_data):
    """Build interior mesh holes for Reflective structures.

    ADR-4 / TASK-1270 routing:
      Reflective → interior mesh hole with reflective wall tags.
                   Sliver-merge applied first (port of run_anuga 5604fc1):
                   shapely unary_union merges adjacent/shared-vertex buildings
                   so Triangle never sees coincident edges that produce
                   near-zero-area (~1e-9 m²) triangles forcing sub-µs CFL.
      Mannings   → friction zone only; skipped here.
      Raised     → post-mesh elevation; skipped here.
    """
    raw_polys = []
    if input_data.get('structure'):
        for structure in input_data['structure']['features']:
            method = structure.get('properties', {}).get('method')
            if method == 'Reflective':
                raw_polys.append(_extract_polygon_outer_ring(structure.get('geometry')))
            elif method in ('Mannings', 'Raised'):
                pass  # handled elsewhere
            else:
                if method is not None:
                    logger.error(f"Unknown structure method: {method!r} — skipping")

    if not raw_polys:
        return None, None

    # Sliver-merge (ported from run_anuga 5604fc1): adjacent buildings that
    # share vertices or edges produce near-zero-area triangles when passed as
    # separate holes to Triangle. Merge them first with shapely unary_union;
    # simplify(0.01) removes sub-centimetre artefacts at shared boundary
    # vertices. For Merewether: 60 buildings → 57 merged, min edge 1.074m.
    try:
        from shapely.geometry import Polygon as _ShapelyPolygon
        from shapely.ops import unary_union as _unary_union
        shapely_polys = [_ShapelyPolygon(c) for c in raw_polys]
        merged = _unary_union(shapely_polys)
        geoms = list(merged.geoms) if merged.geom_type == 'MultiPolygon' else [merged]
        merged_polys = []
        for geom in geoms:
            simplified = geom.simplify(0.01, preserve_topology=True)
            merged_polys.append(list(simplified.exterior.coords))
        logger.critical(
            f"make_interior_holes_and_tags: {len(raw_polys)} raw polys → "
            f"{len(merged_polys)} merged holes (sliver-merge applied)"
        )
    except ImportError:
        logger.warning("shapely not available — sliver-merge skipped; slivers may cause CFL issues")
        merged_polys = raw_polys

    interior_holes = []
    hole_tags = []
    for coords in merged_polys:
        interior_holes.append(coords)
        hole_tags.append({'reflective': list(range(len(coords)))})
    return interior_holes, hole_tags


def compute_yieldstep(duration):
    """Calculate yield step interval for the simulation evolve loop.

    Returns an integer number of seconds, clamped to
    [MIN_YIELDSTEP_S, MAX_YIELDSTEP_S]. Single source of truth for the
    yieldstep used by run.py (ported from run_anuga main; logically identical
    to the prior inline computation).
    """
    base_step = math.floor(duration / defaults.MAX_YIELDSTEPS)
    yieldstep = max(base_step, defaults.MIN_YIELDSTEP_S)
    return min(yieldstep, defaults.MAX_YIELDSTEP_S)


def make_frictions(input_data):
    # Raster precedence (TASK-830 / TASK-1259): a friction raster sets the
    # base Manning's value at full raster resolution across the whole domain.
    # Per-structure Manning's-n patches (method='Mannings') must STILL overlay
    # the raster — ANUGA's composite_quantity_setting_function evaluates entries
    # in order, last-match wins, so appending the structure patches AFTER the
    # raster entry applies them on top.
    #
    # Old behaviour (pre-TASK-1259): early-return dropped all structure patches
    # when a raster was present.  New behaviour: raster first, then structure
    # patches, no 'All' fallback (raster already covers the whole domain).
    #
    # Without a raster: polygon-only path unchanged (structure + friction polys
    # + 'All' fallback).
    # See docs/reports/2026-05-13-q-1-task-830-friction-raster-attachment.html.
    frictions = list()
    if input_data.get('friction_raster_filename'):
        frictions.append(['Extent', input_data['friction_raster_filename']])
        # Overlay per-structure Manning's-n patches on top of the raster.
        if input_data.get('structure'):
            for structure in input_data['structure']['features']:
                if structure.get('properties').get('method') == 'Mannings':
                    structure_polygon = _extract_polygon_outer_ring(structure.get('geometry'))
                    frictions.append((structure_polygon, defaults.BUILDING_MANNINGS_N,))
        return frictions
    if input_data.get('structure'):
        for structure in input_data['structure']['features']:
            if structure.get('properties').get('method') == 'Mannings':
                structure_polygon = _extract_polygon_outer_ring(structure.get('geometry'))
                frictions.append((structure_polygon, defaults.BUILDING_MANNINGS_N,))
    if input_data.get('friction'):
        for friction in input_data['friction']['features']:
            friction_polygon = _extract_polygon_outer_ring(friction.get('geometry'))
            friction_value = friction.get('properties').get('mannings')
            frictions.append((friction_polygon, friction_value,))
    frictions.append(['All', defaults.DEFAULT_MANNINGS_N])
    return frictions


def make_raised_elevation_pairs(input_data):
    """Build (polygon, raised_height) pairs for Raised-method structures.

    TASK-1299: The Raised method applies a post-mesh elevation correction to
    building footprints. This replaces the old universal +5m DEM-burn: only
    structures with method='Raised' get an elevation adjustment, and the height
    is per-structure (defaulting to Scenario.default_raised_height via the
    scenario_config key 'default_raised_height').

    Returns a list of (polygon_coords, height_m) pairs, empty if no Raised
    structures are present. Caller applies these via
    composite_quantity_setting_function AFTER the base DEM elevation is seated.
    """
    default_raised_height = float(
        input_data.get('scenario_config', {}).get('default_raised_height', defaults.BUILDING_BURN_HEIGHT_M)
    )
    pairs = []
    if not input_data.get('structure'):
        return pairs
    for structure in input_data['structure']['features']:
        method = structure.get('properties', {}).get('method')
        if method != 'Raised':
            continue
        height = structure.get('properties', {}).get('raised_height')
        height = float(height) if height is not None else default_raised_height
        polygon = _extract_polygon_outer_ring(structure.get('geometry'))
        pairs.append((polygon, height))
    return pairs


def apply_raised_elevation_correction(domain, raised_pairs):
    """Add per-structure Raised heights to a built domain's centroid elevation.

    TASK-2149 (F1): the point-in-polygon test MUST use ABSOLUTE centroid coordinates
    — make_raised_elevation_pairs returns polygons in absolute UTM. Testing LOCAL
    centroids (get_centroid_coordinates(absolute=False)) silently returned zero hits
    whenever the mesh carried a nonzero geo_reference offset (every local-offset mesh,
    i.e. every prod sim before the mesh geo-ref fix), so Raised heights were never
    applied. Using absolute=True makes the correction offset-independent — it works on
    both the absolute-UTM no-hole mesh and the local-offset hole mesh.

    Returns the number of Raised structures that matched at least one centroid.
    """
    from anuga.geometry.polygon import inside_polygon
    centroids = domain.get_centroid_coordinates(absolute=True)
    elev = domain.get_quantity('elevation').get_values(location='centroids')
    applied = 0
    for poly_coords, height_m in raised_pairs:
        if not poly_coords:
            continue
        inside_idx = inside_polygon(centroids, poly_coords)
        if len(inside_idx) > 0:
            elev[inside_idx] += height_m
            applied += 1
    domain.set_quantity('elevation', elev, location='centroids')
    return applied


def apply_negative_depth_protection(domain):
    """Run ANUGA's own negative-depth protection on ``domain`` iff any centroid
    is dry-above-stage (stage < elevation). Returns True if protection ran.

    TASK-2226 — defense-in-depth for run 1283's negative-inlet-volume assert.
    ANUGA's serial evolve calls protect_against_infinitesimal_and_negative_
    heights() before any operator's first ``__call__``, so a serial
    Inlet_operator never sees a negative inlet volume even when a Raised
    structure (or any bed cell above stage=0.0) leaves the inlet region
    dry-above-stage. The PARALLEL path's Parallel_Inlet_operator asserts on that
    negative volume at the FIRST evolve step — BEFORE the scattered sub-domains
    ever reach protect (run 1283: MPI_ABORT rank 10,
    anuga/parallel/parallel_inlet_operator.py:121). Calling this on the whole
    rank-0 domain, pre-distribute(), makes the serial and parallel paths clip
    identically.

    It is idempotent with evolve's own protect — a successful run is unchanged;
    the only observable effect is preventing that first-step assert. The
    gn_anuga build-time guard (assert_inflow_does_not_overlap_raised_structure,
    epic 2204 W4) blocks the KNOWN geometry conflict at build time; this closes
    the residual class the guard cannot see — any mechanism that leaves an inlet
    region dry-above-stage.
    """
    stage_centroids = domain.quantities['stage'].centroid_values
    elevation_centroids = domain.quantities['elevation'].centroid_values
    # centroid_values are numpy arrays; use the array's own .any() so run_utils
    # needs no module-level numpy import.
    if (stage_centroids < elevation_centroids).any():
        domain.protect_against_infinitesimal_and_negative_heights()
        return True
    return False


def assert_raster_has_no_nodata_inside_boundary(raster_path, boundary_polygon, *, quantity_name):
    """Pre-flight guard: raise a clear error if a raster has nodata cells inside the model boundary.

    ANUGA seats elevation/friction values onto mesh centroids via
    ``composite_quantity_setting_function(..., nan_treatment='exception')``. If
    any sample point lands on a nodata cell, ANUGA raises an opaque exception
    deep inside ``composite_quantity_setting_function`` mid-build, with no hint
    that the cause is a data gap in the input raster. This function runs the
    same check up-front, on rank 0, before the set_quantity calls, and raises a
    message that names the raster, the count of offending cells, and the fix.

    We deliberately KEEP ``nan_treatment='exception'`` everywhere (we never want
    to silently fabricate a bed/friction value); this guard simply surfaces the
    failure earlier and more clearly.

    Nodata detection: a cell is "nodata" if it equals the raster's declared
    nodata tag (e.g. the TASK-1136 standard ``-9999``) OR is NaN. Both are
    handled because GeoTIFFs in the wild use either convention. If the raster
    declares NO nodata value, there is nothing to check and we return (pass) —
    a stray NaN with no declared nodata is left for ANUGA to surface.

    Inside-boundary test: the boundary polygon is rasterised onto the raster's
    own grid/transform via ``rasterio.features.geometry_mask`` (vectorised, one
    pass — far cheaper than a per-cell shapely point-in-polygon test), then
    intersected with the nodata mask.

    Parameters
    ----------
    raster_path : str
        Path to the elevation or friction GeoTIFF (in the raster's projected CRS).
    boundary_polygon : list
        Flat list of ``[x, y]`` vertices for the model boundary, in the raster's
        projected CRS (``input_data['boundary_polygon']``). A no-op if empty.
    quantity_name : str
        Keyword-only label ('elevation' or 'friction') used in the error message.

    Raises
    ------
    ValueError
        If one or more nodata cells fall inside the model boundary.
    """
    if not boundary_polygon:
        # No boundary to test against — nothing this guard can assert.
        return
    rasterio = import_optional("rasterio")
    features = import_optional("rasterio.features")
    numpy = import_optional("numpy")

    with rasterio.open(raster_path) as dataset:
        band = dataset.read(1)
        nodata_value = dataset.nodata
        transform = dataset.transform
        out_shape = (dataset.height, dataset.width)

    # NaN is always treated as nodata. A finite declared nodata tag (e.g. -9999)
    # is matched exactly. If neither applies, there is nothing to flag.
    nodata_mask = numpy.isnan(band)
    if nodata_value is not None and not numpy.isnan(nodata_value):
        nodata_mask = nodata_mask | (band == nodata_value)
    # No nodata cells (no declared tag and no NaN, or a declared tag that no
    # cell matches) — nothing to check.
    if not nodata_mask.any():
        return

    # Close the ring so shapely/rasterio treats it as a polygon, not a line.
    ring = [tuple(point) for point in boundary_polygon]
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    boundary_geometry = {'type': 'Polygon', 'coordinates': [ring]}

    # geometry_mask returns True OUTSIDE the geometry by default (invert=False),
    # so cells inside the boundary are where the mask is False.
    outside_mask = features.geometry_mask(
        [boundary_geometry],
        out_shape=out_shape,
        transform=transform,
        invert=False,
        all_touched=True,
    )
    inside_mask = ~outside_mask
    offending = nodata_mask & inside_mask
    offending_count = int(offending.sum())
    if offending_count > 0:
        raise ValueError(
            f"{quantity_name} raster '{raster_path}' has {offending_count} nodata "
            f"cells inside the model boundary; fill the data gaps "
            f"(e.g. gdal_fillnodata) or reselect the terrain extent. "
            f"ANUGA cannot seat a bed/friction value there."
        )


def compute_mesh_qa(anuga_mesh):
    """Compute mesh-quality metrics from an ANUGA mesh object.

    Returns a dict with:
      triangle_count   — true element count (after W1.1 len() fix)
      node_count       — vertex count
      min_angle_deg    — minimum interior angle across all triangles (degrees)
      max_angle_deg    — maximum interior angle across all triangles (degrees)
      sliver_count     — triangles with any angle < SLIVER_ANGLE_THRESHOLD_DEG
      aspect_ratio_max — maximum ratio of longest to shortest edge per triangle
      duplicate_node_count — number of vertices duplicated (same x, y)
      has_degenerate_triangles — True if any triangle has zero or near-zero area

    All numeric fields default to 0 / False when the mesh has no triangles, so
    callers never need to guard for None.

    This function is anuga-independent: it reads tri_mesh.vertices and
    tri_mesh.triangles (plain numpy arrays) and does all work with numpy.
    """
    numpy = import_optional("numpy")
    SLIVER_ANGLE_THRESHOLD_DEG = 10.0  # leading CFL-risk indicator

    vertices = anuga_mesh.tri_mesh.vertices   # (N, 2) float
    triangles = anuga_mesh.tri_mesh.triangles  # (M, 3) int

    n_triangles = len(triangles)
    n_nodes = len(vertices)

    if n_triangles == 0:
        return {
            'triangle_count': 0,
            'node_count': n_nodes,
            'min_angle_deg': 0.0,
            'max_angle_deg': 0.0,
            'sliver_count': 0,
            'aspect_ratio_max': 0.0,
            'duplicate_node_count': 0,
            'has_degenerate_triangles': False,
            # W3 (TASK-1923) — area metrics
            'min_triangle_area': 0.0,
            'area_histogram': [],
        }

    # Gather vertex coordinates for each triangle corner
    v0 = vertices[triangles[:, 0]]  # (M, 2)
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # Edge vectors
    e0 = v1 - v0  # opposite corner v2
    e1 = v2 - v1  # opposite corner v0
    e2 = v0 - v2  # opposite corner v1

    # Edge lengths
    l0 = numpy.linalg.norm(e0, axis=1)
    l1 = numpy.linalg.norm(e1, axis=1)
    l2 = numpy.linalg.norm(e2, axis=1)

    # Degenerate triangles: any edge length near zero
    EPS = 1e-10
    has_degenerate = bool(numpy.any((l0 < EPS) | (l1 < EPS) | (l2 < EPS)))

    # Interior angles via law of cosines (clamp for numerical safety)
    def _safe_angle(a, b, c):
        """Angle at vertex opposite side c, given side lengths a, b, c."""
        denom = 2.0 * a * b
        safe_denom = numpy.where(denom < EPS, EPS, denom)
        cos_val = numpy.clip((a**2 + b**2 - c**2) / safe_denom, -1.0, 1.0)
        return numpy.degrees(numpy.arccos(cos_val))

    # angle at v0 (between edges e2 and e0, opposite edge l1)
    ang0 = _safe_angle(l2, l0, l1)
    # angle at v1 (between edges e0 and e1, opposite edge l2)
    ang1 = _safe_angle(l0, l1, l2)
    # angle at v2 (between edges e1 and e2, opposite edge l0)
    ang2 = _safe_angle(l1, l2, l0)

    all_angles = numpy.concatenate([ang0, ang1, ang2])

    min_angle = float(numpy.min(all_angles))
    max_angle = float(numpy.max(all_angles))

    # Per-triangle minimum angle
    per_tri_min_angle = numpy.minimum(numpy.minimum(ang0, ang1), ang2)
    sliver_count = int(numpy.sum(per_tri_min_angle < SLIVER_ANGLE_THRESHOLD_DEG))

    # Aspect ratio: longest / shortest edge per triangle
    edge_max = numpy.maximum(numpy.maximum(l0, l1), l2)
    edge_min = numpy.minimum(numpy.minimum(l0, l1), l2)
    safe_min = numpy.where(edge_min < EPS, EPS, edge_min)
    aspect_ratios = edge_max / safe_min
    aspect_ratio_max = float(numpy.max(aspect_ratios))

    # Duplicate nodes: round to 3 decimal places (millimetre precision)
    rounded = numpy.round(vertices, 3)
    unique_rows = numpy.unique(rounded, axis=0)
    duplicate_node_count = int(n_nodes - len(unique_rows))

    # Triangle areas via 2-D cross product (exact for flat triangles).
    # area = 0.5 * |e0 × e2| = 0.5 * |e0x*e2y - e0y*e2x|
    cross = numpy.abs(e0[:, 0] * (-e2[:, 1]) - e0[:, 1] * (-e2[:, 0]))
    areas = 0.5 * cross
    min_triangle_area = float(numpy.min(areas))

    # Log-spaced area histogram over 7+ decades (min_area – 10 M m²).
    # Edges span from 10x below the observed minimum up to 10 M m² so ALL
    # triangles fall within the first/last bin (no under- or overflow losses).
    # We clamp to a floor of 1e-3 m² to avoid log10(0) on degenerate meshes.
    _area_floor = max(float(min_triangle_area) * 0.1, 1e-3)
    HIST_HI = 1.0e7    # upper bound (m²), > MAX_TRIANGLE_AREA 10 M m²
    N_BINS = 22        # 22 edges → 21 log-spaced bins
    log_edges = numpy.logspace(
        numpy.log10(_area_floor), numpy.log10(HIST_HI), N_BINS
    )
    # Clamp areas into [log_edges[0], log_edges[-1]] so np.histogram counts all.
    clamped = numpy.clip(areas, log_edges[0], log_edges[-1] * (1 - 1e-12))
    counts, _ = numpy.histogram(clamped, bins=log_edges)
    area_histogram = [
        {
            "bin_lo": round(float(log_edges[i]), 4),
            "bin_hi": round(float(log_edges[i + 1]), 4),
            "count": int(counts[i]),
        }
        for i in range(len(counts))
    ]

    return {
        'triangle_count': n_triangles,
        'node_count': n_nodes,
        'min_angle_deg': round(min_angle, 2),
        'max_angle_deg': round(max_angle, 2),
        'sliver_count': sliver_count,
        'aspect_ratio_max': round(aspect_ratio_max, 2),
        'duplicate_node_count': duplicate_node_count,
        'has_degenerate_triangles': has_degenerate,
        # W3 (TASK-1923) — area metrics
        'min_triangle_area': round(min_triangle_area, 4),
        'area_histogram': area_histogram,
    }


def extract_boundary_condition_types(domain) -> list:
    """Return the SORTED unique set of boundary-condition type names in use.

    ``domain.boundary`` is a dict mapping ``(edge_index, ...)`` segment keys
    to type-name strings (e.g. ``"exterior"``, ``"Reflective"``, ``"Time"``).
    Returns a sorted list so the value is stable for corpus grouping.

    W3 (TASK-1923).
    """
    try:
        bc_values = domain.boundary.values()
        return sorted(set(str(v) for v in bc_values))
    except Exception:
        return []


def correction_for_polar_quadrants(base, height):
    result = 0
    result = 0 if base > 0 and height > 0 else result
    result = math.pi if base < 0 and height > 0 else result
    result = math.pi if base < 0 and height < 0 else result
    result = 2 * math.pi if base > 0 and height < 0 else result
    return result


def lookup_boundary_tag(index, boundary_tags):
    for key in boundary_tags.keys():
        if index in boundary_tags[key]:
            return key


def _flatten_line_coordinates(geometry):
    """
    Return a flat list of [x, y] points from a LineString or MultiLineString
    GeoJSON geometry. PostGIS / GeoServer normalises every boundary feature
    to MultiLineString regardless of input — even when the source file is
    LineString — so this helper has to handle both. For MultiLineString with
    a single ring we flatten one level; for true multi-rings we concatenate
    (the boundary polygon is sorted clockwise by feature centroid afterwards,
    so the join order within a feature does not matter for sorting).

    Returns [] for missing or empty coordinates; the debug log surfaces
    data-quality issues without aborting the simulation.
    """
    raw_coords = geometry.get('coordinates')
    coords = raw_coords or []
    gtype = geometry.get('type')
    if not coords:
        logger.debug(
            "_flatten_line_coordinates: empty/missing coordinates "
            "(type=%r, raw=%r); returning []",
            gtype, raw_coords,
        )
        return []
    if gtype == 'MultiLineString':
        return [pt for line in coords for pt in line]
    if gtype != 'LineString':
        logger.debug(
            "_flatten_line_coordinates: unsupported geometry type %r; "
            "returning coordinates as-is",
            gtype,
        )
    return coords


def _extract_polygon_outer_ring(geometry):
    """
    Return the 2-D outer ring of a Polygon or MultiPolygon GeoJSON geometry.
    The BE can serialise the same polygon-shaped feature as either Polygon
    or MultiPolygon depending on storage and round-trip path, so ANUGA-side
    callers (set_quantity, Polygonal_rate_operator, shapely.Polygon,
    gdalwarp -cutline) must accept both shapes — they all want an (N, 2)
    list of vertices, not a (1, N, 2) list-of-polygons.
    For MultiPolygon with multiple sub-polygons, the first sub-polygon's
    outer ring is returned and a warning is logged. Inner rings (holes)
    inside the first sub-polygon are also dropped, matching the previous
    Polygon-only behaviour which only ever consumed coordinates[0].

    Returns [] for missing or empty coordinates; the debug log surfaces
    data-quality issues without aborting the simulation.
    """
    raw_coords = geometry.get('coordinates')
    coords = raw_coords or []
    gtype = geometry.get('type')
    if not coords:
        logger.debug(
            "_extract_polygon_outer_ring: empty/missing coordinates "
            "(type=%r, raw=%r); returning []",
            gtype, raw_coords,
        )
        return []
    if gtype == 'MultiPolygon':
        if len(coords) > 1:
            logger.warning(
                "MultiPolygon with %d sub-polygons; only the first will be used",
                len(coords),
            )
        return coords[0][0] if coords[0] else []
    if gtype != 'Polygon':
        logger.debug(
            "_extract_polygon_outer_ring: unsupported geometry type %r; "
            "falling through to coords[0]",
            gtype,
        )
    return coords[0]


def create_boundary_polygon_from_boundaries(boundaries_geojson):
    ogr = import_optional("osgeo.ogr")
    geometry_collection = ogr.Geometry(ogr.wkbGeometryCollection)
    if boundaries_geojson.get('crs'):
        epsg_code = boundaries_geojson.get('crs').get('properties').get('name').split(':')[-1]
    else:
        return list(), dict()
    # Create a dict of the available boundary tags
    boundary_tag_labels = dict()
    all_x_coordinates = list()
    all_y_coordinates = list()
    for index, feature in enumerate(boundaries_geojson.get('features')):
        if feature.get('properties').get('location') != "External":
            continue
        boundary_tag_labels[feature.get('properties').get('boundary')] = []
        geometry = ogr.CreateGeometryFromJson(json.dumps(feature.get('geometry')))
        geometry_collection.AddGeometry(geometry)
        # Collect a list of the coordinates associated with each boundary tag:
        feature_coordinates = _flatten_line_coordinates(feature.get('geometry'))
        for coordinate in feature_coordinates:
            all_x_coordinates.append(coordinate[0])
            all_y_coordinates.append(coordinate[1])
    srs = ogr.osr.SpatialReference()
    epsg_integer = int(epsg_code.split(':')[1] if ':' in epsg_code else epsg_code)
    srs.ImportFromEPSG(epsg_integer)

    # Find the center of our project
    if not all_x_coordinates or not all_y_coordinates:
        raise ValueError(
            "create_boundary_polygon_from_boundaries: no valid External-location boundary coordinates found"
        )
    max_x = max(all_x_coordinates)
    max_y = max(all_y_coordinates)
    min_x = min(all_x_coordinates)
    min_y = min(all_y_coordinates)
    mid_x = max_x - (max_x - min_x) / 2
    mid_y = max_y - (max_y - min_y) / 2
    line_list = list()

    # Now create and sort the line_list of boundary lines in a clockwise direction around it
    for index, feature in enumerate(boundaries_geojson.get('features')):
        if feature.get('properties').get('location') != "External":
            # discard any internal boundaries from the boundary_polygon
            continue
        geometry = ogr.CreateGeometryFromJson(json.dumps(feature.get('geometry')))
        centroid = json.loads(geometry.Centroid().ExportToJson()).get('coordinates')
        base = centroid[0] - mid_x
        height = centroid[1] - mid_y
        # the angle in polar coordinates will sort our boundary lines into the correct order
        angle = math.atan2(height, base)
        line_list.append({
            "centroid": centroid,
            "boundary": feature.get('properties').get('boundary'),
            "id": feature.get('id'),
            "angle": angle,
            "coordinates": _flatten_line_coordinates(feature.get('geometry')),
        })
    line_list.sort(key=lambda line: line.get('angle'), reverse=True)

    # Now join all our lines in clockwise order and create the boundary tags object
    boundary_polygon = list()
    boundary_tags_list = list()
    counter = 0
    boundary_tags = deepcopy(boundary_tag_labels)
    for line in line_list:
        for coordinate in line.get("coordinates"):
            boundary_polygon.append(coordinate)
            boundary_tags[line.get("boundary")].append(counter)
            boundary_tags_list.append(lookup_boundary_tag(counter, boundary_tags))
            counter += 1

    # now sort the boundary_polygon points in clockwise order, in case those original lines were drawn with different
    # directions
    boundary_polygon_with_angle_data = list()
    sorted_boundary_polygon = list()
    sorted_boundary_tags = deepcopy(boundary_tag_labels)
    for index, point in enumerate(boundary_polygon):
        base = point[0] - mid_x
        height = point[1] - mid_y
        angle = math.atan2(height, base)
        boundary_polygon_with_angle_data.append({
            "point": point,
            "boundary": lookup_boundary_tag(index, boundary_tags),
            "angle": angle
        })
    boundary_polygon_with_angle_data.sort(key=lambda point_blob: point_blob.get('angle'), reverse=True)
    for index, point_blob in enumerate(boundary_polygon_with_angle_data):
        sorted_boundary_polygon.append(point_blob.get('point'))
        sorted_boundary_tags[point_blob.get('boundary')].append(index)

    # # Make a dump of the centroids geometry (for debugging only - not returned anywhere).
    # output_driver_centroids = ogr.GetDriverByName('GeoJSON')
    # filepath_geojson_driver_centroids = os.path.join(package_dir, f'outputs_{run_label.split("run_")[1]}', f'{run_label}_boundary_centroids.geojson')
    # output_data_source_centroids = output_driver_centroids.CreateDataSource(filepath_geojson_driver_centroids)
    # output_layer_centroids = output_data_source_centroids.CreateLayer(filepath_geojson_driver_centroids, srs, geom_type=ogr.wkbPolygon)
    # feature_definition_centroids = output_layer_centroids.GetLayerDefn()
    # field_definition_centroids_1 = ogr.FieldDefn('index', ogr.OFTReal)
    # output_layer_centroids.CreateField(field_definition_centroids_1)
    # field_definition_centroids_2 = ogr.FieldDefn('angle', ogr.OFTReal)
    # output_layer_centroids.CreateField(field_definition_centroids_2)
    # field_definition_centroids_3 = ogr.FieldDefn('id', ogr.OFTString)
    # field_definition_centroids_3.SetWidth(1000)
    # output_layer_centroids.CreateField(field_definition_centroids_3)
    # for index, line in enumerate(line_list):
    #     output_feature_centroids = ogr.Feature(feature_definition_centroids)
    #     centroid = line.get('centroid')
    #     point = ogr.Geometry(ogr.wkbPoint)
    #     point.AddPoint(centroid[0], centroid[1])
    #     output_feature_centroids.SetGeometry(point)
    #     output_feature_centroids.SetField('index', index)
    #     output_feature_centroids.SetField('angle', line.get('angle'))
    #     output_feature_centroids.SetField('id', line.get('id'))
    #     output_layer_centroids.CreateFeature(output_feature_centroids)
    #
    # # Make a dump of the boundary polygon geometry (for debugging only - not returned anywhere).
    # output_driver = ogr.GetDriverByName('GeoJSON')
    # filepath_geojson_driver = os.path.join(package_dir, f'outputs_{run_label.split("run_")[1]}', f'{run_label}_boundary_polygon.geojson')
    # output_data_source = output_driver.CreateDataSource(filepath_geojson_driver)
    # output_layer = output_data_source.CreateLayer(filepath_geojson_driver, srs, geom_type=ogr.wkbPolygon)
    # feature_definition = output_layer.GetLayerDefn()
    # field_definition_1 = ogr.FieldDefn('index', ogr.OFTReal)
    # output_layer.CreateField(field_definition_1)
    # field_definition_2 = ogr.FieldDefn('boundary', ogr.OFTString)
    # field_definition_2.SetWidth(1000)
    # output_layer_centroids.CreateField(field_definition_2)
    # for index, coordinate in enumerate(boundary_polygon):
    #     output_feature = ogr.Feature(feature_definition)
    #     point = ogr.Geometry(ogr.wkbPoint)
    #     point.AddPoint(coordinate[0], coordinate[1])
    #     output_feature.SetGeometry(point)
    #     output_feature.SetField('index', index)
    #     output_feature.SetField('boundary', boundary_tags_list[index])
    #     output_layer.CreateFeature(output_feature)

    return sorted_boundary_polygon, sorted_boundary_tags


def build_time_boundary_function(time_boundary_features, defaults=None):
    """Build the function passed to ``anuga.Time_boundary(domain, function=...)``.

    The Anuga Time boundary expects a callable ``f(t_seconds) -> [stage,
    xmom, ymom]`` returning conserved-quantity values for the model time t.

    Inputs:
        time_boundary_features: a list of GeoJSON Feature dicts with
            ``properties.boundary == 'Time'`` and a ``properties.data``
            value that has already been resolved server-side to either:
              * a numeric stage value (constant case), or
              * a list of ``{timestamp: ISO8601, value: number}`` dicts
                (timeseries case).

        defaults: the run_anuga.defaults module (or any object exposing
            float-coercible attributes). Currently unused but accepted for
            forward-compat — leaves room for momentum or factor defaults.

    Anuga's tag system collapses multiple Time-boundary features under a
    single 'Time' tag. We use the first feature's data and log a warning
    if there are multiple distinct Time boundaries — this is an authoring
    issue rather than a code limitation, and it surfaces clearly in logs.
    """
    if not time_boundary_features:
        raise ValueError(
            'build_time_boundary_function() called with no features'
        )
    if len(time_boundary_features) > 1:
        logger.warning(
            'Multiple Time boundary features supplied (%d); only the first '
            "will be applied (Anuga collapses them all under one 'Time' tag)",
            len(time_boundary_features),
        )

    first = time_boundary_features[0]
    data = (first.get('properties') or {}).get('data')

    # Constant case: a single numeric stage value applied for all t.
    if data is None:
        # Don't blow up at parse time — let Anuga surface the failure when
        # it tries to evaluate the function. Constant 0 is the safest default.
        logger.error(
            "Time boundary feature %s has no resolved data; defaulting to 0.0",
            first.get('id'),
        )
        return lambda t: [0.0, 0.0, 0.0]
    if isinstance(data, (int, float)):
        constant = float(data)
        return lambda t: [constant, 0.0, 0.0]
    if isinstance(data, str):
        # Shouldn't happen if Boundary.make_file did its job, but tolerate
        # a numeric string just in case.
        try:
            constant = float(data)
            return lambda t: [constant, 0.0, 0.0]
        except (TypeError, ValueError):
            raise ValueError(
                f'Time boundary data must be numeric or a list of '
                f'{{timestamp, value}} dicts, got string {data!r} that '
                f'could not be float-coerced'
            )

    # TimeSeries case: a list of {timestamp, value} dicts.
    if not isinstance(data, list) or not data:
        raise ValueError(
            f'Time boundary data must be numeric or a non-empty list of '
            f'{{timestamp, value}} dicts, got: {type(data).__name__}'
        )

    numpy = import_optional("numpy")
    pd = import_optional("pandas")

    timestamps = []
    values = []
    for row in data:
        ts = row.get('timestamp') if isinstance(row, dict) else None
        val = row.get('value') if isinstance(row, dict) else None
        if ts is None or val is None:
            raise ValueError(
                f'Time boundary data row missing timestamp or value: {row!r}'
            )
        timestamps.append(pd.to_datetime(ts, utc=True))
        values.append(float(val))

    # Convert timestamps → seconds since the first sample. The Anuga function
    # is called with model-time-in-seconds (starting at 0).
    timestamps_sec = numpy.array(
        [(t - timestamps[0]).total_seconds() for t in timestamps],
        dtype=float,
    )
    values_arr = numpy.array(values, dtype=float)

    def _time_function(t):
        # numpy.interp clamps below[0]→values[0] and above[-1]→values[-1],
        # which matches the Inflow timeseries semantics (forward-fill at edges).
        stage = float(numpy.interp(float(t), timestamps_sec, values_arr))
        return [stage, 0.0, 0.0]

    return _time_function


def apply_inflows_to_domain(
    input_data,
    domain,
    start,
    duration,
    Polygonal_rate_operator,
    Inlet_operator,
    defaults_module=None,
):
    """Apply Rainfall, Surface and Catchment inflows to an Anuga ``domain``.

    Inflow.make_file() server-side resolves each feature's ``properties.data``
    via FeatureDataMixin (TASK-820) to one of:

      * a list of ``{timestamp: ISO8601, value: number}`` dicts (timeseries case),
      * a float (constant case, from ``data_constant`` or numeric legacy value),
      * ``None`` (no resolved value, e.g. a legacy free-text row with neither
        ``data_constant`` nor ``data_timeseries_id`` set).

    This function guards at the consumption boundary, mirroring the contract
    enforced by ``build_time_boundary_function``. Side effects:

      * For each Rainfall inflow, register a ``Polygonal_rate_operator`` on
        ``domain`` with a positive ``RAINFALL_FACTOR``.
      * For each Catchment polygon, register a ``Polygonal_rate_operator``
        with a *negative* ``RAINFALL_FACTOR`` (the catchment absorbs the rain
        falling on it and re-introduces it as a Surface inflow elsewhere).
      * For each Surface inflow line that is fully inside the boundary,
        register an ``Inlet_operator``.

    Raises:
        NotImplementedError: if a catchment is paired with a timeseries
            rainfall (catchments need a single uniform rate), or if more than
            one rainfall polygon is paired with a catchment.

    Returns a dict ``{feature_id: callable}`` of the inflow callables created,
    primarily for test introspection.
    """
    pd = import_optional("pandas")
    if defaults_module is None:
        defaults_module = defaults

    rainfall_inflow_polygons = (input_data.get('rainfall') or {}).get('features', []) or []
    surface_inflow_lines = (input_data.get('inflow') or {}).get('features', []) or []
    catchment_polygons = (
        [feature for feature in input_data.get('catchment').get('features')]
        if input_data.get('catchment') else []
    )
    boundary_polygon = input_data.get('boundary_polygon')

    datetime_range = pd.date_range(start=start, periods=duration + 1, freq='s')
    inflow_dataframe = pd.DataFrame(datetime_range, columns=['timestamp'])
    inflow_functions = dict()

    def create_inflow_function(dataframe, name):
        def rain(time_in_seconds):
            t_sec = int(math.floor(time_in_seconds))
            return dataframe[name][t_sec]
        rain.__name__ = name
        return rain

    def _merge_timeseries(name, rows):
        """Merge a timeseries list of ``{timestamp, value}`` dicts into
        ``inflow_dataframe`` under column ``name``, ffill-aligned to the
        model's per-second timestamp index.
        """
        nonlocal inflow_dataframe
        new_dataframe = pd.DataFrame(rows)
        new_dataframe['timestamp'] = pd.to_datetime(new_dataframe['timestamp'])
        new_dataframe[name] = pd.to_numeric(new_dataframe['value'])
        if inflow_dataframe['timestamp'].dt.tz is None:
            inflow_dataframe['timestamp'] = inflow_dataframe['timestamp'].dt.tz_localize('UTC')
        if new_dataframe['timestamp'].dt.tz is None:
            new_dataframe['timestamp'] = new_dataframe['timestamp'].dt.tz_localize('UTC')
        inflow_dataframe = pd.merge(inflow_dataframe, new_dataframe, how='left', on='timestamp')
        inflow_dataframe.ffill(inplace=True)

        # TASK-2155 (epic 2147 W2) — a series whose timestamps don't overlap
        # the model window AT ALL (e.g. a decades-old model_start mismatch)
        # left-merges to every row NaN; ffill can't invent values before the
        # first real sample, so the column stays entirely NaN. That silently
        # applies ZERO rain/inflow for the whole run with no error or log.
        # Operator decision: ERROR the run, NEVER warn-and-continue. A
        # PARTIAL lead-in gap (series starts partway through the window) is
        # NOT this failure mode and must not raise — only ALL-NaN does.
        if inflow_dataframe[name].isna().all():
            series_start = new_dataframe['timestamp'].min()
            series_end = new_dataframe['timestamp'].max()
            window_start = inflow_dataframe['timestamp'].min()
            window_end = inflow_dataframe['timestamp'].max()
            raise ValueError(
                f"Series '{name}' has ZERO overlap with the model window: "
                f"series spans [{series_start}, {series_end}] but the model "
                f"window is [{window_start}, {window_end}]. Applying this "
                f"series would silently yield zero rain/inflow for the "
                f"entire simulation — aborting instead."
            )

    for inflow_polygon in rainfall_inflow_polygons:
        polygon_name = inflow_polygon.get('id')
        data = inflow_polygon.get('properties').get('data')
        if data is None:
            logger.warning(
                "Rainfall inflow %s has no resolved data (data_constant "
                "and data_timeseries_id both unset); skipping",
                polygon_name,
            )
            continue
        if isinstance(data, list):
            _merge_timeseries(polygon_name, data)
        else:
            inflow_dataframe[polygon_name] = float(data)
        inflow_function = create_inflow_function(inflow_dataframe, polygon_name)
        inflow_functions[polygon_name] = inflow_function
        geometry = _extract_polygon_outer_ring(inflow_polygon.get('geometry'))
        Polygonal_rate_operator(
            domain,
            rate=inflow_function,
            factor=defaults_module.RAINFALL_FACTOR,
            polygon=geometry,
            default_rate=0.00,
        )

    if len(rainfall_inflow_polygons) > 1 and len(catchment_polygons) > 0:
        # Fail-fast BEFORE the catchment loop so no Polygonal_rate_operator
        # side effects are registered on a configuration we cannot honour.
        raise NotImplementedError(
            'Cannot handle multiple rainfall polygons together with catchment '
            'hydrology.'
        )

    if len(rainfall_inflow_polygons) >= 1 and len(catchment_polygons) > 0:
        first_rainfall_data = rainfall_inflow_polygons[0].get('properties').get('data')
        if first_rainfall_data is None:
            raise NotImplementedError(
                'Catchment hydrology requires the first rainfall inflow to '
                'have a resolved data value, got None. Set a data_constant '
                'or data_timeseries_id on the first rainfall, or remove the '
                'catchment.'
            )
        if isinstance(first_rainfall_data, list):
            # Catchments use a SINGLE uniform rate, which is undefined for a
            # time-varying input. Surface the limitation clearly.
            raise NotImplementedError(
                'Catchment hydrology with a timeseries rainfall inflow is '
                'not supported. Use a constant rainfall value or remove the '
                'catchment.'
            )
        for catchment_polygon in catchment_polygons:
            uniform_rainfall_rate = float(first_rainfall_data)
            polygon_name = catchment_polygon.get('id')
            inflow_dataframe[polygon_name] = uniform_rainfall_rate
            inflow_function = create_inflow_function(inflow_dataframe, polygon_name)
            geometry = _extract_polygon_outer_ring(catchment_polygon.get('geometry'))
            # The catchment needs to be wholly in the domain:
            if check_coordinates_are_in_polygon(geometry, boundary_polygon):
                Polygonal_rate_operator(
                    domain,
                    rate=inflow_function,
                    factor=-defaults_module.RAINFALL_FACTOR,
                    polygon=geometry,
                    default_rate=0.00,
                )

    for inflow_line in surface_inflow_lines:
        polyline_name = inflow_line.get('id')
        data = inflow_line.get('properties').get('data')
        if data is None:
            logger.warning(
                "Surface inflow %s has no resolved data (data_constant "
                "and data_timeseries_id both unset); skipping",
                polyline_name,
            )
            continue
        if isinstance(data, list):
            _merge_timeseries(polyline_name, data)
        else:
            inflow_dataframe[polyline_name] = float(data)
        inflow_function = create_inflow_function(inflow_dataframe, polyline_name)
        inflow_functions[polyline_name] = inflow_function
        geometry = _flatten_line_coordinates(inflow_line.get('geometry'))
        if check_coordinates_are_in_polygon(geometry, boundary_polygon):
            Inlet_operator(domain, geometry, Q=inflow_function)

    return inflow_functions


def post_process_sww(package_dir, run_args=None, output_raster_resolution=None):
    # TASK-1143: ANUGA result rasters (depth/velocity/depthIntegratedVelocity/stage)
    # intentionally keep NaN as their nodata value, NOT -9999.  NaN is propagated
    # by Make_Geotif (plot_utils.py nodata=numpy.nan) and relied upon by make_video
    # (np.ma.masked_invalid) and make_raster_diff (--NoDataValue=nan).  Switching to
    # -9999 would break both without any benefit — result layers are single-coverage
    # per run so there is no overlapping-edge problem.  The assertion block below
    # guards against a future anuga_core default flip.
    anuga = import_optional("anuga")
    util = anuga.utilities.plot_utils
    output_quantities = ['depth', 'velocity', 'depthIntegratedVelocity', 'stage']
    input_data = setup_input_data(package_dir)
    logger.critical(f'Generating output rasters on {anuga.myid}...')
    resolutions = list()
    if input_data.get('mesh_region'):
        for feature in input_data.get('mesh_region').get('features') or list():
            # logger.critical(f'{feature=}')
            resolutions.append(feature.get('properties').get('resolution'))
    logger.critical(f'{resolutions=}')
    if len(resolutions) == 0:
        resolutions = [input_data.get('resolution') or 1000]
    finest_grid_resolution = min(resolutions)
    logger.critical(f'raster output resolution: {finest_grid_resolution}m')

    epsg_integer = int(input_data['scenario_config'].get("epsg").split(":")[1]
                       if ":" in input_data['scenario_config'].get("epsg")
                       else input_data['scenario_config'].get("epsg"))
    interior_holes, _ = make_interior_holes_and_tags(input_data)
    util.Make_Geotif(
        swwFile=f"{input_data['output_directory']}/{input_data['run_label']}.sww",
        output_quantities=output_quantities,
        myTimeStep='all',
        CellSize=finest_grid_resolution,
        lower_left=None,
        upper_right=None,
        EPSG_CODE=epsg_integer,
        proj4string=None,
        velocity_extrapolation=True,
        min_allowed_height=defaults.MIN_ALLOWED_HEIGHT_M,
        output_dir=input_data['output_directory'],
        bounding_polygon=input_data['boundary_polygon'],
        internal_holes=interior_holes,
        verbose=False,
        k_nearest_neighbours=defaults.K_NEAREST_NEIGHBOURS,
        creation_options=[]
    )
    util.Make_Geotif(
        swwFile=f"{input_data['output_directory']}/{input_data['run_label']}.sww",
        output_quantities=output_quantities,
        myTimeStep='max',
        CellSize=finest_grid_resolution,
        lower_left=None,
        upper_right=None,
        EPSG_CODE=epsg_integer,
        proj4string=None,
        velocity_extrapolation=True,
        min_allowed_height=defaults.MIN_ALLOWED_HEIGHT_M,
        output_dir=input_data['output_directory'],
        bounding_polygon=input_data['boundary_polygon'],
        internal_holes=interior_holes,
        verbose=False,
        k_nearest_neighbours=defaults.K_NEAREST_NEIGHBOURS,
        creation_options=[]
    )

    # TASK-1143: guard against a future anuga_core default flip away from NaN.
    # Re-open the *_max.tif outputs and assert band 1 nodata is NaN.
    import math
    _rasterio = import_optional("rasterio")
    for _quantity in output_quantities:
        _max_tif = os.path.join(
            input_data['output_directory'],
            f"{input_data['run_label']}_{_quantity}_max.tif",
        )
        if os.path.isfile(_max_tif):
            with _rasterio.open(_max_tif) as _ds:
                _nodata = _ds.nodata
            assert _nodata is not None and math.isnan(_nodata), (
                f"Result raster '{_max_tif}' nodata is {_nodata!r}; expected NaN. "
                "ANUGA result rasters intentionally use NaN nodata — do not switch to -9999."
            )

    video_dir = f"{input_data['output_directory']}/videos/"
    if os.path.isdir(video_dir):
        shutil.rmtree(video_dir)
    logger.critical('Successfully generated depth, velocity, momentum outputs')


def reprocess_from_archived_sww(
    bucket: str,
    sww_key: str,
    *,
    output_dir: "str | Path | None" = None,
    quantity: str = "depth",
    cell_size: float = 10.0,
    epsg_code: int = 32754,
) -> "Path":
    """Fetch a .sww from the cold-archive S3 prefix and regenerate a max raster.

    W2 (TASK-1921) — Reprocess-from-archive entrypoint / smoke.  TASK-1821
    noted "no reprocess-from-sww code path exists"; this function is the seam
    that a future flow-animation or re-render feature builds on.

    Fetches the .sww from ``s3://<bucket>/<sww_key>`` into a temporary
    directory, then runs the existing ``Make_Geotif`` reader (the same one
    ``post_process_sww`` uses) for ONE quantity and ONE time-step ('max') and
    returns the path to the produced raster.

    Parameters
    ----------
    bucket
        S3 bucket name (matches ``RESULT_S3_BUCKET``).
    sww_key
        Full S3 key of the .sww file, e.g.
        ``cold-archive/601_384_1243/601_384_1243.sww``.
    output_dir
        Directory to write the produced raster(s) into.  When ``None`` a
        temporary directory is created (caller is responsible for cleanup if
        persistence is needed).
    quantity
        ANUGA output quantity to render (default ``"depth"``).
    cell_size
        Raster cell size in metres (default 10.0 — coarse, fast, smoke-only).
    epsg_code
        EPSG integer for the output raster CRS (default 32754 — Merewether
        benchmark projection; override for other domains).

    Returns
    -------
    Path
        Path to the produced ``*_max.tif`` raster.  The file is non-empty and
        readable by rasterio / GDAL.

    Raises
    ------
    ImportError
        If ``boto3`` or ``anuga`` are not installed (lazy via import_optional).
    FileNotFoundError
        If the produced raster is missing after ``Make_Geotif`` completes.
    """
    import tempfile

    boto3 = import_optional("boto3")
    anuga = import_optional("anuga")
    util = anuga.utilities.plot_utils

    # Build a working directory.
    if output_dir is None:
        _tmpdir_obj = tempfile.TemporaryDirectory()
        output_dir = Path(_tmpdir_obj.name)
    else:
        _tmpdir_obj = None
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a local filename from the S3 key.
    sww_filename = sww_key.rsplit("/", 1)[-1]  # e.g. "601_384_1243.sww"
    local_sww = output_dir / sww_filename

    # Fetch the .sww from S3 cold archive.
    s3 = boto3.client("s3")
    logger.info(
        "reprocess_from_archived_sww: fetching s3://%s/%s -> %s",
        bucket, sww_key, local_sww,
    )
    s3.download_file(bucket, sww_key, str(local_sww))

    # Derive run_label from the .sww filename (strip .sww extension).
    run_stem = sww_filename[:-4]  # e.g. "601_384_1243"

    # Re-render ONE quantity at ONE time-step using Make_Geotif.
    logger.info(
        "reprocess_from_archived_sww: running Make_Geotif(quantity=%s, myTimeStep=max)",
        quantity,
    )
    util.Make_Geotif(
        swwFile=str(local_sww),
        output_quantities=[quantity],
        myTimeStep="max",
        CellSize=cell_size,
        lower_left=None,
        upper_right=None,
        EPSG_CODE=epsg_code,
        proj4string=None,
        velocity_extrapolation=True,
        min_allowed_height=defaults.MIN_ALLOWED_HEIGHT_M,
        output_dir=str(output_dir),
        bounding_polygon=None,
        internal_holes=None,
        verbose=False,
        k_nearest_neighbours=defaults.K_NEAREST_NEIGHBOURS,
        creation_options=[],
    )

    # Locate the produced raster.  Make_Geotif writes
    # ``<stem>_<quantity>_max.tif`` into output_dir.
    expected_name = f"{run_stem}_{quantity}_max.tif"
    expected_path = output_dir / expected_name
    if not expected_path.exists():
        # Fallback: glob for any *_max.tif produced in output_dir.
        candidates = list(output_dir.glob(f"*_{quantity}_max.tif"))
        if candidates:
            expected_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"reprocess_from_archived_sww: Make_Geotif did not produce "
                f"'{expected_name}' (or any *_{quantity}_max.tif) in {output_dir}"
            )

    logger.info(
        "reprocess_from_archived_sww: produced raster %s (%d bytes)",
        expected_path, expected_path.stat().st_size,
    )
    return expected_path


def make_video(input_directory_1, result_type):
    np = import_optional("numpy")
    rasterio = import_optional("rasterio")
    cv2 = import_optional("cv2")
    plt = import_optional("matplotlib.pyplot")
    pe = import_optional("matplotlib.patheffects")
    run_label_1 = os.path.basename(input_directory_1).replace('outputs', 'run')
    tif_files = glob.glob(f"{input_directory_1}/{run_label_1}_{result_type}_*.tif")
    tif_files = [tif_file for tif_file in tif_files if "_max" not in tif_file]
    tif_files.sort(key=lambda f: int(os.path.splitext(f)[0][-6:]))

    max_file = f"{input_directory_1}/{run_label_1}_{result_type}_max.tif"
    global_min = 0
    raster_data = rasterio.open(max_file).read(1)
    masked_data_raster_data = np.ma.masked_invalid(raster_data)
    global_max = masked_data_raster_data.max()
    image_files = list()
    image_directory = f"{input_directory_1}/videos"
    if not os.path.exists(image_directory):
        os.makedirs(image_directory, exist_ok=True)
    for i, file in enumerate(tif_files):
        raster = rasterio.open(file)
        data = raster.read(1)
        label = str(file).split('/')[-1]
        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap='viridis', vmin=global_min, vmax=global_max)
        plt.axis('off')
        plt.text(0, 0, label, color='white', fontsize=10, ha='left', va='top', path_effects=[pe.withStroke(linewidth=3, foreground='black')])
        img_file = f"{image_directory}/frame_{result_type}_{i:03d}.png"
        plt.savefig(img_file, dpi=300)
        image_files.append(img_file)
        plt.close()

    img = cv2.imread(image_files[0])
    height, width, _ = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"{input_directory_1}/{run_label_1}_{result_type}.mp4", fourcc, 20.0, (width, height))

    for img_file in image_files:
        img = cv2.imread(img_file)
        out.write(img)
    out.release()


def make_comparison_video(input_directory_1, input_directory_2, result_type):
    np = import_optional("numpy")
    rasterio = import_optional("rasterio")
    cv2 = import_optional("cv2")
    plt = import_optional("matplotlib.pyplot")
    pe = import_optional("matplotlib.patheffects")
    run_label_1 = os.path.basename(input_directory_1).replace('outputs', 'run')
    run_label_2 = os.path.basename(input_directory_2).replace('outputs', 'run')

    tif_files_1 = glob.glob(f"{input_directory_1}/{run_label_1}_{result_type}_*.tif")
    tif_files_1 = sorted([tif_file for tif_file in tif_files_1 if "_max" not in tif_file],
                         key=lambda f: int(os.path.splitext(f)[0][-6:]))

    tif_files_2 = glob.glob(f"{input_directory_2}/{run_label_2}_{result_type}_*.tif")
    tif_files_2 = sorted([tif_file for tif_file in tif_files_2 if "_max" not in tif_file],
                         key=lambda f: int(os.path.splitext(f)[0][-6:]))

    assert len(tif_files_1) == len(tif_files_2), "Number of tif files in the two directories are not the same."

    max_file = f"{input_directory_1}/{run_label_1}_{result_type}_max.tif"
    global_min = 0
    raster_data = rasterio.open(max_file).read(1)
    masked_data_raster_data = np.ma.masked_invalid(raster_data)
    global_max = masked_data_raster_data.max()

    image_files_1 = list()
    image_files_2 = list()
    image_files_diff = list()

    image_directory_1 = f"{input_directory_1}/videos"
    image_directory_2 = f"{input_directory_2}/videos"
    image_directory_diff = f"{input_directory_1}/diff_videos"

    for directory in [image_directory_1, image_directory_2, image_directory_diff]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    for i, (file_1, file_2) in enumerate(zip(tif_files_1, tif_files_2)):
        raster_1 = rasterio.open(file_1).read(1)
        raster_2 = rasterio.open(file_2).read(1)

        diff = np.abs(raster_2 - raster_1)

        for file, image_directory, image_files, data in [(file_1, image_directory_1, image_files_1, raster_1),
                                                         (file_2, image_directory_2, image_files_2, raster_2),
                                                         (None, image_directory_diff, image_files_diff, diff)]:
            plt.figure(figsize=(10, 10))
            if "diff" in image_directory:
                cmap = "RdBu"
            else:
                cmap = "viridis"
            plt.imshow(data, cmap=cmap, vmin=global_min, vmax=global_max)
            plt.axis('off')
            if file is not None:
                label = str(file).split('/')[-1]
                plt.text(0, 0, label, color='white', fontsize=10, ha='left', va='top',
                         path_effects=[pe.withStroke(linewidth=3, foreground='black')])
            img_file = f"{image_directory}/frame_{result_type}_{i:03d}.png"
            plt.savefig(img_file, dpi=100)
            image_files.append(img_file)
            plt.close()

    img1 = cv2.imread(image_files_1[0])
    img2 = cv2.imread(image_files_2[0])
    img_diff = cv2.imread(image_files_diff[0])

    largest_width = max([img1.shape[1], img2.shape[1], img_diff.shape[1]])
    height, _, _ = img1.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f"{input_directory_1}/{run_label_1}_{result_type}_difference.avi", fourcc, 20.0,
                          (3 * largest_width, height))

    for file1, file2, file_diff in zip(image_files_1, image_files_2, image_files_diff):
        img1 = cv2.imread(file1)
        img2 = cv2.imread(file2)
        img_diff = cv2.imread(file_diff)

        def pad_image(img):
            padding = ((0, 0), (0, largest_width - img.shape[1]), (0, 0))
            return np.pad(img, padding, mode='constant')

        img1 = pad_image(img1)
        img2 = pad_image(img2)
        img_diff = pad_image(img_diff)

        combined_img = np.concatenate((img1, img_diff, img2), axis=1)
        out.write(combined_img)

    out.release()


class _V2LogHandler(logging.Handler):
    """Logging handler that ships records to the V2 ``/log/`` control endpoint.

    Replaces the legacy ``logging.handlers.HTTPHandler`` (V1 BasicAuth to
    ``/anuga/api/{p}/{s}/run/{r}/log/``) installed by ``setup_logger`` before
    TASK-989. Auth is the site-wide ``X-Internal-Token`` shared secret (RAW,
    no ``Bearer`` prefix — see ``gn_anuga.permissions.IsInternalComputeCaller``),
    carried on a single owned ``requests.Session`` reused across every emit.

    Mirrors ``HydrataCallback`` (callbacks.py): the V2 log endpoint accepts
    ``{message, levelname, created}`` and writes the formatted line to
    ``Run.log``; ``project_id``/``scenario_id`` are inferred server-side from
    the run row, so only ``run_id`` appears in the URL.

    Emit failures never propagate (logging must not break the run loop): a
    transport error is routed through ``logging.Handler.handleError``. The
    owned Session pool is released by ``close()`` (idempotent); ``setup_logger``
    pairs ``addHandler`` with ``removeHandler`` + ``close()`` on re-entry.
    """

    def __init__(self, control_server: str, run_id, token: str):
        super().__init__()
        from run_anuga._http import make_internal_session

        base = str(control_server).rstrip('/')
        self._log_url = f"{base}/api/v2/anuga/runs/{run_id}/log/"
        self._session = make_internal_session(token)

    def emit(self, record: logging.LogRecord) -> None:
        from run_anuga._http import post_to_control_server

        try:
            post_to_control_server(
                self._log_url,
                method='POST',
                data={
                    'message': self.format(record),
                    'levelname': record.levelname,
                    'created': record.created,
                },
                session=self._session,
                timeout=30,
            )
        except Exception:
            # Never let a log shipment break the run — route to the standard
            # logging error hook (respects logging.raiseExceptions, which is
            # False in production).
            self.handleError(record)

    def close(self) -> None:
        """Release the owned ``requests.Session`` and unregister the handler.

        Idempotent: a second call is a no-op (``self._session`` may already be
        closed; ``requests.Session.close()`` tolerates repeat calls).
        """
        try:
            self._session.close()
        except Exception:  # pragma: no cover — defensive
            pass
        super().close()


def setup_logger(input_data, username=None, password=None, batch_number=1):
    if not username or not password:
        username = os.environ.get('COMPUTE_USERNAME')
        password = os.environ.get('COMPUTE_PASSWORD')
    # Create handlers
    file_handler = logging.FileHandler(os.path.join(input_data['output_directory'], f'run_anuga_{batch_number}.log'))
    file_handler.setLevel(logging.DEBUG)

    # Avoid duplicate handlers when run_sim() is called multiple times.
    # Close removed network handlers so their owned Session connection pool
    # is released (the V2 handler below owns a requests.Session); FileHandler
    # also benefits from close() flushing/closing its file descriptor.
    for h in logger.handlers[:]:
        if isinstance(h, (logging.FileHandler, logging.handlers.HTTPHandler, _V2LogHandler)):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:  # pragma: no cover — defensive, never break setup
                pass

    # Add handlers to the logger
    logger.addHandler(file_handler)

    # Ship log lines to the control server over the V2 log endpoint using the
    # site-wide X-Internal-Token shared secret (NOT the legacy V1 BasicAuth
    # HTTPHandler, which 401'd against allauth on localhost and targeted the
    # /anuga/api/.../log/ URL that TASK-1184 will delete). The token is read
    # from the environment, matching how run.py builds HydrataCallback; the
    # username/password parameters are retained for signature back-compat but
    # are no longer used for the web log channel. When the token is absent
    # (e.g. a standalone CLI run) no web handler is installed and logging
    # stays file/console-only.
    token = os.environ.get('HYDRATA_INTERNAL_COMPUTE_TOKEN')
    control_server = input_data['scenario_config'].get('control_server')
    run_id = input_data['scenario_config'].get('run_id')
    if token and control_server and run_id:
        web_handler = _V2LogHandler(
            control_server=control_server,
            run_id=run_id,
            token=token,
        )
        web_handler.setLevel(logging.DEBUG)
        logger.addHandler(web_handler)
    logger.setLevel(logging.DEBUG)
    return logger


def burn_structures_into_raster(structures_filename, raster_filename, backup=True):
    if backup:
        shutil.copyfile(raster_filename, f"{raster_filename[:-4]}_original.tif")
    output = subprocess.run(["gdal_rasterize", "-burn", str(defaults.BUILDING_BURN_HEIGHT_M), "-add", structures_filename, raster_filename], capture_output=True, universal_newlines=True)
    print(output)
    if output.returncode != 0:
        raise RuntimeError(output.stderr)
    return True


def make_shp_from_polygon(boundary_polygon, epsg_code, shapefilepath):
    ogr = import_optional("osgeo.ogr")
    osr = import_optional("osgeo.osr")
    boundary_ring_geom = ogr.Geometry(ogr.wkbLinearRing)
    for point in boundary_polygon:
        boundary_ring_geom.AddPoint(point[0], point[1])
    boundary_ring_geom.AddPoint(boundary_polygon[0][0], boundary_polygon[0][1])
    boundary_polygon_geom = ogr.Geometry(ogr.wkbPolygon)
    boundary_polygon_geom.AddGeometry(boundary_ring_geom)
    logger.critical(f"{boundary_polygon_geom=}")

    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(shapefilepath)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    layer = ds.CreateLayer("mesh_region", srs, ogr.wkbPolygon)
    id_field = ogr.FieldDefn("id", ogr.OFTInteger)
    layer.CreateField(id_field)
    feature_defn = layer.GetLayerDefn()
    feature = ogr.Feature(feature_defn)
    feature.SetGeometry(boundary_polygon_geom)
    feature.SetField("id", 1)
    layer.CreateFeature(feature)
    feature = None
    ds = None


def snap_links_to_nodes(package_dir):
    print('snap_links_to_nodes')
    raise NotImplementedError


def calculate_hydrology(package_dir):
    shapely_geometry = import_optional("shapely.geometry")
    Point, Polygon = shapely_geometry.Point, shapely_geometry.Polygon
    prep = import_optional("shapely.prepared").prep
    input_data = setup_input_data(package_dir)

    # prepare Nodes
    node_points = list()
    for node in input_data.get('nodes').get('features'):
        node_point = Point(node.get('geometry').get('coordinates'))
        node_points.append(node_point)

    # assign catchments to nodes
    for index, catchment in enumerate(input_data.get('catchment').get('features')):
        catchment_polygon = prep(Polygon(_extract_polygon_outer_ring(catchment.get('geometry'))))
        node_candidate = list(filter(catchment_polygon.contains, node_points))
        if len(node_candidate) == 0:
            logger.critical(f"Catchment {catchment.get('id')} has no internal node")
            continue
        if len(node_candidate) > 1:
            raise IndexError(f"Catchment {catchment.get('id')} has more than one node")
        print('find', node_candidate)
        node_index = node_points.index(node_candidate[0])
        input_data['catchment']['features'][index]['node_id'] = input_data.get('nodes').get('features')[node_index].get('id')

    rainfall_features = (input_data.get('rainfall') or {}).get('features', []) or []
    if not rainfall_features:
        logger.warning(
            "calculate_hydrology: no rainfall features supplied; "
            "skipping surface_flow_m3_s assignment for catchments"
        )
        return True
    rainfall_steady_state_intensity_mm_hr = float(rainfall_features[0].get('properties').get('data'))
    rainfall_steady_state_intensity_m_s = rainfall_steady_state_intensity_mm_hr * 0.001 / 3600
    for index, catchment in enumerate(input_data.get('catchment').get('features')):
        area_m2 = Polygon(_extract_polygon_outer_ring(catchment.get('geometry'))).area
        input_data['catchment']['features'][index]['surface_flow_m3_s'] = 1.0 * rainfall_steady_state_intensity_m_s * area_m2

    # create surface inflows at catchment nodes
    for index, catchment in enumerate(input_data.get('catchment').get('features')):
        if catchment.get('surface_flow_m3_s'):
            node_id = catchment.get('node_id')
            location = None
            for node in input_data.get('nodes').get('features'):
                if node.get('id') == node_id:
                    location = node.get('geometry').get('coordinates')
            # create an inflow object
            start_point = [location[0] - 10, location[1]]
            end_point = [location[0] + 10, location[1]]
            coordinates = [start_point, end_point]
            inflow_object = make_new_inflow(node_id, coordinates, catchment.get('surface_flow_m3_s'))
            add_inflow_to_file(inflow_object, input_data.get('inflow_filename'))

    return True


def make_new_inflow(inflow_id, coordinates, flow):
    return {
        'type': 'Feature',
        'id': f'inf_16_inflow_01.{inflow_id}',
        'geometry': {
            'type': 'LineString',
            'coordinates': coordinates
        },
        'geometry_name': 'the_geom',
        'properties': {
            'fid': 1,
            'type': 'Surface',
            'data': str(flow),
            'description': None
        }
    }


def add_inflow_to_file(inflow_object, filepath):
    with open(filepath) as f:
        file_contents = json.load(f)
    file_contents['features'].append(inflow_object)
    with open(filepath, 'w') as json_file:
        json.dump(file_contents, json_file)
    return True


def check_coordinates_are_in_polygon(coordinates, polygon):
    """
    TASK-2187 (epic 2147 W2): a caller can hand this a CLOSED RING (e.g. a
    Rainfall Polygon's outer ring mistakenly routed through the Surface-
    inflow geometry path — ``_flatten_line_coordinates`` falls through to
    ``coords`` unchanged for a non-LineString geometry type, ~run_utils.py:
    892) or another nested/GeoJSON-shaped ``coordinates`` list. Passing that
    straight to ``shapely.geometry.Point()`` raises an opaque
    "Point() takes only scalar or 1-size vector arguments" crash that gives
    the operator no idea which feature or file caused it.

    Every entry of ``coordinates`` (after the single-point normalisation
    below) MUST be a flat ``[x, y]`` / ``[x, y, z]`` pair of real numbers —
    exactly what ``_flatten_line_coordinates`` produces for the supported
    LineString/MultiLineString inflow shapes (TASK-1113). Anything else
    raises a CLEAR, named ``ValueError`` instead of either crashing
    opaquely or silently returning False (a silent False would skip
    registering the operator with no signal at all — the same
    never-warn-and-continue guard as the TASK-2155 NaN guard).
    """
    if not coordinates:
        return False
    shapely_geometry = import_optional("shapely.geometry")
    Point, Polygon = shapely_geometry.Point, shapely_geometry.Polygon
    shapely_polgyon = Polygon(polygon)
    if isinstance(coordinates[0], Real):
        coordinates = [coordinates]
    for point in coordinates:
        if not (
            isinstance(point, (list, tuple))
            and len(point) in (2, 3)
            and all(isinstance(c, Real) for c in point)
        ):
            raise ValueError(
                f"check_coordinates_are_in_polygon: expected a flat [x, y] "
                f"point but got {point!r} — coordinates looks like a nested "
                f"ring/polygon was passed where a flat point list was "
                f"expected (full coordinates={coordinates!r})."
            )
        shapely_point = Point(point)
        if not shapely_polgyon.contains(shapely_point):
            return False
    return True


def generate_stac(output_directory, run_label, output_quantities, initial_time_iso_string,
                  aws_access_key_id=None, aws_secret_access_key=None, s3_bucket_name=None):
    # Deterministic no-op when required scenario_config inputs are missing.
    # Without this guard, datetime.fromisoformat(None) and the for-loop over
    # output_quantities raise on every legacy run (no initial_time/output_quantities
    # in older packages). Log once so operators can spot the coverage gap.
    if not initial_time_iso_string or not output_quantities:
        logger.info(
            "STAC generation skipped: missing required inputs "
            "(run_label=%s, initial_time_iso_string=%r, output_quantities=%r)",
            run_label, initial_time_iso_string, output_quantities,
        )
        return
    _explicit_creds = aws_access_key_id is not None
    # Fall back to Django settings if params not provided
    if not _explicit_creds:
        if isinstance(settings, dict):
            return  # No Django settings and no explicit creds — silently skip
        aws_access_key_id = getattr(settings, 'AWS_ACCESS_KEY_ID', None)
        aws_secret_access_key = getattr(settings, 'AWS_SECRET_ACCESS_KEY', None)
        s3_bucket_name = getattr(settings, 'ANUGA_S3_STAC_BUCKET_NAME', None)
    if not aws_access_key_id or not aws_secret_access_key or not s3_bucket_name:
        if not _explicit_creds:
            return  # Django settings incomplete — silently skip (matches old behavior)
        raise ValueError("AWS credentials required: pass aws_access_key_id, aws_secret_access_key, and s3_bucket_name")

    boto3 = import_optional("boto3")
    rasterio = import_optional("rasterio")
    pystac = import_optional("pystac")
    Item, Asset, Collection, MediaType = pystac.Item, pystac.Asset, pystac.Collection, pystac.MediaType
    Extent, SpatialExtent, TemporalExtent = pystac.Extent, pystac.SpatialExtent, pystac.TemporalExtent
    CatalogType, Catalog = pystac.CatalogType, pystac.Catalog
    from pystac.stac_io import DefaultStacIO, StacIO

    class S3StacIO(DefaultStacIO):
        def __init__(self):
            self.s3 = boto3.resource(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            super().__init__()

        def read_text(self, source, *args, **kwargs) -> str:
            parsed = urlparse(source)
            if parsed.scheme == "s3":
                bucket = parsed.netloc
                key = parsed.path[1:]
                obj = self.s3.Object(bucket, key)
                return obj.get()["Body"].read().decode("utf-8")
            else:
                return super().read_text(source, *args, **kwargs)

        def write_text(self, dest, txt, *args, **kwargs) -> None:
            parsed = urlparse(dest)
            if parsed.scheme == "s3":
                bucket = parsed.netloc
                key = parsed.path[1:]
                self.s3.Object(bucket, key).put(Body=txt, ContentEncoding="utf-8")
            else:
                super().write_text(dest, txt, *args, **kwargs)

    StacIO.set_default(S3StacIO)

    s3_catalog_uri = f"s3://{s3_bucket_name}/{run_label}"
    catalog = Catalog(id=run_label, description=f"{run_label} - {initial_time_iso_string}")
    min_left, min_bottom, max_right, max_top = None, None, None, None
    min_datetime, max_datetime = None, None
    initial_time = datetime.datetime.fromisoformat(initial_time_iso_string)

    for result_type in output_quantities:
        tif_files = glob.glob(f"{output_directory}/{run_label}_{result_type}_*.tif")
        tif_files = [tif_file for tif_file in tif_files if "_max" not in tif_file]
        tif_files.sort(key=lambda f: int(os.path.splitext(f)[0][-6:]))
        items = []
        collection = Collection(
            id=f"{run_label}_{result_type}",
            description="test",
            extent=Extent(
                spatial=SpatialExtent([[-180, -90, 180, 90]]),
                temporal=TemporalExtent([["2028-01-01T00:00:00Z", None]]),
            ),
        )

        for tif_file in tif_files:
            item_name = os.path.basename(tif_file).split('.')[0]
            s3_resource = boto3.resource(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            s3_bucket = s3_resource.Bucket(s3_bucket_name)
            key = f"{run_label}/{result_type}/{os.path.basename(tif_file)}"
            with open(tif_file, 'rb') as data:
                s3_bucket.upload_fileobj(data, key)
            s3_tif_url = f"https://{s3_bucket_name}.s3.us-west-2.amazonaws.com/{key}"
            model_time_sec = int(tif_file[-10:-4])
            time_elapsed = initial_time + datetime.timedelta(seconds=model_time_sec)
            with rasterio.open(tif_file) as dataset:
                bbox = dataset.bounds
            item = Item(
                id=item_name,
                geometry={},
                bbox=[bbox.left, bbox.bottom, bbox.right, bbox.top],
                datetime=time_elapsed,
                properties={}
            )
            asset = Asset(
                href=s3_tif_url,
                media_type=MediaType.GEOTIFF
            )
            item.add_asset(key='data', asset=asset)
            items.append(item)
            collection.add_item(item)

            if min_left is None or bbox.left < min_left:
                min_left = bbox.left
            if min_bottom is None or bbox.bottom < min_bottom:
                min_bottom = bbox.bottom
            if max_right is None or bbox.right > max_right:
                max_right = bbox.right
            if max_top is None or bbox.top > max_top:
                max_top = bbox.top
            if min_datetime is None or time_elapsed < min_datetime:
                min_datetime = time_elapsed
            if max_datetime is None or time_elapsed > max_datetime:
                max_datetime = time_elapsed
        collection.extent = Extent(
            spatial=SpatialExtent([min_left, min_bottom, max_right, max_top]),
            temporal=TemporalExtent([[min_datetime, max_datetime]])
        )
        catalog.add_child(collection)

    catalog.extent = Extent(
        spatial=SpatialExtent([min_left, min_bottom, max_right, max_top]),
        temporal=TemporalExtent([[min_datetime, max_datetime]])
    )

    catalog.normalize_and_save(
        root_href=s3_catalog_uri,
        catalog_type=CatalogType.SELF_CONTAINED
    )
