"""Gmsh-based mesh generation for ANUGA simulations.

Two sizing modes:
- ``flat``:  uniform element size = resolution (burned DEM, no building holes)
- ``proximity``: adaptive sizing near building edges (original DEM, buildings as holes)
"""

import json
import logging

import gmsh
import numpy as np
import rasterio

from . import MeshResult

logger = logging.getLogger(__name__)

# ANUGA edge convention: edge k is opposite vertex k in the triangle
_EDGE_VERTS = [(1, 2), (0, 2), (0, 1)]


def generate_mesh(input_data):
    """Generate a 2D triangular mesh using Gmsh.

    Parameters
    ----------
    input_data : dict
        The ``input_data`` dict produced by ``setup_input_data()``, containing
        ``scenario_config``, ``package_dir``, ``elevation_filename``, etc.

    Returns
    -------
    MeshResult
    """
    config = input_data["scenario_config"]
    params = config.get("mesher_params", {})
    sizing = params.get("sizing", "flat")
    resolution = config.get("resolution", 2.0)

    # Load boundary polygon and tags
    boundary_segments, boundary_ring = _load_boundary(
        input_data["boundary_filename"]
    )

    # Load buildings for hole mode
    buildings = []
    if sizing == "proximity" and "structure_filename" in input_data:
        buildings = _load_buildings(input_data["structure_filename"])

    # DEM path for elevation sampling
    dem_path = input_data["elevation_filename"]

    # Generate mesh with Gmsh
    points, triangles = _run_gmsh(
        boundary_ring, buildings, resolution, sizing, params
    )

    logger.info(
        f"Gmsh ({sizing}): {len(points)} vertices, {len(triangles)} triangles"
    )

    # Sample elevation from DEM
    elevation = _sample_elevation(points, dem_path)

    # Compute boundary tags
    boundary_tags = _compute_boundary_tags(
        points, triangles, boundary_segments
    )

    return MeshResult(
        points=points,
        triangles=triangles,
        elevation=elevation,
        boundary_tags=boundary_tags,
    )


def _load_boundary(path):
    """Load boundary segments and build the outer ring.

    Returns
    -------
    segments : list of (tag, coords)
        Each segment is (tag_string, [[x1,y1],[x2,y2]]).
    ring : list of [x, y]
        Closed polygon ring (first == last).
    """
    with open(path) as f:
        data = json.load(f)

    segments = []
    ring = []
    for feat in data["features"]:
        tag = feat["properties"]["boundary"]
        coords = feat["geometry"]["coordinates"]
        segments.append((tag, coords))
        # Build ring from segment start points
        ring.append(coords[0])
    # Close the ring
    ring.append(ring[0])
    return segments, ring


def _load_buildings(path):
    """Load building polygons from structure GeoJSON.

    Returns list of coordinate rings (each is list of [x, y], closed).
    """
    with open(path) as f:
        data = json.load(f)

    buildings = []
    for feat in data["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            ring = geom["coordinates"][0]
            buildings.append(ring)
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                buildings.append(poly[0])
    return buildings


def _run_gmsh(boundary_ring, buildings, resolution, sizing, params):
    """Run Gmsh and return (points, triangles) as numpy arrays.

    points:    (N, 2) float64
    triangles: (M, 3) int, 0-based
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("mesh")

    # --- Outer boundary ---
    outer_pts = []
    for x, y in boundary_ring[:-1]:  # skip closing duplicate
        tag = gmsh.model.occ.addPoint(x, y, 0)
        outer_pts.append(tag)

    outer_lines = []
    n = len(outer_pts)
    for i in range(n):
        line = gmsh.model.occ.addLine(outer_pts[i], outer_pts[(i + 1) % n])
        outer_lines.append(line)

    outer_loop = gmsh.model.occ.addCurveLoop(outer_lines)

    # --- Building holes ---
    hole_loops = []
    for bld_ring in buildings:
        bld_pts = []
        for x, y in bld_ring[:-1]:
            tag = gmsh.model.occ.addPoint(x, y, 0)
            bld_pts.append(tag)
        bld_lines = []
        nb = len(bld_pts)
        for i in range(nb):
            line = gmsh.model.occ.addLine(bld_pts[i], bld_pts[(i + 1) % nb])
            bld_lines.append(line)
        loop = gmsh.model.occ.addCurveLoop(bld_lines)
        hole_loops.append(loop)

    # Create surface with holes
    surface = gmsh.model.occ.addPlaneSurface([outer_loop] + hole_loops)
    gmsh.model.occ.synchronize()

    # Physical group required for getElements to return anything
    gmsh.model.addPhysicalGroup(2, [surface], tag=1)

    # --- Sizing ---
    if sizing == "flat":
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", resolution)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", resolution * 0.5)
    elif sizing == "proximity":
        size_min = params.get("size_min", 1.0)
        size_max = params.get("size_max", 4.0)
        dist_min = params.get("dist_min", 2.0)
        dist_max = params.get("dist_max", 20.0)

        # Distance field from building curves
        if hole_loops:
            # Get all building curve tags
            bld_curve_tags = []
            for loop_tag in hole_loops:
                # Get curves in this loop
                curves = gmsh.model.getBoundary(
                    [(1, abs(c)) for c in [loop_tag]], oriented=False
                )
                # Actually, the curves are the lines we added for buildings
                pass

            # Use all curves in the model except the outer boundary
            all_curves = gmsh.model.getEntities(dim=1)
            outer_curve_set = set(outer_lines)
            bld_curves = [
                c[1] for c in all_curves if c[1] not in outer_curve_set
            ]

            if bld_curves:
                f_dist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(
                    f_dist, "CurvesList", bld_curves
                )
                gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 100)

                f_thresh = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
                gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", size_min)
                gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", size_max)
                gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", dist_min)
                gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", dist_max)

                gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)
                # Disable default sizing so field controls everything
                gmsh.option.setNumber(
                    "Mesh.MeshSizeExtendFromBoundary", 0
                )
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                gmsh.option.setNumber(
                    "Mesh.MeshSizeFromCurvature", 0
                )
            else:
                # No building curves found — fall back to uniform
                gmsh.option.setNumber(
                    "Mesh.CharacteristicLengthMax", resolution
                )
        else:
            gmsh.option.setNumber(
                "Mesh.CharacteristicLengthMax", resolution
            )

    # Frontal-Delaunay algorithm for quality triangles
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # Minimum angle constraint
    min_angle = params.get("min_angle", 28.0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # --- Extract mesh data ---
    # Get nodes: tags are 1-based
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    # node_coords is flat: [x1, y1, z1, x2, y2, z2, ...]
    node_coords = np.array(node_coords).reshape(-1, 3)[:, :2]

    # Build tag-to-index mapping (1-based → 0-based)
    tag_to_idx = {}
    for i, tag in enumerate(node_tags):
        tag_to_idx[int(tag)] = i

    # Get triangles (element type 2 = 3-node triangle)
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    tri_node_tags = None
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype == 2:  # 3-node triangle
            tri_node_tags = np.array(enodes, dtype=int).reshape(-1, 3)
            break

    if tri_node_tags is None:
        gmsh.finalize()
        raise RuntimeError("Gmsh produced no triangles")

    # Remap 1-based node tags to 0-based indices
    triangles = np.vectorize(tag_to_idx.get)(tri_node_tags)

    gmsh.finalize()

    return node_coords.astype(np.float64), triangles.astype(np.int64)


def _sample_elevation(points, dem_path):
    """Sample DEM elevation at mesh vertex locations."""
    with rasterio.open(dem_path) as src:
        # rasterio.sample expects iterable of (x, y)
        coords = [(x, y) for x, y in points]
        samples = list(src.sample(coords))
    elev = np.array([s[0] for s in samples], dtype=np.float64)
    # Replace nodata with 0
    nodata = None
    with rasterio.open(dem_path) as src:
        nodata = src.nodata
    if nodata is not None:
        elev[elev == nodata] = 0.0
    # Replace NaN with 0
    elev = np.nan_to_num(elev, nan=0.0)
    return elev


def _compute_boundary_tags(points, triangles, boundary_segments):
    """Classify exterior edges by proximity to boundary segments.

    Outer boundary edges (close to a boundary segment) get the segment's tag.
    Interior hole edges (far from all boundary segments) get 'Reflective'
    since they represent solid building walls.

    Returns
    -------
    dict : {(tri_id, edge_id): tag_string}
    """
    # Find exterior edges
    edge_map = {}  # (v_min, v_max) -> list of (tri_id, edge_id)
    for tri_id, tri in enumerate(triangles):
        for edge_id, (a, b) in enumerate(_EDGE_VERTS):
            key = (min(tri[a], tri[b]), max(tri[a], tri[b]))
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append((tri_id, edge_id))

    exterior_edges = []
    for key, entries in edge_map.items():
        if len(entries) == 1:
            exterior_edges.append((entries[0], key))

    # Precompute boundary segment vectors
    seg_data = []
    for tag, coords in boundary_segments:
        p0 = np.array(coords[0], dtype=float)
        p1 = np.array(coords[1], dtype=float)
        seg_data.append((tag, p0, p1))

    # Distance threshold: edges further than this from any boundary segment
    # are interior hole edges (building walls). Resolution-scale tolerance.
    max_boundary_dist = 5.0  # metres

    # Classify each exterior edge
    boundary_tags = {}
    for (tri_id, edge_id), (v0, v1) in exterior_edges:
        mid = (points[v0] + points[v1]) / 2.0
        best_dist = float("inf")
        best_tag = "exterior"
        for tag, p0, p1 in seg_data:
            d = _point_to_segment_dist(mid, p0, p1)
            if d < best_dist:
                best_dist = d
                best_tag = tag
        # Edges far from any boundary segment are building hole walls
        if best_dist > max_boundary_dist:
            best_tag = "Reflective"
        boundary_tags[(tri_id, edge_id)] = best_tag

    return boundary_tags


def _point_to_segment_dist(p, a, b):
    """Distance from point p to line segment a-b."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / max(np.dot(ab, ab), 1e-12)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))
