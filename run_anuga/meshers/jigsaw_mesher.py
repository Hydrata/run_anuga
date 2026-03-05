"""JIGSAW-based mesh generation for ANUGA simulations.

Two sizing modes:
- ``flat``:  uniform element size = resolution (burned DEM, no building holes)
- ``topo``: DEM slope-based adaptive sizing (original DEM, buildings as holes)
"""

import json
import logging

import jigsawpy
import numpy as np
import rasterio

from . import MeshResult

logger = logging.getLogger(__name__)

_EDGE_VERTS = [(1, 2), (0, 2), (0, 1)]


def generate_mesh(input_data):
    """Generate a 2D triangular mesh using JIGSAW.

    Parameters
    ----------
    input_data : dict
        The ``input_data`` dict produced by ``setup_input_data()``.

    Returns
    -------
    MeshResult
    """
    config = input_data["scenario_config"]
    params = config.get("mesher_params", {})
    sizing = params.get("sizing", "flat")
    resolution = config.get("resolution", 2.0)

    # Load boundary
    boundary_segments, boundary_ring = _load_boundary(
        input_data["boundary_filename"]
    )

    # Load buildings for hole mode
    buildings = []
    if sizing == "topo" and "structure_filename" in input_data:
        buildings = _load_buildings(input_data["structure_filename"])

    # DEM path
    dem_path = input_data["elevation_filename"]

    # Build geometry
    geo = _build_geometry(boundary_ring, buildings)

    # Build sizing function
    hfn = None
    if sizing == "topo":
        hfn = _build_slope_hfun(dem_path, params)

    # Run JIGSAW
    points, triangles = _run_jigsaw(geo, resolution, sizing, params, hfn)

    logger.info(
        f"JIGSAW ({sizing}): {len(points)} vertices, {len(triangles)} triangles"
    )

    # Sample elevation
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
    """Load boundary segments and build the outer ring."""
    with open(path) as f:
        data = json.load(f)

    segments = []
    ring = []
    for feat in data["features"]:
        tag = feat["properties"]["boundary"]
        coords = feat["geometry"]["coordinates"]
        segments.append((tag, coords))
        ring.append(coords[0])
    ring.append(ring[0])
    return segments, ring


def _load_buildings(path):
    """Load building polygons from structure GeoJSON."""
    with open(path) as f:
        data = json.load(f)

    buildings = []
    for feat in data["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            buildings.append(geom["coordinates"][0])
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                buildings.append(poly[0])
    return buildings


def _build_geometry(boundary_ring, buildings):
    """Build JIGSAW geometry (PSLG) with outer boundary and building holes."""
    VERT2_t = jigsawpy.jigsaw_msh_t.VERT2_t
    EDGE2_t = jigsawpy.jigsaw_msh_t.EDGE2_t

    verts = []
    edges = []
    vid = 0

    # Outer boundary (ring without closing duplicate)
    ring = boundary_ring[:-1]
    n_outer = len(ring)
    for x, y in ring:
        verts.append(((x, y), 0))
    for i in range(n_outer):
        edges.append(((vid + i, vid + (i + 1) % n_outer), 0))
    vid += n_outer

    # Building holes
    for bld_ring in buildings:
        bld = bld_ring[:-1] if bld_ring[0] == bld_ring[-1] else bld_ring
        nb = len(bld)
        for x, y in bld:
            verts.append(((x, y), 1))
        for i in range(nb):
            edges.append(((vid + i, vid + (i + 1) % nb), 1))
        vid += nb

    geo = jigsawpy.jigsaw_msh_t()
    geo.mshID = "euclidean-mesh"
    geo.ndims = 2
    geo.vert2 = np.array(verts, dtype=VERT2_t)
    geo.edge2 = np.array(edges, dtype=EDGE2_t)

    return geo


def _build_slope_hfun(dem_path, params):
    """Build a grid-based sizing function from DEM slope magnitude.

    Steeper terrain → finer mesh. Flat areas → coarser mesh.
    """
    hfun_hmin = params.get("hfun_hmin", 1.0)
    hfun_hmax = params.get("hfun_hmax", 4.0)

    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform

        # Replace nodata with NaN for gradient computation
        if nodata is not None:
            elev[elev == nodata] = np.nan

        nrows, ncols = elev.shape
        dx = abs(transform.a)
        dy = abs(transform.e)

        # Compute slope magnitude via numpy gradient
        gy, gx = np.gradient(elev, dy, dx)
        slope = np.sqrt(gx ** 2 + gy ** 2)
        slope = np.nan_to_num(slope, nan=0.0)

        # Normalise slope to [0, 1] range
        smax = np.percentile(slope[slope > 0], 99) if np.any(slope > 0) else 1.0
        slope_norm = np.clip(slope / max(smax, 1e-6), 0.0, 1.0)

        # Size = hmax where flat, hmin where steep
        size_grid = hfun_hmax - (hfun_hmax - hfun_hmin) * slope_norm

        # Build grid coordinates (cell centres)
        # rasterio: transform maps pixel (col, row) to (x, y)
        xs = np.array([transform.c + (j + 0.5) * dx for j in range(ncols)])
        ys = np.array([transform.f + (i + 0.5) * transform.e for i in range(nrows)])

    # JIGSAW requires xgrid and ygrid in strictly ascending order.
    # rasterio rows go top-to-bottom (descending y), so flip both.
    if ys[0] > ys[-1]:
        ys = ys[::-1]
        size_grid = size_grid[::-1, :]

    hfn = jigsawpy.jigsaw_msh_t()
    hfn.mshID = "euclidean-grid"
    hfn.ndims = 2
    hfn.xgrid = xs.astype(np.float64)
    hfn.ygrid = ys.astype(np.float64)
    hfn.value = size_grid.astype(np.float32)

    return hfn


def _run_jigsaw(geo, resolution, sizing, params, hfn):
    """Run JIGSAW and return (points, triangles) as numpy arrays."""
    jig = jigsawpy.jigsaw_jig_t()
    jig.mesh_dims = 2
    jig.geom_feat = True
    jig.verbosity = 0

    # JIGSAW defaults to relative scaling — we always want absolute (metres)
    jig.hfun_scal = "absolute"

    if sizing == "flat":
        jig.hfun_hmax = float(resolution)
        jig.hfun_hmin = float(resolution)
    elif sizing == "topo":
        jig.hfun_hmax = params.get("hfun_hmax", 4.0)
        jig.hfun_hmin = params.get("hfun_hmin", 1.0)
        jig.hfun_scal = "absolute"

    # Quality: optimise for good angles
    jig.optm_iter = 64

    mesh = jigsawpy.jigsaw_msh_t()

    if hfn is not None:
        jigsawpy.lib.jigsaw(jig, geo, mesh, hfun=hfn)
    else:
        jigsawpy.lib.jigsaw(jig, geo, mesh)

    # Extract vertex coordinates and triangle indices
    coords = np.array([v["coord"] for v in mesh.vert2], dtype=np.float64)
    tris = np.array([t["index"] for t in mesh.tria3], dtype=np.int64)

    return coords, tris


def _sample_elevation(points, dem_path):
    """Sample DEM elevation at mesh vertex locations."""
    with rasterio.open(dem_path) as src:
        coords = [(x, y) for x, y in points]
        samples = list(src.sample(coords))
        nodata = src.nodata
    elev = np.array([s[0] for s in samples], dtype=np.float64)
    if nodata is not None:
        elev[elev == nodata] = 0.0
    elev = np.nan_to_num(elev, nan=0.0)
    return elev


def _compute_boundary_tags(points, triangles, boundary_segments):
    """Classify exterior edges by proximity to boundary segments.

    Outer boundary edges get the segment's tag. Interior hole edges
    (far from all boundary segments) get 'Reflective' (building walls).
    """
    edge_map = {}
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

    seg_data = []
    for tag, coords in boundary_segments:
        p0 = np.array(coords[0], dtype=float)
        p1 = np.array(coords[1], dtype=float)
        seg_data.append((tag, p0, p1))

    max_boundary_dist = 5.0  # metres

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
