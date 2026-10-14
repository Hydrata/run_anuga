"""D15 exchange regions: generated from the mesh, never drawn (TASK-3638).

At each end a rectangle sits on the approach side (outside the barrel): it
starts at the end point and extends away from the barrel, perpendicular to the
culvert line. Width = max(footprint width, 2 local cell edges); depth = 2 local
cell edges. When it holds fewer than MIN_REGION_TRIANGLES triangle centroids it
is expanded (width and depth together) until it does, or the build fails.

The "local cell edge" at a point is sqrt(2 * area) of the triangle containing
it: the same edge-length convention as scenario.resolution (an ANUGA max area
of res**2 / 2), so a 5 m mesh reads as a 5 m cell edge.

numpy only, on absolute coordinates; callers adapt a Domain or a pmesh to
(nodes, triangles) via :class:`MeshView`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from run_anuga._imports import import_optional
from run_anuga.culverts import constants


#: containing_triangle tests this many nearest centroids before a full scan.
_NEAREST_CANDIDATES = 64
#: Triangles per block in the full-scan fallback (bounds the temporary arrays).
_SCAN_CHUNK = 500_000


@dataclass
class MeshView:
    """Absolute node coordinates (N, 2) + triangle vertex indices (M, 3)."""

    nodes: object
    triangles: object
    centroids: object = field(init=False)
    areas: object = field(init=False)

    def __post_init__(self):
        numpy = import_optional("numpy")
        self.nodes = numpy.asarray(self.nodes, dtype=float)
        self.triangles = numpy.asarray(self.triangles, dtype=int)
        a, b, c = (self.nodes[self.triangles[:, k]] for k in range(3))
        self.centroids = (a + b + c) / 3.0
        e1, e2 = b - a, c - a
        self.areas = 0.5 * numpy.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])

    @classmethod
    def from_domain(cls, domain):
        return cls(domain.get_nodes(absolute=True), domain.triangles)

    @classmethod
    def from_pmesh(cls, anuga_mesh):
        """A pmesh from create_mesh_from_regions: vertices are relative to its geo_reference."""
        numpy = import_optional("numpy")
        vertices = numpy.asarray(anuga_mesh.tri_mesh.vertices, dtype=float)
        geo = getattr(anuga_mesh, "geo_reference", None)
        if geo is not None:
            vertices = vertices + numpy.array([geo.get_xllcorner(), geo.get_yllcorner()])
        return cls(vertices, anuga_mesh.tri_mesh.triangles)

    def containing_triangle(self, point):
        """Index of the triangle containing ``point`` (on an edge counts), or None.

        Tests the nearest centroids first (the containing triangle is almost
        always among them), then falls back to a chunked scan of the whole mesh
        so a large triangle beside a fine Mesh region is never missed. Never
        materialises an (M, 3, 2) array: a production mesh has millions of
        triangles on rank 0.
        """
        numpy = import_optional("numpy")
        x, y = float(point[0]), float(point[1])
        count = len(self.triangles)
        if count == 0:
            return None
        distance2 = (self.centroids[:, 0] - x) ** 2 + (self.centroids[:, 1] - y) ** 2
        nearest = min(_NEAREST_CANDIDATES, count)
        first = numpy.argpartition(distance2, nearest - 1)[:nearest]
        found = self._first_containing(first[numpy.argsort(distance2[first])], x, y)
        if found is not None or nearest == count:
            return found
        for start in range(0, count, _SCAN_CHUNK):
            found = self._first_containing(numpy.arange(start, min(start + _SCAN_CHUNK, count)), x, y)
            if found is not None:
                return found
        return None

    def _first_containing(self, indices, x, y):
        """First of ``indices`` whose triangle contains (x, y), else None (vectorised)."""
        numpy = import_optional("numpy")
        a, b, c = (self.nodes[self.triangles[indices, k]] for k in range(3))
        d1 = (x - b[:, 0]) * (a[:, 1] - b[:, 1]) - (a[:, 0] - b[:, 0]) * (y - b[:, 1])
        d2 = (x - c[:, 0]) * (b[:, 1] - c[:, 1]) - (b[:, 0] - c[:, 0]) * (y - c[:, 1])
        d3 = (x - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (y - a[:, 1])
        # Sign tolerance scales with the triangle (cross products are length**2).
        scale = 1e-9 * numpy.maximum(1.0, self.areas[indices])
        has_neg = (d1 < -scale) | (d2 < -scale) | (d3 < -scale)
        has_pos = (d1 > scale) | (d2 > scale) | (d3 > scale)
        hits = numpy.nonzero(~(has_neg & has_pos))[0]
        return int(indices[hits[0]]) if len(hits) else None

    def local_cell_edge(self, point):
        """sqrt(2 * area) of the triangle containing ``point``, or None outside the mesh."""
        index = self.containing_triangle(point)
        if index is None:
            return None
        return math.sqrt(2.0 * float(self.areas[index]))


@dataclass
class ExchangeRegion:
    end: str  # 'inlet' | 'outlet'
    polygon: list  # absolute [[x, y], ...] closed ring
    triangle_ids: list
    width_m: float
    depth_m: float
    expansions: int

    @property
    def triangle_count(self) -> int:
        return len(self.triangle_ids)


def rectangle(point, outward, width, depth):
    """Closed ring of the rectangle at ``point`` extending ``depth`` along ``outward``."""
    ux, uy = outward
    nx, ny = -uy, ux
    px, py = float(point[0]), float(point[1])
    half = width / 2.0
    return [
        [px + nx * half, py + ny * half],
        [px + ux * depth + nx * half, py + uy * depth + ny * half],
        [px + ux * depth - nx * half, py + uy * depth - ny * half],
        [px - nx * half, py - ny * half],
        [px + nx * half, py + ny * half],
    ]


def triangles_in_rectangle(mesh, point, outward, width, depth, candidates=None):
    """Triangle ids whose centroid lies inside the rectangle (in the local frame).

    ``candidates`` (an index array) restricts the test to a pre-selected subset.
    """
    numpy = import_optional("numpy")
    ux, uy = outward
    ids = numpy.arange(len(mesh.centroids)) if candidates is None else candidates
    rel = mesh.centroids[ids] - numpy.array([float(point[0]), float(point[1])])
    along = rel[:, 0] * ux + rel[:, 1] * uy
    across = rel[:, 0] * -uy + rel[:, 1] * ux
    inside = (along >= 0.0) & (along <= depth) & (numpy.abs(across) <= width / 2.0)
    return [int(i) for i in ids[inside]]


def build_exchange_region(mesh, end, point, outward, footprint_width, cell_edge):
    """D15 region at one end, expanded until it holds MIN_REGION_TRIANGLES (or the cap).

    The caller fails the build when the returned region still holds too few.
    """
    numpy = import_optional("numpy")
    width = max(footprint_width, constants.REGION_MIN_WIDTH_CELL_EDGES * cell_edge)
    depth = constants.REGION_DEPTH_CELL_EDGES * cell_edge
    # Every rectangle the expansion can reach fits in this circle: select its
    # centroids once instead of rescanning the whole mesh per step.
    grow = constants.REGION_EXPANSION_FACTOR ** constants.MAX_REGION_EXPANSIONS
    reach2 = (width * grow / 2.0) ** 2 + (depth * grow) ** 2
    distance2 = ((mesh.centroids - numpy.array([float(point[0]), float(point[1])])) ** 2).sum(axis=1)
    candidates = numpy.nonzero(distance2 <= reach2)[0]
    expansions = 0
    ids = triangles_in_rectangle(mesh, point, outward, width, depth, candidates)
    while len(ids) < constants.MIN_REGION_TRIANGLES and expansions < constants.MAX_REGION_EXPANSIONS:
        width *= constants.REGION_EXPANSION_FACTOR
        depth *= constants.REGION_EXPANSION_FACTOR
        expansions += 1
        ids = triangles_in_rectangle(mesh, point, outward, width, depth, candidates)
    return ExchangeRegion(
        end=end,
        polygon=rectangle(point, outward, width, depth),
        triangle_ids=ids,
        width_m=width,
        depth_m=depth,
        expansions=expansions,
    )


#: Written beside culverts/metrics.json (TASK-3646): the D15 regions, for the map.
EXCHANGE_REGIONS_FILENAME = "exchange_regions.geojson"


def exchange_regions_geojson(results, epsg=None) -> dict:
    """One Polygon Feature per culvert end, in the model's absolute coordinates.

    ``results`` are checks.CulvertMeshResult. Properties: feature_id, name, end,
    triangle_count, coarseness_ratio (the culvert's value, exactly as in
    metrics.json) and cell_edge_m (this end's local cell edge). Read-only over
    the regions already built: writing it changes nothing in the hydraulics.
    """
    features = []
    for result in results:
        culvert = result.culvert
        for end in ("inlet", "outlet"):
            region = result.regions[end]
            cell_edge = (result.cell_edges_m or {}).get(end)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[round(float(x), 3), round(float(y), 3)] for x, y in region.polygon]],
                },
                "properties": {
                    "feature_id": culvert.feature_id,
                    "name": culvert.name,
                    "end": end,
                    "triangle_count": int(region.triangle_count),
                    "coarseness_ratio": round(float(result.coarseness_ratio), 4),
                    "cell_edge_m": round(float(cell_edge), 4) if cell_edge is not None else None,
                },
            })
    document = {"type": "FeatureCollection", "features": features}
    if epsg:
        code = str(epsg).strip()
        code = code if code.upper().startswith("EPSG:") else f"EPSG:{code}"
        document["crs"] = {"type": "name", "properties": {"name": code.upper()}}
    return document


def write_exchange_regions(output_directory, results, epsg=None) -> str:
    """Write culverts/exchange_regions.geojson (atomic); returns its path."""
    path = os.path.join(output_directory, constants.METRICS_DIRNAME, EXCHANGE_REGIONS_FILENAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(exchange_regions_geojson(results, epsg), f)
    os.replace(tmp, path)
    return path
