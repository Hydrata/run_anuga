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

import math
from dataclasses import dataclass, field

from run_anuga._imports import import_optional
from run_anuga.culverts import constants


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
        corners = self.nodes[self.triangles]  # (M, 3, 2)
        self.centroids = corners.mean(axis=1)
        e1 = corners[:, 1] - corners[:, 0]
        e2 = corners[:, 2] - corners[:, 0]
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
        """Index of the triangle containing ``point`` (on an edge counts), or None."""
        numpy = import_optional("numpy")
        x, y = float(point[0]), float(point[1])
        corners = self.nodes[self.triangles]
        lo, hi = corners.min(axis=1), corners.max(axis=1)
        tol = 1e-9 * max(1.0, abs(x), abs(y))
        candidates = numpy.nonzero(
            (lo[:, 0] - tol <= x) & (x <= hi[:, 0] + tol) & (lo[:, 1] - tol <= y) & (y <= hi[:, 1] + tol)
        )[0]
        for index in candidates:
            a, b, c = corners[index]
            d1 = (x - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (y - b[1])
            d2 = (x - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (y - c[1])
            d3 = (x - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (y - a[1])
            scale = tol * max(1.0, float(self.areas[index]))
            has_neg = (d1 < -scale) or (d2 < -scale) or (d3 < -scale)
            has_pos = (d1 > scale) or (d2 > scale) or (d3 > scale)
            if not (has_neg and has_pos):
                return int(index)
        return None

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


def triangles_in_rectangle(mesh, point, outward, width, depth):
    """Triangle ids whose centroid lies inside the rectangle (in the local frame)."""
    numpy = import_optional("numpy")
    ux, uy = outward
    rel = mesh.centroids - numpy.array([float(point[0]), float(point[1])])
    along = rel[:, 0] * ux + rel[:, 1] * uy
    across = rel[:, 0] * -uy + rel[:, 1] * ux
    inside = (along >= 0.0) & (along <= depth) & (numpy.abs(across) <= width / 2.0)
    return [int(i) for i in numpy.nonzero(inside)[0]]


def build_exchange_region(mesh, end, point, outward, footprint_width, cell_edge):
    """D15 region at one end, expanded until it holds MIN_REGION_TRIANGLES (or the cap).

    The caller fails the build when the returned region still holds too few.
    """
    width = max(footprint_width, constants.REGION_MIN_WIDTH_CELL_EDGES * cell_edge)
    depth = constants.REGION_DEPTH_CELL_EDGES * cell_edge
    expansions = 0
    ids = triangles_in_rectangle(mesh, point, outward, width, depth)
    while len(ids) < constants.MIN_REGION_TRIANGLES and expansions < constants.MAX_REGION_EXPANSIONS:
        width *= constants.REGION_EXPANSION_FACTOR
        depth *= constants.REGION_EXPANSION_FACTOR
        expansions += 1
        ids = triangles_in_rectangle(mesh, point, outward, width, depth)
    return ExchangeRegion(
        end=end,
        polygon=rectangle(point, outward, width, depth),
        triangle_ids=ids,
        width_m=width,
        depth_m=depth,
        expansions=expansions,
    )
