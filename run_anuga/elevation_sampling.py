"""Footprint-integrated bed elevation seating (TASK-2186).

ANUGA holds one bed elevation per triangle (``set_values`` with ``location='centroids'``
then ``extrapolate_first_order``: a flat plate per triangle). Hydrata used to pick that
number as the ONE DEM cell under the triangle's centroid: the callable path
(``composite_quantity_setting_function`` -> ``rasterValuesAtPoints``, ``interpolation='pixel'``)
ignores ``alpha``, so a sub-metre mesh-origin shift that carries a centroid across a cell
edge flipped the bed by the full step on a knife-edge DEM (TASK-2149 F4).

Here each triangle's bed is the AREA-WEIGHTED MEAN of the DEM cells its footprint covers.
A shift moves a triangle's bed in proportion to the area that crossed a step, never by
the whole step. On a flat DEM the two methods agree exactly.

The triangle-to-cell coverage is a named object, :class:`CoverageIndex`: (triangle, cell,
weight) triples where ``weight`` is the fraction of the triangle's in-raster footprint
inside ``cell`` (exact polygon clipping, so one triangle's weights sum to 1). It is built
in blocks of consecutive triangle rows, each block written straight into a compressed
``.npz`` beside the mesh, and the seating streams the blocks back one at a time, so the
whole index is never resident beside the DEM band (TASK-1910: the band is already the
build phase's big memory consumer).

Rules kept from the centroid seating:

* nodata never becomes a bed: a covered cell equal to the raster's nodata tag, or NaN,
  raises (``nan_treatment='exception'``); the TASK-1138 pre-flight still runs first;
* a triangle whose centroid lies outside the raster raises, as ``rasterValuesAtPoints`` did;
* a zero-area triangle takes the cell under its centroid (the centroid fallback). A
  triangle smaller than one cell that sits inside one cell gets that cell, which is the
  centroid sample; ``min_footprint_cells`` widens the fallback to every triangle smaller
  than that many cells (off by default: a small triangle across a cell edge gets the
  area-weighted mean, which is what stops the coin flip on fine meshes);
* TASK-1299 Raised buildings are applied by the caller AFTER this seating.

Footprint cells outside the raster extent (a triangle poking past the DEM edge while its
centroid is inside) are left out of that triangle's mean, as no data exists there.
"""
from __future__ import annotations

import json
import os
import zipfile
from dataclasses import asdict, dataclass

import numpy as np

from run_anuga._imports import import_optional

#: Suffix of the persisted coverage index, written beside the run's ``.msh``.
COVERAGE_INDEX_SUFFIX = ".elevation_coverage.npz"
#: Format tag stored in the index's ``meta`` record; :meth:`CoverageIndex.open` checks it.
COVERAGE_INDEX_FORMAT = "run_anuga.elevation_coverage/1"
#: Candidate (triangle, cell) pairs per block. The clip's working arrays cost a few
#: hundred bytes per pair, so a block peaks near 10 MiB. Measured on Merewether (100k
#: triangles): 200k pairs per block raised the raster-read peak RSS 20 % above the
#: centroid seating; 20k pairs kept it below, at the same speed.
DEFAULT_MAX_PAIRS_PER_BLOCK = 20_000
#: A triangle (or a clipped piece) with less area than this, in cells, counts as zero area.
_ZERO_AREA_CELLS = 1e-9
#: rasterValuesAtPoints' tolerance for matching the nodata tag (relative to the tag).
_NODATA_REL_TOL = 1.0e-08


def coverage_index_path(mesh_filepath: str) -> str:
    """Where the coverage index of the mesh at ``mesh_filepath`` is persisted."""
    return os.path.splitext(mesh_filepath)[0] + COVERAGE_INDEX_SUFFIX


@dataclass(frozen=True)
class RasterGrid:
    """A north-up raster's cell geometry: cell (row, col) spans pixel coords [col, col+1] x [row, row+1]."""

    x_origin: float
    y_origin: float
    pixel_width: float
    pixel_height: float  # negative for the usual north-up raster
    width: int
    height: int

    @classmethod
    def from_transform(cls, transform, width: int, height: int) -> "RasterGrid":
        """From a rasterio/affine transform. A rotated or sheared raster is refused."""
        if transform.b != 0.0 or transform.d != 0.0:
            raise ValueError(
                "elevation raster is rotated or sheared; Hydrata seats the bed from a north-up "
                "raster only. Warp it north-up first (e.g. gdalwarp to the model CRS)."
            )
        return cls(float(transform.c), float(transform.f), float(transform.a), float(transform.e),
                   int(width), int(height))

    def to_pixel(self, xy: np.ndarray) -> np.ndarray:
        """Absolute (x, y) -> fractional pixel (u = column, v = row) coordinates."""
        return np.stack([(xy[..., 0] - self.x_origin) / self.pixel_width,
                         (xy[..., 1] - self.y_origin) / self.pixel_height], axis=-1)


class CoverageIndex:
    """The persisted triangle-to-DEM-cell coverage: (triangle, cell, weight) triples.

    ``triangle`` indexes the mesh's triangles (ANUGA order), ``cell`` is the flat raster
    index ``row * width + col``, and ``weight`` is the fraction of the triangle's in-raster
    footprint area inside that cell (the weights of one triangle sum to 1). Triples are
    stored in blocks of consecutive triangles ``[t0, t1)``; the instance holds only the
    file path and metadata, and :meth:`iter_blocks` reads one block at a time.
    """

    def __init__(self, path: str, n_triangles: int, grid: RasterGrid, block_ranges):
        self.path = path
        self.n_triangles = n_triangles
        self.grid = grid
        self.block_ranges = [tuple(r) for r in block_ranges]

    @property
    def n_blocks(self) -> int:
        return len(self.block_ranges)

    @classmethod
    def build(cls, path: str, vertex_xy: np.ndarray, grid: RasterGrid, *, offset=(0.0, 0.0),
              max_pairs_per_block: int = DEFAULT_MAX_PAIRS_PER_BLOCK,
              min_footprint_cells: float = 0.0) -> "CoverageIndex":
        """Build the index for a mesh and persist it to ``path`` (a compressed ``.npz``).

        ``vertex_xy`` is ANUGA's ``(3 * n_triangles, 2)`` vertex layout (rows ``3k..3k+2``
        are triangle ``k``) or ``(n_triangles, 3, 2)``, in coordinates that ``offset`` turns
        absolute (an ANUGA domain's relative vertices plus its geo_reference corner).
        Raises ValueError when a triangle's centroid lies outside the raster.
        """
        tris = np.asarray(vertex_xy).reshape(-1, 3, 2)
        offset = np.asarray(offset, dtype=float)
        ranges = _plan_blocks(tris, offset, grid, max_pairs_per_block)
        partial_path = path + ".partial"
        try:
            with zipfile.ZipFile(partial_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                for k, (t0, t1) in enumerate(ranges):
                    tri, cell, weight = _block_triples(tris[t0:t1], t0, offset, grid, min_footprint_cells)
                    _write_member(zf, f"tri_{k:06d}", tri)
                    _write_member(zf, f"cell_{k:06d}", cell)
                    _write_member(zf, f"weight_{k:06d}", weight)
                meta = {"format": COVERAGE_INDEX_FORMAT, "n_triangles": len(tris), "grid": asdict(grid),
                        "block_ranges": ranges, "min_footprint_cells": min_footprint_cells}
                _write_member(zf, "meta", np.array(json.dumps(meta)))
            os.replace(partial_path, path)
        finally:
            if os.path.exists(partial_path):
                os.remove(partial_path)
        return cls(path, len(tris), grid, ranges)

    @classmethod
    def open(cls, path: str) -> "CoverageIndex":
        """Reopen a persisted index from its file alone (metadata only; no triples are read)."""
        with np.load(path, allow_pickle=False) as npz:
            meta = json.loads(str(npz["meta"].ravel()[0])) if "meta" in npz.files else {}
        if meta.get("format") != COVERAGE_INDEX_FORMAT:
            raise ValueError(f"{path} is not a Hydrata elevation coverage index")
        return cls(path, int(meta["n_triangles"]), RasterGrid(**meta["grid"]), meta["block_ranges"])

    def iter_blocks(self):
        """Yield ``(t0, t1, triangle, cell, weight)`` per block, reading one block at a time."""
        with np.load(self.path, allow_pickle=False) as npz:
            for k, (t0, t1) in enumerate(self.block_ranges):
                yield t0, t1, npz[f"tri_{k:06d}"], npz[f"cell_{k:06d}"], npz[f"weight_{k:06d}"]


def footprint_mean(index: CoverageIndex, band: np.ndarray, nodata=None) -> np.ndarray:
    """Per-triangle area-weighted mean of ``band`` over each footprint, streamed block by block.

    Raises ValueError when any covered cell is nodata (the declared tag or NaN): Hydrata
    never fabricates a bed elevation (``nan_treatment='exception'``).
    """
    if band.shape != (index.grid.height, index.grid.width):
        raise ValueError(f"DEM band shape {band.shape} does not match the coverage index grid "
                         f"({index.grid.height}, {index.grid.width})")
    flat = band.reshape(-1)  # a view of the C-ordered band, never a copy
    check_tag = nodata is not None and np.isfinite(nodata)
    out = np.empty(index.n_triangles, dtype=float)
    bad_triangles = 0
    for t0, t1, tri, cell, weight in index.iter_blocks():
        z = flat[cell].astype(float)
        bad = np.isnan(z)
        if check_tag:  # exact (as the TASK-1138 pre-flight) or within rasterValuesAtPoints' tolerance
            bad |= (z == nodata) | (np.abs(z - nodata) < _NODATA_REL_TOL * abs(nodata))
        if bad.any():
            bad_triangles += len(np.unique(tri[bad]))
            continue
        local = tri - t0
        w = weight.astype(float)
        out[t0:t1] = (np.bincount(local, w * z, minlength=t1 - t0)
                      / np.bincount(local, w, minlength=t1 - t0))
    if bad_triangles:
        raise ValueError(
            f"{bad_triangles} triangle(s) cover nodata cells of the elevation raster; fill the data "
            f"gaps (e.g. gdal_fillnodata) or reselect the terrain extent. Hydrata never fabricates "
            f"a bed elevation (nan_treatment='exception')."
        )
    return out


def seat_elevation(domain, elevation_filename: str, coverage_path: str) -> CoverageIndex:
    """Seat ``domain``'s bed from the DEM: the footprint mean per triangle, at the centroids.

    Builds and persists the coverage index at ``coverage_path``, streams it against the DEM
    band, and sets ``elevation`` with ``location='centroids'`` (one value per triangle, as
    the centroid seating did), so TASK-1299 Raised buildings apply on top afterwards.
    """
    rasterio = import_optional("rasterio")
    with rasterio.open(elevation_filename) as dataset:
        grid = RasterGrid.from_transform(dataset.transform, dataset.width, dataset.height)
        nodata = dataset.nodata
        band = dataset.read(1)  # the raster's own dtype; only per-block samples go float64
    geo = domain.geo_reference
    index = CoverageIndex.build(
        coverage_path,
        domain.get_vertex_coordinates(absolute=False),  # ANUGA's own array, not a copy
        grid,
        offset=(geo.get_xllcorner(), geo.get_yllcorner()),
    )
    values = footprint_mean(index, band, nodata)
    del band
    domain.set_quantity('elevation', values, location='centroids')
    return index


# ── builder internals ────────────────────────────────────────────────────


def _write_member(zf: zipfile.ZipFile, name: str, array: np.ndarray) -> None:
    with zf.open(name + ".npy", "w", force_zip64=True) as fh:
        np.lib.format.write_array(fh, np.ascontiguousarray(array), allow_pickle=False)


def _pixel_bounds(uv: np.ndarray, grid: RasterGrid):
    """Per-triangle candidate cell ranges [c0, c1] x [r0, r1] (inclusive), clipped to the raster."""
    c0 = np.maximum(np.floor(uv[:, :, 0].min(axis=1)), 0).astype(np.int64)
    c1 = np.minimum(np.ceil(uv[:, :, 0].max(axis=1)) - 1, grid.width - 1).astype(np.int64)
    r0 = np.maximum(np.floor(uv[:, :, 1].min(axis=1)), 0).astype(np.int64)
    r1 = np.minimum(np.ceil(uv[:, :, 1].max(axis=1)) - 1, grid.height - 1).astype(np.int64)
    return c0, c1, r0, r1


def _plan_blocks(tris, offset, grid, max_pairs):
    """Split the triangles into consecutive ranges of at most ``max_pairs`` candidate pairs
    (a triangle with more candidates than that gets a block of its own)."""
    n = len(tris)
    cum = np.empty(n, dtype=np.int64)
    step = 200_000
    for s in range(0, n, step):
        c0, c1, r0, r1 = _pixel_bounds(grid.to_pixel(tris[s:s + step] + offset), grid)
        cum[s:s + step] = np.maximum(np.maximum(c1 - c0 + 1, 0) * np.maximum(r1 - r0 + 1, 0), 1)
    np.cumsum(cum, out=cum)  # in place: one int64 per triangle
    ranges, t0, done = [], 0, 0
    while t0 < n:
        t1 = max(int(np.searchsorted(cum, done + max_pairs, side="right")), t0 + 1)
        ranges.append([t0, t1])
        done = int(cum[t1 - 1])
        t0 = t1
    return ranges


def _block_triples(tris, t0, offset, grid, min_footprint_cells):
    """(triangle, cell, weight) arrays for one block of triangles whose ids start at ``t0``."""
    uv = grid.to_pixel(tris + offset)  # (m, 3, 2) pixel coordinates
    m = len(uv)
    centroid = uv.mean(axis=1)
    ccol = np.floor(centroid[:, 0]).astype(np.int64)
    crow = np.floor(centroid[:, 1]).astype(np.int64)
    outside = (ccol < 0) | (ccol >= grid.width) | (crow < 0) | (crow >= grid.height)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} triangle centroid(s) lie outside the elevation raster extent; "
            f"extend the terrain to cover the whole model boundary."
        )
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    signed2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]  # twice the signed area, in cells
    tri_area = 0.5 * np.abs(signed2)
    fallback = (tri_area <= _ZERO_AREA_CELLS) | (tri_area < min_footprint_cells)

    # Every (triangle, cell) pair in each triangle's bounding box of cells.
    c0, c1, r0, r1 = _pixel_bounds(uv, grid)
    ncols = np.maximum(c1 - c0 + 1, 1)
    counts = np.where(fallback, 0, np.maximum(c1 - c0 + 1, 0) * np.maximum(r1 - r0 + 1, 0))
    pair_tri = np.repeat(np.arange(m), counts)
    k = np.arange(len(pair_tri)) - np.repeat(np.cumsum(counts) - counts, counts)
    pair_col = c0[pair_tri] + k % ncols[pair_tri]
    pair_row = r0[pair_tri] + k // ncols[pair_tri]
    del k

    corner = np.stack([pair_col, pair_row], axis=-1)[:, None, :]
    area = _overlap_area(uv[pair_tri] - corner, np.sign(signed2)[pair_tri])
    keep = area > _ZERO_AREA_CELLS
    pair_tri, pair_col, pair_row, area = pair_tri[keep], pair_col[keep], pair_row[keep], area[keep]

    # Normalise by the in-raster footprint; a sliver whose pieces all vanished falls back too.
    covered = np.bincount(pair_tri, area, minlength=m)
    fallback |= covered <= _ZERO_AREA_CELLS
    fb = np.flatnonzero(fallback)
    live = ~fallback[pair_tri]
    tri_ids = np.concatenate([pair_tri[live], fb])
    cells = np.concatenate([pair_row[live] * grid.width + pair_col[live], crow[fb] * grid.width + ccol[fb]])
    weights = np.concatenate([area[live] / covered[pair_tri[live]], np.ones(len(fb))])
    order = np.argsort(tri_ids, kind="stable")
    cell_dtype = np.int32 if grid.width * grid.height < 2**31 else np.int64
    tri_dtype = np.int32 if t0 + m < 2**31 else np.int64
    return ((tri_ids[order] + t0).astype(tri_dtype), cells[order].astype(cell_dtype),
            weights[order].astype(np.float32))


def _overlap_area(local, orient):
    """Area inside the unit cell [0, 1]^2 of each triangle ``local`` (N, 3, 2), given relative
    to its cell's corner. ``orient`` is each triangle's winding sign (+1 / -1).

    A cell wholly inside the triangle counts 1 and a cell wholly beyond one edge counts 0
    without clipping; the rest are clipped (Sutherland-Hodgman, vectorised over the pairs).
    """
    area = np.zeros(len(local))
    if not len(local):
        return area
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    all_in = np.ones(len(local), dtype=bool)
    beyond_an_edge = np.zeros(len(local), dtype=bool)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        p, d = local[:, a], local[:, b] - local[:, a]
        # >= 0: the corner is on the triangle's side of edge a -> b
        side = orient[:, None] * (d[:, None, 0] * (corners[None, :, 1] - p[:, None, 1])
                                  - d[:, None, 1] * (corners[None, :, 0] - p[:, None, 0]))
        all_in &= (side >= 0).all(axis=1)
        beyond_an_edge |= (side <= 0).all(axis=1)
    area[all_in] = 1.0
    partial = ~all_in & ~beyond_an_edge
    if partial.any():
        area[partial] = _clipped_area(local[partial])
    return area


def _clipped_area(tri):
    """Shoelace area of each triangle (N, 3, 2) clipped to the unit cell."""
    poly = np.zeros((len(tri), 7, 2))  # a triangle clipped by 4 half-planes has <= 7 vertices
    poly[:, :3] = tri
    count = np.full(len(tri), 3)
    for axis, bound, sign in ((0, 0.0, 1.0), (0, 1.0, -1.0), (1, 0.0, 1.0), (1, 1.0, -1.0)):
        poly, count = _clip_half_plane(poly, count, axis, bound, sign)
    k = np.arange(poly.shape[1])
    nxt = (k[None, :] + 1) % np.maximum(count, 1)[:, None]
    x, y = poly[..., 0], poly[..., 1]
    cross = x * np.take_along_axis(y, nxt, axis=1) - np.take_along_axis(x, nxt, axis=1) * y
    return 0.5 * np.abs(np.where(k[None, :] < count[:, None], cross, 0.0).sum(axis=1))


def _clip_half_plane(poly, count, axis, bound, sign):
    """Clip convex polygons (N, M, 2) with ``count`` live vertices to sign * (p[axis] - bound) >= 0."""
    out = np.zeros_like(poly)
    out_count = np.zeros(len(poly), dtype=np.int64)
    rows = np.arange(len(poly))
    safe = np.maximum(count, 1)
    for k in range(int(count.max(initial=0))):
        live = k < count
        cur, nxt = poly[:, k], poly[rows, (k + 1) % safe]
        dc, dn = sign * (cur[:, axis] - bound), sign * (nxt[:, axis] - bound)
        cin, nin = dc >= 0, dn >= 0
        r = rows[live & cin]
        out[r, out_count[r]] = cur[r]
        out_count[r] += 1
        r = rows[live & (cin != nin)]
        t = (dc[r] / (dc[r] - dn[r]))[:, None]
        out[r, out_count[r]] = cur[r] + t * (nxt[r] - cur[r])
        out_count[r] += 1
    return out, out_count
