"""TASK-2186: footprint-integrated bed elevation seating.

Each triangle's bed is the area-weighted mean of the DEM cells its footprint covers,
read through a persisted coverage index of (triangle, cell, weight) triples, instead of
the one DEM cell under its centroid. The centroid point sample flips by the full step
height when a sub-metre origin shift carries a centroid across a cell edge on a
knife-edge DEM (TASK-2149 F4); the footprint mean moves in proportion to the area that
crossed.

The pure-numpy tests run in CI's plain job (numpy + rasterio, no ANUGA). The tests that
seat a real ANUGA Domain carry ``requires_anuga`` and run in the ``test-geo`` job.
"""
import zipfile

import numpy as np
import pytest

KERB_M = 0.3          # AC1: a LOW step (a 0.3 m kerb) in a 1 m DEM
ORIGIN_SHIFT_M = 0.45  # AC1: the measured median centroid drift between mesh origins
BASE_Z = 20.0


# ── helpers ──────────────────────────────────────────────────────────────


def _grid(x_origin=0.0, y_top=40.0, width=40, height=40, cell=1.0):
    from run_anuga.elevation_sampling import RasterGrid
    return RasterGrid(x_origin=x_origin, y_origin=y_top, pixel_width=cell, pixel_height=-cell,
                      width=width, height=height)


def _kerb_band(grid, kerb_axis, kerb_at):
    """A 1 m DEM at BASE_Z with a KERB_M step up on the far side of x (or y) = kerb_at."""
    band = np.full((grid.height, grid.width), BASE_Z, dtype=np.float32)
    if kerb_axis == "x":
        cols = np.arange(grid.width)
        band[:, grid.x_origin + cols * grid.pixel_width >= kerb_at] += KERB_M
    else:
        rows = np.arange(grid.height)
        cell_bottom = grid.y_origin + (rows + 1) * grid.pixel_height
        band[cell_bottom >= kerb_at, :] += KERB_M
    return band


def _structured_mesh(x0, y0, nx, ny, size):
    """(3n, 2) vertex rows (ANUGA layout) of a 2 m right-triangle mesh, alternating diagonals."""
    tris = []
    for j in range(ny):
        for i in range(nx):
            a = (x0 + i * size, y0 + j * size)
            b = (x0 + (i + 1) * size, y0 + j * size)
            c = (x0 + (i + 1) * size, y0 + (j + 1) * size)
            d = (x0 + i * size, y0 + (j + 1) * size)
            if (i + j) % 2:
                tris += [(a, b, c), (a, c, d)]
            else:
                tris += [(a, b, d), (b, c, d)]
    return np.asarray(tris, dtype=float).reshape(-1, 2)


def _clip_area_beyond(tri, axis, at):
    """Independent oracle: area of one triangle on the far side of the line axis = at."""
    pts = [tuple(p) for p in tri]
    out = []
    for k in range(3):
        p, q = pts[k], pts[(k + 1) % 3]
        pin, qin = p[axis] >= at, q[axis] >= at
        if pin:
            out.append(p)
        if pin != qin:
            t = (at - p[axis]) / (q[axis] - p[axis])
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    if len(out) < 3:
        return 0.0
    # Shoelace about the first vertex: raw UTM products (~1e12) would cost ~1e-4 m2.
    xs = np.array([p[0] for p in out]) - out[0][0]
    ys = np.array([p[1] for p in out]) - out[0][1]
    return 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))


def _high_side_fraction(vertex_xy, axis, at):
    tris = vertex_xy.reshape(-1, 3, 2)
    out = np.empty(len(tris))
    for k, tri in enumerate(tris):
        full = 0.5 * abs((tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
                         - (tri[2, 0] - tri[0, 0]) * (tri[1, 1] - tri[0, 1]))
        out[k] = _clip_area_beyond(tri, axis, at) / full
    return out


def _seat_pure(tmp_path, name, vertex_xy, grid, band, nodata=None, **build_kwargs):
    from run_anuga.elevation_sampling import CoverageIndex, footprint_mean
    index = CoverageIndex.build(str(tmp_path / f"{name}.elevation_coverage.npz"), vertex_xy, grid, **build_kwargs)
    return footprint_mean(index, band, nodata)


def _centroid_sample(vertex_xy, grid, band):
    """Today's seating: the one DEM cell under each centroid (rasterValuesAtPoints 'pixel')."""
    c = vertex_xy.reshape(-1, 3, 2).mean(axis=1)
    cols = np.floor((c[:, 0] - grid.x_origin) / grid.pixel_width).astype(int)
    rows = np.floor((c[:, 1] - grid.y_origin) / grid.pixel_height).astype(int)
    return band[rows, cols].astype(float)


def _write_tif(path, band, grid, nodata=-9999.0):
    import rasterio
    from rasterio.transform import Affine
    transform = Affine(grid.pixel_width, 0.0, grid.x_origin, 0.0, grid.pixel_height, grid.y_origin)
    with rasterio.open(path, "w", driver="GTiff", height=band.shape[0], width=band.shape[1], count=1,
                       dtype=band.dtype, transform=transform, nodata=nodata) as ds:
        ds.write(band, 1)
    return str(path)


# ── AC1: continuity in the mesh origin ──────────────────────────────────


@pytest.mark.parametrize("axis", ["x", "y"])
def test_footprint_mean_moves_in_proportion_to_the_area_that_crossed(tmp_path, axis):
    """Pure twin of AC1 (runs without ANUGA): a 0.45 m shift over a 0.3 m kerb moves each
    triangle by exactly KERB_M x (the fraction of its area that crossed), never a full flip."""
    grid = _grid()
    kerb_at = 20.0
    band = _kerb_band(grid, axis, kerb_at)
    mesh = _structured_mesh(4.3, 4.3, 16, 16, 2.0)  # off the kerb line, so centroids cross it
    shift = np.array([ORIGIN_SHIFT_M, 0.0] if axis == "x" else [0.0, ORIGIN_SHIFT_M])
    k = 0 if axis == "x" else 1

    before = _seat_pure(tmp_path, "before", mesh, grid, band)
    after = _seat_pure(tmp_path, "after", mesh + shift, grid, band)
    crossed = _high_side_fraction(mesh + shift, k, kerb_at) - _high_side_fraction(mesh, k, kerb_at)

    np.testing.assert_allclose(after - before, KERB_M * crossed, atol=1e-6)
    assert np.abs(after - before).max() < KERB_M
    assert np.abs(after - before).max() > 0.1 * KERB_M, "the shift must actually move the kerb triangles"
    # The defect being fixed: today's centroid sample flips by the whole kerb.
    flips = np.abs(_centroid_sample(mesh + shift, grid, band) - _centroid_sample(mesh, grid, band))
    assert np.isclose(flips.max(), KERB_M)


def _anuga_kerb_domain(xll, yll, mesh_from=None):
    """A Hydrata-resolution-2 (max area 2 m2) Triangle mesh over a 30 x 30 m square."""
    import anuga
    if mesh_from is None:
        poly = [[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]]
        domain = anuga.create_domain_from_regions(
            poly, boundary_tags={"bottom": [0], "right": [1], "top": [2], "left": [3]},
            maximum_triangle_area=2.0, verbose=False)
        points, triangles, boundary = domain.get_nodes(), domain.get_triangles(), domain.boundary
    else:
        points, triangles, boundary = mesh_from
    geo = anuga.Geo_reference(zone=56, xllcorner=xll, yllcorner=yll)
    return anuga.Domain(points, triangles, boundary, geo_reference=geo), (points, triangles, boundary)


@pytest.mark.requires_anuga
def test_footprint_average_is_continuous_in_origin(tmp_path):
    """AC1: seat a real Domain over a 1 m DEM with a 0.3 m kerb, shift the mesh origin
    0.45 m, and every triangle's bed moves by at most the kerb times the fraction of its
    area that crossed it, never by the whole kerb. RED on centroid sampling (a full flip)."""
    from run_anuga.elevation_sampling import seat_elevation

    xll, yll = 382000.0, 6354000.0
    grid = _grid(x_origin=xll - 5.0, y_top=yll + 40.0, width=45, height=45)
    kerb_at = xll + 15.0
    tif = _write_tif(tmp_path / "kerb.tif", _kerb_band(grid, "x", kerb_at), grid)

    base, mesh = _anuga_kerb_domain(xll, yll)
    shifted, _ = _anuga_kerb_domain(xll + ORIGIN_SHIFT_M, yll, mesh_from=mesh)
    seat_elevation(base, tif, str(tmp_path / "base.elevation_coverage.npz"))
    seat_elevation(shifted, tif, str(tmp_path / "shifted.elevation_coverage.npz"))

    dz = (shifted.get_quantity("elevation").get_values(location="centroids")
          - base.get_quantity("elevation").get_values(location="centroids"))
    crossed = (_high_side_fraction(shifted.get_vertex_coordinates(absolute=True), 0, kerb_at)
               - _high_side_fraction(base.get_vertex_coordinates(absolute=True), 0, kerb_at))
    worst = int(np.argmax(np.abs(dz) - KERB_M * np.abs(crossed)))
    assert np.all(np.abs(dz) <= KERB_M * np.abs(crossed) + 1e-6), (
        f"triangle {worst} moved {dz[worst]:+.3f} m but only {crossed[worst]:.3f} of its area "
        f"crossed the {KERB_M} m kerb (bound {KERB_M * abs(crossed[worst]):.3f} m)")
    assert np.abs(dz).max() < KERB_M, "a triangle flipped by the full kerb height"
    assert np.abs(dz).max() > 0.0


# ── AC2: flat DEM, nodata, small triangles ──────────────────────────────


def test_footprint_average_matches_flat_dem(tmp_path):
    grid = _grid()
    band = np.full((grid.height, grid.width), 17.25, dtype=np.float32)
    rng = np.random.default_rng(2186)
    mesh = _structured_mesh(3.3, 2.7, 12, 12, 2.5)
    mesh = mesh + rng.uniform(-0.2, 0.2, size=(1, 2))  # an arbitrary sub-cell offset
    np.testing.assert_allclose(_seat_pure(tmp_path, "flat", mesh, grid, band),
                               _centroid_sample(mesh, grid, band), atol=1e-6)


@pytest.mark.requires_anuga
def test_footprint_average_matches_centroid_on_flat_dem_on_a_domain(tmp_path):
    """The same agreement through seat_elevation on a real Domain, against anuga's own seating."""
    from anuga.utilities import quantity_setting_functions as qs
    from run_anuga.elevation_sampling import seat_elevation

    xll, yll = 382000.0, 6354000.0
    grid = _grid(x_origin=xll - 5.0, y_top=yll + 40.0, width=45, height=45)
    tif = _write_tif(tmp_path / "flat.tif", np.full((45, 45), 12.5, dtype=np.float32), grid)
    footprint, mesh = _anuga_kerb_domain(xll, yll)
    centroid, _ = _anuga_kerb_domain(xll, yll, mesh_from=mesh)
    seat_elevation(footprint, tif, str(tmp_path / "f.elevation_coverage.npz"))
    centroid.set_quantity("elevation", qs.composite_quantity_setting_function(
        [["Extent", tif]], centroid, nan_treatment="exception"), location="centroids")
    np.testing.assert_allclose(footprint.get_quantity("elevation").get_values(location="centroids"),
                               centroid.get_quantity("elevation").get_values(location="centroids"), atol=1e-6)


@pytest.mark.parametrize("bad", ["declared", "nan"])
def test_nodata_inside_boundary_still_raises(tmp_path, bad):
    """nan_treatment='exception' semantics survive: a covered nodata cell refuses, and the
    TASK-1138 pre-flight still names the gap before seating starts."""
    from run_anuga.run_utils import assert_raster_has_no_nodata_inside_boundary

    grid = _grid()
    band = np.full((grid.height, grid.width), BASE_Z, dtype=np.float32)
    band[20, 20] = -9999.0 if bad == "declared" else np.nan
    mesh = _structured_mesh(10.0, 10.0, 10, 10, 2.0)

    with pytest.raises(ValueError, match="nodata"):
        _seat_pure(tmp_path, "gap", mesh, grid, band, nodata=-9999.0)

    tif = _write_tif(tmp_path / "gap.tif", band, grid)
    boundary = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]
    with pytest.raises(ValueError, match="nodata"):
        assert_raster_has_no_nodata_inside_boundary(tif, boundary, quantity_name="elevation")


def test_a_zero_nodata_tag_is_matched_exactly(tmp_path):
    """A relative tolerance alone never matches a 0 tag; the exact match (as the pre-flight) does."""
    grid = _grid()
    band = np.full((grid.height, grid.width), BASE_Z, dtype=np.float32)
    band[15, 15] = 0.0
    with pytest.raises(ValueError, match="nodata"):
        _seat_pure(tmp_path, "zero", _structured_mesh(10.0, 10.0, 10, 10, 2.0), grid, band, nodata=0.0)


def test_nodata_outside_every_footprint_is_ignored(tmp_path):
    grid = _grid()
    band = np.full((grid.height, grid.width), BASE_Z, dtype=np.float32)
    band[0, 0] = -9999.0  # far from the mesh
    mesh = _structured_mesh(10.0, 10.0, 4, 4, 2.0)
    np.testing.assert_allclose(_seat_pure(tmp_path, "ok", mesh, grid, band, nodata=-9999.0), BASE_Z)


def test_small_triangle_falls_back_to_centroid(tmp_path):
    """A triangle smaller than one cell seats the centroid sample: inside one cell the
    footprint is that one cell, and a zero-area triangle uses the cell under its centroid."""
    grid = _grid(width=4, height=4)
    band = np.arange(16, dtype=np.float32).reshape(4, 4)
    inside_one_cell = [(1.1, 37.1), (1.6, 37.2), (1.3, 37.8)]   # inside cell row 2, col 1
    degenerate = [(2.2, 38.5), (2.4, 38.5), (2.6, 38.5)]         # zero area, centroid in row 1, col 2
    mesh = np.asarray([inside_one_cell, degenerate], dtype=float).reshape(-1, 2)

    seated = _seat_pure(tmp_path, "small", mesh, grid, band)
    np.testing.assert_allclose(seated, _centroid_sample(mesh, grid, band))
    np.testing.assert_allclose(seated, [band[2, 1], band[1, 2]])


def test_sub_cell_triangle_across_a_cell_edge_is_area_weighted_unless_the_fallback_is_asked_for(tmp_path):
    """A sub-cell triangle straddling a cell edge gets the area-weighted mean by default (no
    coin flip); min_footprint_cells=1 restores the card's literal centroid fallback."""
    grid = _grid(width=4, height=4)
    band = np.zeros((4, 4), dtype=np.float32)
    band[:, 2:] = 1.0                                   # step at x = 2
    tri = np.asarray([(1.5, 37.2), (2.5, 37.2), (1.5, 37.8)], dtype=float)  # 0.3 m2, a quarter of it right of x=2
    seated = _seat_pure(tmp_path, "weighted", tri, grid, band)
    np.testing.assert_allclose(seated, [0.25], atol=1e-6)
    literal = _seat_pure(tmp_path, "literal", tri, grid, band, min_footprint_cells=1.0)
    np.testing.assert_allclose(literal, _centroid_sample(tri, grid, band))


def test_exact_area_weights(tmp_path):
    """Triangle (0,0),(2,0),(0,2) in cell units: one full cell plus two half cells."""
    from run_anuga.elevation_sampling import CoverageIndex
    grid = _grid(width=4, height=4)
    tri = np.asarray([(0.0, 36.0), (2.0, 36.0), (0.0, 38.0)], dtype=float)
    index = CoverageIndex.build(str(tmp_path / "w.elevation_coverage.npz"), tri, grid)
    (t0, t1, tris, cells, weights), = list(index.iter_blocks())
    got = {int(c): float(w) for c, w in zip(cells, weights)}
    # rows count down from the top (y=40): y in [36,37] is row 3, [37,38] row 2.
    assert got == pytest.approx({3 * 4 + 0: 0.5, 3 * 4 + 1: 0.25, 2 * 4 + 0: 0.25})
    assert (t0, t1) == (0, 1) and set(tris.tolist()) == {0}


def test_centroid_outside_raster_raises(tmp_path):
    grid = _grid(width=10, height=10)  # x 0..10, y 30..40
    tri = np.asarray([(9.0, 31.0), (14.0, 31.0), (14.0, 36.0)], dtype=float)  # centroid x = 12.3
    with pytest.raises(ValueError, match="outside the elevation raster"):
        _seat_pure(tmp_path, "outside", tri, grid, np.zeros((10, 10), dtype=np.float32))


def test_footprint_partly_outside_raster_is_the_mean_of_the_cells_that_exist(tmp_path):
    grid = _grid(width=10, height=10)
    band = np.full((10, 10), 5.0, dtype=np.float32)
    tri = np.asarray([(8.0, 31.0), (10.6, 31.0), (8.0, 33.6)], dtype=float)  # pokes 0.6 m past x=10
    np.testing.assert_allclose(_seat_pure(tmp_path, "edge", tri, grid, band), [5.0])


def test_rotated_raster_is_refused():
    from rasterio.transform import Affine
    from run_anuga.elevation_sampling import RasterGrid
    with pytest.raises(ValueError, match="north-up"):
        RasterGrid.from_transform(Affine(1.0, 0.2, 0.0, 0.0, -1.0, 40.0), 10, 10)


# ── AC3: the coverage index ──────────────────────────────────────────────


def test_coverage_index_roundtrip(tmp_path):
    """Built in row blocks, persisted as a compressed .npz beside the mesh, reopened from
    the file alone, and never resident: the object holds a path and metadata, not arrays."""
    from run_anuga.elevation_sampling import CoverageIndex, coverage_index_path

    grid = _grid()
    mesh = _structured_mesh(3.3, 2.7, 12, 12, 2.5)
    path = coverage_index_path(str(tmp_path / "run_1_1_1.msh"))
    assert path == str(tmp_path / "run_1_1_1.elevation_coverage.npz")

    blocked = CoverageIndex.build(path, mesh, grid, max_pairs_per_block=200)
    whole = CoverageIndex.build(str(tmp_path / "whole.npz"), mesh, grid)
    assert blocked.n_blocks > 1 and whole.n_blocks == 1
    assert not [k for k, v in vars(blocked).items() if isinstance(v, np.ndarray)], "index arrays are resident"

    reopened = CoverageIndex.open(path)
    assert (reopened.n_triangles, reopened.grid, reopened.n_blocks) == (len(mesh) // 3, grid, blocked.n_blocks)

    def triples(index):
        blocks = list(index.iter_blocks())
        assert [b[0] for b in blocks] == sorted(b[0] for b in blocks)
        assert blocks[-1][1] == index.n_triangles
        return [np.concatenate([b[i] for b in blocks]) for i in (2, 3, 4)]

    for a, b in zip(triples(reopened), triples(whole)):
        np.testing.assert_array_equal(a, b)
    tri, _, weight = triples(reopened)
    np.testing.assert_allclose(np.bincount(tri, weight.astype(float)), 1.0, atol=1e-6)

    with zipfile.ZipFile(path) as zf:
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in zf.infolist())
    with np.load(path) as npz:
        assert "meta" in npz.files


def test_open_refuses_a_file_that_is_not_a_coverage_index(tmp_path):
    from run_anuga.elevation_sampling import CoverageIndex
    path = tmp_path / "other.npz"
    np.savez_compressed(path, a=np.arange(3))
    with pytest.raises(ValueError, match="coverage index"):
        CoverageIndex.open(str(path))


def test_result_zip_leaves_the_coverage_index_out():
    """The index is build scratch like the .msh: it must not ride the result zip to S3."""
    from pathlib import Path

    from run_anuga._handoff import _is_excluded_from_result
    from run_anuga.elevation_sampling import coverage_index_path
    index = Path(coverage_index_path("outputs_1_1_1/run_1_1_1.msh"))
    assert _is_excluded_from_result(index, "result.zip")


# ── TASK-1299 still applies after the seating ────────────────────────────


@pytest.mark.requires_anuga
def test_raised_buildings_still_apply_after_footprint_seating(tmp_path):
    from run_anuga.elevation_sampling import seat_elevation
    from run_anuga.run_utils import apply_raised_elevation_correction

    xll, yll = 382000.0, 6354000.0
    grid = _grid(x_origin=xll - 5.0, y_top=yll + 40.0, width=45, height=45)
    tif = _write_tif(tmp_path / "flat.tif", np.full((45, 45), 10.0, dtype=np.float32), grid)
    domain, _ = _anuga_kerb_domain(xll, yll)
    seat_elevation(domain, tif, str(tmp_path / "d.elevation_coverage.npz"))
    poly = [[xll + 10, yll + 10], [xll + 20, yll + 10], [xll + 20, yll + 20], [xll + 10, yll + 20]]
    assert apply_raised_elevation_correction(domain, [(poly, 4.0)]) == 1
    elev = domain.get_quantity("elevation").get_values(location="centroids")
    assert elev.max() == pytest.approx(14.0) and elev.min() == pytest.approx(10.0)


# ── TASK-4357: the sampling method is a scenario option ──────────────────


def test_elevation_sampling_defaults_to_footprint():
    from run_anuga.elevation_sampling import elevation_sampling_method
    assert elevation_sampling_method({}) == "footprint"
    assert elevation_sampling_method(None) == "footprint"
    assert elevation_sampling_method({"elevation_sampling": None}) == "footprint"
    assert elevation_sampling_method({"elevation_sampling": "footprint"}) == "footprint"
    assert elevation_sampling_method({"elevation_sampling": "centroid"}) == "centroid"


@pytest.mark.parametrize("bad", ["", "Centroid", "bilinear", 1, ["centroid"]])
def test_an_unknown_elevation_sampling_is_refused_naming_the_allowed_values(bad):
    from run_anuga.elevation_sampling import elevation_sampling_method
    with pytest.raises(ValueError) as err:
        elevation_sampling_method({"elevation_sampling": bad})
    assert "'footprint'" in str(err.value) and "'centroid'" in str(err.value)


def _seat_like_cd78e08(domain, elevation_filename):
    """The pre-2186 seating, copied from run_anuga cd78e08 run.py (the AC2 oracle)."""
    from anuga.utilities import quantity_setting_functions as qs
    poly_fun_pairs = [['Extent', elevation_filename]]
    elevation_function = qs.composite_quantity_setting_function(
        poly_fun_pairs,
        domain,
        nan_treatment='exception',
    )
    domain.set_quantity('elevation', elevation_function, verbose=False, alpha=0.99, location='centroids')


@pytest.mark.requires_anuga
def test_centroid_sampling_is_bit_identical_to_the_pre_2186_seating(tmp_path):
    """AC2 + AC5: 'centroid' seats every triangle exactly as cd78e08 did (the one DEM cell
    under its centroid) on a Triangle mesh over a rough DEM, and writes no coverage index."""
    from run_anuga.elevation_sampling import seat_elevation_by_method

    xll, yll = 382000.0, 6354000.0
    grid = _grid(x_origin=xll - 5.0, y_top=yll + 40.0, width=45, height=45)
    rough = (BASE_Z + np.random.default_rng(4357).normal(0.0, 0.5, (45, 45))).astype(np.float32)
    tif = _write_tif(tmp_path / "rough.tif", rough, grid)
    seated, mesh = _anuga_kerb_domain(xll, yll)
    oracle, _ = _anuga_kerb_domain(xll, yll, mesh_from=mesh)
    coverage = tmp_path / "seated.elevation_coverage.npz"
    assert seat_elevation_by_method(seated, "centroid", tif, str(coverage)) == "centroid"
    _seat_like_cd78e08(oracle, tif)
    got = seated.get_quantity("elevation").get_values(location="centroids")
    want = oracle.get_quantity("elevation").get_values(location="centroids")
    assert np.array_equal(got, want)
    np.testing.assert_array_equal(  # and it is the one cell under each centroid
        got, _centroid_sample(seated.get_vertex_coordinates(absolute=True), grid, rough))
    assert not coverage.exists() and not list(tmp_path.glob("*.npz"))


@pytest.mark.requires_anuga
def test_footprint_sampling_by_method_is_seat_elevation(tmp_path):
    """AC1: 'footprint' is today's seat_elevation, unchanged (and it persists the index)."""
    from run_anuga.elevation_sampling import seat_elevation, seat_elevation_by_method

    xll, yll = 382000.0, 6354000.0
    grid = _grid(x_origin=xll - 5.0, y_top=yll + 40.0, width=45, height=45)
    tif = _write_tif(tmp_path / "kerb.tif", _kerb_band(grid, "x", xll + 15.0), grid)
    by_method, mesh = _anuga_kerb_domain(xll, yll)
    direct, _ = _anuga_kerb_domain(xll, yll, mesh_from=mesh)
    coverage = tmp_path / "m.elevation_coverage.npz"
    assert seat_elevation_by_method(by_method, "footprint", tif, str(coverage)) == "footprint"
    seat_elevation(direct, tif, str(tmp_path / "d.elevation_coverage.npz"))
    assert coverage.exists()
    assert np.array_equal(by_method.get_quantity("elevation").get_values(location="centroids"),
                          direct.get_quantity("elevation").get_values(location="centroids"))


def _run_small_test(package, **scenario):
    import json
    import subprocess
    import sys
    path = package / "scenario.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "duration": 60, **scenario}))
    result = subprocess.run([sys.executable, "-m", "run_anuga.cli", "run", str(package)],
                            capture_output=True, text=True, timeout=900)
    return result, result.stdout + result.stderr


@pytest.mark.requires_anuga
@pytest.mark.parametrize("method", ["centroid", None])
def test_a_run_logs_and_records_its_elevation_sampling(small_test_copy, method):
    """AC4 + AC5 end to end: run_anuga_1.log states the method, run_summary_1.json records
    it as terrain.elevation_sampling, and only the footprint run writes a coverage index."""
    import json
    extra = {} if method is None else {"elevation_sampling": method}
    result, output = _run_small_test(small_test_copy, **extra)
    assert result.returncode == 0, output[-3000:]
    [out] = small_test_copy.glob("outputs_*")
    expected = method or "footprint"
    assert f"elevation sampling: {expected}" in (out / "run_anuga_1.log").read_text()
    summary = json.loads((out / "run_summary_1.json").read_text())
    assert summary["terrain"]["elevation_sampling"] == expected
    assert bool(list(out.glob("*.elevation_coverage.npz"))) == (expected == "footprint")


@pytest.mark.requires_anuga
def test_an_unknown_elevation_sampling_stops_the_run_before_the_mesh(small_test_copy):
    """AC3: refused before any mesh or DEM work, with the allowed values named."""
    result, output = _run_small_test(small_test_copy, elevation_sampling="nearest")
    assert result.returncode != 0
    assert "'footprint'" in output and "'centroid'" in output
    assert not list(small_test_copy.glob("outputs_*/*.msh"))
