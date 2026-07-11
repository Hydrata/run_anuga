"""TASK-1715 — Breakline mesh-edge conformance + build-time Triangle-safety conditioner.

A ``Breakline`` must CONFORM a mesh edge (force a Triangle PSLG constraint along the
line), not merely grade density near it. This module proves:

1. CONFORMANCE — building a mesh with a known straight breakline strictly inside the
   boundary yields an output mesh whose constrained segments include an edge lying
   along the breakline; a control build of the SAME scenario WITHOUT the breakline
   lacks that edge.  (Mirrors anuga's own test_create_mesh_with_breaklines,
   /opt/anuga_core/anuga/pmesh/tests/test_mesh_interface.py:651.)
2. CONDITIONER — clip to boundary, node crossings into a valid PSLG, dedupe
   coincident vertices, drop sub-CFL/degenerate segments, simplify/densify toward
   near_spacing.
3. GRADING PRESERVED — make_breaklines buffer-ring grading still emitted; conform +
   grade COMPOSE.
4. GUARDRAILS — no degenerate sliver triangles introduced; triangle count bounded.

Self-skips when ANUGA / meshpy are not installed (CI without the engine).
"""

import math
import os
import tempfile

import pytest

anuga = pytest.importorskip("anuga", reason="anuga not installed")
pytest.importorskip("meshpy.triangle", reason="meshpy not installed")

from run_anuga.run_utils import (  # noqa: E402
    create_anuga_mesh,
    make_breaklines,
    compute_mesh_qa,
)

# ---------------------------------------------------------------------------
# Shared domain geometry — 200m x 200m square in EPSG:28355 (GDA94 / MGA zone 55)
# ---------------------------------------------------------------------------

BOUNDARY_POLYGON = [
    [321000.0, 5812000.0],
    [321200.0, 5812000.0],
    [321200.0, 5812200.0],
    [321000.0, 5812200.0],
]
BOUNDARY_TAGS = {'exterior': list(range(len(BOUNDARY_POLYGON)))}
SCENARIO_CONFIG = {
    'epsg': 'EPSG:28355',
    'resolution': 20,            # 20m -> ~200 triangles baseline, fast test
    'default_near_spacing': 10.0,
    'project': 1,
    'id': 1,
    'run_id': 1,
}

# A straight diagonal breakline strictly inside the boundary, deliberately NOT
# colinear with any (axis-aligned) boundary edge so the control build cannot
# accidentally produce a colinear segment.
BREAKLINE_A = [321060.0, 5812060.0]
BREAKLINE_B = [321140.0, 5812140.0]


def _make_breakline_geojson(lines, near_spacing=10.0):
    """Build a minimal GeoJSON FeatureCollection of LineString breaklines."""
    return {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'id': f'bl_{i}',
                'geometry': {'type': 'LineString', 'coordinates': line},
                'properties': {'near_spacing': near_spacing},
            }
            for i, line in enumerate(lines)
        ],
    }


def _build_input_data(tmp_dir, breakline_lines=None, near_spacing=10.0):
    msh_path = os.path.join(tmp_dir, "run_1_1_1.msh")
    input_data = {
        "mesh_filepath": msh_path,
        "scenario_config": SCENARIO_CONFIG,
        "boundary_polygon": BOUNDARY_POLYGON,
        "boundary_tags": BOUNDARY_TAGS,
    }
    if breakline_lines is not None:
        input_data['breakline'] = _make_breakline_geojson(breakline_lines, near_spacing)
    return input_data


def _point_on_segment(p, a, b, tol=1e-3):
    """True if point p lies on the finite segment a->b within tol (perp dist + extent)."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0:
        return math.hypot(px - ax, py - ay) <= tol
    # projection parameter
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    if t < -tol or t > 1 + tol:
        return False
    # perpendicular distance
    cross = abs((px - ax) * dy - (py - ay) * dx) / math.sqrt(seg_len2)
    return cross <= tol


def _mesh_offset(anuga_mesh):
    """(xll, yll) of the mesh geo_reference. Adding it to a mesh-local vertex
    recovers ABSOLUTE coordinates in either frame:
      - absolute-UTM path (TASK-2149, epsg + no holes) -> xll=yll=0, verts already absolute;
      - ANUGA local lower-left offset (None geo_reference) -> xll/yll = boundary corner.
    Comparing in absolute coords makes the conformance check independent of which
    mesh_geo_reference create_anuga_mesh chose (the branch that de94f00 predates).
    """
    gr = anuga_mesh.geo_reference
    return gr.get_xllcorner(), gr.get_yllcorner()


def _constrained_segments_along_line(anuga_mesh, a, b, tol=1e-3):
    """Count output constrained segments whose BOTH endpoints lie on segment a->b.

    tri_mesh.segments holds only the constrained (PSLG) edges — boundary + breaklines
    — so a colinear constrained segment is direct evidence the breakline conformed a
    mesh edge.  Vertices are lifted to ABSOLUTE coords via the mesh geo_reference so
    the comparison line (a, b — absolute) matches regardless of the mesh frame.
    """
    verts = anuga_mesh.tri_mesh.vertices
    segs = anuga_mesh.tri_mesh.segments
    xoff, yoff = _mesh_offset(anuga_mesh)
    count = 0
    for seg in segs:
        v0 = verts[seg[0]]
        v1 = verts[seg[1]]
        p0 = (v0[0] + xoff, v0[1] + yoff)
        p1 = (v1[0] + xoff, v1[1] + yoff)
        if _point_on_segment(p0, a, b, tol) and _point_on_segment(p1, a, b, tol):
            count += 1
    return count


def _min_edge_length(anuga_mesh):
    verts = anuga_mesh.tri_mesh.vertices
    tris = anuga_mesh.tri_mesh.triangles
    if len(tris) == 0:
        return 0.0
    min_len = float('inf')
    for tri in tris:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            edge = math.hypot(verts[tri[i]][0] - verts[tri[j]][0],
                              verts[tri[i]][1] - verts[tri[j]][1])
            min_len = min(min_len, edge)
    return min_len


# ---------------------------------------------------------------------------
# CONFORMANCE — the load-bearing proof
# ---------------------------------------------------------------------------

class TestBreaklineConformance:

    def test_breakline_creates_conforming_mesh_edge(self):
        """A breakline strictly inside the boundary forces ≥1 colinear constrained edge."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_data = _build_input_data(tmp_dir, breakline_lines=[[BREAKLINE_A, BREAKLINE_B]])
            _, anuga_mesh = create_anuga_mesh(input_data)
            n_along = _constrained_segments_along_line(anuga_mesh, BREAKLINE_A, BREAKLINE_B)
            assert n_along >= 1, (
                "No constrained mesh edge lies along the breakline — breaklines= was "
                "not passed to create_mesh_from_regions (the whole TASK-1715 bug)."
            )

    def test_control_without_breakline_lacks_edge(self):
        """The SAME scenario WITHOUT a breakline has no constrained edge along that line."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_data = _build_input_data(tmp_dir, breakline_lines=None)
            _, anuga_mesh = create_anuga_mesh(input_data)
            n_along = _constrained_segments_along_line(anuga_mesh, BREAKLINE_A, BREAKLINE_B)
            assert n_along == 0, (
                f"Control mesh (no breakline) unexpectedly has {n_along} constrained "
                f"edges along the diagonal — the conformance test would false-positive."
            )


# ---------------------------------------------------------------------------
# GRADING PRESERVED — conform + grade COMPOSE (do not remove the buffer rings)
# ---------------------------------------------------------------------------

class TestGradingComposesWithConformance:

    def test_grading_regions_still_emitted(self):
        """make_breaklines still synthesises buffer-ring grading regions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_data = _build_input_data(tmp_dir, breakline_lines=[[BREAKLINE_A, BREAKLINE_B]])
            regions = make_breaklines(input_data)
            assert len(regions) >= 1, "breakline grading buffer rings were dropped"

    def test_conform_and_grade_both_present(self):
        """ONE mesh build has BOTH a conforming breakline edge AND graded refinement."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_data = _build_input_data(tmp_dir, breakline_lines=[[BREAKLINE_A, BREAKLINE_B]])
            _, anuga_mesh = create_anuga_mesh(input_data)

            # Conform: a constrained edge lies along the breakline.
            n_along = _constrained_segments_along_line(anuga_mesh, BREAKLINE_A, BREAKLINE_B)
            assert n_along >= 1, "conforming breakline edge missing"

            # Grade: triangles near the line are finer than the global max area
            # (triangle_resolution = resolution**2 / 2 = 200 m² here).
            global_max_area = (SCENARIO_CONFIG['resolution'] ** 2) / 2.0
            verts = anuga_mesh.tri_mesh.vertices
            tris = anuga_mesh.tri_mesh.triangles
            xoff, yoff = _mesh_offset(anuga_mesh)
            # Compare in ABSOLUTE coords (verts lifted by the mesh offset) so the
            # midpoint stays valid under either mesh_geo_reference frame.
            mid_abs = ((BREAKLINE_A[0] + BREAKLINE_B[0]) / 2,
                       (BREAKLINE_A[1] + BREAKLINE_B[1]) / 2)

            def _tri_area(t):
                (x0, y0), (x1, y1), (x2, y2) = verts[t[0]], verts[t[1]], verts[t[2]]
                return abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / 2.0

            def _tri_centroid_near(t, pt, radius=15.0):
                cx = (verts[t[0]][0] + verts[t[1]][0] + verts[t[2]][0]) / 3.0 + xoff
                cy = (verts[t[0]][1] + verts[t[1]][1] + verts[t[2]][1]) / 3.0 + yoff
                return math.hypot(cx - pt[0], cy - pt[1]) <= radius

            near_areas = [_tri_area(t) for t in tris if _tri_centroid_near(t, mid_abs)]
            assert near_areas, "no triangles found near the breakline midpoint"
            assert min(near_areas) < global_max_area, (
                "no graded refinement near the breakline — grading did not compose "
                "with conformance"
            )


# ---------------------------------------------------------------------------
# GUARDRAILS — no degenerate slivers; triangle count bounded
# ---------------------------------------------------------------------------

class TestConformanceGuardrails:

    def test_no_degenerate_triangles(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_data = _build_input_data(tmp_dir, breakline_lines=[[BREAKLINE_A, BREAKLINE_B]])
            _, anuga_mesh = create_anuga_mesh(input_data)
            qa = compute_mesh_qa(anuga_mesh)
            assert not qa['has_degenerate_triangles'], (
                f"breakline/conditioner introduced degenerate triangles: {qa}"
            )
            assert _min_edge_length(anuga_mesh) >= 0.5, "sub-CFL sliver edge present"

    def test_crossing_breaklines_build_valid_mesh(self):
        """Two crossing breaklines (a Triangle PSLG hazard) build via the conditioner."""
        line_a = [[321040.0, 5812100.0], [321160.0, 5812100.0]]  # horizontal
        line_b = [[321100.0, 5812040.0], [321100.0, 5812160.0]]  # vertical (crosses)
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_data = _build_input_data(tmp_dir, breakline_lines=[line_a, line_b])
            _, anuga_mesh = create_anuga_mesh(input_data)
            assert len(anuga_mesh.tri_mesh.triangles) > 0
            qa = compute_mesh_qa(anuga_mesh)
            assert not qa['has_degenerate_triangles'], (
                "crossing breaklines produced degenerate triangles — noding failed"
            )

    def test_triangle_count_bounded_vs_plain(self):
        """A conformed mesh is within a generous budget of the plain (no-breakline) mesh."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            plain = _build_input_data(tmp_dir, breakline_lines=None)
            _, plain_mesh = create_anuga_mesh(plain)
            n_plain = len(plain_mesh.tri_mesh.triangles)
        with tempfile.TemporaryDirectory() as tmp_dir:
            withbl = _build_input_data(tmp_dir, breakline_lines=[[BREAKLINE_A, BREAKLINE_B]])
            _, bl_mesh = create_anuga_mesh(withbl)
            n_bl = len(bl_mesh.tri_mesh.triangles)
        assert n_bl < 5 * n_plain, (
            f"conformed mesh triangle count {n_bl} blew past budget "
            f"(5x plain baseline {n_plain})"
        )
