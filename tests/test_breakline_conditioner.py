"""TASK-1715 — Unit tests for the build-time Triangle-safety breakline conditioner.

PURE (shapely-only) — deliberately NOT gated behind ``import anuga`` so these run
in CI on a box without the ANUGA engine.  Covers the four conditioning duties:
clip-to-boundary, node-crossings, dedupe, drop-sub-CFL, plus simplify/densify and
defensive handling of malformed input.
"""

import math

import pytest

pytest.importorskip("shapely", reason="shapely not installed")

from run_anuga.breakline_conditioner import (  # noqa: E402
    condition_breaklines,
    _dedupe_and_drop_short,
    _line_length,
)

# A simple 100 x 100 square domain in planar (CRS-agnostic) coords.
BOUNDARY = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]


def _fc(lines, near_spacing=10.0):
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


def _all_vertices(polylines):
    return [tuple(round(c, 6) for c in pt) for line in polylines for pt in line]


def _min_segment(polylines):
    m = float('inf')
    for line in polylines:
        for i in range(len(line) - 1):
            m = min(m, math.hypot(line[i + 1][0] - line[i][0],
                                  line[i + 1][1] - line[i][1]))
    return m


def _point_in_square(pt, lo=-1e-6, hi=100.0 + 1e-6):
    return lo <= pt[0] <= hi and lo <= pt[1] <= hi


# ---------------------------------------------------------------------------
# Helper: dedupe + sub-CFL drop (the precise responsibility)
# ---------------------------------------------------------------------------

class TestDedupeAndDropShort:

    def test_drops_exact_duplicate_vertex(self):
        out = _dedupe_and_drop_short([[0, 0], [0, 0], [10, 0]], 0.5)
        assert out == [[0, 0], [10, 0]]

    def test_drops_subcfl_interior_segment(self):
        # (0.1, 0) is 0.1m from (0,0) — below the 0.5m floor — so it is absorbed.
        out = _dedupe_and_drop_short([[0, 0], [0.1, 0], [10, 0]], 0.5)
        assert out == [[0, 0], [10, 0]]

    def test_snaps_subcfl_tail_onto_true_endpoint(self):
        out = _dedupe_and_drop_short([[0, 0], [10, 0], [10.1, 0]], 0.5)
        # No sub-floor tail segment survives; the true endpoint is preserved.
        assert out[-1] == [10.1, 0]
        assert _min_segment([out]) >= 0.5

    def test_never_truncates_below_two_vertices(self):
        out = _dedupe_and_drop_short([[0, 0], [0.1, 0]], 0.5)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Public condition_breaklines — the four conditioning duties end-to-end
# ---------------------------------------------------------------------------

class TestConditionBreaklines:

    def test_clips_to_boundary(self):
        """A line running outside the boundary is clipped to inside."""
        line = [[-50.0, 50.0], [150.0, 50.0]]  # spans well past both x edges
        out = condition_breaklines(_fc([line]), BOUNDARY)
        assert out, "expected at least one conditioned line"
        for pt in _all_vertices(out):
            assert _point_in_square(pt), f"vertex {pt} escaped the boundary clip"

    def test_nodes_crossings(self):
        """Two crossing lines are noded — a shared vertex is inserted at the crossing."""
        line_a = [[10.0, 50.0], [90.0, 50.0]]   # horizontal
        line_b = [[50.0, 10.0], [50.0, 90.0]]   # vertical, crosses at (50,50)
        out = condition_breaklines(_fc([line_a, line_b]), BOUNDARY)
        verts = _all_vertices(out)
        assert (50.0, 50.0) in verts, (
            "crossing point (50,50) was not inserted as a shared PSLG vertex"
        )
        # PSLG validity: the noded collection's only intersections are at endpoints.
        from shapely.geometry import MultiLineString
        from shapely.ops import unary_union
        mls = MultiLineString([[(x, y) for x, y in line] for line in out])
        # unary_union of an already-noded set adds no further nodes -> same length.
        assert math.isclose(unary_union(mls).length, mls.length, rel_tol=1e-9)

    def test_dedupes_coincident_vertices(self):
        line = [[10.0, 50.0], [10.0, 50.0], [90.0, 50.0]]  # leading exact duplicate
        out = condition_breaklines(_fc([line], near_spacing=200.0), BOUNDARY)
        assert out
        assert _min_segment(out) > 0.0, "a zero-length (duplicate) segment survived"

    def test_drops_subcfl_segments(self):
        line = [[10.0, 50.0], [10.05, 50.0], [90.0, 50.0]]  # 0.05m sub-CFL stub
        out = condition_breaklines(_fc([line], near_spacing=200.0), BOUNDARY)
        assert out
        assert _min_segment(out) >= 0.5, "a sub-CFL (<0.5m) segment survived conditioning"

    def test_simplify_densify_toward_near_spacing(self):
        """A long straight line is densified so no segment exceeds near_spacing."""
        line = [[5.0, 50.0], [95.0, 50.0]]  # 90m long
        out = condition_breaklines(_fc([line], near_spacing=10.0), BOUNDARY)
        assert out
        # densify floor: every segment <= near_spacing (+ float slack)
        assert _min_segment(out) <= 10.0 + 1e-6
        max_seg = max(
            math.hypot(line[i + 1][0] - line[i][0], line[i + 1][1] - line[i][1])
            for line in out for i in range(len(line) - 1)
        )
        assert max_seg <= 10.0 + 1e-6, f"a segment of {max_seg}m exceeds near_spacing"

    def test_empty_inputs_return_empty(self):
        assert condition_breaklines(None, BOUNDARY) == []
        assert condition_breaklines({'features': []}, BOUNDARY) == []
        assert condition_breaklines({}, BOUNDARY) == []

    def test_malformed_geometry_does_not_raise(self):
        bad = {
            'type': 'FeatureCollection',
            'features': [
                {'type': 'Feature', 'id': 'nogeom', 'geometry': None, 'properties': {}},
                {'type': 'Feature', 'id': 'bad', 'geometry': {'type': 'Nonsense'},
                 'properties': {}},
                {'type': 'Feature', 'id': 'ok',
                 'geometry': {'type': 'LineString', 'coordinates': [[10, 50], [90, 50]]},
                 'properties': {'near_spacing': 20.0}},
            ],
        }
        out = condition_breaklines(bad, BOUNDARY)  # must not raise
        assert out, "the one valid line should still be conditioned"

    def test_line_length_helper(self):
        assert math.isclose(_line_length([[0, 0], [3, 4]]), 5.0)
        assert _line_length([[0, 0]]) == 0.0
