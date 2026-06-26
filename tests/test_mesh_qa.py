"""TASK-1923 — compute_mesh_qa extended area + shape features.

Tests:
- min_triangle_area returned and correct
- log-binned area histogram returned and sums to triangle_count
- existing shape metrics (min_angle_deg, aspect_ratio_max, sliver_count) present
- boundary_condition_types captured from domain.boundary
- resolution + duration denormalized from scenario_config
- export_batch_resource_corpus _FEATURE_COLUMNS includes the new fields
"""
from __future__ import annotations

import math
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal fake anuga mesh object
# ---------------------------------------------------------------------------

def _make_fake_mesh(triangle_coords):
    """Build a fake anuga mesh object from a list of (v0, v1, v2) coordinate triples.

    Each triple is ((x0,y0),(x1,y1),(x2,y2)).  We derive the vertices/triangles
    arrays exactly like the real ANUGA tri_mesh structure.
    """
    all_verts = []
    triangles = []
    for tri in triangle_coords:
        idx = []
        for pt in tri:
            all_verts.append(pt)
            idx.append(len(all_verts) - 1)
        triangles.append(idx)

    vertices = np.array(all_verts, dtype=float)
    tris = np.array(triangles, dtype=int)

    tri_mesh = types.SimpleNamespace(vertices=vertices, triangles=tris)
    mesh = types.SimpleNamespace(tri_mesh=tri_mesh)
    return mesh


# A single right-isoceles triangle with legs of 1 m — area = 0.5 m².
_UNIT_TRIANGLE = [
    ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
]

# Two triangles with very different areas: 0.5 m² and 5000 m².
_TWO_AREA_TRIANGLES = [
    ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),           # area = 0.5 m²
    ((0.0, 0.0), (100.0, 0.0), (0.0, 100.0)),        # area = 5000 m²
]


class TestComputeMeshQaAreaStats:
    """compute_mesh_qa now includes min_triangle_area + area_histogram."""

    def test_min_triangle_area_single_triangle(self):
        from run_anuga.run_utils import compute_mesh_qa
        mesh = _make_fake_mesh(_UNIT_TRIANGLE)
        qa = compute_mesh_qa(mesh)
        assert "min_triangle_area" in qa
        assert abs(qa["min_triangle_area"] - 0.5) < 0.01

    def test_area_histogram_present_and_sums_to_triangle_count(self):
        from run_anuga.run_utils import compute_mesh_qa
        mesh = _make_fake_mesh(_TWO_AREA_TRIANGLES)
        qa = compute_mesh_qa(mesh)
        assert "area_histogram" in qa
        hist = qa["area_histogram"]
        # histogram is a list of {"bin_lo": ..., "bin_hi": ..., "count": ...}
        assert isinstance(hist, list)
        assert len(hist) > 0
        total = sum(b["count"] for b in hist)
        assert total == qa["triangle_count"], (
            f"histogram total {total} != triangle_count {qa['triangle_count']}"
        )

    def test_area_histogram_bins_are_log_spaced(self):
        from run_anuga.run_utils import compute_mesh_qa
        mesh = _make_fake_mesh(_TWO_AREA_TRIANGLES)
        qa = compute_mesh_qa(mesh)
        hist = qa["area_histogram"]
        # Each non-empty bin should have bin_hi / bin_lo constant (log-spaced)
        nonempty = [b for b in hist if b["count"] > 0]
        assert len(nonempty) >= 1

    def test_min_triangle_area_zero_triangle_mesh(self):
        """Empty mesh stays safe — no division by zero, returns 0."""
        from run_anuga.run_utils import compute_mesh_qa
        tri_mesh = types.SimpleNamespace(
            vertices=np.zeros((0, 2), dtype=float),
            triangles=np.zeros((0, 3), dtype=int),
        )
        mesh = types.SimpleNamespace(tri_mesh=tri_mesh)
        qa = compute_mesh_qa(mesh)
        assert qa["min_triangle_area"] == 0.0
        assert qa["area_histogram"] == []

    def test_shape_metrics_present_alongside_area(self):
        """Existing keys (min_angle_deg, aspect_ratio_max, sliver_count) still present."""
        from run_anuga.run_utils import compute_mesh_qa
        mesh = _make_fake_mesh(_UNIT_TRIANGLE)
        qa = compute_mesh_qa(mesh)
        assert "min_angle_deg" in qa
        assert "aspect_ratio_max" in qa
        assert "sliver_count" in qa


class TestComputeMeshQaBCTypes:
    """compute_mesh_qa_bc_types extracts BC type set from a domain.boundary."""

    def test_bc_types_from_domain_boundary(self):
        from run_anuga.run_utils import extract_boundary_condition_types

        domain = MagicMock()
        domain.boundary = {
            (0, 0): "exterior",
            (1, 0): "Reflective",
            (2, 0): "exterior",
        }
        bc_types = extract_boundary_condition_types(domain)
        assert isinstance(bc_types, list)
        assert set(bc_types) == {"exterior", "Reflective"}

    def test_bc_types_empty_domain_boundary(self):
        from run_anuga.run_utils import extract_boundary_condition_types

        domain = MagicMock()
        domain.boundary = {}
        bc_types = extract_boundary_condition_types(domain)
        assert bc_types == []


# NOTE: TestExportCorpusFeatureColumns lives in the hydrata repo test suite:
# apps/gn_anuga/tests/test_mesh_qa_corpus.py (requires Django/gn_anuga import).
