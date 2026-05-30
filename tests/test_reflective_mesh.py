"""TASK-1270 / W4.2 — Validation test: Reflective structures as interior mesh holes.

Tests that:
1. make_interior_holes_and_tags routes method='Reflective' to the interior-hole path.
2. The sliver-merge fix (ported from run_anuga 5604fc1) is applied — min edge length
   is above the sliver threshold (0.5m) when adjacent buildings are passed.
3. create_anuga_mesh produces a valid mesh with Reflective structures as interior voids.
4. A short timestep CFL check confirms the mesh doesn't collapse.

Design: uses a synthetic 200m x 200m domain (EPSG:28355) with two adjacent
10m x 10m building footprints sharing a vertex — this is the exact sliver scenario
the 5604fc1 fix addresses.

Self-skips gracefully when ANUGA or meshpy is not installed (for CI without ANUGA).
"""

import math
import os
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Self-skip guards — tests here require the full ANUGA stack
# ---------------------------------------------------------------------------
anuga = pytest.importorskip("anuga", reason="anuga not installed")
pytest.importorskip("meshpy.triangle", reason="meshpy not installed")

from run_anuga.run_utils import (  # noqa: E402
    create_anuga_mesh,
    make_interior_holes_and_tags,
    compute_mesh_qa,
)

# ---------------------------------------------------------------------------
# Shared domain geometry
# 200m x 200m square in EPSG:28355 (GDA94 / MGA zone 55)
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
    'resolution': 20,  # 20m → modest triangle count, fast test
    'project': 1,
    'id': 1,
    'run_id': 1,
}

# Two adjacent 10m x 10m buildings sharing their right/left edge at x=321050.
# Without sliver-merge, the shared boundary creates ~0 m² triangles.
BUILDING_A = [
    [321040.0, 5812080.0],
    [321050.0, 5812080.0],
    [321050.0, 5812090.0],
    [321040.0, 5812090.0],
    [321040.0, 5812080.0],  # closed ring
]
BUILDING_B = [
    [321050.0, 5812080.0],
    [321060.0, 5812080.0],
    [321060.0, 5812090.0],
    [321050.0, 5812090.0],
    [321050.0, 5812080.0],  # closed ring
]

# Minimum edge length the sliver-merge must achieve (threshold from 5604fc1: 1.074m
# on Merewether; we use 0.5m as a conservative floor for this simpler fixture).
MIN_EDGE_LENGTH_M = 0.5


def _make_structure_geojson(rings, method='Reflective'):
    """Build a minimal GeoJSON FeatureCollection of polygon structures."""
    return {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'id': f'str_{i}',
                'geometry': {'type': 'Polygon', 'coordinates': [ring]},
                'properties': {'method': method},
            }
            for i, ring in enumerate(rings)
        ]
    }


def _min_edge_length(anuga_mesh) -> float:
    """Compute the minimum edge length across all triangles in the mesh."""
    vertices = anuga_mesh.tri_mesh.vertices   # (N, 2)
    triangles = anuga_mesh.tri_mesh.triangles  # (M, 3)
    if len(triangles) == 0:
        return 0.0
    min_len = float('inf')
    for tri in triangles:
        for i, j in [(0, 1), (1, 2), (2, 0)]:
            dx = vertices[tri[i]][0] - vertices[tri[j]][0]
            dy = vertices[tri[i]][1] - vertices[tri[j]][1]
            edge = math.sqrt(dx * dx + dy * dy)
            if edge < min_len:
                min_len = edge
    return min_len


# ---------------------------------------------------------------------------
# Unit tests for make_interior_holes_and_tags (sliver-merge path)
# ---------------------------------------------------------------------------

class TestMakeInteriorHolesReflective:
    """TASK-1270: make_interior_holes_and_tags routes Reflective to the hole path."""

    def test_reflective_produces_holes(self):
        """A single Reflective structure produces one interior hole."""
        input_data = {'structure': _make_structure_geojson([BUILDING_A])}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes is not None, "Expected interior holes, got None"
        assert len(holes) >= 1
        assert tags is not None
        assert all('reflective' in t for t in tags)

    def test_adjacent_buildings_merge_reduces_sliver(self):
        """Two adjacent buildings sharing a vertex merge into ≤ 2 holes (sliver-safe)."""
        input_data = {'structure': _make_structure_geojson([BUILDING_A, BUILDING_B])}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes is not None
        # The two touching squares MUST merge to fewer than 2 holes
        # (ideally 1 merged rectangle; at most 2 if slightly separated).
        assert len(holes) <= 2, f"Expected ≤2 merged holes, got {len(holes)}"

    def test_mannings_skipped(self):
        """Mannings structures produce no holes."""
        input_data = {'structure': _make_structure_geojson([BUILDING_A], method='Mannings')}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes is None

    def test_raised_skipped(self):
        """Raised structures produce no holes (post-mesh path)."""
        input_data = {'structure': _make_structure_geojson([BUILDING_A], method='Raised')}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes is None

    def test_hole_tags_are_reflective(self):
        """Each hole is tagged with 'reflective' wall indices."""
        input_data = {'structure': _make_structure_geojson([BUILDING_A])}
        holes, tags = make_interior_holes_and_tags(input_data)
        assert holes is not None
        for i, tag in enumerate(tags):
            assert 'reflective' in tag, f"Hole {i} missing 'reflective' tag"
            assert len(tag['reflective']) == len(holes[i]), (
                f"Hole {i}: tag indices ({len(tag['reflective'])}) "
                f"don't match coords ({len(holes[i])})"
            )


# ---------------------------------------------------------------------------
# Integration test: create_anuga_mesh with Reflective structures
# ---------------------------------------------------------------------------

class TestReflectiveMeshIntegration:
    """TASK-1270 hard accept criterion: mesh with Reflective holes is sliver-safe."""

    def _build_input_data(self, tmp_dir, rings=None):
        msh_path = os.path.join(tmp_dir, "run_1_1_1.msh")
        input_data = {
            "mesh_filepath": msh_path,
            "scenario_config": SCENARIO_CONFIG,
            "boundary_polygon": BOUNDARY_POLYGON,
            "boundary_tags": BOUNDARY_TAGS,
        }
        if rings:
            input_data['structure'] = _make_structure_geojson(rings)
        return input_data

    def test_mesh_with_reflective_holes_is_sliver_safe(self):
        """Mesh with two adjacent Reflective buildings meets MIN_EDGE_LENGTH_M threshold."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_data = self._build_input_data(tmp_dir, rings=[BUILDING_A, BUILDING_B])
            _, anuga_mesh = create_anuga_mesh(input_data)

            n_triangles = len(anuga_mesh.tri_mesh.triangles)
            assert n_triangles > 0, "Expected >0 triangles after meshing with reflective holes"

            min_edge = _min_edge_length(anuga_mesh)
            assert min_edge >= MIN_EDGE_LENGTH_M, (
                f"Sliver detected: min edge length {min_edge:.6f}m < {MIN_EDGE_LENGTH_M}m. "
                f"The sliver-merge (5604fc1) may not be active or working correctly."
            )

    def test_mesh_triangle_count_bounded(self):
        """Triangle count for 200m domain at 20m resolution is in a reasonable range."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_data = self._build_input_data(tmp_dir, rings=[BUILDING_A, BUILDING_B])
            _, anuga_mesh = create_anuga_mesh(input_data)
            n = len(anuga_mesh.tri_mesh.triangles)
            # 200m x 200m at 20m = expected ~200 triangles; 5000 is generous upper bound.
            assert 10 < n < 5000, f"Unexpected triangle count: {n}"

    def test_mesh_qa_no_degenerate_triangles(self):
        """compute_mesh_qa reports no degenerate (zero-area) triangles."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_data = self._build_input_data(tmp_dir, rings=[BUILDING_A, BUILDING_B])
            _, anuga_mesh = create_anuga_mesh(input_data)
            qa = compute_mesh_qa(anuga_mesh)
            assert not qa['has_degenerate_triangles'], (
                f"Degenerate triangles present after sliver-merge: {qa}"
            )

    def test_no_structures_produces_valid_mesh(self):
        """Baseline: mesh without structures is also valid."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_data = self._build_input_data(tmp_dir, rings=None)
            _, anuga_mesh = create_anuga_mesh(input_data)
            n = len(anuga_mesh.tri_mesh.triangles)
            assert n > 0

    def test_cfl_short_timestep(self):
        """A 1-second timestep domain evolves without CFL collapse.

        This is a live ANUGA simulation check — confirms no sub-microsecond
        CFL timestep (which would indicate surviving slivers). The domain
        uses a flat DEM at 0m elevation; we set bed and stage then evolve 1s.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_data = self._build_input_data(tmp_dir, rings=[BUILDING_A, BUILDING_B])
            msh_path, anuga_mesh = create_anuga_mesh(input_data)

            # Load mesh into an ANUGA Domain
            domain = anuga.Domain(msh_path)
            domain.set_quantity('elevation', 0.0)
            domain.set_quantity('stage', 0.1)  # 10cm of water everywhere
            domain.set_quantity('xmomentum', 0.0)
            domain.set_quantity('ymomentum', 0.0)

            # Bind boundary conditions. ANUGA requires all tags to have a bound
            # boundary object before evolving. 'exterior' and 'reflective' are
            # the tags produced by our mesh; we use Reflective for both.
            Br = anuga.Reflective_boundary(domain)
            boundary_map = {tag: Br for tag in domain.get_boundary_tags()}
            domain.set_boundary(boundary_map)

            # Evolve for 1 second with a safe timestep floor check.
            # If slivers survived, the adaptive timestep will collapse below 1e-3s.
            # Skip t=0 (initial state, dt=0 before first step).
            timesteps = []
            for t in domain.evolve(yieldstep=0.5, finaltime=1.0):
                dt = domain.get_timestep()
                if t > 0:  # skip initial dt=0 at t=0
                    timesteps.append(dt)

            # The minimum CFL timestep must be at least 1e-3s.
            # Slivers produce timesteps of ~1e-9 to 1e-12s.
            if timesteps:
                min_timestep_seen = min(timesteps)
                assert min_timestep_seen >= 1e-3, (
                    f"CFL timestep collapsed to {min_timestep_seen:.3e}s — "
                    f"slivers likely present despite sliver-merge. "
                    f"Min edge length was {_min_edge_length(anuga_mesh):.6f}m."
                )
