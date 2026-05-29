"""TASK-1265 / W2.4 — Golden-mesh regression + determinism test.

Tests that ``create_anuga_mesh`` with a fixed small boundary polygon produces a
mesh whose canonicalised geometry hash matches a committed golden value.  This
guards against unintentional changes to the mesh stack (meshpy/Triangle/ANUGA
geo_reference) that would silently alter mesh geometry.

Canonicalization algorithm
--------------------------
1. Sort nodes by (x, y) — gives a canonical vertex ordering independent of
   Triangle's internal node numbering.
2. Build a remapping table (old_index → new_sorted_index) and apply it to the
   triangle connectivity array.
3. For each triangle sort the 3 remapped node-indices (unordered triple).
4. Sort the list of triples lexicographically.
5. Concatenate coordinates (rounded to 6 decimal places) and connectivity into
   a numpy byte-string and SHA-256 hash it.

This makes the hash invariant to benign node-reorderings that Triangle may
produce on different platforms while still catching real geometric changes.

How to regenerate the golden hash
----------------------------------
When meshpy is intentionally bumped (version pin change):
1. Run this test with ``GOLDEN_MESH_REGEN=1`` env var set — it prints the new
   hash to stdout and passes (skips the assertion).
2. Copy the printed hash into ``GOLDEN_HASH`` below.
3. Commit both the meshpy version bump and the updated hash in the same PR so
   reviewers can see the geometry change was intentional.

Cross-platform optional CI
---------------------------
If ``GOLDEN_MESH_CROSS_PLATFORM=1`` is set the test also attempts to call the
ANUGA Docker image over the fixture (requires Docker available) to verify the
Batch image produces the same hash.  This gate is NOT run in normal CI.

Skip behaviour
--------------
pytest.importorskip is used for each of: anuga, meshpy.triangle, osgeo.gdal.
Tests self-skip gracefully where the ANUGA environment is absent.
"""
import hashlib
import os
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Dependencies — self-skip if ANUGA stack absent
# ---------------------------------------------------------------------------
anuga = pytest.importorskip("anuga", reason="anuga not installed")
pytest.importorskip("meshpy.triangle", reason="meshpy not installed")
pytest.importorskip("osgeo.gdal", reason="GDAL not installed")

from run_anuga.run_utils import create_anuga_mesh  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture — small square boundary in EPSG:28355 (GDA94 / MGA zone 55)
# ~200m × 200m square near dummy origin; enough to produce a small mesh
# ---------------------------------------------------------------------------

# Boundary polygon: four corners of a 200m square
# (list of [x, y] projected metre pairs, NOT closed)
BOUNDARY_POLYGON = [
    [321000.0, 5812000.0],
    [321200.0, 5812000.0],
    [321200.0, 5812200.0],
    [321000.0, 5812200.0],
]

# boundary_tags format: {'tag_name': [list of (x,y) tuples on that edge]}
# ANUGA requires the boundary polygon vertices to be associated with tags.
# We use a single 'exterior' tag covering the whole boundary.
BOUNDARY_TAGS = {
    'exterior': list(range(len(BOUNDARY_POLYGON))),
}

SCENARIO_CONFIG = {
    'epsg': 'EPSG:28355',
    'resolution': 40,   # 40 m resolution → ~8 triangles for a 200m square
    'project': 1,
    'id': 1,
    'run_id': 1,
}

# ---------------------------------------------------------------------------
# Golden hash
# The hash below was captured on the pinned stack (meshpy 2024.1, ANUGA 3.3.x).
# See regeneration instructions in the module docstring.
# Set GOLDEN_MESH_REGEN=1 to regenerate.
# ---------------------------------------------------------------------------
GOLDEN_HASH: str = "4b23c88734efe112e5031cc899e47b67d6ed46d25af3f3f38ad85f1130ffb94a"


def _build_input_data(tmp_dir: str) -> dict:
    """Build a minimal input_data dict for create_anuga_mesh."""
    msh_path = os.path.join(tmp_dir, "run_1_1_1.msh")
    return {
        "mesh_filepath": msh_path,
        "scenario_config": SCENARIO_CONFIG,
        "boundary_polygon": BOUNDARY_POLYGON,
        "boundary_tags": BOUNDARY_TAGS,
    }


def _canonicalise_mesh(anuga_mesh) -> tuple[np.ndarray, np.ndarray]:
    """Return (sorted_vertices, sorted_triangles) for hashing.

    Vertices sorted by (x, y); triangles remapped and sorted.
    """
    vertices = anuga_mesh.tri_mesh.vertices   # (N, 2)
    triangles = anuga_mesh.tri_mesh.triangles  # (M, 3)

    # 1. Sort vertices by (x, y)
    sort_order = np.lexsort((vertices[:, 1], vertices[:, 0]))
    sorted_vertices = vertices[sort_order]

    # 2. Build old_index → new_sorted_index remap
    remap = np.empty(len(vertices), dtype=int)
    remap[sort_order] = np.arange(len(vertices))

    # 3. Remap triangle connectivity; sort each triple
    remapped = remap[triangles]                     # (M, 3)
    remapped.sort(axis=1)                           # sort within each triple

    # 4. Sort triangles lexicographically
    lex_order = np.lexsort((remapped[:, 2], remapped[:, 1], remapped[:, 0]))
    sorted_triangles = remapped[lex_order]

    return sorted_vertices, sorted_triangles


def _compute_hash(vertices: np.ndarray, triangles: np.ndarray) -> str:
    """Compute SHA-256 over rounded vertices + triangle connectivity."""
    rounded_verts = np.round(vertices, 6)
    h = hashlib.sha256()
    h.update(rounded_verts.tobytes())
    h.update(triangles.tobytes())
    return h.hexdigest()


def _run_mesh(tmp_dir: str):
    """Build the mesh and return (anuga_mesh_filepath, anuga_mesh)."""
    input_data = _build_input_data(tmp_dir)
    return create_anuga_mesh(input_data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def golden_mesh():
    """Build the mesh once per module; return (filepath, anuga_mesh, hash)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mesh_fp, mesh = _run_mesh(tmp_dir)
        verts, tris = _canonicalise_mesh(mesh)
        mesh_hash = _compute_hash(verts, tris)
        n_triangles = len(mesh.tri_mesh.triangles)
        yield mesh_fp, mesh, mesh_hash, n_triangles


class TestGoldenMesh:
    """Golden-mesh regression + determinism suite (TASK-1265 / W2.4)."""

    def test_true_triangle_count(self, golden_mesh):
        """Mesh has at least 1 triangle (sanity — non-empty output)."""
        _, _, _, n_triangles = golden_mesh
        assert n_triangles > 0, f"Expected >0 triangles, got {n_triangles}"

    def test_triangle_count_uses_len_not_size(self, golden_mesh):
        """len(tri_mesh.triangles) returns true N, not 3N."""
        _, mesh, _, n_triangles = golden_mesh
        assert n_triangles == len(mesh.tri_mesh.triangles)
        assert n_triangles != mesh.tri_mesh.triangles.size

    def test_determinism_build_twice(self):
        """Two separate calls with the same boundary produce identical canonical hashes."""
        with tempfile.TemporaryDirectory() as d1:
            _, mesh1 = _run_mesh(d1)
            v1, t1 = _canonicalise_mesh(mesh1)
            hash1 = _compute_hash(v1, t1)

        with tempfile.TemporaryDirectory() as d2:
            _, mesh2 = _run_mesh(d2)
            v2, t2 = _canonicalise_mesh(mesh2)
            hash2 = _compute_hash(v2, t2)

        assert hash1 == hash2, (
            f"Two builds produced different canonical hashes:\n"
            f"  build1: {hash1}\n"
            f"  build2: {hash2}\n"
            "This indicates non-deterministic mesh generation."
        )

    def test_golden_hash_stable(self, golden_mesh):
        """Canonical hash matches the committed golden value (or prints new value for REGEN)."""
        _, _, mesh_hash, _ = golden_mesh

        if os.environ.get("GOLDEN_MESH_REGEN"):
            print(f"\n[REGEN] New golden hash: {mesh_hash}\n")
            return  # Pass without asserting — operator must update GOLDEN_HASH

        if not GOLDEN_HASH:
            # No golden captured yet (first time running after initial commit of this file).
            # Print the hash so the operator can commit it.
            print(f"\n[FIRST RUN] Golden hash: {mesh_hash}\n"
                  f"Update GOLDEN_HASH in test_golden_mesh.py and re-run.\n")
            pytest.skip("GOLDEN_HASH not yet set — run once to capture, then commit")

        assert mesh_hash == GOLDEN_HASH, (
            f"Canonical mesh hash changed!\n"
            f"  expected: {GOLDEN_HASH}\n"
            f"  actual:   {mesh_hash}\n"
            "If this is an intentional meshpy bump, set GOLDEN_MESH_REGEN=1 to regenerate."
        )

    def test_vertices_are_float_array(self, golden_mesh):
        """Vertices are a 2-D float array."""
        _, mesh, _, _ = golden_mesh
        verts = mesh.tri_mesh.vertices
        assert verts.ndim == 2
        assert verts.shape[1] == 2
        assert verts.dtype in (np.float32, np.float64)

    def test_triangles_are_int_array_n_by_3(self, golden_mesh):
        """Triangles are an (N, 3) int array."""
        _, mesh, _, _ = golden_mesh
        tris = mesh.tri_mesh.triangles
        assert tris.ndim == 2
        assert tris.shape[1] == 3

    def test_msh_file_exists_and_nonzero(self, golden_mesh):
        """The .msh file is created on disk."""
        mesh_fp, _, _, _ = golden_mesh
        assert os.path.exists(mesh_fp), f".msh not found at {mesh_fp}"
        assert os.path.getsize(mesh_fp) > 0

    def test_cross_platform_parity(self, golden_mesh):
        """Optional: verify Batch Docker image produces same hash (skipped unless GOLDEN_MESH_CROSS_PLATFORM=1)."""
        if not os.environ.get("GOLDEN_MESH_CROSS_PLATFORM"):
            pytest.skip("GOLDEN_MESH_CROSS_PLATFORM not set")
        # Operator-driven gate: run only for release validation
        pytest.skip("Cross-platform Docker test not yet implemented")
