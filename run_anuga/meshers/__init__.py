"""External mesher dispatch for ANUGA simulations.

Provides a common interface for alternative mesh generators (Gmsh, JIGSAW)
that return mesh data compatible with ``anuga.Domain(points, triangles, boundary=...)``.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class MeshResult:
    """Mesh data ready for ANUGA Domain construction."""
    points: np.ndarray       # (N, 2) vertex coordinates
    triangles: np.ndarray    # (M, 3) vertex indices (0-based)
    elevation: np.ndarray    # (N,) elevation at each vertex
    boundary_tags: dict      # {(vol_id, edge_id): tag_string}


def get_mesher(name):
    """Return the generate_mesh function for the named mesher."""
    if name == "gmsh":
        from .gmsh_mesher import generate_mesh
    elif name == "jigsaw":
        from .jigsaw_mesher import generate_mesh
    else:
        raise ValueError(f"Unknown mesher: {name}")
    return generate_mesh
