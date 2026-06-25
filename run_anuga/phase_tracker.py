"""Thread-safe build-phase tracker for sub-phase memory attribution (TASK-1910).

The cgroup ``memory.peak`` the resource sampler reads is a MONOTONIC whole-run
high-water mark — a per-phase peak cannot be read from it directly. Instead the
sampler thread TAGS each periodic RSS sample with the build phase that is active
at sample time, and the per-phase peak is ``max(samples tagged to that phase)``.

This module is the phase SIGNAL: ``run.py`` calls :func:`set_phase` (or the
:func:`phase` context manager) at each Domain-build boundary, and the sampler
reads the active phase via the ``phase_provider`` callable it is handed
(``get_phase``). The mesh-size features (``mesh_triangle_count`` …) ride alongside
via :func:`set_mesh_features` / :func:`get_mesh_features`, so the corpus joins
peak memory to mesh size + outcome (this absorbs TASK-1808-P1).

Design points:

* **Django-free + dependency-free.** Pure stdlib (``threading``). run_anuga must
  import this without Django (it runs inside the Batch container AND a localhost
  celery worker).
* **In-process, rank-0.** The sampler runs in-process on rank 0; a module-level
  variable guarded by a lock is read by the sampler thread and written by the
  main thread. No marker file / no IPC needed (cross-backend: identical on AWS
  Batch and a localhost run).
* **Best-effort signal.** A very-fast phase that gets no ~5 s sample legitimately
  reports nothing for that phase — that is acceptable; the mechanism is what
  matters, a real heavy run fills the dominant phases.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Dict, Iterator, Optional

# The five Domain-build sub-phases attributed by the sampler (TASK-1907 W1).
PHASE_MESH_GEN = "mesh-gen"
PHASE_PARTITION = "partition"
PHASE_RASTER_READ = "raster-read"
PHASE_DISTRIBUTE = "distribute"
PHASE_EVOLVE = "evolve"

# Stable order for reporting; also the membership set the sampler validates.
BUILD_PHASES = (
    PHASE_MESH_GEN,
    PHASE_PARTITION,
    PHASE_RASTER_READ,
    PHASE_DISTRIBUTE,
    PHASE_EVOLVE,
)

_lock = threading.Lock()
_current_phase: Optional[str] = None
_mesh_features: Dict[str, object] = {}


def set_phase(phase: Optional[str]) -> None:
    """Set the currently-active build phase (read by the sampler thread).

    ``phase`` is normally one of :data:`BUILD_PHASES`; ``None`` clears the tag so
    samples taken outside a known phase are NOT attributed to any phase.
    """
    global _current_phase
    with _lock:
        _current_phase = phase


def get_phase() -> Optional[str]:
    """Return the currently-active build phase (or ``None``).

    This is the callable handed to the sampler as its ``phase_provider``.
    """
    with _lock:
        return _current_phase


@contextlib.contextmanager
def phase(name: Optional[str]) -> Iterator[None]:
    """Context manager that sets ``name`` for its body then restores the prior phase.

    Restores (not blanks) on exit — nested phases pop back to their parent — and
    restores even when the body raises.
    """
    with _lock:
        previous = _current_phase
    set_phase(name)
    try:
        yield
    finally:
        set_phase(previous)


def set_mesh_features(**features: object) -> None:
    """Merge mesh-size features (e.g. ``mesh_triangle_count=8_160_000``) into the bag.

    Accumulative: later calls add/overwrite keys without dropping earlier ones.
    The sampler reads the merged bag via :func:`get_mesh_features`.
    """
    with _lock:
        _mesh_features.update(features)


def get_mesh_features() -> Dict[str, object]:
    """Return a shallow copy of the accumulated mesh-feature bag.

    This is the callable handed to the sampler as its ``mesh_features_provider``.
    """
    with _lock:
        return dict(_mesh_features)


def reset() -> None:
    """Clear the phase and the mesh-feature bag (test hygiene / new run)."""
    global _current_phase
    with _lock:
        _current_phase = None
        _mesh_features.clear()
