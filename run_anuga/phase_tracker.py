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

TASK-1954 (epic 1952): adds per-phase DURATION tracking as the symmetric sibling
of the per-phase peak-memory attribution. Each :func:`set_phase` call accumulates
``time.perf_counter()`` elapsed into the PREVIOUS phase's bucket, giving
:func:`get_phase_durations` a deterministic (non-sampling, always non-zero for any
real phase) view of how long each phase took. Rollup taxonomy:

    build   = mesh-gen + raster-read + partition + distribute
    solve   = evolve
    publish = cog-export + archive

Design points:

* **Django-free + dependency-free.** Pure stdlib (``threading``, ``time``).
  run_anuga must import this without Django (it runs inside the Batch container
  AND a localhost celery worker).
* **In-process, rank-0.** The sampler runs in-process on rank 0; a module-level
  variable guarded by a lock is read by the sampler thread and written by the
  main thread. No marker file / no IPC needed (cross-backend: identical on AWS
  Batch and a localhost run).
* **Best-effort signal.** A very-fast phase that gets no ~5 s sample legitimately
  reports nothing for that phase in ``phase_peaks_mib`` — that is acceptable;
  the mechanism is what matters. ``phase_durations_s`` is DETERMINISTIC (perf_counter
  resolution ~1 µs) so even a sub-second phase yields a non-zero value.
"""
from __future__ import annotations

import contextlib
import threading
import time
from typing import Dict, Iterator, Optional

# The five Domain-build sub-phases attributed by the sampler (TASK-1907 W1).
PHASE_MESH_GEN = "mesh-gen"
PHASE_PARTITION = "partition"
PHASE_RASTER_READ = "raster-read"
PHASE_DISTRIBUTE = "distribute"
PHASE_EVOLVE = "evolve"

# Post-sim publish phases (TASK-1954): cog-export (post_process_sww in run.py)
# and archive (cold-archive + zip/upload in _handoff.run_and_report).
PHASE_COG_EXPORT = "cog-export"
PHASE_ARCHIVE = "archive"

# Stable order for reporting; also the membership set the sampler validates
# for peak-memory attribution (BUILD_PHASES + EVOLVE stay as-is for back-compat).
BUILD_PHASES = (
    PHASE_MESH_GEN,
    PHASE_PARTITION,
    PHASE_RASTER_READ,
    PHASE_DISTRIBUTE,
    PHASE_EVOLVE,
)

# All phases in canonical order (build phases + publish phases).
# Used for rollup derivation: build = 0:4, solve = 4, publish = 5:7.
ALL_PHASES = BUILD_PHASES + (PHASE_COG_EXPORT, PHASE_ARCHIVE)

_lock = threading.Lock()
_current_phase: Optional[str] = None
_mesh_features: Dict[str, object] = {}

# Per-phase duration accumulators (TASK-1954).
# _phase_start_time is perf_counter at the moment _current_phase was last set
# to a non-None value; None means no phase is active / duration not started.
_phase_durations: Dict[str, float] = {}
_phase_start_time: Optional[float] = None


def set_phase(phase: Optional[str]) -> None:
    """Set the currently-active build phase (read by the sampler thread).

    Also accumulates the elapsed ``time.perf_counter()`` duration for the phase
    that is being LEFT into :data:`_phase_durations`, giving
    :func:`get_phase_durations` a deterministic per-phase timing view (TASK-1954).

    ``phase`` is normally one of :data:`BUILD_PHASES` or the publish-phase
    constants (:data:`PHASE_COG_EXPORT`, :data:`PHASE_ARCHIVE`); ``None`` clears
    the tag so samples taken outside a known phase are NOT attributed to any phase
    and no duration is accumulated.
    """
    global _current_phase, _phase_start_time
    with _lock:
        now = time.perf_counter()
        # Accumulate the duration for the phase we are leaving.
        if _current_phase is not None and _phase_start_time is not None:
            elapsed = now - _phase_start_time
            _phase_durations[_current_phase] = (
                _phase_durations.get(_current_phase, 0.0) + elapsed
            )
        _current_phase = phase
        _phase_start_time = now if phase is not None else None


def get_phase() -> Optional[str]:
    """Return the currently-active build phase (or ``None``).

    This is the callable handed to the sampler as its ``phase_provider``.
    """
    with _lock:
        return _current_phase


def get_phase_durations() -> Dict[str, float]:
    """Return a snapshot of per-phase elapsed durations in seconds (TASK-1954).

    Values are accumulated by :func:`set_phase` via ``time.perf_counter()``
    at each phase transition, so they are deterministic and non-zero for any
    phase that had real work. The currently-active phase's PARTIAL duration
    (time since :func:`set_phase` last set it) is included in the snapshot so
    a mid-run call still yields useful information.

    This is the callable handed to the sampler as its ``phase_durations_provider``.
    The sampler calls it lazily when :meth:`~ResourceSampler.summary` is invoked
    so that post-sim phases (cog-export, archive) are captured if the summary is
    taken after they complete.
    """
    with _lock:
        result = dict(_phase_durations)
        # Include partial duration for the currently-active phase.
        if _current_phase is not None and _phase_start_time is not None:
            partial = time.perf_counter() - _phase_start_time
            result[_current_phase] = result.get(_current_phase, 0.0) + partial
        return result


@contextlib.contextmanager
def phase(name: Optional[str]) -> Iterator[None]:
    """Context manager that sets ``name`` for its body then restores the prior phase.

    Restores (not blanks) on exit — nested phases pop back to their parent — and
    restores even when the body raises.  The duration of ``name`` is accumulated
    by the :func:`set_phase` call on exit.
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
    """Clear the phase, duration accumulators, and the mesh-feature bag.

    Called at the start of each :func:`~run_anuga.run.run_sim` (test hygiene /
    new run on a long-lived celery worker).
    """
    global _current_phase, _phase_start_time
    with _lock:
        _current_phase = None
        _phase_start_time = None
        _mesh_features.clear()
        _phase_durations.clear()
