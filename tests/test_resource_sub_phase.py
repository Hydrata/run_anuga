"""TASK-1910 (epic 1907 W1/Track 0) — sub-phase Domain-build memory attribution.

The existing resource sampler reads cgroup ``memory.peak`` as ONE whole-run
high-water mark. cgroup ``memory.peak`` is monotonic, so a per-PHASE peak cannot
be read directly — instead each periodic RSS sample is TAGGED with the
currently-active build phase, and the per-phase peak is ``max(samples tagged to
that phase)``.

This test pins:

1. The Django-free ``run_anuga.phase_tracker`` thread-safe current-phase holder
   (set/get/reset + a ``phase()`` context manager + a mesh-feature bag).
2. The sampler's ``phase_provider`` injection seam: when a provider callable is
   injected, each sample's peak is attributed to the phase the provider reports,
   and ``summary()['observed']['phase_peaks_mib']`` carries the per-phase peaks.
3. The mesh-size feature (``mesh_triangle_count``) rides into the summary's
   ``features`` bag via the injected ``mesh_features_provider``.
4. Back-compat: a sampler with NO provider behaves exactly as before (no phase
   tagging, empty ``phase_peaks_mib``) — the tool-agnostic contract for
   terrain-merge / IDF is preserved.

The sampler lives in ``gn_anuga.batch_common`` which is NOT on the run_anuga
test path (run_anuga is Django-free). The sampler tests that need it import it
under a skip-guard; the phase_tracker tests are pure run_anuga.
"""
from __future__ import annotations

import threading
import time

import pytest

from run_anuga import phase_tracker


@pytest.fixture(autouse=True)
def _reset_phase_tracker():
    """Each test starts and ends with a clean module-level phase tracker."""
    phase_tracker.reset()
    yield
    phase_tracker.reset()


# ── phase_tracker (pure run_anuga, Django-free) ─────────────────────────────

def test_phase_tracker_defaults_to_none():
    assert phase_tracker.get_phase() is None


def test_phase_tracker_set_get():
    phase_tracker.set_phase(phase_tracker.PHASE_MESH_GEN)
    assert phase_tracker.get_phase() == "mesh-gen"
    phase_tracker.set_phase(phase_tracker.PHASE_RASTER_READ)
    assert phase_tracker.get_phase() == "raster-read"


def test_phase_tracker_reset_clears_phase_and_features():
    phase_tracker.set_phase(phase_tracker.PHASE_EVOLVE)
    phase_tracker.set_mesh_features(mesh_triangle_count=123)
    phase_tracker.reset()
    assert phase_tracker.get_phase() is None
    assert phase_tracker.get_mesh_features() == {}


def test_phase_constants_are_the_five_build_phases():
    expected = {"mesh-gen", "partition", "raster-read", "distribute", "evolve"}
    assert set(phase_tracker.BUILD_PHASES) == expected


def test_phase_context_manager_sets_and_restores():
    phase_tracker.set_phase(phase_tracker.PHASE_MESH_GEN)
    with phase_tracker.phase(phase_tracker.PHASE_RASTER_READ):
        assert phase_tracker.get_phase() == "raster-read"
    # restored to the previous phase, not blanked
    assert phase_tracker.get_phase() == "mesh-gen"


def test_phase_context_manager_restores_on_exception():
    with phase_tracker.phase(phase_tracker.PHASE_DISTRIBUTE):
        try:
            with phase_tracker.phase(phase_tracker.PHASE_EVOLVE):
                raise ValueError("boom")
        except ValueError:
            pass
        assert phase_tracker.get_phase() == "distribute"


def test_phase_tracker_thread_safe_set():
    """Concurrent set_phase calls must not corrupt the holder (no torn state)."""
    errors = []

    def worker(name):
        try:
            for _ in range(200):
                phase_tracker.set_phase(name)
                got = phase_tracker.get_phase()
                assert got in phase_tracker.BUILD_PHASES
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(p,))
        for p in phase_tracker.BUILD_PHASES
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_mesh_features_accumulate():
    phase_tracker.set_mesh_features(mesh_triangle_count=42)
    phase_tracker.set_mesh_features(mesh_node_count=99)
    feats = phase_tracker.get_mesh_features()
    assert feats["mesh_triangle_count"] == 42
    assert feats["mesh_node_count"] == 99


# ── sampler phase-tagging (needs gn_anuga.batch_common) ─────────────────────

def _import_sampler():
    try:
        from gn_anuga.batch_common.resource_sampler import ResourceSampler
        return ResourceSampler
    except Exception:
        pytest.skip("gn_anuga.batch_common not on path (Django-free run_anuga env)")


def test_sampler_with_no_provider_has_empty_phase_peaks(tmp_path):
    """Back-compat: no phase_provider → no phase tagging, empty phase_peaks_mib."""
    ResourceSampler = _import_sampler()
    with ResourceSampler(tmp_path, tool="terrain-merge", job_id="j",
                         interval_s=0.02) as s:
        time.sleep(0.05)
    summary = s.summary()
    assert summary["observed"].get("phase_peaks_mib") == {}


def test_sampler_tags_samples_by_active_phase(tmp_path):
    """Each sample's peak is attributed to the phase the provider reports.

    Drive the provider through two phases while the sampler thread runs; both
    phases must appear in phase_peaks_mib with a non-decreasing (monotonic
    high-water) peak. On a non-cgroup box memory.peak may be None for every
    sample — in that case phase_peaks_mib values are None, which is acceptable
    (the MECHANISM — a key per visited phase — is what we assert).
    """
    ResourceSampler = _import_sampler()
    current = {"phase": None}

    def provider():
        return current["phase"]

    s = ResourceSampler(
        tmp_path, tool="anuga", job_id="j", interval_s=0.02,
        phase_provider=provider,
    )
    with s:
        current["phase"] = "mesh-gen"
        time.sleep(0.1)
        current["phase"] = "raster-read"
        time.sleep(0.1)

    summary = s.summary()
    peaks = summary["observed"]["phase_peaks_mib"]
    # both visited phases recorded a key
    assert "mesh-gen" in peaks
    assert "raster-read" in peaks
    # None-phase samples must NOT pollute the per-phase map
    assert None not in peaks
    assert "None" not in peaks


def test_sampler_phase_peak_is_max_of_phase_samples(tmp_path):
    """Per-phase peak = max of samples tagged to that phase (not whole-run)."""
    ResourceSampler = _import_sampler()
    # Feed synthetic samples directly via the internal accumulator to avoid
    # depending on real cgroup growth timing.
    s = ResourceSampler(tmp_path, tool="anuga", job_id="j", interval_s=999)
    mib = 1024 * 1024
    s._record_phase_peak("mesh-gen", 100 * mib)
    s._record_phase_peak("mesh-gen", 250 * mib)   # higher → wins
    s._record_phase_peak("mesh-gen", 180 * mib)   # lower → ignored
    s._record_phase_peak("raster-read", 700 * mib)
    s._record_phase_peak(None, 9999 * mib)        # untagged → dropped
    peaks = s.summary()["observed"]["phase_peaks_mib"]
    assert peaks["mesh-gen"] == 250
    assert peaks["raster-read"] == 700
    assert None not in peaks


def test_sampler_carries_mesh_features(tmp_path):
    """mesh_triangle_count (+ friends) ride into summary['features']."""
    ResourceSampler = _import_sampler()

    def mesh_features():
        return {"mesh_triangle_count": 8_160_000, "mesh_node_count": 4_090_000}

    s = ResourceSampler(
        tmp_path, tool="anuga", job_id="j", interval_s=999,
        mesh_features_provider=mesh_features,
    )
    with s:
        pass
    summary = s.summary()
    assert summary["features"]["mesh_triangle_count"] == 8_160_000
    assert summary["features"]["mesh_node_count"] == 4_090_000


def test_sampler_features_empty_without_provider(tmp_path):
    ResourceSampler = _import_sampler()
    s = ResourceSampler(tmp_path, tool="terrain-merge", job_id="j", interval_s=999)
    with s:
        pass
    assert s.summary()["features"] == {}


def test_handoff_injects_phase_and_mesh_providers(tmp_path, monkeypatch):
    """_make_resource_sampler wires run_anuga.phase_tracker into the sampler.

    The sampler's phase_provider must reflect run_anuga.phase_tracker.get_phase
    and the mesh_features_provider must reflect get_mesh_features — so a real
    run that calls set_phase()/set_mesh_features() lands per-phase peaks +
    features in the summary.
    """
    _import_sampler()  # skip if batch_common absent
    from run_anuga import _handoff

    sampler = _handoff._make_resource_sampler(
        str(tmp_path),
        control_server="https://hydrata.com",
        ids={"run_id": 1, "project_id": 2, "scenario_id": 3},
    )
    assert sampler is not None, "sampler should construct when batch_common present"

    phase_tracker.set_phase(phase_tracker.PHASE_MESH_GEN)
    phase_tracker.set_mesh_features(mesh_triangle_count=555)
    assert sampler._phase_provider() == "mesh-gen"
    assert sampler._mesh_features_provider()["mesh_triangle_count"] == 555
