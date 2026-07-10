"""TASK-2197 (epic 2190 W4.1) — GPU compute-target mode plumbing.

Three concerns, each a pure/getattr-defensive helper extracted from
``run_sim`` so they are unit-testable WITHOUT a real ANUGA ``Domain`` (no MPI,
no mesh, no GPU hardware — this box has none of those):

1. ``_resolve_multiprocessor_mode`` — ``RUN_ANUGA_MULTIPROCESSOR_MODE`` env
   OVERRIDES ``scenario_config['multiprocessor_mode']`` (the dispatcher,
   ``gn_anuga.services.resolve_target_dispatch``, already injects this env for
   the ``batch-gpu-a10g`` target so the scenario PACKAGE itself stays
   immutable). Preserves the TASK-1954 falsy-mode->1 coercion (5d5328f) on
   BOTH sources.

2. ``_assert_gpu_engaged`` — a mode-2 (GPU) request that did not actually
   engage GPU offload must FAIL the run, not silently continue in mode 1
   (today's ``set_multiprocessor_mode`` GPU-unavailable path is
   warn-and-continue). getattr-defensive across engine variants: prefers a
   ``gpu_offload_enabled()`` probe if the engine exposes one, falls back to
   checking whether ``multiprocessor_mode`` actually stuck at 2, and FAILS
   CLOSED when neither signal is available (mode 2 was requested; "cannot
   prove engagement" must not read as "engaged"). No-op when mode != 2 (a CPU
   image has no GPU probe at all — this must never be invoked there).

3. GPU feature capture — ``_capture_gpu_model`` (best-effort NVML / nvidia-smi,
   never raises) and the ``phase_tracker.set_mesh_features(mode=..., gpu_model=...)``
   call that rides the (already-shipped) ``features`` bag into
   ``BatchJobResourceRecord.raw`` — ``AdminRunResourceRecordSerializer.get_mode``
   / ``get_gpu_model`` already read exactly this location (TASK-2195 predates
   this task; the WRITE side was the gap).
"""
from __future__ import annotations

import logging
import os
import sys
import types
from unittest import mock

import pytest

from run_anuga import phase_tracker
from run_anuga.run import (
    _MULTIPROCESSOR_GPU,
    _MULTIPROCESSOR_OPENMP,
    _assert_gpu_engaged,
    _capture_gpu_model,
    _resolve_multiprocessor_mode,
)


@pytest.fixture(autouse=True)
def _reset_tracker():
    phase_tracker.reset()
    yield
    phase_tracker.reset()


def _input_data(multiprocessor_mode=None):
    scenario_config = {"duration": 10, "run_id": 1}
    if multiprocessor_mode is not None:
        scenario_config["multiprocessor_mode"] = multiprocessor_mode
    return {"scenario_config": scenario_config}


# ---------------------------------------------------------------------------
# 1. _resolve_multiprocessor_mode — env beats scenario.json
# ---------------------------------------------------------------------------

def test_env_override_beats_scenario_config(monkeypatch):
    monkeypatch.setenv("RUN_ANUGA_MULTIPROCESSOR_MODE", "2")
    assert _resolve_multiprocessor_mode(_input_data(multiprocessor_mode=1)) == 2


def test_scenario_config_used_when_env_absent(monkeypatch):
    monkeypatch.delenv("RUN_ANUGA_MULTIPROCESSOR_MODE", raising=False)
    assert _resolve_multiprocessor_mode(_input_data(multiprocessor_mode=2)) == 2


def test_default_is_openmp_when_neither_set(monkeypatch):
    monkeypatch.delenv("RUN_ANUGA_MULTIPROCESSOR_MODE", raising=False)
    assert _resolve_multiprocessor_mode(_input_data()) == _MULTIPROCESSOR_OPENMP


def test_falsy_env_value_defaults_to_openmp(monkeypatch):
    """TASK-1954 review (5d5328f): a falsy coerced mode ('0') must default to
    1, not stay 0 — preserved for the env source too."""
    monkeypatch.setenv("RUN_ANUGA_MULTIPROCESSOR_MODE", "0")
    assert _resolve_multiprocessor_mode(_input_data(multiprocessor_mode=2)) == 1


def test_invalid_env_falls_back_to_scenario_config(monkeypatch, caplog):
    monkeypatch.setenv("RUN_ANUGA_MULTIPROCESSOR_MODE", "not-a-number")
    with caplog.at_level(logging.WARNING):
        mode = _resolve_multiprocessor_mode(_input_data(multiprocessor_mode=2))
    assert mode == 2
    assert "RUN_ANUGA_MULTIPROCESSOR_MODE" in caplog.text


def test_falsy_scenario_value_defaults_to_openmp(monkeypatch):
    monkeypatch.delenv("RUN_ANUGA_MULTIPROCESSOR_MODE", raising=False)
    assert _resolve_multiprocessor_mode(_input_data(multiprocessor_mode=0)) == 1


def test_invalid_scenario_value_defaults_to_openmp(monkeypatch):
    monkeypatch.delenv("RUN_ANUGA_MULTIPROCESSOR_MODE", raising=False)
    assert _resolve_multiprocessor_mode(_input_data(multiprocessor_mode="abc")) == 1


def test_blank_env_is_treated_as_absent(monkeypatch):
    """An empty-string env (e.g. a rendered-but-unset Ansible var) must defer
    to scenario_config, not be treated as a present-but-invalid override."""
    monkeypatch.setenv("RUN_ANUGA_MULTIPROCESSOR_MODE", "")
    assert _resolve_multiprocessor_mode(_input_data(multiprocessor_mode=2)) == 2


# ---------------------------------------------------------------------------
# 2. _assert_gpu_engaged — FAIL the run if mode 2 didn't actually engage
# ---------------------------------------------------------------------------

def test_noop_when_mode_not_gpu():
    """CPU images / mode-1 runs never reach a GPU probe (which may not exist
    there at all) — a bare object with zero GPU attributes must not raise."""
    domain = object()
    _assert_gpu_engaged(domain, _MULTIPROCESSOR_OPENMP)  # must not raise


def test_passes_when_gpu_offload_enabled_true():
    domain = mock.Mock()
    domain.gpu_offload_enabled.return_value = True
    _assert_gpu_engaged(domain, _MULTIPROCESSOR_GPU)  # must not raise


def test_raises_when_gpu_offload_enabled_false():
    domain = mock.Mock()
    domain.gpu_offload_enabled.return_value = False
    with pytest.raises(RuntimeError, match="gpu_offload_enabled"):
        _assert_gpu_engaged(domain, _MULTIPROCESSOR_GPU)


def test_raises_when_gpu_offload_enabled_probe_itself_raises():
    domain = mock.Mock()
    domain.gpu_offload_enabled.side_effect = RuntimeError("device query failed")
    with pytest.raises(RuntimeError, match="gpu_offload_enabled"):
        _assert_gpu_engaged(domain, _MULTIPROCESSOR_GPU)


def test_falls_back_to_get_multiprocessor_mode_when_no_probe():
    domain = mock.Mock(spec=["get_multiprocessor_mode"])
    domain.get_multiprocessor_mode.return_value = _MULTIPROCESSOR_GPU
    _assert_gpu_engaged(domain, _MULTIPROCESSOR_GPU)  # must not raise


def test_raises_when_multiprocessor_mode_silently_fell_back_to_cpu():
    """Mirrors the fork's REAL behaviour today: set_gpu_interface() falls
    back to set_multiprocessor_mode(1) on missing cupy/GPU — silently. The
    getter reflects the fallback; this must now be a hard failure."""
    domain = mock.Mock(spec=["get_multiprocessor_mode"])
    domain.get_multiprocessor_mode.return_value = _MULTIPROCESSOR_OPENMP
    with pytest.raises(RuntimeError, match="did not engage"):
        _assert_gpu_engaged(domain, _MULTIPROCESSOR_GPU)


def test_falls_back_to_multiprocessor_mode_attribute_when_no_getter():
    class _Domain:
        multiprocessor_mode = _MULTIPROCESSOR_GPU

    _assert_gpu_engaged(_Domain(), _MULTIPROCESSOR_GPU)  # must not raise


def test_raises_when_no_signal_available_at_all():
    """Fail CLOSED: mode 2 was requested and NOTHING can verify it engaged."""
    domain = object()
    with pytest.raises(RuntimeError, match="no gpu_offload_enabled"):
        _assert_gpu_engaged(domain, _MULTIPROCESSOR_GPU)


# ---------------------------------------------------------------------------
# 3a. _capture_gpu_model — best-effort, never raises
# ---------------------------------------------------------------------------

def test_capture_via_pynvml_when_available(monkeypatch):
    fake_pynvml = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: "handle0",
        nvmlDeviceGetName=lambda h: b"NVIDIA A10G",
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    assert _capture_gpu_model() == "NVIDIA A10G"


def test_capture_via_nvidia_smi_when_pynvml_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import raises ImportError

    def _fake_run(*args, **kwargs):
        return mock.Mock(returncode=0, stdout="NVIDIA A10G\n")

    monkeypatch.setattr("subprocess.run", _fake_run)
    assert _capture_gpu_model() == "NVIDIA A10G"


def test_capture_returns_none_when_neither_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)

    def _fake_run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr("subprocess.run", _fake_run)
    assert _capture_gpu_model() is None


# ---------------------------------------------------------------------------
# 3b. mode/gpu_model land in the resource_summary features bag
# ---------------------------------------------------------------------------

def _import_sampler():
    try:
        from gn_anuga.batch_common.resource_sampler import ResourceSampler
        return ResourceSampler
    except Exception:
        pytest.skip("gn_anuga.batch_common not on path")


def test_mode_and_gpu_model_land_in_resource_summary_features(tmp_path):
    """phase_tracker.set_mesh_features(mode=..., gpu_model=...) — the write
    side this task adds — rides the SAME features bag AdminRunResourceRecord
    Serializer.get_mode/get_gpu_model already read (TASK-2195, pre-existing).
    """
    ResourceSampler = _import_sampler()

    phase_tracker.set_mesh_features(mode="gpu", gpu_model="NVIDIA A10G")

    s = ResourceSampler(
        tmp_path, tool="anuga", job_id="j", interval_s=999,
        mesh_features_provider=phase_tracker.get_mesh_features,
    )
    with s:
        pass
    features = s.summary()["features"]
    assert features["mode"] == "gpu"
    assert features["gpu_model"] == "NVIDIA A10G"


def test_cpu_mode_does_not_fabricate_a_gpu_model(tmp_path):
    """A CPU-mode run stamps mode='cpu' and must NEVER fabricate a gpu_model
    (no-vaporware) — the key stays absent, not a null placeholder."""
    ResourceSampler = _import_sampler()

    phase_tracker.set_mesh_features(mode="cpu")

    s = ResourceSampler(
        tmp_path, tool="anuga", job_id="j", interval_s=999,
        mesh_features_provider=phase_tracker.get_mesh_features,
    )
    with s:
        pass
    features = s.summary()["features"]
    assert features["mode"] == "cpu"
    assert "gpu_model" not in features


# ---------------------------------------------------------------------------
# 3c. End-to-end proof against a REAL (non-GPU) Domain: mode=2 on THIS fork
#     (no cupy, no GPU) is exactly the silent-fallback scenario the epic
#     calls out — set_multiprocessor_mode(2) here falls back to 1 internally
#     (see shallow_water_domain.Domain.set_gpu_interface). Before TASK-2197
#     the run would have completed silently in CPU mode; it must now FAIL.
# ---------------------------------------------------------------------------

@pytest.mark.requires_anuga
@pytest.mark.slow
def test_real_domain_mode_2_without_gpu_fails_the_run(small_test_copy):
    """Run as a SEPARATE process (mirrors test_cli.py's subprocess pattern):
    the assertion's raise reaches run_sim's outer except-Exception handler,
    which calls ``MPI.COMM_WORLD.Abort(1)`` — a hard process kill that a
    same-process ``pytest.raises`` cannot observe (confirmed: it takes the
    whole pytest worker down with it, mid-run, no test report at all).
    """
    import subprocess

    env = dict(os.environ, RUN_ANUGA_MULTIPROCESSOR_MODE="2")
    result = subprocess.run(
        [sys.executable, "-m", "run_anuga.cli", "run", str(small_test_copy)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode != 0
    assert "did not engage (silent CPU fallback)" in result.stderr


# ---------------------------------------------------------------------------
# 4. Structural: run_sim wires the three helpers (mirrors the house style in
#    test_phase_durations.py's multiprocessor_mode section).
# ---------------------------------------------------------------------------

def test_run_sim_wires_the_gpu_helpers():
    import run_anuga.run as run_module
    src = open(run_module.__file__).read()
    assert "_resolve_multiprocessor_mode(input_data)" in src
    assert "_assert_gpu_engaged(domain, _multiprocessor_mode)" in src
    assert "phase_tracker.set_mesh_features(" in src
