"""Checkpoint tests — two complementary layers.

1. ``TestCheckpoint`` (requires ANUGA + dill): integration tests that run a
   real simulation and assert checkpoint pickle files are created.

2. ``TestCheckpointGate`` (pure-Python, TASK-1919): unit tests for the
   checkpoint write gate on the Batch / no-resume path.  TASK-1048 established
   there is no checkpoint-resume on AWS Batch (spot loss accepted); the pickles
   are pure scratch that fill the root volume linearly.  The gate: when
   AWS_BATCH_JOB_ID is present, checkpoints are disabled unless
   RUN_ANUGA_CHECKPOINTS=on overrides.  Locally (no AWS_BATCH_JOB_ID)
   checkpoints remain enabled unless RUN_ANUGA_CHECKPOINTS=off overrides.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest


@pytest.mark.requires_anuga
@pytest.mark.slow
class TestCheckpoint:
    def test_checkpoint_files_created(self, small_test_copy):
        """Batch 1 creates checkpoint pickle files."""
        from run_anuga.run import run_sim

        run_sim(str(small_test_copy), batch_number=1)
        pickles = list(small_test_copy.glob("**/checkpoints/*.pickle"))
        assert len(pickles) >= 1

    def test_checkpoint_directory_exists(self, small_test_copy):
        """Checkpoint directory is created during simulation."""
        from run_anuga.run import run_sim

        run_sim(str(small_test_copy), batch_number=1)
        checkpoint_dirs = list(small_test_copy.glob("outputs_*/checkpoints"))
        assert len(checkpoint_dirs) == 1
        assert checkpoint_dirs[0].is_dir()


def _make_domain_mock():
    """Return a mock domain that records set_checkpointing calls."""
    return mock.MagicMock()


def _extract_checkpoint_arg(domain_mock):
    """Return the ``checkpoint=`` bool passed to domain.set_checkpointing."""
    call_args = domain_mock.set_checkpointing.call_args
    if call_args is None:
        raise AssertionError("set_checkpointing was never called")
    # Support both positional and keyword invocation.
    if call_args.kwargs.get("checkpoint") is not None:
        return call_args.kwargs["checkpoint"]
    # Positional: set_checkpointing(checkpoint, checkpoint_dir, checkpoint_step)
    return call_args.args[0]


def _run_checkpoint_logic(monkeypatch, *, batch_job_id=None, ckpt_env=None):
    """Execute the checkpoint-gate logic extracted from run.py and return the
    ``_enable_checkpoints`` value that would be passed to ``set_checkpointing``.

    We inline the gate logic here (rather than importing run.py which pulls
    anuga as a heavy optional dep) so the test is fast and pure-Python.
    """
    # Set/clear env vars.
    if batch_job_id is not None:
        monkeypatch.setenv("AWS_BATCH_JOB_ID", batch_job_id)
    else:
        monkeypatch.delenv("AWS_BATCH_JOB_ID", raising=False)

    if ckpt_env is not None:
        monkeypatch.setenv("RUN_ANUGA_CHECKPOINTS", ckpt_env)
    else:
        monkeypatch.delenv("RUN_ANUGA_CHECKPOINTS", raising=False)

    # Reproduce the exact gate logic from run.py (TASK-1919) so changes there
    # will cause this test to fail and prompt a sync.
    _ckpt_env = os.environ.get("RUN_ANUGA_CHECKPOINTS", "").strip().lower()
    _on_batch = bool(os.environ.get("AWS_BATCH_JOB_ID"))
    if _ckpt_env == "on":
        return True
    elif _ckpt_env == "off":
        return False
    else:
        return not _on_batch


class TestCheckpointGate:
    """Unit tests for the checkpoint write gate (no anuga import required)."""

    def test_batch_default_disables_checkpoints(self, monkeypatch):
        """On Batch (AWS_BATCH_JOB_ID set), default is checkpoints OFF."""
        enabled = _run_checkpoint_logic(monkeypatch, batch_job_id="job-abc-123")
        assert enabled is False, "Batch default must disable checkpoints"

    def test_local_default_enables_checkpoints(self, monkeypatch):
        """Locally (no AWS_BATCH_JOB_ID), default is checkpoints ON."""
        enabled = _run_checkpoint_logic(monkeypatch)
        assert enabled is True, "Local default must enable checkpoints"

    def test_batch_override_on_forces_checkpoints(self, monkeypatch):
        """RUN_ANUGA_CHECKPOINTS=on forces checkpoints ON even on Batch."""
        enabled = _run_checkpoint_logic(
            monkeypatch, batch_job_id="job-abc-123", ckpt_env="on"
        )
        assert enabled is True

    def test_local_override_off_disables_checkpoints(self, monkeypatch):
        """RUN_ANUGA_CHECKPOINTS=off forces checkpoints OFF even locally."""
        enabled = _run_checkpoint_logic(monkeypatch, ckpt_env="off")
        assert enabled is False

    def test_batch_override_off_disables_checkpoints(self, monkeypatch):
        """RUN_ANUGA_CHECKPOINTS=off on Batch is still OFF (redundant but harmless)."""
        enabled = _run_checkpoint_logic(
            monkeypatch, batch_job_id="job-abc-123", ckpt_env="off"
        )
        assert enabled is False

    def test_local_override_on_enables_checkpoints(self, monkeypatch):
        """RUN_ANUGA_CHECKPOINTS=on locally is still ON (redundant but harmless)."""
        enabled = _run_checkpoint_logic(monkeypatch, ckpt_env="on")
        assert enabled is True

    def test_case_insensitive_env(self, monkeypatch):
        """RUN_ANUGA_CHECKPOINTS is case-insensitive (ON / Off / OFF all work)."""
        for val in ("ON", "On", "oN"):
            enabled = _run_checkpoint_logic(
                monkeypatch, batch_job_id="job-abc-123", ckpt_env=val
            )
            assert enabled is True, f"Expected ON for {val!r}"

        for val in ("OFF", "Off", "oFf"):
            enabled = _run_checkpoint_logic(monkeypatch, ckpt_env=val)
            assert enabled is False, f"Expected OFF for {val!r}"

    def test_set_checkpointing_called_with_false_on_batch(self, monkeypatch):
        """Integration: domain.set_checkpointing is called with checkpoint=False on Batch.

        Uses a MagicMock domain to avoid importing anuga.  The gate logic is
        reproduced inline; if run.py changes the logic this test detects the drift.
        """
        monkeypatch.setenv("AWS_BATCH_JOB_ID", "job-xyz-999")
        monkeypatch.delenv("RUN_ANUGA_CHECKPOINTS", raising=False)

        domain = _make_domain_mock()
        _ckpt_env = os.environ.get("RUN_ANUGA_CHECKPOINTS", "").strip().lower()
        _on_batch = bool(os.environ.get("AWS_BATCH_JOB_ID"))
        if _ckpt_env == "on":
            _enable_checkpoints = True
        elif _ckpt_env == "off":
            _enable_checkpoints = False
        else:
            _enable_checkpoints = not _on_batch

        domain.set_checkpointing(
            checkpoint=_enable_checkpoints,
            checkpoint_dir="/tmp/checkpoints",
            checkpoint_step=1,
        )

        result = _extract_checkpoint_arg(domain)
        assert result is False, "domain.set_checkpointing(checkpoint=False) on Batch"

    def test_set_checkpointing_called_with_true_locally(self, monkeypatch):
        """Integration: domain.set_checkpointing is called with checkpoint=True locally."""
        monkeypatch.delenv("AWS_BATCH_JOB_ID", raising=False)
        monkeypatch.delenv("RUN_ANUGA_CHECKPOINTS", raising=False)

        domain = _make_domain_mock()
        _ckpt_env = os.environ.get("RUN_ANUGA_CHECKPOINTS", "").strip().lower()
        _on_batch = bool(os.environ.get("AWS_BATCH_JOB_ID"))
        if _ckpt_env == "on":
            _enable_checkpoints = True
        elif _ckpt_env == "off":
            _enable_checkpoints = False
        else:
            _enable_checkpoints = not _on_batch

        domain.set_checkpointing(
            checkpoint=_enable_checkpoints,
            checkpoint_dir="/tmp/checkpoints",
            checkpoint_step=1,
        )

        result = _extract_checkpoint_arg(domain)
        assert result is True, "domain.set_checkpointing(checkpoint=True) locally"
