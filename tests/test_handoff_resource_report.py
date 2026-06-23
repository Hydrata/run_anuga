"""Tests for the TASK-1846 ANUGA resource-report retrofit (epic 1830 W4).

Covers run_anuga._handoff's new resource_summary emit wiring:

* ``RESOURCE_REPORT_TOOL`` is the ``"anuga"`` discriminator.
* ``_make_resource_sampler`` returns ``None`` (no raise) when the staged
  ``gn_anuga.batch_common`` leaf is absent (localhost / non-Batch) — the
  guarded-import contract.
* ``report_resource_summary`` POSTs the summary to ``/jobs/resource-report/``
  with ``tool='anuga'`` when a job_id is present, SKIPs the POST when there is
  no AWS_BATCH_JOB_ID, and NEVER raises on a POST failure (best-effort ledger).
* The module still imports cleanly without Django (run_anuga stays Django-free).

These are pure unit tests — no real AWS / network / ANUGA sim. The sampler is a
lightweight stand-in (a real ResourceSampler needs the cgroup filesystem).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from run_anuga._handoff import (
    RESOURCE_REPORT_TOOL,
    _make_resource_sampler,
    report_resource_summary,
)


class _FakeSampler:
    """Minimal sampler stand-in: returns a fixed summary dict from ``.summary()``."""

    def __init__(self, summary):
        self._summary = summary

    def summary(self):
        return self._summary


def test_resource_report_tool_constant():
    """The discriminator stamped onto the ANUGA resource_summary must be 'anuga'."""
    assert RESOURCE_REPORT_TOOL == "anuga"


def test_make_resource_sampler_none_without_batch_common():
    """No staged gn_anuga.batch_common on the path → None, never raises.

    Simulate the localhost / non-Batch image (the leaf is not bundled) by making
    the import fail; the helper must degrade to None so run_and_report still runs
    the sim with no ledger.
    """
    with mock.patch.dict(sys.modules, {"gn_anuga.batch_common.resource_sampler": None}):
        # Setting the module to None makes `import gn_anuga.batch_common...` raise
        # ImportError, which the helper catches.
        sampler = _make_resource_sampler(
            "/tmp",
            control_server="https://hydrata.com",
            ids={"run_id": 1, "project_id": 2, "scenario_id": 3},
        )
    assert sampler is None


def test_report_resource_summary_posts_with_anuga_tool():
    """A summary carrying a job_id is POSTed to /jobs/resource-report/ as 'anuga'."""
    sampler = _FakeSampler({"tool": "anuga", "job_id": "abc-123", "observed": {}})
    fake_resp = mock.Mock(ok=True, status_code=201)
    fake_session = mock.Mock()
    fake_session.post.return_value = fake_resp

    with mock.patch(
        "run_anuga._http.make_internal_session", return_value=fake_session,
    ):
        report_resource_summary("https://hydrata.com/", "tok", sampler)

    fake_session.post.assert_called_once()
    args, kwargs = fake_session.post.call_args
    assert args[0] == "https://hydrata.com/api/v2/anuga/jobs/resource-report/"
    assert kwargs["json"]["tool"] == "anuga"
    assert kwargs["json"]["job_id"] == "abc-123"
    fake_session.close.assert_called_once()


def test_report_resource_summary_skips_without_job_id():
    """No AWS_BATCH_JOB_ID (local run) → no POST (BE would 400 an empty job_id)."""
    sampler = _FakeSampler({"tool": "anuga", "job_id": "", "observed": {}})
    fake_session = mock.Mock()

    with mock.patch(
        "run_anuga._http.make_internal_session", return_value=fake_session,
    ):
        report_resource_summary("https://hydrata.com", "tok", sampler)

    fake_session.post.assert_not_called()


def test_report_resource_summary_none_sampler_is_noop():
    """A None sampler (batch_common absent) → silent no-op, never raises."""
    # Must not touch make_internal_session at all.
    with mock.patch(
        "run_anuga._http.make_internal_session",
    ) as mk:
        report_resource_summary("https://hydrata.com", "tok", None)
    mk.assert_not_called()


def test_report_resource_summary_never_raises_on_post_failure():
    """A POST failure must NOT mask the run outcome — best-effort ledger."""
    sampler = _FakeSampler({"tool": "anuga", "job_id": "abc-123", "observed": {}})
    fake_session = mock.Mock()
    fake_session.post.side_effect = RuntimeError("network down")

    with mock.patch(
        "run_anuga._http.make_internal_session", return_value=fake_session,
    ):
        # No exception should escape.
        report_resource_summary("https://hydrata.com", "tok", sampler)

    # session.close() still runs in the finally even though post raised.
    fake_session.close.assert_called_once()


def test_report_resource_summary_never_raises_on_summary_failure():
    """A sampler whose .summary() raises is swallowed (never masks the run)."""
    bad_sampler = mock.Mock()
    bad_sampler.summary.side_effect = RuntimeError("sampler broke")
    # No mock of make_internal_session needed — we never reach the POST.
    report_resource_summary("https://hydrata.com", "tok", bad_sampler)


def test_module_imports_without_django():
    """run_anuga must stay Django-free; import the module in a fresh subprocess."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", "import run_anuga._handoff"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
    )
    assert result.returncode == 0, (
        f"run_anuga._handoff failed to import standalone:\nstderr: {result.stderr}"
    )
