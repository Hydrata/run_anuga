"""Tests for the TASK-1846 ANUGA resource-report retrofit (epic 1830 W4).

Updated for TASK-1879: ``report_resource_summary`` now delegates to the shared
``gn_anuga.batch_common.emit.emit_resource_summary`` helper.  The helper is
imported via a guarded try/except inside the function (mirrors the existing
``_make_resource_sampler`` pattern), so tests that exercise the POST path must
inject a fake ``gn_anuga.batch_common.emit`` module via ``mock.patch.dict``.

Covers run_anuga._handoff's resource_summary emit wiring:

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
import types
from pathlib import Path
from unittest import mock


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


def _fake_emit_module():
    """Return a fake ``gn_anuga.batch_common.emit`` module with a real emit helper.

    The helper directly does the POST via ``session`` (as the real one does),
    so the per-test session mocks see the call.
    """
    import logging

    def _emit(sampler, *, session, resource_report_url, timeout=30):
        try:
            summary = sampler.summary()
            if not summary.get("job_id"):
                logging.getLogger("run_anuga._handoff").info(
                    "emit_resource_summary: no AWS_BATCH_JOB_ID"
                    " — skipping resource-report POST",
                )
                return
            resp = session.post(resource_report_url, json=summary, timeout=timeout)
            if not getattr(resp, "ok", False):
                logging.getLogger("run_anuga._handoff").warning(
                    "emit_resource_summary: %s responded %s — %s",
                    resource_report_url,
                    getattr(resp, "status_code", "?"),
                    (getattr(resp, "text", "") or "")[:200],
                )
        except Exception:
            logging.getLogger("run_anuga._handoff").warning(
                "emit_resource_summary: resource-report POST failed; suppressed",
                exc_info=True,
            )

    mod = types.ModuleType("gn_anuga.batch_common.emit")
    mod.emit_resource_summary = _emit
    return mod


def _with_emit(fn):
    """Decorator: inject a working fake gn_anuga.batch_common.emit for the test."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        fake_mod = _fake_emit_module()
        patch = {
            "gn_anuga": types.ModuleType("gn_anuga"),
            "gn_anuga.batch_common": types.ModuleType("gn_anuga.batch_common"),
            "gn_anuga.batch_common.emit": fake_mod,
        }
        with mock.patch.dict(sys.modules, patch):
            return fn(*args, **kwargs)

    return wrapper


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


def _capture_sampler_kwargs(code_shas_json):
    """Call ``_make_resource_sampler`` with a staged fake ``ResourceSampler`` that
    records its constructor kwargs, driving ``CODE_SHAS_JSON`` from the env.

    ``code_shas_json=None`` means the env var is unset (empty string).
    Returns the captured kwargs dict.
    """
    import os as _os

    captured: dict = {}

    def _factory(scratch_dir, **kwargs):
        captured.update(kwargs)
        return _FakeSampler({"tool": "anuga", "job_id": "j"})

    fake_mod = types.ModuleType("gn_anuga.batch_common.resource_sampler")
    fake_mod.ResourceSampler = _factory
    module_patch = {
        "gn_anuga": types.ModuleType("gn_anuga"),
        "gn_anuga.batch_common": types.ModuleType("gn_anuga.batch_common"),
        "gn_anuga.batch_common.resource_sampler": fake_mod,
    }
    env_patch = {"CODE_SHAS_JSON": "" if code_shas_json is None else code_shas_json}
    with mock.patch.dict(sys.modules, module_patch), \
            mock.patch.dict(_os.environ, env_patch, clear=False):
        _make_resource_sampler(
            "/tmp",
            control_server="https://hydrata.com",
            ids={"run_id": 1, "project_id": 2, "scenario_id": 3},
        )
    return captured


def test_make_resource_sampler_parses_code_shas_into_sampler():
    """TASK-2105: CODE_SHAS_JSON is parsed and handed to the sampler's code_shas
    kwarg, so it lands in summary()['ids']['code_shas'] for the staff API."""
    import json as _json

    shas = {"hydrata": "abc1234", "geonode": "def5678", "run_anuga": "9900aab"}
    captured = _capture_sampler_kwargs(_json.dumps(shas))
    assert captured["code_shas"] == shas


def test_make_resource_sampler_code_shas_none_when_env_absent():
    """No CODE_SHAS_JSON in the env → code_shas is None (never fabricated)."""
    captured = _capture_sampler_kwargs(None)
    assert captured["code_shas"] is None


def test_make_resource_sampler_code_shas_none_when_env_malformed():
    """A malformed CODE_SHAS_JSON → code_shas None, but the sampler is still built
    (a bad provenance env must not cost us the whole resource ledger)."""
    captured = _capture_sampler_kwargs("{not valid json")
    assert captured.get("code_shas") is None
    assert "ids" in captured  # sampler still constructed


def test_make_resource_sampler_code_shas_none_when_env_not_a_dict():
    """A non-object CODE_SHAS_JSON (e.g. a JSON list) → code_shas None."""
    captured = _capture_sampler_kwargs("[1, 2, 3]")
    assert captured.get("code_shas") is None


@_with_emit
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


@_with_emit
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


@_with_emit
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


@_with_emit
def test_report_resource_summary_never_raises_on_summary_failure():
    """A sampler whose .summary() raises is swallowed (never masks the run)."""
    bad_sampler = mock.Mock()
    bad_sampler.summary.side_effect = RuntimeError("sampler broke")
    # No mock of make_internal_session needed — we never reach the POST.
    with mock.patch("run_anuga._http.make_internal_session"):
        report_resource_summary("https://hydrata.com", "tok", bad_sampler)


def test_report_resource_summary_skips_when_emit_absent():
    """No gn_anuga.batch_common on path → emit_resource_summary = None → no-op."""
    sampler = _FakeSampler({"tool": "anuga", "job_id": "abc-123", "observed": {}})
    # Simulate gn_anuga absent (the localhost / non-Batch case).
    with mock.patch.dict(sys.modules, {
        "gn_anuga": None,
        "gn_anuga.batch_common": None,
        "gn_anuga.batch_common.emit": None,
    }):
        with mock.patch("run_anuga._http.make_internal_session") as mk:
            report_resource_summary("https://hydrata.com", "tok", sampler)
    mk.assert_not_called()


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
