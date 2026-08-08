"""``python -m run_anuga.run`` refuses — TASK-2692 (epic 2662 W5).

This file used to test ``run_anuga.run._report_run_error`` (TASK-1078, W6.2 of
TASK-1048), which POSTed the originating run failure to
``/api/v2/anuga/runs/<id>/error/`` from ``main()``'s exception handler.

W5 turns that route into a 410 tombstone, and the function had no honest
successor:

* it could not be CONVERTED to the events protocol — the protocol is keyed on a
  TaskMonitor Process uuid that a dispatcher hands to the container, and nothing
  dispatches ``python -m run_anuga.run`` (the packaged console script is
  ``run-anuga = run_anuga.cli:main``; both dispatchers shell
  ``run_anuga.cli run-and-report``). The entry point also has NO result channel
  at all, so an events port would be an error-only half-dialect that can flip a
  Run to ERROR but never complete one;
* it could not be SILENTLY DROPPED either — that would leave a duplicate of
  ``run_anuga.cli run`` still wearing the legacy username/password signature, a
  live trap where a developer points it at a real ``control_server`` and the
  server learns NOTHING, not even the failure.

So the entry point refuses, and these tests pin that refusal plus the absence of
any surviving caller of a tombstoned route anywhere in the shipped package.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "run_anuga"


def test_report_run_error_is_gone():
    import run_anuga.run as run_module

    assert not hasattr(run_module, "_report_run_error"), (
        "_report_run_error POSTed /api/v2/anuga/runs/<id>/error/, a 410 "
        "tombstone since TASK-2692; it must not come back"
    )


def test_main_refuses_and_returns_non_zero(capsys):
    import run_anuga.run as run_module

    assert run_module.main() == 2
    err = capsys.readouterr().err
    assert "no longer a supported entry point" in err
    # It must name BOTH supported replacements, or the refusal is a dead end.
    assert "run_anuga.cli run " in err or "run_anuga.cli run <" in err
    assert "run_anuga.cli run-and-report" in err


def test_module_invocation_exits_non_zero(tmp_path):
    """End to end: `python -m run_anuga.run` must not silently run a sim."""
    proc = subprocess.run(
        [sys.executable, "-m", "run_anuga.run"],
        capture_output=True, text=True, timeout=300,
        cwd=str(tmp_path),
        # run_anuga is not necessarily pip-installed for this interpreter
        # (same constraint test_handoff.test_module_imports_without_django
        # works around) — point at the repo root explicitly.
        env={"PYTHONPATH": str(PACKAGE_ROOT.parent), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 2, (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
    assert "no longer a supported entry point" in proc.stderr


@pytest.mark.parametrize("module_path", sorted(PACKAGE_ROOT.rglob("*.py")),
                         ids=lambda p: p.name)
def test_no_module_builds_a_tombstoned_run_scoped_url(module_path: Path):
    """Repo-wide: nothing in the shipped package may BUILD a legacy per-run URL.

    Prose may still NAME the dead routes when explaining the deletion (several
    docstrings do). What must never come back is a live URL, which every
    deleted caller built by interpolating the run id — so the ban is on exactly
    that shape.
    """
    text = module_path.read_text()
    offenders = re.findall(r"anuga/runs/\{[^}]*\}", text)
    assert not offenders, (
        f"{module_path.name} builds a legacy /api/v2/anuga/runs/<id>/ URL "
        f"({offenders!r}); those routes are 410 tombstones since TASK-2692"
    )
