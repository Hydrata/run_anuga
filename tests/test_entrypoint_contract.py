"""Contract test for the /process-result/ wire field name.

TASK-1158 (Phase F0): the entrypoint previously POSTed the JSON field ``key``,
but the V2 ``/process-result/`` endpoint reads ``result_package_key``. A Batch
run that SUCCEEDED at simulation therefore lost its result key, and the run
wedged in COMPUTING until the 1h zombie watchdog flipped it to ERROR.

TASK-1159 (Phase F1) removed that drift class structurally by:

* Moving the POST out of ``batch/entrypoint.sh`` into typed Python
  (``run_anuga._handoff.report_result``).
* Routing the field name through ONE constant (``RESULT_PACKAGE_KEY_FIELD``)
  that both the sender and the Django receiver
  (``gn_anuga.api_v2.HydrataAnugaRunsViewSet.process_result``) import.

This test pins those two properties: (1) the constant's value matches the
receiver's expected field name, and (2) the entrypoint shell no longer
contains a ``/process-result/`` POST or a bare ``key`` JSON field.
"""

from __future__ import annotations

from pathlib import Path

from run_anuga._handoff import RESULT_PACKAGE_KEY_FIELD

ENTRYPOINT = Path(__file__).resolve().parent.parent / "batch" / "entrypoint.sh"


def test_entrypoint_file_exists():
    assert ENTRYPOINT.is_file(), f"entrypoint.sh not found at {ENTRYPOINT}"


def test_shared_constant_value():
    """The constant's value must match what gn_anuga.api_v2 reads."""
    assert RESULT_PACKAGE_KEY_FIELD == "result_package_key"


def test_entrypoint_invokes_run_and_report_cli():
    """Entrypoint shrinks to download + invoke; no shell-level handoff curl."""
    text = ENTRYPOINT.read_text()
    assert "run-and-report" in text, (
        "entrypoint.sh must invoke 'run_anuga.cli run-and-report' for the post-sim "
        "handoff; the F1 refactor moved zip + upload + /process-result/ into Python."
    )


def test_entrypoint_does_not_curl_process_result():
    """No curl-driven /process-result/ POST may live in the shell anymore."""
    text = ENTRYPOINT.read_text()
    # Match `curl ... process-result` as a single shell command (across line
    # continuations). The literal token still appears in the explanatory comment
    # block describing what F1 moved out, so a bare `in` check is too strict.
    import re
    matches = re.findall(r"curl\b[^#\n]*process-result", text, flags=re.DOTALL)
    assert not matches, (
        "entrypoint.sh still curl-POSTs /process-result/; that path is owned "
        f"by run_anuga._handoff.report_result now (F1). Found: {matches!r}"
    )


def test_entrypoint_does_not_send_legacy_key_field():
    """The old bare 'key' JSON field is silently dropped by the endpoint — gone."""
    text = ENTRYPOINT.read_text()
    assert '"key":' not in text and '\\"key\\":' not in text, (
        "entrypoint.sh still references the legacy bare 'key' JSON field"
    )
