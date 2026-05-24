"""Contract test for the Batch entrypoint /process-result/ POST payload.

TASK-1158 (Phase F0): the entrypoint previously POSTed the JSON field
``key``, but the V2 ``/process-result/`` endpoint reads ``result_package_key``
(gn_anuga/api_v2.py — ``data.get('result_package_key')``; the docstring
documents the field as ``result_package_key`` and there is no ``key`` alias).
A Batch run that SUCCEEDED at simulation therefore lost its result key:
``process_result_async`` was never dispatched and the run wedged in COMPUTING
until the 1h zombie watchdog flipped it to ERROR.

This test pins the wire contract so the field cannot silently drift again. It
parses ``batch/entrypoint.sh`` as text (no shell execution needed): it asserts
the ``/process-result/`` POST sends ``result_package_key`` and that the old
bare ``key`` field is gone from that payload.

Plain ``import`` style (no ``importorskip``) to match test_http_helper.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parent.parent / "batch" / "entrypoint.sh"


def _process_result_data_payload() -> str:
    """Return the JSON body of the --data arg on the /process-result/ POST.

    Locates the ``curl`` invocation whose URL ends in ``/process-result/`` and
    returns the (un-shell-escaped) ``--data`` JSON string for it. Raises if the
    payload cannot be found so the test fails loudly rather than passing vacuously.
    """
    text = ENTRYPOINT.read_text()

    # Grab every --data "..." argument with its (shell-escaped) JSON body.
    # The payloads use \" escapes inside a double-quoted shell string.
    data_args = re.findall(r'--data\s+"(.*?)"\s*\\?\n', text)

    for raw in data_args:
        # Un-escape the shell-level \" -> " so we can inspect the JSON.
        unescaped = raw.replace('\\"', '"')
        # The result POST is the one carrying the result key (RESULT_KEY var).
        if "RESULT_KEY" in unescaped or "result_package_key" in unescaped or '"key"' in unescaped:
            return unescaped

    raise AssertionError(
        "Could not locate the /process-result/ --data payload in entrypoint.sh"
    )


def test_entrypoint_file_exists():
    assert ENTRYPOINT.is_file(), f"entrypoint.sh not found at {ENTRYPOINT}"


def test_process_result_sends_result_package_key():
    """The endpoint reads result_package_key — the payload must send it."""
    payload = _process_result_data_payload()
    assert '"result_package_key"' in payload, (
        "entrypoint /process-result/ POST must send the 'result_package_key' "
        f"field that the endpoint reads; got payload: {payload!r}"
    )


def test_process_result_does_not_send_legacy_key_field():
    """The old bare 'key' field is silently dropped by the endpoint — it must be gone."""
    payload = _process_result_data_payload()
    # Parse the (shell-variable-bearing) JSON by neutralising the ${...} expansion
    # so json.loads can read the key names. The value is irrelevant to the contract.
    neutralised = re.sub(r"\$\{[^}]*\}", "PLACEHOLDER", payload)
    fields = json.loads(neutralised)
    assert "key" not in fields, (
        "entrypoint /process-result/ POST still sends the legacy 'key' field, "
        f"which the endpoint ignores; got fields: {sorted(fields)}"
    )
    assert "result_package_key" in fields


def test_process_result_payload_is_valid_json_shape():
    """Defence in depth: the payload must be a single-object JSON with the right key."""
    payload = _process_result_data_payload()
    neutralised = re.sub(r"\$\{[^}]*\}", "PLACEHOLDER", payload)
    fields = json.loads(neutralised)
    assert isinstance(fields, dict)
    assert fields.get("result_package_key") == "PLACEHOLDER"
