"""Contract tests for batch/terrain-compute-entrypoint.sh's EXIT trap.

TASK-2692 (epic 2662 W5). The terrain entrypoint's EXIT trap is the ONLY
reporting channel for a *pre-Python* container death — an S3 manifest download
failure, an unzip failure, an import crash, an OOM before the sampler arms.
TASK-2681 (W4.1) turned the route it was posting to
(``/api/v2/anuga/analysis-surfaces/<id>/derive-error/``) into an unconditional
``410 Gone`` tombstone, which made the trap a silent no-op in production: those
deaths reached an operator only via the W3.3 reaper's ~1h clock.

The trap now posts a terminal ``error`` event to the protocol endpoint
``POST /api/v2/tasks/processes/<uuid>/events/``.

Shell on the critical path is not provable by unit-testing Python, so these
tests are deliberately split:

* textual pins — the tombstoned routes are gone, the events endpoint is there;
* a BEHAVIOURAL harness that runs the real shipped script with a fake ``aws``
  that fails, against a raw-socket HTTP stub, and asserts the request line and
  the JSON envelope byte-for-byte. That is the only way to catch a quoting bug,
  a ``set -u`` explosion inside the trap, or a trap that masks the exit code.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import pytest

ENTRYPOINT = (
    Path(__file__).resolve().parent.parent / "batch" / "terrain-compute-entrypoint.sh"
)

#: The protocol endpoint, mirroring
#: ``gn_anuga.batch_common.telemetry_protocol.EVENTS_PATH_TEMPLATE``. That module
#: is baked into the terrain image but is deliberately NOT importable from a bare
#: run_anuga env (same constraint tests/test_events_dialect.py works around), so
#: the path is re-stated here and the behavioural test below proves the shell
#: actually produces it.
EVENTS_PATH_RE = r"/api/v2/tasks/processes/\$\{HYDRATA_PROCESS_ID\}/events/"

PROCESS_ID = "6f1c0c2e-6b1e-4d3e-9a55-0c9d1a2b3c4d"
TOKEN = "test-internal-compute-token"


# ---------------------------------------------------------------------------
# Textual pins
# ---------------------------------------------------------------------------

def test_entrypoint_file_exists():
    assert ENTRYPOINT.is_file(), f"terrain entrypoint not found at {ENTRYPOINT}"


def test_trap_does_not_curl_any_tombstoned_derive_route():
    """No curl may address a W4.1 tombstone (derive-error/-progress/-result)."""
    text = ENTRYPOINT.read_text()
    matches = re.findall(r"curl\b[^#\n]*derive-(?:error|progress|result)", text, re.DOTALL)
    assert not matches, (
        "terrain-compute-entrypoint.sh still curl-POSTs a legacy derive-* route; "
        "those answer 410 Gone since TASK-2681 (W4.1). Found: %r" % (matches,)
    )


def test_trap_curls_the_events_endpoint():
    text = ENTRYPOINT.read_text()
    assert re.search(EVENTS_PATH_RE, text), (
        "terrain-compute-entrypoint.sh must POST its terminal error event to "
        "the protocol endpoint /api/v2/tasks/processes/<uuid>/events/"
    )


def test_trap_never_masks_the_exit_code():
    """`|| true` on the curl is load-bearing: a telemetry outage must not
    change the container's exit status (the reaper reads Batch job status)."""
    text = ENTRYPOINT.read_text()
    trap = text.split("trap '", 1)[1].split("' EXIT", 1)[0]
    assert "|| true" in trap, "the trap's curl must be suffixed `|| true`"


def test_trap_guards_on_process_id():
    """A mis-provisioned container must degrade to silence, not curl a URL with
    an empty uuid segment."""
    text = ENTRYPOINT.read_text()
    trap = text.split("trap '", 1)[1].split("' EXIT", 1)[0]
    assert '[ -n "${HYDRATA_PROCESS_ID:-}" ]' in trap, (
        "the trap must guard on a non-empty HYDRATA_PROCESS_ID"
    )


def test_entrypoint_is_syntactically_valid_bash():
    proc = subprocess.run(
        ["bash", "-n", str(ENTRYPOINT)], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Behavioural harness — run the REAL script, capture the RAW request
# ---------------------------------------------------------------------------

_TELEMETRY_PROTOCOL_STUB = "SCHEMA_VERSION = 1\n"


def _stage_fake_gn_anuga(root: Path, schema_version: int) -> Path:
    """Mirror the terrain image's staged layout (PYTHONPATH=/app, gn_anuga/)."""
    app = root / "app"
    pkg = app / "gn_anuga"
    (pkg / "batch_common").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "batch_common" / "__init__.py").write_text("")
    (pkg / "batch_common" / "telemetry_protocol.py").write_text(
        f"SCHEMA_VERSION = {schema_version}\n"
    )
    tc = pkg / "terrain_compute"
    tc.mkdir()
    (tc / "__init__.py").write_text("")
    (tc / "__main__.py").write_text("import sys\nprint('stub merge', sys.argv[1:])\n")
    return app


def _run_entrypoint(tmp_path, *, process_id, schema_version=1, aws_fails=True):
    """Run the shipped entrypoint against a one-shot raw-socket HTTP stub.

    Returns ``(completed_process, captured)`` where ``captured`` is ``{}`` when
    the trap posted nothing.
    """
    captured: dict = {}

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    sock.settimeout(30)
    port = sock.getsockname()[1]

    def serve():
        try:
            conn, _ = sock.accept()
        except OSError:
            return
        conn.settimeout(10)
        try:
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            head, _, rest = buf.partition(b"\r\n\r\n")
            length = 0
            for line in head.split(b"\r\n")[1:]:
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1].strip())
            while len(rest) < length:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                rest += chunk
            captured["head"] = head.decode()
            captured["body"] = rest.decode()
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Content-Length: 33\r\n\r\n{\"accepted\":1,\"status\":\"error\"}\n"
            )
        finally:
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    if aws_fails:
        aws = '#!/bin/bash\necho "fake aws: simulated S3 failure" >&2\nexit 1\n'
    else:
        aws = '#!/bin/bash\necho \'{"analysis_surface_id": 42}\' > "${!#}"\nexit 0\n'
    (fakebin / "aws").write_text(aws)
    (fakebin / "aws").chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["CONTROL_SERVER"] = f"http://127.0.0.1:{port}/"
    env["HYDRATA_INTERNAL_COMPUTE_TOKEN"] = TOKEN
    env["RESULT_S3_BUCKET"] = "hydrata-results"
    env["MANIFEST_S3_URI"] = "s3://bucket/key.json"
    env["PYTHONPATH"] = str(_stage_fake_gn_anuga(tmp_path, schema_version))
    if process_id is None:
        env.pop("HYDRATA_PROCESS_ID", None)
    else:
        env["HYDRATA_PROCESS_ID"] = process_id

    proc = subprocess.run(
        ["bash", str(ENTRYPOINT)],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=120,
    )
    thread.join(timeout=5)
    sock.close()
    return proc, captured


needs_curl = pytest.mark.skipif(
    shutil.which("curl") is None, reason="curl not installed"
)


@needs_curl
def test_pre_python_failure_posts_terminal_error_event(tmp_path):
    """The manifest download fails -> the trap POSTs a spec-shaped error event.

    This is the exact window the old surface-id-keyed trap could never report
    (the id was extracted FROM the manifest that failed to download).
    """
    proc, cap = _run_entrypoint(tmp_path, process_id=PROCESS_ID)

    assert cap, "trap posted nothing on a pre-Python failure"
    request_line = cap["head"].splitlines()[0]
    assert request_line == (
        f"POST /api/v2/tasks/processes/{PROCESS_ID}/events/ HTTP/1.1"
    ), request_line
    assert f"X-Internal-Token: {TOKEN}" in cap["head"]
    assert "Content-Type: application/json" in cap["head"]

    body = json.loads(cap["body"])
    # Envelope must match process-telemetry-spec.md §2/§2.1 EXACTLY.
    assert sorted(body) == ["message", "schema_version", "ts", "type"], body
    assert body["type"] == "error"
    assert body["schema_version"] == 1
    assert isinstance(body["ts"], (int, float)) and not isinstance(body["ts"], bool)
    assert "exit code 1" in body["message"]
    assert "terrain-compute-entrypoint.sh" in body["message"]

    # The trap must never mask the originating exit code.
    assert proc.returncode == 1, proc.returncode


@needs_curl
def test_schema_version_is_read_from_the_baked_protocol_module(tmp_path):
    """Known-positive: stage SCHEMA_VERSION=7 and require it on the wire.

    Without this the hardcoded `1` fallback could be doing all the work and the
    happy-path assertion above would still pass.
    """
    _, cap = _run_entrypoint(tmp_path, process_id=PROCESS_ID, schema_version=7)
    assert cap, "trap posted nothing"
    assert json.loads(cap["body"])["schema_version"] == 7


@needs_curl
def test_missing_process_id_degrades_to_silence(tmp_path):
    """A mis-provisioned container must not curl a malformed URL."""
    proc, cap = _run_entrypoint(tmp_path, process_id=None)
    assert not cap, f"trap POSTed despite an unset HYDRATA_PROCESS_ID: {cap}"
    assert proc.returncode == 1


@needs_curl
def test_successful_run_posts_nothing(tmp_path):
    """The trap is an error path only — a clean exit reports nothing."""
    proc, cap = _run_entrypoint(tmp_path, process_id=PROCESS_ID, aws_fails=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert not cap, f"trap POSTed on a successful run: {cap}"
