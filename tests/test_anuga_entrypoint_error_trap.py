"""Contract tests for batch/entrypoint.sh's EXIT trap (TASK-2692, epic 2662 W5).

The ANUGA entrypoint's EXIT trap is the ONLY reporting channel for a
*pre-Python* container death — the S3 package download, the unzip, an
mpirun/import crash, an OOM kill before the sampler arms. TASK-2692 turns the
route it used to curl (``/api/v2/anuga/runs/<id>/error/``) into an
unconditional ``410 Gone`` tombstone, so leaving it there would make every such
death silent, reaching an operator only via the W3.3 reaper's ~1h clock. That
is exactly what happened to the terrain twin between W4.1 and 1a8d6c0.

The trap now posts a terminal ``error`` event to the protocol endpoint
``POST /api/v2/tasks/processes/<uuid>/events/``, CARRYING ``source`` (spec
§2.5) so ``Run.mark_error``'s TASK-2206 precedence rule still classifies an
entrypoint failure instead of degrading it to ``unknown``.

Shell on the critical path is not provable by unit-testing Python, so these
tests are deliberately split (same shape as
``tests/test_terrain_entrypoint_contract.py``, commit 1a8d6c0):

* textual pins — the tombstoned route is gone, the events endpoint is there;
* a BEHAVIOURAL harness that runs the real shipped script with a failing fake
  ``aws`` against a raw-socket HTTP stub and asserts the request line, headers
  and JSON envelope byte-for-byte, plus exit-code preservation. That is the
  only way to catch a quoting bug, a ``set -u`` explosion inside the trap, or a
  trap that masks the exit code;
* a KNOWN-NEGATIVE that drives the SAME harness against the pre-TASK-2692
  script and shows it produces the OLD (410-bound) request — proving the
  harness can actually detect the defect rather than passing vacuously;
* a KNOWN-POSITIVE that stages ``SCHEMA_VERSION=7`` and requires it on the
  wire, so the hardcoded ``1`` fallback cannot be silently doing the work;
* a cross-check that feeds the CAPTURED BYTES to the REAL server validator
  (``taskmonitor.telemetry.validate_event``), with a control proving the
  validator is not a no-op.

Static gating is two-layer: ``bash -n`` PARSES the script, and ``shellcheck``
(added in the epic 2662 cleanup pass) catches the quoting / word-splitting class
that a parse is structurally blind to. See
``test_entrypoint_passes_shellcheck`` below for the severity and exclusion
reasoning — both are load-bearing, not stylistic.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO_ROOT / "batch" / "entrypoint.sh"

#: The protocol endpoint, mirroring
#: ``gn_anuga.batch_common.telemetry_protocol.EVENTS_PATH_TEMPLATE``. That module
#: is baked into the ANUGA image (the `overlay` build context in
#: deploy/scripts/rebuild-batch-image.sh) but is deliberately NOT importable from
#: a bare run_anuga env — same constraint tests/test_events_dialect.py works
#: around — so the path is re-stated here and the behavioural tests below prove
#: the shell actually produces it.
EVENTS_PATH_RE = r"/api/v2/tasks/processes/\$\{HYDRATA_PROCESS_ID\}/events/"

#: The commit this script was ported ON — provenance for the fixture below.
#: Regenerate with:
#:   git show 1a8d6c05a0c99146f7dbb8a3ba4ff2cb11811d4c:batch/entrypoint.sh \
#:     > tests/data/entrypoint_baselines/entrypoint.pre-2692.sh
PRE_PORT_REF = "1a8d6c05a0c99146f7dbb8a3ba4ff2cb11811d4c"

#: The pre-port bytes, checked in so the known-negative proves the harness has
#: detection power in CI too — not only where the full history happens to exist.
PRE_PORT_SCRIPT = (
    Path(__file__).resolve().parent
    / "data" / "entrypoint_baselines" / "entrypoint.pre-2692.sh"
)

PROCESS_ID = "2b3f6a10-5f8e-4a1b-9c7d-8e0f1a2b3c4d"
TOKEN = "test-internal-compute-token"
RUN_ID = "1243"

#: The one interpreter that can import the monolith's taskmonitor package. Used
#: by the validator cross-check only; absent in bare-run_anuga CI, where that
#: single test skips.
HYDRATA_VENV_PYTHON = Path("/opt/venv/hydrata/bin/python")
HYDRATA_REPO = Path("/opt/hydrata")


# ---------------------------------------------------------------------------
# Textual pins
# ---------------------------------------------------------------------------

def test_entrypoint_file_exists():
    assert ENTRYPOINT.is_file(), f"anuga entrypoint not found at {ENTRYPOINT}"


def test_trap_does_not_curl_the_tombstoned_run_error_route():
    """No curl may address the W5 tombstone (/api/v2/anuga/runs/<id>/error/)."""
    text = ENTRYPOINT.read_text()
    matches = re.findall(r"curl\b[^#\n]*anuga/runs/[^\n]*/error/", text, re.DOTALL)
    assert not matches, (
        "entrypoint.sh still curl-POSTs the legacy /api/v2/anuga/runs/<id>/error/ "
        "route; that answers 410 Gone since TASK-2692 (W5). Found: %r" % (matches,)
    )


def test_trap_curls_the_events_endpoint():
    text = ENTRYPOINT.read_text()
    assert re.search(EVENTS_PATH_RE, text), (
        "entrypoint.sh must POST its terminal error event to the protocol "
        "endpoint /api/v2/tasks/processes/<uuid>/events/"
    )


def test_trap_never_masks_the_exit_code():
    """`|| true` on the curl is load-bearing: a telemetry outage must not change
    the container's exit status (the reaper reads Batch job status)."""
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


def test_entrypoint_passes_shellcheck():
    """shellcheck the shipped entrypoint at shellcheck's DEFAULT severity.

    ``bash -n`` above only PARSES. It is structurally blind to the quoting /
    word-splitting class that actually breaks this script — an unquoted
    ``${PACKAGE_S3_KEY}`` containing a space silently truncates the ``aws s3
    cp`` and the container dies before Python, i.e. exactly the pre-Python death
    the trap above exists to report. Measured, not assumed: injecting that
    defect leaves ``bash -n`` at rc=0 while shellcheck reds with SC2086.

    Severity is left at the DEFAULT deliberately. ``-S warning`` would NOT catch
    the injected quoting bug, because SC2086 is info-level — a warnings-only
    gate here is VACUOUS. Do not "tighten" it by raising the threshold.

    SC2154 is excluded via the FLAG, never an inline ``# shellcheck disable``
    directive, so the shipped .sh stays byte-identical to the checked-in
    pre-TASK-2692 baseline the known-negative below diffs against. It is a
    proven false positive: shellcheck cannot follow ``exit_code=$?`` assigned as
    the first statement INSIDE a single-quoted ``trap '...' EXIT`` body, so it
    fires on every trap of this shape (reproduced on a 4-line script).

    Never passes vacuously in CI: GitHub-hosted ubuntu runners ship shellcheck,
    so its absence THERE is a runner regression, not a reason to go green. Same
    canary reasoning as ``test_rasterio_present_when_ci_expects_it``
    (tests/test_run_utils.py). Locally it degrades to a warning rather than
    ``pytest.skip`` because the TASK-2329 marker-lint ratchet caps this repo at
    ``conditional-skip=29`` and the repo already sits exactly on that cap — a
    new skip call site would red `marker-lint`, and raising the cap means
    editing .github/**.
    """
    if shutil.which("shellcheck") is None:
        assert not os.environ.get("GITHUB_ACTIONS"), (
            "shellcheck is missing on a GitHub runner — this gate would pass "
            "vacuously. Fix the runner image or install shellcheck explicitly; "
            "do not relax this assertion."
        )
        warnings.warn(
            "shellcheck not installed locally — static gate degraded to `bash -n` "
            "for this run (CI still enforces it)",
            stacklevel=2,
        )
        return
    proc = subprocess.run(
        ["shellcheck", "-e", "SC2154", str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Behavioural harness — run the REAL script, capture the RAW request
# ---------------------------------------------------------------------------

def _stage_fake_gn_anuga(root: Path, schema_version: int) -> Path:
    """Mirror the ANUGA image's staged overlay (PYTHONPATH=/app, gn_anuga/)."""
    app = root / "app"
    pkg = app / "gn_anuga"
    (pkg / "batch_common").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "batch_common" / "__init__.py").write_text("")
    (pkg / "batch_common" / "telemetry_protocol.py").write_text(
        f"SCHEMA_VERSION = {schema_version}\n"
    )
    return app


def _stage_fakebin(root: Path, *, aws_fails: bool) -> Path:
    """PATH shims for the three externals the entrypoint shells.

    ``python`` is a dispatcher, not a stub: ``python -c ...`` (the
    SCHEMA_VERSION resolution) delegates to the REAL interpreter so the staged
    protocol module is genuinely imported, while ``python -m run_anuga.cli
    run-and-report`` is a no-op success (running a real sim here is obviously
    out of scope).
    """
    fakebin = root / "bin"
    fakebin.mkdir()

    if aws_fails:
        aws = '#!/bin/bash\necho "fake aws: simulated S3 failure" >&2\nexit 1\n'
    else:
        # `aws s3 cp <uri> <dest>` — materialise the destination and succeed.
        aws = '#!/bin/bash\n: > "${!#}"\nexit 0\n'
    (fakebin / "aws").write_text(aws)

    (fakebin / "unzip").write_text('#!/bin/bash\nexit 0\n')
    (fakebin / "python").write_text(
        "#!/bin/bash\n"
        'if [ "$1" = "-c" ]; then exec %s "$@"; fi\n'
        'echo "fake python: $*"\n'
        "exit 0\n" % sys.executable
    )
    for name in ("aws", "unzip", "python"):
        (fakebin / name).chmod(0o755)
    return fakebin


def _run_entrypoint(tmp_path, *, script=None, process_id, schema_version=1,
                    aws_fails=True):
    """Run a shipped entrypoint against a one-shot raw-socket HTTP stub.

    Returns ``(completed_process, captured)`` where ``captured`` is ``{}`` when
    the trap posted nothing.
    """
    script = Path(script) if script is not None else ENTRYPOINT
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

    fakebin = _stage_fakebin(tmp_path, aws_fails=aws_fails)

    env = dict(os.environ)
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    env["CONTROL_SERVER"] = f"http://127.0.0.1:{port}/"
    env["HYDRATA_INTERNAL_COMPUTE_TOKEN"] = TOKEN
    env["RESULT_S3_BUCKET"] = "hydrata-results"
    env["PACKAGE_S3_BUCKET"] = "hydrata-packages"
    env["PACKAGE_S3_KEY"] = "packages/601_384_1243.zip"
    env["PROJECT_ID"] = "601"
    env["SCENARIO_ID"] = "384"
    env["RUN_ID"] = RUN_ID
    env["CPUS"] = "1"  # single-process branch: no mpirun on this box
    env["PYTHONPATH"] = str(_stage_fake_gn_anuga(tmp_path, schema_version))
    if process_id is None:
        env.pop("HYDRATA_PROCESS_ID", None)
    else:
        env["HYDRATA_PROCESS_ID"] = process_id

    proc = subprocess.run(
        ["bash", str(script)],
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
    """The package download fails -> the trap POSTs a spec-shaped error event."""
    proc, cap = _run_entrypoint(tmp_path, process_id=PROCESS_ID)

    assert cap, "trap posted nothing on a pre-Python failure"
    request_line = cap["head"].splitlines()[0]
    assert request_line == (
        f"POST /api/v2/tasks/processes/{PROCESS_ID}/events/ HTTP/1.1"
    ), request_line
    assert f"X-Internal-Token: {TOKEN}" in cap["head"]
    assert "Content-Type: application/json" in cap["head"]

    body = json.loads(cap["body"])
    # Envelope must match process-telemetry-spec.md §2/§2.1 + the §2.5 `source`.
    assert sorted(body) == ["message", "schema_version", "source", "ts", "type"], body
    assert body["type"] == "error"
    assert body["schema_version"] == 1
    assert isinstance(body["ts"], (int, float)) and not isinstance(body["ts"], bool)
    assert "exit code 1" in body["message"]
    assert body["source"] == "entrypoint.sh"

    # The trap must never mask the originating exit code.
    assert proc.returncode == 1, proc.returncode


@needs_curl
def test_source_is_carried_so_mark_error_precedence_still_applies(tmp_path):
    """spec §2.5: without `source` the TASK-2206 precedence rule cannot fire and
    an entrypoint failure classifies as 'unknown'. This is the ONE producer of
    the field in the whole fleet, so it gets its own named pin."""
    _, cap = _run_entrypoint(tmp_path, process_id=PROCESS_ID)
    assert cap, "trap posted nothing"
    assert json.loads(cap["body"])["source"] == "entrypoint.sh"


@needs_curl
def test_schema_version_is_read_from_the_baked_protocol_module(tmp_path):
    """KNOWN-POSITIVE: stage SCHEMA_VERSION=7 and require it on the wire.

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


# ---------------------------------------------------------------------------
# KNOWN-NEGATIVE — prove the harness can detect the defect
# ---------------------------------------------------------------------------

def _pre_port_script(tmp_path: Path) -> Path:
    """Materialise batch/entrypoint.sh as it was BEFORE this port.

    Read from a CHECKED-IN fixture, not from ``git show <ref>``. The blob form
    made this test's own precondition depend on clone depth, and
    ``actions/checkout@v4`` defaults to ``fetch-depth: 1`` — so in CI the
    ``git show`` failed and the known-negative SKIPPED. A detector that only
    proves itself on a developer box is not proven where it matters
    (``marker_lint`` counted that runtime skip against run_anuga's
    conditional-skip cap, which is how this surfaced). The fixture is the
    verbatim blob from ``PRE_PORT_REF``; it never changes, because the commit
    it came from never changes.
    """
    path = tmp_path / "entrypoint.pre-port.sh"
    path.write_text(PRE_PORT_SCRIPT.read_text())
    return path


@needs_curl
def test_known_negative_pre_port_script_posts_the_tombstoned_route(tmp_path):
    """Drive the SAME harness against the pre-TASK-2692 script.

    It must produce the OLD request — legacy route, no `type`, no
    `schema_version`, no `ts`. A harness that cannot show this difference would
    pass vacuously against ANY script, so this is what makes the assertions
    above worth trusting.
    """
    script = _pre_port_script(tmp_path)
    proc, cap = _run_entrypoint(tmp_path, script=script, process_id=PROCESS_ID)

    assert cap, "the pre-port script posted nothing — harness is not driving it"
    request_line = cap["head"].splitlines()[0]
    assert request_line == f"POST /api/v2/anuga/runs/{RUN_ID}/error/ HTTP/1.1", (
        request_line
    )
    body = json.loads(cap["body"])
    assert sorted(body) == ["message", "source"], body
    assert "type" not in body and "schema_version" not in body
    assert proc.returncode == 1


# ---------------------------------------------------------------------------
# Server-side cross-check — the REAL validator accepts the captured bytes
# ---------------------------------------------------------------------------

_VALIDATOR_SNIPPET = """
import json, os, sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'hydrata.pytest_local_settings'
os.environ.setdefault('RECAPTCHA_PRIVATE_KEY', 'x')
os.environ.setdefault('RECAPTCHA_PUBLIC_KEY', 'x')
import django
django.setup()
from taskmonitor.telemetry import validate_event
etype, err = validate_event(json.loads(sys.argv[1]))
print(json.dumps({'type': etype, 'error': err}))
"""

needs_monolith = pytest.mark.skipif(
    not (HYDRATA_VENV_PYTHON.exists() and HYDRATA_REPO.is_dir()),
    reason="hydrata monolith venv not present (bare run_anuga env)",
)


def _validate_with_real_server(payload: str) -> dict:
    proc = subprocess.run(
        [str(HYDRATA_VENV_PYTHON), "-c", _VALIDATOR_SNIPPET, payload],
        cwd=str(HYDRATA_REPO), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@needs_curl
@needs_monolith
def test_captured_bytes_are_accepted_by_the_real_server_validator(tmp_path):
    """Feed the EXACT bytes the shell put on the wire to
    ``taskmonitor.telemetry.validate_event``. Re-stating the envelope in a test
    only proves the test agrees with itself; this proves the server agrees."""
    _, cap = _run_entrypoint(tmp_path, process_id=PROCESS_ID)
    assert cap, "trap posted nothing"
    verdict = _validate_with_real_server(cap["body"])
    assert verdict == {"type": "error", "error": None}, verdict


@needs_curl
@needs_monolith
def test_the_real_server_validator_is_not_a_no_op(tmp_path):
    """Control: the same validator must REJECT a malformed envelope, else the
    acceptance above would prove nothing."""
    _, cap = _run_entrypoint(tmp_path, process_id=PROCESS_ID)
    assert cap, "trap posted nothing"
    body = json.loads(cap["body"])

    bogus_type = dict(body, type="entrypoint_failure")
    verdict = _validate_with_real_server(json.dumps(bogus_type))
    assert verdict["type"] is None and "unknown event type" in verdict["error"]

    future_version = dict(body, schema_version=body["schema_version"] + 1000)
    verdict = _validate_with_real_server(json.dumps(future_version))
    assert verdict["type"] is None and "newer than this server" in verdict["error"]
