#!/bin/bash
set -euo pipefail

# AWS Batch entrypoint for ANUGA simulations.
#
# Required environment variables (set via job definition):
#   PACKAGE_S3_BUCKET                 - S3 bucket containing the simulation package ZIP
#   PACKAGE_S3_KEY                    - S3 key of the simulation package ZIP
#   RESULT_S3_BUCKET                  - S3 bucket for result uploads
#   CONTROL_SERVER                    - Hydrata control server URL (e.g. https://hydrata.com/)
#   PROJECT_ID                        - Hydrata project ID
#   SCENARIO_ID                       - Hydrata scenario ID
#   RUN_ID                            - Hydrata run ID
#   HYDRATA_INTERNAL_COMPUTE_TOKEN    - Shared secret for V2 IsInternalComputeCaller (raw token,
#                                       sent in X-Internal-Token header; NOT a Bearer token)
#   HYDRATA_PROCESS_ID                - TaskMonitor Process uuid (TASK-2672, epic 2662
#                                       W2.2). The container's ONLY telemetry channel:
#                                       POST /api/v2/tasks/processes/<uuid>/events/.
#                                       Was optional during the D9 migration window;
#                                       TASK-2692 (W5) tombstoned the legacy
#                                       /api/v2/anuga/runs/<id>/{process-result,error}/
#                                       routes, so it is REQUIRED now and fail-closed at
#                                       BOTH ends — the dispatchers
#                                       (gn_anuga.services._dispatch_batch for Batch,
#                                       gn_anuga.tasks.dispatch_local_anuga_run for the
#                                       celery-native localhost path) refuse to dispatch
#                                       without it, and run_anuga._handoff.
#                                       _make_telemetry_client raises on it. Deliberately
#                                       NOT `:?`-asserted here: the python raise carries
#                                       the actionable operator message, and asserting
#                                       first would only replace it with a terser shell
#                                       error. Passes through this entrypoint into the
#                                       python process env — no explicit export needed
#                                       (run_anuga._handoff reads it from os.environ).
#
# Optional:
#   CPUS                              - Number of MPI processes (default: nproc)
#
# D-decisions (TASK-1048):
#   D4.c  No SIGTERM trap, no checkpoint resume. Operator accepts spot loss.
#   D6.c  Token-only auth. Legacy BasicAuth env vars are NOT read.
#   Subprocess CLI invocation (not in-process import), aligning with the
#   2026-05-20 subprocess unification across localhost + prod.

: "${CONTROL_SERVER:?CONTROL_SERVER env var is required}"
: "${HYDRATA_INTERNAL_COMPUTE_TOKEN:?HYDRATA_INTERNAL_COMPUTE_TOKEN env var is required}"
: "${RESULT_S3_BUCKET:?RESULT_S3_BUCKET env var is required}"
: "${PROJECT_ID:?PROJECT_ID env var is required}"
: "${SCENARIO_ID:?SCENARIO_ID env var is required}"
: "${RUN_ID:?RUN_ID env var is required}"

WORK_DIR="/tmp/simulation"
CPUS="${CPUS:-$(nproc)}"
CONTROL_BASE="${CONTROL_SERVER%/}"

# Telemetry schema version for the entrypoint-level error event. Resolved from
# the baked, Django-free protocol module (the ONE source of truth, spec §0/§3)
# so a SCHEMA_VERSION bump propagates on rebake without editing this shell. The
# `1` fallback covers the case where the interpreter itself is broken — which is
# precisely one of the failures the trap below exists to report, so the fallback
# must never be removed. Resolved HERE, at startup, before anything risky runs.
TELEMETRY_SCHEMA_VERSION="$(python -c 'from gn_anuga.batch_common.telemetry_protocol import SCHEMA_VERSION; print(SCHEMA_VERSION)' 2>/dev/null || true)"
: "${TELEMETRY_SCHEMA_VERSION:=1}"

# Secondary safety net: if any step below fails before run_and_report can post
# its own terminal `error` event (the S3 package download, the unzip, an
# mpirun/import crash, or an OOM kill before the sampler arms), POST a terminal
# `error` event so the Process — and therefore the Run — does not wedge in
# COMPUTING.
#
# TASK-2692 (epic 2662 W5) — RE-POINTED from the legacy
# `/api/v2/anuga/runs/<id>/error/` route, which this same task turns into an
# unconditional 410 tombstone. This is the ONLY reporter for a death before
# Python is reachable, so leaving it on the tombstoned route would make every
# such failure silent, reaching an operator only via the W3.3 reaper's ~1h
# clock (exactly what happened to the terrain twin between W4.1 and 1a8d6c0).
#
# Addressing the events endpoint is safe because the anuga tool is now
# fail-closed at BOTH ends: the dispatchers (gn_anuga.services._dispatch_batch,
# gn_anuga.tasks.dispatch_local_anuga_run) refuse to dispatch a Run with no
# TaskMonitor Process, and run_anuga._handoff._make_telemetry_client raises
# without the env var. The `-n` guard is for the mis-provisioned case only:
# degrade to silence rather than curl a URL with an empty uuid segment.
#
# `source` is CARRIED (spec §2.5): Run.mark_error's TASK-2206 precedence rule
# keys on it so this generic "failed with exit code N" message can never
# clobber a richer one already known. The events `error` fan-out passes the
# field straight into that rule (taskmonitor.telemetry._fan_out_error_to_run),
# and TelemetryClient.error grew the matching parameter in the same task —
# dropping it here would regress ENTRYPOINT_FAILURE classification to
# 'unknown'.
#
# Double-reporting is harmless: run_and_report's own `error` event fires first
# for in-process failures, and the server fold is guarded
# (`if process.status not in terminal`), so this second event is a no-op.
#
# curl failures are swallowed (|| true) so they never mask the originating exit
# code (the reaper reads Batch job status). Envelope matches
# docs/strategy/process-telemetry-spec.md §2/§2.1: type + schema_version + ts +
# message, plus the optional §2.5 `source`.
trap 'exit_code=$?; if [ $exit_code -ne 0 ] && [ -n "${HYDRATA_PROCESS_ID:-}" ]; then
  echo "[entrypoint] failing with exit code ${exit_code}; posting terminal error event" >&2
  curl -sS -X POST \
    -H "X-Internal-Token: ${HYDRATA_INTERNAL_COMPUTE_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "{\"type\":\"error\",\"schema_version\":${TELEMETRY_SCHEMA_VERSION},\"ts\":$(date +%s),\"message\":\"Batch entrypoint failed with exit code ${exit_code}\",\"source\":\"entrypoint.sh\"}" \
    "${CONTROL_BASE}/api/v2/tasks/processes/${HYDRATA_PROCESS_ID}/events/" || true
fi' EXIT

echo "[entrypoint] === ANUGA Batch Simulation ==="
echo "[entrypoint] Package: s3://${PACKAGE_S3_BUCKET}/${PACKAGE_S3_KEY}"
echo "[entrypoint] Control: ${CONTROL_BASE}"
echo "[entrypoint] Run:     ${PROJECT_ID}/${SCENARIO_ID}/${RUN_ID}"
echo "[entrypoint] CPUs:    ${CPUS}"
# TASK-2672: one greppable line naming the active telemetry dialect — the
# container-log twin of run_and_report's own arming line. TASK-2692: the absent
# case is no longer a legacy degrade, it is NO CHANNEL AT ALL.
echo "[entrypoint] Process: ${HYDRATA_PROCESS_ID:-<none - NO TELEMETRY CHANNEL; run-and-report will refuse to run>}"
echo "[entrypoint] =============================="

# 1. Download package from S3
mkdir -p "${WORK_DIR}"
PACKAGE_ZIP="${WORK_DIR}/package.zip"

echo "[entrypoint] Downloading simulation package..."
aws s3 cp "s3://${PACKAGE_S3_BUCKET}/${PACKAGE_S3_KEY}" "${PACKAGE_ZIP}"
echo "[entrypoint] Download complete."

# 2. Extract package
echo "[entrypoint] Extracting package..."
cd "${WORK_DIR}"
unzip -q "${PACKAGE_ZIP}"
rm "${PACKAGE_ZIP}"

# 3. Run simulation + result handoff via run_anuga (TASK-1159 / F1).
# run_anuga.cli run-and-report owns the whole post-sim handoff (zip + S3
# upload + a terminal `result` event, with a terminal `error` event on any
# failure) so the wire shape is typed Python with a shared field-name constant
# (RESULT_PACKAGE_KEY_FIELD) instead of two diverging shell+Python copies. The
# TASK-1158 (F0) drift class is structurally impossible from this point on.
# TASK-2692 (W5): the legacy /process-result/ + /error/ dialect this comment
# used to name is deleted — everything rides the events endpoint now.
echo "[entrypoint] Starting simulation + handoff (cpus=${CPUS})..."
export HYDRATA_INTERNAL_COMPUTE_TOKEN
export RESULT_S3_BUCKET
# OpenMPI refuses to run as root by default. The Batch container is single-purpose
# (Fargate-style: run sim, upload result, exit) so the standard non-root hardening
# does not apply here. These two env vars are the documented escape hatch.
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
if [ "${CPUS}" -gt 1 ]; then
    mpirun -np "${CPUS}" --use-hwthread-cpus python -m run_anuga.cli run-and-report "${WORK_DIR}"
else
    python -m run_anuga.cli run-and-report "${WORK_DIR}"
fi
echo "[entrypoint] === Simulation + handoff complete ==="
