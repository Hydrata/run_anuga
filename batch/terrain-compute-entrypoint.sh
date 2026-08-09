#!/bin/bash
set -euo pipefail

# AWS Batch entrypoint for the Django-free terrain compute runner (TASK-1833).
#
# Modelled on batch/entrypoint.sh (the ANUGA sim entrypoint) but for the terrain
# merge: it downloads a JSON *compute manifest* (not a sim package zip), and
# execs `python -m gn_anuga.terrain_compute merge-and-report <manifest>` with
# NO mpirun (the merge is a single-process numpy/rasterio pipeline).
#
# Required environment variables (set via the terrain-compute job definition):
#   CONTROL_SERVER                    - Hydrata control server URL (e.g. https://hydrata.com/)
#   RESULT_S3_BUCKET                  - S3 bucket for the COG result upload (read by the
#                                       runner from the manifest's result_bucket; exported
#                                       here too for parity with the ANUGA contract)
#   HYDRATA_INTERNAL_COMPUTE_TOKEN    - Shared secret for V2 IsInternalComputeCaller (raw
#                                       token, sent in X-Internal-Token header; NOT Bearer)
#   HYDRATA_PROCESS_ID                - TaskMonitor Process uuid (TASK-2677, epic 2662
#                                       W3.1). The container's ONLY telemetry channel:
#                                       POST /api/v2/tasks/processes/<uuid>/events/.
#                                       Was optional during the D9 migration window;
#                                       TASK-2681 (W4.1) tombstoned the legacy derive-*
#                                       fallback, so it is REQUIRED now and fail-closed
#                                       at BOTH ends — the dispatcher
#                                       (gn_anuga.services_terrain_merge.
#                                       dispatch_terrain_merge_batch) refuses to submit
#                                       without it, and merge_and_report raises on it.
#                                       Deliberately NOT `:?`-asserted here: the python
#                                       raise carries the actionable operator message,
#                                       and asserting first would only replace it with a
#                                       terser shell error. Passes through this
#                                       entrypoint into the python process env — no
#                                       explicit export needed (mirrors the ANUGA
#                                       entrypoint, where run_anuga._handoff reads it
#                                       from os.environ).
#   The manifest location, ONE of:
#     MANIFEST_S3_URI                 - full s3://bucket/key URI, OR
#     MANIFEST_S3_BUCKET + MANIFEST_S3_KEY
#
# The manifest carries everything else (analysis_surface_id, project_crs, the
# ordered DEM input stack, result_bucket/result_key, etc.) — see the schema in
# gn_anuga/terrain_compute/merge.py. The runner POSTs its own typed events back
# to CONTROL_SERVER itself; this entrypoint only stages the manifest + provides
# the entrypoint-level terminal-`error` safety net.
#
# Mirrors the ANUGA contract (TASK-1048): no SIGTERM/checkpoint (operator
# accepts spot loss), token-only auth, subprocess CLI invocation.

: "${CONTROL_SERVER:?CONTROL_SERVER env var is required}"
: "${HYDRATA_INTERNAL_COMPUTE_TOKEN:?HYDRATA_INTERNAL_COMPUTE_TOKEN env var is required}"
: "${RESULT_S3_BUCKET:?RESULT_S3_BUCKET env var is required}"

WORK_DIR="/tmp/terrain_compute"
CONTROL_BASE="${CONTROL_SERVER%/}"

# Resolve the manifest S3 URI from either MANIFEST_S3_URI or the
# bucket+key pair.
if [ -n "${MANIFEST_S3_URI:-}" ]; then
  MANIFEST_URI="${MANIFEST_S3_URI}"
elif [ -n "${MANIFEST_S3_BUCKET:-}" ] && [ -n "${MANIFEST_S3_KEY:-}" ]; then
  MANIFEST_URI="s3://${MANIFEST_S3_BUCKET}/${MANIFEST_S3_KEY}"
else
  echo "[terrain-entrypoint] ERROR: provide MANIFEST_S3_URI or MANIFEST_S3_BUCKET+MANIFEST_S3_KEY" >&2
  exit 2
fi

# Telemetry schema version for the entrypoint-level error event. Resolved from
# the baked, Django-free protocol module (the ONE source of truth, spec §0/§3)
# so a SCHEMA_VERSION bump propagates on rebake without editing this shell. The
# `1` fallback covers the case where the interpreter itself is broken — which is
# precisely one of the failures the trap below exists to report, so the fallback
# must never be removed. Resolved HERE, at startup, before anything risky runs.
TELEMETRY_SCHEMA_VERSION="$(python -c 'from gn_anuga.batch_common.telemetry_protocol import SCHEMA_VERSION; print(SCHEMA_VERSION)' 2>/dev/null || true)"
: "${TELEMETRY_SCHEMA_VERSION:=1}"

# Secondary safety net: if a step below fails before merge_and_report can post
# its own terminal `error` event (e.g. the manifest download fails, the runner
# crashes during import, or the container is OOM-killed before the sampler
# arms), POST a terminal `error` event so the Process — and therefore the
# AnalysisSurface — does not wedge.
#
# TASK-2692 (epic 2662 W5) — RE-POINTED from the legacy
# `/api/v2/anuga/analysis-surfaces/<id>/derive-error/` route, which TASK-2681
# (W4.1) turned into an unconditional 410 tombstone. Between W4.1 and this
# commit the trap was a silent no-op in production: every pre-Python container
# death reported NOTHING and reached an operator only via the W3.3 reaper's
# ~1h clock.
#
# Addressing the events endpoint is safe here because terrain is fail-closed at
# BOTH ends as of W4.1 (dispatcher refuses to submit without a Process row;
# merge_and_report raises without the env var), so HYDRATA_PROCESS_ID is
# guaranteed present in a legitimately-dispatched container. The `-n` guard is
# for the mis-provisioned case only: degrade to silence rather than curl a
# malformed URL. (The ANUGA twin in batch/entrypoint.sh could not make this move
# when the above was written — its dispatcher did not yet guarantee
# HYDRATA_PROCESS_ID. W5 closed that: both dispatchers now refuse a Run with no
# TaskMonitor Process, and the twin posts the same events dialect.)
#
# Double-reporting is harmless: merge_and_report's own `error` event fires first
# for in-process failures, and the server fold is guarded
# (`if process.status not in terminal`), so this second event is a no-op.
#
# `source` is CARRIED (spec §2.5/§8.2), naming THIS entrypoint. Today it is
# inert on the server: a terrain Process's source_object is an AnalysisSurface,
# so taskmonitor.telemetry._fan_out_error_to_run returns before `Run.mark_error`
# — and the TASK-2206 precedence rule keys on the literal `entrypoint.sh`
# (BATCH_ENTRYPOINT_SOURCE) anyway, which this value is deliberately NOT. It is
# sent because §8.2 requires BOTH shell traps to name their reporter: the field
# is the only thing that distinguishes a pre-Python container death from
# merge_and_report's own in-process `error` once both are folded onto the same
# Process, and it is already on the wire the day terrain telemetry grows a
# precedence rule of its own. `validate_event` ignores unknown keys, so this is
# additive on every deployed server version.
#
# curl failures are swallowed (|| true) so they never mask the originating exit
# code. Envelope matches docs/strategy/process-telemetry-spec.md §2/§2.1 exactly:
# type + schema_version + ts + message, plus the optional §2.5 `source`.
trap 'exit_code=$?; if [ $exit_code -ne 0 ] && [ -n "${HYDRATA_PROCESS_ID:-}" ]; then
  echo "[terrain-entrypoint] failing with exit code ${exit_code}; posting terminal error event" >&2
  curl -sS -X POST \
    -H "X-Internal-Token: ${HYDRATA_INTERNAL_COMPUTE_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "{\"type\":\"error\",\"schema_version\":${TELEMETRY_SCHEMA_VERSION},\"ts\":$(date +%s),\"message\":\"terrain-compute-entrypoint.sh failed with exit code ${exit_code}\",\"source\":\"terrain-compute-entrypoint.sh\"}" \
    "${CONTROL_BASE}/api/v2/tasks/processes/${HYDRATA_PROCESS_ID}/events/" || true
fi' EXIT

echo "[terrain-entrypoint] === Terrain Compute (merge-and-report) ==="
echo "[terrain-entrypoint] Manifest: ${MANIFEST_URI}"
echo "[terrain-entrypoint] Control:  ${CONTROL_BASE}"
# TASK-2677: one greppable line naming the active telemetry dialect — the
# container-log twin of merge_and_report's own arming line. TASK-2692: the
# absent case is no longer a legacy degrade, it is NO CHANNEL AT ALL.
echo "[terrain-entrypoint] Process:  ${HYDRATA_PROCESS_ID:-<none - NO TELEMETRY CHANNEL; merge_and_report will refuse to run>}"
echo "[terrain-entrypoint] ============================================"

# 1. Download the manifest from S3.
mkdir -p "${WORK_DIR}"
MANIFEST_PATH="${WORK_DIR}/manifest.json"
echo "[terrain-entrypoint] Downloading manifest..."
aws s3 cp "${MANIFEST_URI}" "${MANIFEST_PATH}"
echo "[terrain-entrypoint] Download complete."

# (TASK-2692: the best-effort analysis_surface_id extraction that used to live
# here is gone with the derive-error trap it fed. The events endpoint is keyed
# on HYDRATA_PROCESS_ID, which is present from container start — which is also
# why the trap now covers the manifest download itself, a window the old
# surface-id-keyed trap could never report.)

# 2. Run the merge + report. NO mpirun — the merge is a single-process
# numpy/rasterio pipeline. merge_and_report owns the whole handoff (download
# the DEM stack, union-grid + streaming reproject + feather merge + COG, S3
# upload, `result` event, with an `error` event on any failure).
export HYDRATA_INTERNAL_COMPUTE_TOKEN
export RESULT_S3_BUCKET
echo "[terrain-entrypoint] Starting terrain merge..."
cd "${WORK_DIR}"
python -m gn_anuga.terrain_compute merge-and-report "${MANIFEST_PATH}"
echo "[terrain-entrypoint] === Terrain merge complete ==="
