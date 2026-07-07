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
#   The manifest location, ONE of:
#     MANIFEST_S3_URI                 - full s3://bucket/key URI, OR
#     MANIFEST_S3_BUCKET + MANIFEST_S3_KEY
#
# The manifest carries everything else (analysis_surface_id, project_crs, the
# ordered DEM input stack, result_bucket/result_key, etc.) — see the schema in
# gn_anuga/terrain_compute/merge.py. The runner POSTs progress/result/error
# back to CONTROL_SERVER itself; this entrypoint only stages the manifest +
# provides the entrypoint-level /derive-error/ safety net.
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

# Secondary safety net: if a step below fails before merge_and_report can POST
# its own /derive-error/ (e.g. the manifest download fails, or the runner
# crashes during import), POST a terminal /error/ so the AnalysisSurface does
# not wedge. merge_and_report's own /derive-error/ fires first for in-process
# failures; this trap covers only the entrypoint-level cases. We extract the
# analysis_surface_id from the downloaded manifest when available so the error
# lands on the right surface; if the manifest never downloaded, ANALYSIS_SURFACE_ID
# stays empty and we skip the POST (no surface id to address). curl failures are
# swallowed (|| true) so they never mask the originating exit code.
ANALYSIS_SURFACE_ID=""
trap 'exit_code=$?; if [ $exit_code -ne 0 ] && [ -n "${ANALYSIS_SURFACE_ID}" ]; then
  echo "[terrain-entrypoint] failing with exit code ${exit_code}; posting /derive-error/" >&2
  curl -sS -X POST \
    -H "X-Internal-Token: ${HYDRATA_INTERNAL_COMPUTE_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "{\"analysis_surface_id\":${ANALYSIS_SURFACE_ID},\"message\":\"Terrain-compute entrypoint failed with exit code ${exit_code}\"}" \
    "${CONTROL_BASE}/api/v2/anuga/analysis-surfaces/${ANALYSIS_SURFACE_ID}/derive-error/" || true
fi' EXIT

echo "[terrain-entrypoint] === Terrain Compute (merge-and-report) ==="
echo "[terrain-entrypoint] Manifest: ${MANIFEST_URI}"
echo "[terrain-entrypoint] Control:  ${CONTROL_BASE}"
echo "[terrain-entrypoint] ============================================"

# 1. Download the manifest from S3.
mkdir -p "${WORK_DIR}"
MANIFEST_PATH="${WORK_DIR}/manifest.json"
echo "[terrain-entrypoint] Downloading manifest..."
aws s3 cp "${MANIFEST_URI}" "${MANIFEST_PATH}"
echo "[terrain-entrypoint] Download complete."

# Best-effort extract analysis_surface_id for the entrypoint-level error trap.
ANALYSIS_SURFACE_ID="$(python -c "import json,sys; print(json.load(open('${MANIFEST_PATH}')).get('analysis_surface_id',''))" 2>/dev/null || true)"

# 2. Run the merge + report. NO mpirun — the merge is a single-process
# numpy/rasterio pipeline. merge_and_report owns the whole handoff (download
# the DEM stack, union-grid + streaming reproject + feather merge + COG, S3
# upload, POST /derive-result/, with /derive-error/ on any failure).
export HYDRATA_INTERNAL_COMPUTE_TOKEN
export RESULT_S3_BUCKET
echo "[terrain-entrypoint] Starting terrain merge..."
cd "${WORK_DIR}"
python -m gn_anuga.terrain_compute merge-and-report "${MANIFEST_PATH}"
echo "[terrain-entrypoint] === Terrain merge complete ==="
