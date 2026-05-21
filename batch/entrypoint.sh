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
: "${PROJECT_ID:?PROJECT_ID env var is required}"
: "${SCENARIO_ID:?SCENARIO_ID env var is required}"
: "${RUN_ID:?RUN_ID env var is required}"

WORK_DIR="/tmp/simulation"
CPUS="${CPUS:-$(nproc)}"
RESULT_KEY="${PROJECT_ID}_${SCENARIO_ID}_${RUN_ID}_results.zip"
CONTROL_BASE="${CONTROL_SERVER%/}"

# Secondary safety net: if any step below fails before run.py can call its
# own _report_run_error (or if run.py itself crashes during import / before
# its inner try/except is entered), POST a terminal /error/ so the run does
# not wedge in COMPUTING. run.py's _report_run_error fires first for
# in-process failures; this trap covers only the entrypoint-level cases.
# Failures of the curl itself are swallowed (|| true) so they never mask
# the originating exit code.
trap 'exit_code=$?; if [ $exit_code -ne 0 ]; then
  echo "[entrypoint] failing with exit code ${exit_code}; posting /error/" >&2
  curl -sS -X POST \
    -H "X-Internal-Token: ${HYDRATA_INTERNAL_COMPUTE_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "{\"message\":\"Batch entrypoint failed with exit code ${exit_code}\",\"source\":\"entrypoint.sh\"}" \
    "${CONTROL_BASE}/api/v2/anuga/runs/${RUN_ID}/error/" || true
fi' EXIT

echo "[entrypoint] === ANUGA Batch Simulation ==="
echo "[entrypoint] Package: s3://${PACKAGE_S3_BUCKET}/${PACKAGE_S3_KEY}"
echo "[entrypoint] Control: ${CONTROL_BASE}"
echo "[entrypoint] Run:     ${PROJECT_ID}/${SCENARIO_ID}/${RUN_ID}"
echo "[entrypoint] CPUs:    ${CPUS}"
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

# 3. Run simulation via subprocess (aligns with 2026-05-20 subprocess
# unification: no in-process MPI taint, watchdog-protected finalize).
# The HYDRATA_INTERNAL_COMPUTE_TOKEN is exported into run.py's environment
# for any future token-aware HydrataCallback path (W1 / TASK-1049). Today
# run.py invoked without positional username/password constructs a
# NullCallback at runtime; the terminal /process-result/ + /error/ POSTs
# below carry the token directly and satisfy V2 IsInternalComputeCaller.
echo "[entrypoint] Starting simulation (subprocess, cpus=${CPUS})..."
export HYDRATA_INTERNAL_COMPUTE_TOKEN
if [ "${CPUS}" -gt 1 ]; then
    mpirun -np "${CPUS}" --use-hwthread-cpus python /app/run_anuga_src/run_anuga/run.py \
        --package_dir "${WORK_DIR}" \
        --batch_number 1 \
        --checkpoint_time 0
else
    python /app/run_anuga_src/run_anuga/run.py \
        --package_dir "${WORK_DIR}" \
        --batch_number 1 \
        --checkpoint_time 0
fi
echo "[entrypoint] Simulation complete."

# 4. Zip results and upload to S3
echo "[entrypoint] Packaging results..."
RESULT_ZIP="${WORK_DIR}/${RESULT_KEY}"
cd "${WORK_DIR}"
zip -q -r "${RESULT_ZIP}" . -x "package.zip" "run_anuga/*" "${RESULT_KEY}"

echo "[entrypoint] Uploading results to s3://${RESULT_S3_BUCKET}/${RESULT_KEY}..."
aws s3 cp "${RESULT_ZIP}" "s3://${RESULT_S3_BUCKET}/${RESULT_KEY}"
echo "[entrypoint] Upload complete."

# 5. POST V2 /process-result/ with the result key. IsInternalComputeCaller
# accepts the raw token in X-Internal-Token (NOT Bearer). On non-2xx, set -e
# fails the script and the EXIT trap above posts /error/.
echo "[entrypoint] Notifying control server via V2 /process-result/..."
HTTP_CODE=$(curl -sS -o /tmp/process_result.out -w "%{http_code}" -X POST \
    -H "X-Internal-Token: ${HYDRATA_INTERNAL_COMPUTE_TOKEN}" \
    -H "Content-Type: application/json" \
    --data "{\"key\":\"${RESULT_KEY}\"}" \
    "${CONTROL_BASE}/api/v2/anuga/runs/${RUN_ID}/process-result/")
echo "[entrypoint] /process-result/ returned ${HTTP_CODE}"
cat /tmp/process_result.out || true
echo
if [ "${HTTP_CODE}" -ge 400 ]; then
    echo "[entrypoint] /process-result/ POST failed (HTTP ${HTTP_CODE})" >&2
    exit 1
fi

echo "[entrypoint] === Simulation complete ==="
