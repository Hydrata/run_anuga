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
: "${RESULT_S3_BUCKET:?RESULT_S3_BUCKET env var is required}"
: "${PROJECT_ID:?PROJECT_ID env var is required}"
: "${SCENARIO_ID:?SCENARIO_ID env var is required}"
: "${RUN_ID:?RUN_ID env var is required}"

WORK_DIR="/tmp/simulation"
CPUS="${CPUS:-$(nproc)}"
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

# 3. Run simulation + result handoff via run_anuga (TASK-1159 / F1).
# run_anuga.cli run-and-report owns the whole post-sim handoff (zip + S3
# upload + POST /process-result/, with /error/ on any failure) so the
# wire-shape of /process-result/ is typed Python with a shared field-name
# constant (RESULT_PACKAGE_KEY_FIELD) instead of two diverging shell+Python
# copies. The TASK-1158 (F0) drift class is structurally impossible from
# this point on.
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
