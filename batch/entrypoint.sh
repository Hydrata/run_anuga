#!/bin/bash
set -euo pipefail

# AWS Batch entrypoint for ANUGA simulations.
#
# Required environment variables (set via job definition):
#   PACKAGE_S3_BUCKET  — S3 bucket containing the simulation package ZIP
#   PACKAGE_S3_KEY     — S3 key of the simulation package ZIP
#   RESULT_S3_BUCKET   — S3 bucket for result uploads
#   CONTROL_SERVER     — Hydrata control server URL (e.g. https://hydrata.com/)
#   PROJECT_ID         — Hydrata project ID
#   SCENARIO_ID        — Hydrata scenario ID
#   RUN_ID             — Hydrata run ID
#   ANUGA_USERNAME     — HTTP Basic Auth username for control server
#   ANUGA_PASSWORD     — HTTP Basic Auth password for control server
#
# Optional:
#   CPUS               — Number of MPI processes (default: nproc)

WORK_DIR="/tmp/simulation"
CPUS="${CPUS:-$(nproc)}"

echo "=== ANUGA Batch Simulation ==="
echo "Package: s3://${PACKAGE_S3_BUCKET}/${PACKAGE_S3_KEY}"
echo "Control: ${CONTROL_SERVER}"
echo "Run:     ${PROJECT_ID}/${SCENARIO_ID}/${RUN_ID}"
echo "CPUs:    ${CPUS}"
echo "=============================="

# 1. Download package from S3
mkdir -p "${WORK_DIR}"
PACKAGE_ZIP="${WORK_DIR}/package.zip"

echo "Downloading simulation package..."
python3 -c "
import boto3
s3 = boto3.client('s3')
s3.download_file('${PACKAGE_S3_BUCKET}', '${PACKAGE_S3_KEY}', '${PACKAGE_ZIP}')
print('Download complete.')
"

# 2. Extract package
echo "Extracting package..."
cd "${WORK_DIR}"
unzip -q "${PACKAGE_ZIP}"
rm "${PACKAGE_ZIP}"

# 3. Run simulation with HydrataCallback
echo "Starting simulation..."
python3 -c "
from run_anuga.run import run_sim
from run_anuga.callbacks import HydrataCallback
import os

callback = HydrataCallback(
    username=os.environ['ANUGA_USERNAME'],
    password=os.environ['ANUGA_PASSWORD'],
    control_server=os.environ['CONTROL_SERVER'],
    project=int(os.environ['PROJECT_ID']),
    scenario=int(os.environ['SCENARIO_ID']),
    run_id=int(os.environ['RUN_ID']),
)

run_sim(
    package_dir='${WORK_DIR}',
    callback=callback,
    batch_number=1,
    checkpoint_time=0,
)
print('Simulation complete.')
"

# 4. Zip results and upload to S3
echo "Packaging results..."
RESULT_KEY="${PROJECT_ID}_${SCENARIO_ID}_${RUN_ID}_results"
RESULT_ZIP="${WORK_DIR}/${RESULT_KEY}.zip"

cd "${WORK_DIR}"
zip -q -r "${RESULT_ZIP}" . -x "package.zip" "run_anuga/*"

echo "Uploading results to S3..."
python3 -c "
import boto3
s3 = boto3.client('s3')
s3.upload_file('${RESULT_ZIP}', '${RESULT_S3_BUCKET}', '${RESULT_KEY}.zip')
print('Upload complete.')
"

# 5. POST result to control server
echo "Notifying control server..."
python3 -c "
import os
import requests

client = requests.Session()
client.auth = requests.auth.HTTPBasicAuth(
    os.environ['ANUGA_USERNAME'],
    os.environ['ANUGA_PASSWORD'],
)
url = os.environ['CONTROL_SERVER'].rstrip('/') + \
    f'/anuga/api/{os.environ[\"PROJECT_ID\"]}/{os.environ[\"SCENARIO_ID\"]}/run/{os.environ[\"RUN_ID\"]}/process-result/'
response = client.post(url, data={
    'result_package_key': '${RESULT_KEY}.zip',
})
print(f'Control server response: {response.status_code}')
"

echo "=== Simulation complete ==="
