#!/usr/bin/env bash
# Phase 5: Visualization
# Tests: viz subcommand generates video from result TIFFs

echo "=== Phase 5: Visualization ==="
echo ""

# Copy source to writable location (bind mount is :ro)
cp -r /app /tmp/run_anuga_src

# Install everything needed (system deps + sim + viz + anuga)
echo "== Setting up: system deps + run_anuga[sim,viz] + anuga =="

apt-get update -qq && apt-get install -y -qq \
    build-essential gfortran libopenmpi-dev openmpi-bin \
    libhdf5-dev libnetcdf-dev 2>&1 | tail -3

ANUGA_SRC="anuga @ git+https://github.com/Hydrata/anuga_core.git@main"
pip install '/tmp/run_anuga_src[sim,viz]' mpi4py pymetis scipy triangle netCDF4 "$ANUGA_SRC" 2>&1 | tail -5

# First, run the simulation to get output TIFFs
echo ""
echo "== Pre-requisite: running simulation to generate output TIFFs =="
cp -r /app/examples/small_test /tmp/workdir/small_test_viz

run-anuga run /tmp/workdir/small_test_viz/ 2>&1 | tail -10
rc_run=$?

if [ $rc_run -ne 0 ]; then
    echo "    Simulation failed (exit $rc_run) — cannot test viz"
    count_fail=1
    print_summary
    exit 1
fi

# Find the output directory
OUTPUT_DIR=$(find /tmp/workdir/small_test_viz -maxdepth 1 -name 'outputs_*' -type d | head -1)

if [ -z "$OUTPUT_DIR" ]; then
    echo "    No outputs_* directory found — cannot test viz"
    count_fail=1
    print_summary
    exit 1
fi

echo ""
echo "== Viz tests =="

# Step 1: Generate depth video
test_step 1 "run-anuga viz (depth video)" \
    "run-anuga viz $OUTPUT_DIR depth"

# Step 2: Check for video output
if [ "$last_rc" -eq 0 ]; then
    test_step 2 "Check for depth video file" \
        "find $OUTPUT_DIR -name '*depth*.mp4' -o -name '*depth*.avi' | head -1"
else
    echo "    Step 2: SKIPPED (viz failed)"
fi

print_summary
