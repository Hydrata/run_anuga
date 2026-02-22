#!/usr/bin/env bash
# Phase 3: Post-process (SWW to GeoTIFF)
# Tests: post-process subcommand, output GeoTIFF files

echo "=== Phase 3: Post-process ==="
echo ""

# Copy source to writable location (bind mount is :ro)
cp -r /app /tmp/run_anuga_src

# Install everything needed (system deps + sim + anuga)
echo "== Setting up: system deps + run_anuga[sim] + anuga =="

apt-get update -qq && apt-get install -y -qq \
    build-essential gfortran libopenmpi-dev openmpi-bin \
    libhdf5-dev libnetcdf-dev 2>&1 | tail -3

# Install anuga from main branch (replaces GDAL with rasterio)
ANUGA_SRC="anuga @ git+https://github.com/Hydrata/anuga_core.git@main"
pip install '/tmp/run_anuga_src[sim]' mpi4py matplotlib pymetis scipy triangle netCDF4 "$ANUGA_SRC" 2>&1 | tail -5

# First, run the simulation to get .sww output
echo ""
echo "== Pre-requisite: running simulation to generate .sww =="
cp -r /app/examples/small_test /tmp/workdir/small_test

run-anuga run /tmp/workdir/small_test/ 2>&1 | tail -10
rc_run=$?

if [ $rc_run -ne 0 ]; then
    echo "    Simulation failed (exit $rc_run) — cannot test post-process"
    count_fail=1
    print_summary
    exit 1
fi

echo ""
echo "== Post-process tests =="

# Step 1: run-anuga post-process
test_step 1 "run-anuga post-process examples/small_test/" \
    "run-anuga post-process /tmp/workdir/small_test/"

# Step 2: Check for GeoTIFF outputs
if [ "$last_rc" -eq 0 ]; then
    test_step 2 "Check for *_depth_max.tif, *_velocity_max.tif" \
        "find /tmp/workdir/small_test -name '*_depth_max.tif' -o -name '*_velocity_max.tif'"

    # Also list all output files for the report
    echo ""
    echo "== All output files =="
    find /tmp/workdir/small_test/outputs_* -type f 2>/dev/null | sort || echo "(no outputs_* directory found)"
else
    echo "    Step 2: Check for GeoTIFFs — SKIPPED (post-process failed)"
fi

print_summary
