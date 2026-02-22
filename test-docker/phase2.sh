#!/usr/bin/env bash
# Phase 2: Sim Install + Run
# Tests: [sim] install (pure pip, no system geo deps), anuga install, run simulation

echo "=== Phase 2: Sim Install + Run ==="
echo ""

# Copy source to writable location (bind mount is :ro)
cp -r /app /tmp/run_anuga_src

# --- 2a: Install system deps (MPI + HDF5/NetCDF for anuga) ---
echo "== 2a: Install system deps (MPI for mpi4py, HDF5/NetCDF for netCDF4) =="

test_step 1 "Install system deps (MPI + build tools)" \
    "apt-get update -qq && apt-get install -y -qq build-essential gfortran libopenmpi-dev openmpi-bin libhdf5-dev libnetcdf-dev 2>&1 | tail -3"

# --- 2b: pip install run_anuga[sim] — all geo deps from binary wheels ---
echo ""
echo "== 2b: pip install run_anuga[sim] (no system GDAL needed) =="

test_step 2 "pip install run_anuga[sim]" \
    "pip install '/tmp/run_anuga_src[sim]' 2>&1 | tail -10"

# Step 3: pip install anuga from main branch + its undeclared deps
# Use our fork which replaces GDAL with rasterio
ANUGA_SRC="anuga @ git+https://github.com/Hydrata/anuga_core.git@main"
test_step 3 "pip install anuga (main branch) + deps" \
    "pip install mpi4py matplotlib pymetis scipy triangle netCDF4 '$ANUGA_SRC' 2>&1 | tail -10"

if [ "$last_rc" -ne 0 ]; then
    echo ""
    echo "    FINDING: anuga install failed"
fi

# Step 4: Verify anuga imports correctly
test_step 4 "python -c 'import anuga'" \
    "python -c 'import anuga; print(\"anuga\", anuga.__version__)'"

# Step 5: Verify no GDAL/osgeo in the dependency tree
test_step 5 "Verify no osgeo/GDAL installed" \
    "python -c 'import importlib; importlib.import_module(\"osgeo\")' 2>&1 && echo 'UNEXPECTED: osgeo found' && exit 1 || echo 'Confirmed: osgeo not installed (expected)'"

# Step 6: run-anuga run
# Copy example to writable location
cp -r /app/examples/small_test /tmp/workdir/small_test

test_step 6 "run-anuga run examples/small_test/" \
    "cd /tmp/workdir && run-anuga run /tmp/workdir/small_test/"

# Step 7: Check output files
if [ "$last_rc" -eq 0 ]; then
    test_step 7 "Check for output files (.sww, .tif)" \
        "find /tmp/workdir/small_test -name '*.sww' -o -name '*.tif' -not -name 'dem.tif'"

    echo ""
    echo "== Output directory listing =="
    find /tmp/workdir/small_test/outputs_* -type f 2>/dev/null | sort || echo "(no outputs_* directory found)"
else
    echo "    Step 7: Check for output files — SKIPPED (run failed)"
fi

print_summary
