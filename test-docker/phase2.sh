#!/usr/bin/env bash
# Phase 2: Sim Install + Run
# Tests: [sim] install (with and without sys deps), anuga install, run simulation

echo "=== Phase 2: Sim Install + Run ==="
echo ""

# Copy source to writable location (bind mount is :ro)
cp -r /app /tmp/run_anuga_src

# --- 2a: Try README commands as-is (expect failures) ---
echo "== 2a: README commands verbatim (no system deps) =="

# Step 1: pip install "run_anuga[sim]" — README command, no sys deps
test_step 1 "pip install run_anuga[sim] (README verbatim, no sys deps)" \
    "pip install '/tmp/run_anuga_src[sim]' 2>&1 | tail -20"

if [ "$last_rc" -ne 0 ]; then
    echo ""
    echo "    FINDING: pip install run_anuga[sim] fails without system deps"
    echo "    README does not mention required system packages."
fi

# --- 2b: Install system deps (not in README) and retry ---
echo ""
echo "== 2b: With system deps (undocumented prerequisite) =="

test_step 2 "Install system deps (NOT in README)" \
    "apt-get update -qq && apt-get install -y -qq build-essential gfortran libopenmpi-dev openmpi-bin gdal-bin libgdal-dev libproj-dev libgeos-dev libhdf5-dev libnetcdf-dev 2>&1 | tail -3"

export GDAL_CONFIG=/usr/bin/gdal-config

# Step 3: Try again — still fails due to GDAL version mismatch
test_step 3 "pip install run_anuga[sim] (with sys deps, unpinned GDAL)" \
    "pip install '/tmp/run_anuga_src[sim]' 2>&1 | tail -10"

if [ "$last_rc" -ne 0 ]; then
    echo ""
    echo "    FINDING: GDAL version mismatch"
    echo "    Debian bookworm ships libgdal $(gdal-config --version)"
    echo "    pip pulls latest GDAL Python bindings which requires matching libgdal"
    echo "    Fix: pip install GDAL==\$(gdal-config --version) BEFORE run_anuga[sim]"
fi

# Step 4: Install numpy first, then build GDAL from source, then run_anuga[sim]
GDAL_VER=$(gdal-config --version)
test_step 4 "pip install numpy + GDAL==$GDAL_VER + run_anuga[sim]" \
    "pip install numpy setuptools && pip install --no-build-isolation --no-binary GDAL GDAL==$GDAL_VER && pip install '/tmp/run_anuga_src[sim]' 2>&1 | tail -10"

# Step 5: pip install anuga + matplotlib
# README says "run-anuga run ... (requires [sim] + anuga)"
# But anuga unconditionally imports matplotlib in __init__.py
test_step 5 "pip install anuga + deps (anuga needs matplotlib + pymetis)" \
    "pip install mpi4py matplotlib pymetis scipy triangle netCDF4 anuga 2>&1 | tail -10"

if [ "$last_rc" -ne 0 ]; then
    echo ""
    echo "    FINDING: anuga install failed"
fi

# Step 6: Verify anuga imports correctly
test_step 6 "python -c 'import anuga'" \
    "python -c 'import anuga; print(\"anuga\", anuga.__version__)'"

# Step 7: run-anuga run
# Copy example to writable location
cp -r /app/examples/australian_floodplain /tmp/workdir/australian_floodplain

test_step 7 "run-anuga run examples/australian_floodplain/" \
    "cd /tmp/workdir && run-anuga run /tmp/workdir/australian_floodplain/"

# Step 8: Check output files
if [ "$last_rc" -eq 0 ]; then
    test_step 8 "Check for output files (.sww, .tif)" \
        "find /tmp/workdir/australian_floodplain -name '*.sww' -o -name '*.tif' -not -name 'dem.tif'"

    echo ""
    echo "== Output directory listing =="
    find /tmp/workdir/australian_floodplain/outputs_* -type f 2>/dev/null | sort || echo "(no outputs_* directory found)"
else
    echo "    Step 8: Check for output files — SKIPPED (run failed)"
fi

print_summary
