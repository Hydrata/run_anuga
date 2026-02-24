#!/usr/bin/env bash
# Phase 4: MPI Parallel Run
# Tests: simulation with 2 MPI processes, verifies merged output

echo "=== Phase 4: MPI Parallel Run ==="
echo ""

# Copy source to writable location (bind mount is :ro)
cp -r /app /tmp/run_anuga_src

# Install everything needed (system deps + sim + anuga)
echo "== Setting up: system deps + run_anuga[sim] + anuga =="

apt-get update -qq && apt-get install -y -qq \
    build-essential gfortran libopenmpi-dev openmpi-bin \
    libhdf5-dev libnetcdf-dev 2>&1 | tail -3

ANUGA_SRC="anuga @ git+https://github.com/Hydrata/anuga_core.git@main"
pip install '/tmp/run_anuga_src[sim]' mpi4py matplotlib pymetis scipy triangle netCDF4 "$ANUGA_SRC" 2>&1 | tail -5

# Copy example to writable location
cp -r /app/examples/small_test /tmp/workdir/small_test_mpi

echo ""
echo "== MPI parallel run tests =="

# Step 1: Run simulation with 2 MPI processes
test_step 1 "mpirun -np 2 run-anuga run (parallel)" \
    "mpirun -np 2 --allow-run-as-root run-anuga run /tmp/workdir/small_test_mpi/"

# Step 2: Verify merged SWW output
if [ "$last_rc" -eq 0 ]; then
    test_step 2 "Check for merged .sww file" \
        "find /tmp/workdir/small_test_mpi -name '*.sww' | head -1"

    # Step 3: Verify GeoTIFF outputs from parallel run
    test_step 3 "Check for *_depth_max.tif from parallel run" \
        "find /tmp/workdir/small_test_mpi -name '*_depth_max.tif'"

    echo ""
    echo "== MPI output files =="
    find /tmp/workdir/small_test_mpi/outputs_* -type f 2>/dev/null | sort || echo "(no outputs_* directory found)"
else
    echo "    Steps 2-3: SKIPPED (MPI run failed)"
fi

print_summary
