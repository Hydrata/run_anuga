FROM python:3.12-bookworm

# System dependencies for ANUGA (C extensions, MPI, GDAL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenmpi-dev \
    openmpi-bin \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL config for pip install
ENV GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

# Copy only pyproject.toml first so the heavy pip-install layers below remain
# cached when only run_anuga source files change.
COPY pyproject.toml /app/run_anuga_src/

# Install GDAL Python bindings matching system GDAL
RUN pip install --no-cache-dir "numpy>=2.0" setuptools wheel && \
    pip install --no-cache-dir --no-binary GDAL --no-build-isolation \
    "GDAL==$(gdal-config --version).*"

# Install ANUGA runtime dependencies (not declared in pyproject.toml) then
# ANUGA from Hydrata fork (has lazy pymetis fix). Pinned to a specific
# anuga_core SHA for reproducible builds; bump intentionally via operator
# workflow, do NOT use @main.
# Pin 8abd3aef = fork main synced to upstream anuga-community/main @2d3696ea
# (epic 2147 W1.1 / TASK-2152; PR #5). Prev: b09d47bb.
#
# TASK-1261: meshpy, pymetis, GDAL are pinned to the versions proven to work
# on the sandbox/prod venv (meshpy==2025.1.1, pymetis==2025.2.2, GDAL==3.8.4).
# Bump intentionally — floating these versions risks breaking the W2.4 golden
# mesh determinism guarantee.
RUN pip install --no-cache-dir \
    scipy matplotlib "meshpy==2025.1.1" "pymetis==2025.2.2" netcdf4 dill utm pyproj affine \
    mpi4py cython pybind11 requests boto3 awscli && \
    pip install --no-cache-dir \
    "anuga @ git+https://github.com/Hydrata/anuga_core.git@8abd3aefcc6df808372043a7ce1fbe523f5abbf3"

# Unbuffered stdout/stderr so CloudWatch shows every print/logger output as
# it happens, not in 4KB chunks. Placed after the heavy pip-install layers so
# adding this ENV doesn't bust their cache.
ENV PYTHONUNBUFFERED=1

# entrypoint.sh BEFORE the source-tree COPY so its layer stays cached when
# only run_anuga/*.py changes.
COPY batch/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Staged Django-free gn_anuga.batch_common leaf (TASK-1846, epic 1830 W4).
# deploy/scripts/rebuild-batch-image.sh --tool anuga OVERLAYS gn_anuga/ (a minimal
# namespace shim + the batch_common/ leaf) into the run_anuga build context, the
# same minimal way the terrain-compute image stages it. PYTHONPATH=/app (set
# below) puts /app/gn_anuga on sys.path so run_anuga._handoff's guarded
# `from gn_anuga.batch_common.resource_sampler import ResourceSampler` resolves
# in-container and emits the resource_summary ledger. COPYed BEFORE the run_anuga
# source so this layer stays cached when only run_anuga/*.py changes. On a build
# context WITHOUT the overlay (e.g. a plain `docker build .`) this COPY no-ops
# absent gn_anuga/ — but the unified rebuild script always stages it.
ENV PYTHONPATH=/app
COPY gn_anuga /app/gn_anuga

# Source tree + local-package install. The heavy layers above stay cached
# across source-only changes; only these two layers re-run.
COPY . /app/run_anuga_src/
RUN pip install --no-cache-dir "/app/run_anuga_src[sim,platform]"

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
