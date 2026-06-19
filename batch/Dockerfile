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
#
# TASK-1261: meshpy, pymetis, GDAL are pinned to the versions proven to work
# on the sandbox/prod venv (meshpy==2025.1.1, pymetis==2025.2.2, GDAL==3.8.4).
# Bump intentionally — floating these versions risks breaking the W2.4 golden
# mesh determinism guarantee.
RUN pip install --no-cache-dir \
    scipy matplotlib "meshpy==2025.1.1" "pymetis==2025.2.2" netcdf4 dill utm pyproj affine \
    mpi4py cython pybind11 requests boto3 awscli && \
    pip install --no-cache-dir \
    "anuga @ git+https://github.com/Hydrata/anuga_core.git@4ae0ec0de652ce0af34c185f59b7809a594fded9"

# Unbuffered stdout/stderr so CloudWatch shows every print/logger output as
# it happens, not in 4KB chunks. Placed after the heavy pip-install layers so
# adding this ENV doesn't bust their cache.
ENV PYTHONUNBUFFERED=1

# entrypoint.sh BEFORE the source-tree COPY so its layer stays cached when
# only run_anuga/*.py changes.
COPY batch/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Source tree + local-package install. The heavy layers above stay cached
# across source-only changes; only these two layers re-run.
COPY . /app/run_anuga_src/
RUN pip install --no-cache-dir "/app/run_anuga_src[sim,platform]"

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
