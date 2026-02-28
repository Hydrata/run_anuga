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

# Copy the run_anuga package
COPY . /app/run_anuga_src

# Install GDAL Python bindings matching system GDAL
RUN pip install --no-cache-dir "numpy>=2.0" setuptools wheel && \
    pip install --no-cache-dir --no-binary GDAL --no-build-isolation \
    "GDAL==$(gdal-config --version).*"

# Install ANUGA runtime dependencies (not declared in pyproject.toml)
# then ANUGA from Hydrata fork (has lazy pymetis fix)
RUN pip install --no-cache-dir \
    scipy matplotlib meshpy pymetis netcdf4 dill utm pyproj affine \
    mpi4py cython pybind11 && \
    pip install --no-cache-dir \
    "anuga @ git+https://github.com/Hydrata/anuga_core.git@main" && \
    pip install --no-cache-dir "/app/run_anuga_src[sim,platform]"

# Copy the entrypoint script
COPY batch/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
