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
    && rm -rf /var/lib/apt/lists/*

# Set GDAL config for pip install
ENV GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

# Copy the run_anuga package
COPY . /app/run_anuga_src

# Install run_anuga[sim] + anuga from PyPI (clean pip install test)
RUN pip install --no-cache-dir "anuga>=3.2" mpi4py && \
    pip install --no-cache-dir "/app/run_anuga_src[sim,dev]"

# Copy the test script
COPY test-docker/run_test.py /app/run_test.py

WORKDIR /app
CMD ["python", "run_test.py"]
