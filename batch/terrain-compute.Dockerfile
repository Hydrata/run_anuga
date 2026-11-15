# terrain-compute.Dockerfile — shared AWS Batch image for the Django-free
# terrain compute runner (TASK-1833, epic 1830 W1).
#
# Forked from batch/Dockerfile (the ANUGA simulation image) but stripped to the
# terrain-merge leaf set: it KEEPS python:3.12-bookworm + GDAL and ADDS
# rasterio/scipy/numpy/shapely/whitebox/networkx, and DROPS MPI
# (libopenmpi-dev/openmpi-bin/mpi4py), ANUGA, meshpy/pymetis, and netcdf4 —
# none of which the terrain merge touches. There is no mpirun in the
# entrypoint.
#
# ── CROSS-REPO BUILD CONTEXT (read this before `docker build`) ──────────────
# The runnable package — `gn_anuga.terrain_compute` (+ its only intra-repo
# import, `gn_anuga.terrain_assembly`) — lives in the *hydrata* repo
# (apps/gn_anuga/), but this Dockerfile lives in *run_anuga*. They are never in
# the same git tree. So this Dockerfile is NOT built against the run_anuga repo
# root: it expects a STAGED build context assembled by
# deploy/scripts/rebuild-terrain-compute-image.sh, which copies into a temp dir:
#
#     <ctx>/terrain-compute.Dockerfile            (this file)
#     <ctx>/terrain-compute-entrypoint.sh         (the entrypoint)
#     <ctx>/gn_anuga/__init__.py                  (minimal namespace shim)
#     <ctx>/gn_anuga/terrain_assembly.py          (from hydrata apps/gn_anuga/)
#     <ctx>/gn_anuga/terrain_compute/             (from hydrata apps/gn_anuga/)
#
# That minimal layout makes `python -m gn_anuga.terrain_compute` resolve inside
# the image WITHOUT pulling in the rest of the Django monolith. The COPY paths
# below are relative to that staged context, NOT to either git repo.
# ───────────────────────────────────────────────────────────────────────────

FROM python:3.12-bookworm

# System dependencies for the terrain merge (GDAL only — NO MPI, NO gfortran).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL config for the from-source rasterio/GDAL Python build below.
ENV GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

# Heavy pip-install layer FIRST (before the source COPY) so it stays cached
# when only the staged gn_anuga source changes. numpy + GDAL bindings pinned to
# the system GDAL, then the terrain-merge leaf set. Explicitly DROPPED vs the
# ANUGA image: mpi4py, meshpy, pymetis, netcdf4, anuga, matplotlib, gfortran.
RUN pip install --no-cache-dir "numpy>=2.0" setuptools wheel && \
    pip install --no-cache-dir --no-binary GDAL --no-build-isolation \
    "GDAL==$(gdal-config --version).*" && \
    pip install --no-cache-dir \
    rasterio scipy shapely "whitebox==2.3.6" networkx \
    pyproj affine requests boto3 awscli

# Air-gap WhiteboxTools: constructing WhiteboxTools() downloads the WBT binary
# into the whitebox package dir on first use. Doing it here BAKES the binary
# into the image so a Batch run (which has no outbound internet to the WBT
# download host) finds it already present. The WBT working dir is left at the
# container CWD; the entrypoint and hydro_enforce both run under writable /tmp,
# and hydro_enforce passes absolute dem/output paths + its own TemporaryDirectory.
# whitebox.WhiteboxTools().exe_path is the package DIR; the native binary is
# <exe_path>/<exe_name> (exe_name == "whitebox_tools"). Construct it, ensure the
# exec bit (WBT only chmod+x's it on first tool run, not on construction), then
# symlink it onto PATH so `whitebox_tools --version` resolves directly.
#
# TASK-4431 (epic 4274 W4.5): the wrapper is pinned above (whitebox==2.3.6, as in
# hydrata requirements/base.txt), and the PROGRAM is pinned here to the same build
# the web server's ansible task pins (ansible-geonode/tasks/whitebox-tools.yml):
# version and sha256 of the linux amd64 binary, read from the binary on the dev
# workstation 15-11-26 ("WhiteboxTools v2.4.0"). The wrapper downloads from an
# UNVERSIONED URL, so the build FAILS unless the download is that build: a new
# WhiteboxTools never enters the image silently. Each delineation result records
# the version it ran with (result['wbt_version'], gn_anuga/delineation/wbt.py).
# TODO(TASK-4431): no versioned download URL for v2.4.0 is recorded in the repos;
# when whiteboxgeo.com's "latest" moves past it, point this at a versioned archive.
ARG WBT_VERSION=2.4.0
ARG WBT_SHA256=96bb47a9d1f0fe8a3cb46df524242c990eab9bc2ae1930e540e09328331172e4
ENV WBT_WORKING_DIR=/tmp
RUN set -eu && \
    python -c "import whitebox; whitebox.WhiteboxTools()" && \
    WBT_BIN="$(python -c 'import os, whitebox; w=whitebox.WhiteboxTools(); print(os.path.join(w.exe_path, w.exe_name))')" && \
    test -f "$WBT_BIN" && \
    chmod +x "$WBT_BIN" && \
    echo "${WBT_SHA256}  ${WBT_BIN}" | sha256sum -c - && \
    ln -sf "$WBT_BIN" /usr/local/bin/whitebox_tools && \
    whitebox_tools --version | grep -q "^WhiteboxTools v${WBT_VERSION} " && \
    whitebox_tools --version

# Unbuffered stdout/stderr so CloudWatch shows every print/logger line live.
# Placed after the heavy layers so adding it does not bust their cache.
ENV PYTHONUNBUFFERED=1

# Entrypoint BEFORE the package COPY so its layer stays cached when only the
# gn_anuga source changes.
COPY terrain-compute-entrypoint.sh /app/terrain-compute-entrypoint.sh
RUN chmod +x /app/terrain-compute-entrypoint.sh

# Staged Django-free package (see CROSS-REPO BUILD CONTEXT above). PYTHONPATH=/app
# (NOT just WORKDIR) puts the package on sys.path regardless of CWD — the
# entrypoint cd's to /tmp/terrain_compute before running `python -m
# gn_anuga.terrain_compute`, so relying on CWD-on-sys.path alone would
# ModuleNotFoundError. NO `pip install` of the monolith — only this leaf.
ENV PYTHONPATH=/app
COPY gn_anuga /app/gn_anuga

ENTRYPOINT ["/app/terrain-compute-entrypoint.sh"]
