# ONE Dockerfile for the ANUGA Batch image family — cpu + gpu targets
# (TASK-2671, epic 2662 W2.1; supersedes batch/Dockerfile.gpu, DELETED).
#
#   docker build --target cpu ...   -> the 4-vCPU / x32 CPU image (no CUDA)
#   docker build --target gpu ...   -> the A10G/L40S GPU image (CUDA base)
#
# STRUCTURE (D8: same-code guarantee): a shared `src` stage holds the ONE
# pinned checkout set — anuga_core cloned at ANUGA_CORE_SHA, plus the
# run_anuga tree + gn_anuga.batch_common overlay (telemetry lib) + entrypoint
# from the build context. BOTH targets consume their engine + dispatch code
# EXCLUSIVELY via `COPY --from=src`, so the cpu and gpu images cannot bake
# different {run_anuga, anuga_core, hydrata-leaves} code by construction.
#
# WHY (the mislabel trap this kills, memory
# reference-gpu-batch-image-engine-from-separate-clone): the OLD gpu image
# (batch/Dockerfile.gpu) inherited its ENGINE from a SEPARATE clone baked
# into the anuga-gpu-dev BASE image, so retagging/rebuilding the batch layer
# without a base rebuild silently shipped whatever engine the base happened
# to bake, mislabeled with fresh-looking tags. Here the gpu target RECOMPILES
# the engine from the shared `src` checkout on every build — the base
# supplies only the toolchain (nvhpc/CUDA/venv), never the engine.
#
# The three SHA LABELs are declared per-target from the SAME global ARGs
# (LABELs cannot be inherited across different FROM bases); the
# assert-image-sha-lock script (batch/assert-image-sha-lock.sh) compares
# them across both built targets, and CI lints the single-pin structure.
#
# Build via deploy/scripts/rebuild-batch-image.sh (stages the gn_anuga
# overlay + passes the SHA args/labels). A plain `docker build --target cpu .`
# still works: the overlay COPY degrades to an empty gn_anuga/ package and
# the provenance manifest stays "{}" (provenance-incomplete, never a failure).
#
# LOCAL BUILD + RUN (D8 portability — no AWS needed):
#   docker build --target cpu -t anuga-sim:local -f batch/Dockerfile .
#   docker run --rm --network host \
#     -v /path/to/package:/tmp/simulation \
#     -e CONTROL_SERVER=http://localhost:8000/ \
#     -e HYDRATA_INTERNAL_COMPUTE_TOKEN=<token> \
#     -e HYDRATA_PROCESS_ID=<process uuid>   \
#     -e RESULT_S3_BUCKET=<bucket> -e PROJECT_ID=.. -e SCENARIO_ID=.. -e RUN_ID=.. \
#     --entrypoint python anuga-sim:local -m run_anuga.cli run-and-report /tmp/simulation
#   (the stock ENTRYPOINT instead downloads the package itself from
#   PACKAGE_S3_BUCKET/KEY — use it when S3 credentials are provided;
#   HYDRATA_PROCESS_ID arms the epic-2662 events dialect, absent = legacy.)

# ---- the ONE engine pin, shared by BOTH targets ---------------------------
# There is deliberately NO second pin anywhere in this file; both targets
# move together.
# Pin 7f1a4847 = Hydrata/anuga_core fork main, the TASK-2652 881-commit
# upstream-develop merge — the SAME engine the live GPU image bakes
# (TASK-2673 engine unification: cpu joins gpu on fork main; towradgi
# parity gate on the cpu build is the acceptance evidence). Prev: 8abd3aef
# (fork main @ upstream anuga-community/main 2d3696ea, epic 2147 W1.1).
ARG ANUGA_CORE_SHA=7f1a4847df34cd2527101d4e10dbbf0b81668e23

# Provenance ARGs (stamped as LABELs in both targets; passed by
# rebuild-batch-image.sh / CI — "unknown" on a plain local build).
ARG RUN_ANUGA_SHA=unknown
ARG HYDRATA_SHA=unknown

# GPU toolchain base (pre-built locally by deploy scripts/gpu-dev/
# build-gpu-dev-image.sh --multi-arch; supplies nvhpc + CUDA + venv +
# spatial deps — its baked engine is REPLACED below, see gpu stage).
ARG GPU_BASE=anuga-gpu-dev:multiarch

# ===========================================================================
# src — the single pinned checkout set both targets consume
# ===========================================================================
FROM python:3.12-bookworm AS src
ARG ANUGA_CORE_SHA
# anuga_core: clone the fork at the ONE pinned SHA. Hydrata/anuga_core is a
# public fork, so no credentials are needed in-build (run_anuga + the
# gn_anuga overlay are private and therefore arrive via the build context,
# staged by rebuild-batch-image.sh from clean, origin-merged trees).
RUN git clone https://github.com/Hydrata/anuga_core.git /src/anuga_core \
    && git -C /src/anuga_core checkout --detach "${ANUGA_CORE_SHA}" \
    && rm -rf /src/anuga_core/.git
# run_anuga tree (+ batch/entrypoint.sh + the gn_anuga overlay when staged).
COPY . /src/run_anuga
# Plain-build degrade: guarantee gn_anuga/ exists so the targets' COPY
# --from=src never fails on an un-staged context (empty package -> the
# guarded `from gn_anuga.batch_common...` imports simply miss, as today).
RUN mkdir -p /src/run_anuga/gn_anuga

# ===========================================================================
# cpu — python:3.12-bookworm + gcc-built engine (no CUDA)
# ===========================================================================
FROM python:3.12-bookworm AS cpu

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

# Install GDAL Python bindings matching system GDAL. meson/ninja/meson-python
# join the build toolchain here because anuga_core builds via the mesonpy
# backend with --no-build-isolation (mirrors the gpu-dev base).
RUN pip install --no-cache-dir "numpy>=2.0" setuptools wheel cython pybind11 \
        meson ninja meson-python && \
    pip install --no-cache-dir --no-binary GDAL --no-build-isolation \
    "GDAL==$(gdal-config --version).*"

# ANUGA runtime dependencies (not declared in anuga's pyproject).
# TASK-1261: meshpy, pymetis, GDAL are pinned to the versions proven to work
# on the sandbox/prod venv (meshpy==2025.1.1, pymetis==2025.2.2, GDAL==3.8.4).
# Bump intentionally — floating these versions risks breaking the W2.4 golden
# mesh determinism guarantee.
# fiona: anuga.utilities.spatialInputUtil gates readListOfRiverWalls /
# readListOfBreakLines behind `try: import fiona, rasterio, shapely` — the
# gpu-dev base installs the trio explicitly; rasterio/shapely arrive via
# run_anuga[sim] below, fiona must be explicit (TASK-2671 unification: the
# cpu image must run riverwall scenarios identically to the gpu image).
RUN pip install --no-cache-dir \
    scipy matplotlib "meshpy==2025.1.1" "pymetis==2025.2.2" netcdf4 dill utm pyproj affine \
    mpi4py requests boto3 awscli fiona "zarr>=3.0"

# ---- the engine, from the SHARED checkout (never a second clone/pin) ------
COPY --from=src /src/anuga_core /app/anuga_core
RUN pip install --no-cache-dir --no-build-isolation /app/anuga_core

# Unbuffered stdout/stderr so CloudWatch shows every print/logger output as
# it happens, not in 4KB chunks.
ENV PYTHONUNBUFFERED=1

# entrypoint.sh BEFORE the source-tree COPY so its layer stays cached when
# only run_anuga/*.py changes.
COPY --from=src /src/run_anuga/batch/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Staged Django-free gn_anuga.batch_common leaf (TASK-1846, epic 1830 W4;
# since epic 2662 it also carries the telemetry client + protocol).
# PYTHONPATH=/app puts /app/gn_anuga on sys.path so the guarded
# `from gn_anuga.batch_common...` imports resolve in-container.
ENV PYTHONPATH=/app
COPY --from=src /src/run_anuga/gn_anuga /app/gn_anuga

# Source tree + local-package install. The heavy layers above stay cached
# across source-only changes; only these layers re-run.
COPY --from=src /src/run_anuga /app/run_anuga_src
RUN pip install --no-cache-dir "/app/run_anuga_src[sim,platform,playback]"

# --- Baked build-provenance manifest (epic 2280 W3.1 / TASK-2286) -----------
# The CONTAINER reads THIS FILE at runtime to self-report the exact cross-repo
# code it ran (run_anuga._handoff). A running container CANNOT read its own
# docker LABELs, so the manifest FILE — not the labels — is the container's
# source of truth (§1); the labels feed the deploy gate + sha-lock instead.
# Passed as a build-arg by deploy/scripts/rebuild-batch-image.sh; the default
# "{}" keeps a plain `docker build .` working (provenance marked incomplete,
# NEVER a build failure).
ARG BUILD_PROVENANCE_JSON="{}"
RUN mkdir -p /opt/hydrata \
    && printf '%s' "$BUILD_PROVENANCE_JSON" > /opt/hydrata/BUILD_PROVENANCE.json

# SHA-lock labels (TASK-2671): declared from the SAME global ARGs as the gpu
# target; assert-image-sha-lock.sh compares these across both built targets.
ARG ANUGA_CORE_SHA
ARG RUN_ANUGA_SHA
ARG HYDRATA_SHA
LABEL org.hydrata.anuga_core_sha=${ANUGA_CORE_SHA} \
      org.hydrata.run_anuga_sha=${RUN_ANUGA_SHA} \
      org.hydrata.gn_anuga_sha=${HYDRATA_SHA} \
      org.hydrata.image_target=cpu

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]

# ===========================================================================
# gpu — nvhpc/CUDA base + nvc-recompiled engine from the SAME src checkout
# ===========================================================================
FROM ${GPU_BASE} AS gpu

# curl: entrypoint.sh's terminal /error/ POST safety net (absent from the
# nvhpc base). awscli: the entrypoint's S3 package/result ops.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir awscli

# Unbuffered stdout/stderr so CloudWatch shows every line live (the base sets
# this too — repeated here so a base change can't silently drop it).
ENV PYTHONUNBUFFERED=1

# ---- the engine, RECOMPILED from the SHARED checkout ----------------------
# This is the structural kill of the mislabel trap: the base's baked engine
# (editable /app/anuga_core-gpu, label org.hydrata.anuga_gpu_sha) is
# REPLACED by an nvc + OpenMP-target-offload build of the SAME
# /src/anuga_core checkout the cpu target installs. The base supplies only
# the toolchain. Which GPU arch(es): cc70/75/86/89 (V100/T4/Ampere/Ada) —
# same list the multiarch base used.
ARG GPU_ARCH=cc70,cc75,cc86,cc89
ARG GPU_AWARE_MPI=false
COPY --from=src /src/anuga_core /app/anuga_core
RUN cd /app/anuga_core && rm -rf build && \
    CC=nvc CXX=nvc++ pip install --no-cache-dir --no-build-isolation /app/anuga_core \
      -Csetup-args=-Dgpu_offload=true \
      -Csetup-args=-Dgpu_arch=${GPU_ARCH} \
      -Csetup-args=-Dgpu_aware_mpi=${GPU_AWARE_MPI} \
      -Csetup-args=-Duse_nvtx=true \
    && rm -rf /app/anuga_core-gpu
# Baked fail-loud detector for the silent-CPU-engine class: a gpu-target
# image whose engine was compiled WITHOUT offload must never leave the
# build. The check is DEVICE-INDEPENDENT on purpose — an offload-compiled
# sw_domain_gpu_ext links the NVIDIA HPC offload runtime
# (libacchost/libaccdevice), a CPU_ONLY_MODE build links libgomp only.
# NEVER use anuga's gpu_offload_supported() here: it also requires a
# physical GPU device (omp_get_num_devices()>0), which no docker build has —
# it would fail EVERY build, including perfect ones (TASK-2671 bring-up
# false-positive). The runtime device probe belongs to run-towradgi-gate.sh.
RUN SO=$(python -c "from anuga.shallow_water import sw_domain_gpu_ext as m; print(m.__file__)") \
    && ldd "$SO" | grep -qE 'libacc(host|device)' \
    && echo "gpu offload runtime linkage OK: $SO"

# entrypoint.sh BEFORE the source-tree COPY so its layer stays cached when
# only run_anuga/*.py changes.
COPY --from=src /src/run_anuga/batch/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Staged Django-free gn_anuga.batch_common leaf (same overlay contract as the
# cpu target).
ENV PYTHONPATH=/app
COPY --from=src /src/run_anuga/gn_anuga /app/gn_anuga

# Source tree + local-package install (into the base's /opt/venv).
COPY --from=src /src/run_anuga /app/run_anuga_src
RUN pip install --no-cache-dir "/app/run_anuga_src[sim,platform]"

# --- Baked build-provenance manifest (same contract as the cpu target) -----
ARG BUILD_PROVENANCE_JSON="{}"
RUN mkdir -p /opt/hydrata \
    && printf '%s' "$BUILD_PROVENANCE_JSON" > /opt/hydrata/BUILD_PROVENANCE.json

# SHA-lock labels — the SAME global ARGs as the cpu target. NOTE
# org.hydrata.anuga_gpu_sha (inherited from the base) is OVERRIDDEN to the
# recompiled engine's SHA: the base's baked engine no longer exists in this
# image, so its label must not survive to mislabel the engine.
ARG ANUGA_CORE_SHA
ARG RUN_ANUGA_SHA
ARG HYDRATA_SHA
LABEL org.hydrata.anuga_core_sha=${ANUGA_CORE_SHA} \
      org.hydrata.anuga_gpu_sha=${ANUGA_CORE_SHA} \
      org.hydrata.run_anuga_sha=${RUN_ANUGA_SHA} \
      org.hydrata.gn_anuga_sha=${HYDRATA_SHA} \
      org.hydrata.image_target=gpu

WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
