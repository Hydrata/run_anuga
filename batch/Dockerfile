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
# SECOND STRUCTURAL GUARANTEE (TASK-2700): both targets set ANUGA_VERSION from
# the pin (so the engine self-reports `0.0.0+g<sha>` instead of `0.0.0+unknown`)
# and then run batch/assert-anuga-provenance.py after their LAST pip install —
# a build-time proof, via PEP 610 direct_url.json, that the importable anuga is
# still the one built from /app/anuga_core and not a PyPI wheel pip substituted
# in to satisfy some later `anuga>=` constraint.
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
ARG ANUGA_CORE_SHA=93368189eb35eef847094dde027d6d6dcb2b0059

# ---- BARE-TAG RULE: a `v`-prefixed anuga_core tag is a SILENT no-op --------
# (TASK-2700. Recorded here and in deploy's install-anuga.yml pin comment —
# the two places a person deciding to tag the fork will actually be looking.)
#
# If Hydrata/anuga_core is ever TAGGED for release, the tag MUST be BARE
# NUMERIC — `3.3.11`, never `v3.3.11`. anuga_core is `dynamic = ['version']`:
# meson.build calls _git_version.py, which maps `git describe` output to PEP 440
# with `^(\d+\.\d+(?:\.\d+)?(?:[a-z]\w*)?)...` — the base tag must START WITH A
# DIGIT. A leading `v` fails that regex and falls through to the
# no-reachable-tag branch, so `git tag v3.3.11` does NOT raise the version; the
# engine keeps reporting 0.0.0. The tagger sees a tag, the fleet sees nothing.
#
# The trap is armed RIGHT NOW: the fork's tags are 3.1.1 3.1.3 3.1.4 3.1.5
# v3.2.0-wheels.8 v3.2.0-wheels.9 — the two NEWEST are v-prefixed, and
# `git describe` always picks the newest reachable tag. Measured on a deployed
# web box 2026-08-10: describe -> `v3.2.0-wheels.9-1221-g7f1a4847`, __version__ ->
# `0.0.0+gv3.2.0-wheels.9-1221-g7f1a4847`. anuga_core's own meson.build comment
# ("no `v` prefix, matching the existing tag convention") is ASPIRATIONAL, not
# descriptive — the existing convention contains v tags.
#
# Nothing needs a tag today: TASK-2695 removed the `anuga>=3.3.0` floor from the
# web (and this file has never carried one), so tagging is optional hygiene, not
# a fix. A CORRECT bare-numeric tag is safe for both provenance gates below —
# they key on the `+g<sha>` local segment, which survives as `3.3.11.dev4+g<sha>`.

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
# TASK-2863: cython was pinned <3.3 here (3.3.0 rejected the fork's pyx
# dict-view assignments, sparse_matrix_ext.pyx:73-74) until the engine pin
# moved past the fork-side fix. ANUGA_CORE_SHA above is now 93368189
# (TASK-2758's sync of upstream tag 4.0.0 into the fork's main, baked in
# PR #19), which carries the fix — verified with
# `git -C anuga_core grep -nE 'cdef (list|dict) [a-zA-Z_]+ *= *[a-zA-Z_.]+\.(items|keys|values)\(\)' origin/main -- '*.pyx'`
# returning no matches — so the cap is dropped here (TASK-2863 AC4).
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
# ANUGA_VERSION (TASK-2700): the src stage deletes .git, so `git describe` fails
# in-build and the engine used to self-report the literal `0.0.0+unknown` — an
# image that could not say WHICH anuga it carried. _git_version.py has always
# supported an ANUGA_VERSION override for exactly this case ("a Docker build
# whose context excludes .git"); nobody used it. Feed it the ONE pin.
# The `0.0.0+g<sha>` shape is _git_version.py's OWN no-reachable-tag mapping,
# and is the dialect deploy's roles/ansible-geonode/tasks/anuga-engine-verify.yml
# already parses (prefix-tolerant `g<sha>` tokens) — so the web box and the
# container report engine provenance identically. The 0.0.0 base is correct
# rather than a placeholder: the fork has no bare-numeric reachable tag (see the
# BARE-TAG RULE at the pin above).
ARG ANUGA_CORE_SHA
COPY --from=src /src/anuga_core /app/anuga_core
RUN ANUGA_VERSION="0.0.0+g${ANUGA_CORE_SHA}" \
    pip install --no-cache-dir --no-build-isolation /app/anuga_core

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

# ---- playback-capability guard (TASK-2749) --------------------------------
# Symmetric with the gpu target's guard. The cpu target has always carried the
# playback extra, but the two stages must stay symmetric or the next
# unification can drop one silently — which is precisely how TASK-2749
# happened. See the gpu stage for the full rationale.
RUN python -c "import run_anuga.playback_store as p; assert p.zarr_available(), \
'TASK-2749: playback extra missing from this image — export_playback_store() would silently skip'"

# ---- substitution guard (TASK-2700) ---------------------------------------
# KEEP THIS AFTER THE LAST pip install OF THE STAGE. It asserts (via PEP 610
# direct_url.json, NOT a version compare) that the anuga this image will import
# is still the one built from /app/anuga_core above. Its entire purpose is to
# turn a LATER pip line silently swapping in PyPI anuga — one `[full]` away —
# into a red build. Rationale in the script's docstring.
RUN python /app/run_anuga_src/batch/assert-anuga-provenance.py \
        /app/anuga_core "${ANUGA_CORE_SHA}"

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
# (ANUGA_CORE_SHA is already in ARG scope from the engine build above — kept
# here so the LABEL's inputs stay readable at the point of use.)
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
# the toolchain.
#
# GPU_ARCH — WIDENED 2026-09-10 (TASK-2758 AC1/AC5), and the widening is NOT
# cosmetic. Until upstream 5b7dc93f this flag was INERT: nvc takes device
# architectures from `-gpu=` on the LINK line and `gpu_link_args` was empty,
# so the built .so carried whatever CUDA 12.9 in the nvhpc base chose —
# MEASURED on the live image as 8 architectures (sm_70 75 80 86 89 90 100 120)
# and ZERO PTX. 5b7dc93f arrives with the anuga_core 4.0.0 sync and makes the
# flag real and EXCLUSIVE, so the old 4-entry list would have SILENTLY dropped
# sm_80/90/100/120 from the rebuilt image — a strictly worse artefact than the
# one it replaces, with no PTX to JIT from. This list is an exact superset of
# the measured baseline. nvc 25.5 / CUDA 12.9 accepts 35 50 60 61 70 75 80 86
# 89 90 100 120 (`nvc -gpu=help`); the four oldest are omitted deliberately —
# they were never in the baseline and only add build time.
#
# PTX (TASK-2758 AC6) — RECORDED DECISION, not silence: nvc offers NO keyword
# that embeds PTX. `-gpu=` exposes only SASS arch selection plus `ptxinfo`
# (print-only). So SASS-only is forced by the toolchain, and forward
# compatibility is bought with architecture BREADTH instead of JIT. That is
# why the superset above is load-bearing rather than tidy.
ARG GPU_ARCH=cc70,cc75,cc80,cc86,cc89,cc90,cc100,cc120
ARG GPU_AWARE_MPI=false
# ANUGA_VERSION: same override, same `0.0.0+g<sha>` dialect as the cpu target
# (TASK-2700 — see the note there). MEASURED, not assumed: the gpu target's
# engine really does come from the shared `src` stage, not the GPU_BASE image.
# The live gpu image's dist-info records direct_url.json url
# "file:///app/anuga_core" and the base's baked /app/anuga_core-gpu is gone —
# so this line, and the guard below, both bite on the gpu target.
ARG ANUGA_CORE_SHA
COPY --from=src /src/anuga_core /app/anuga_core
RUN cd /app/anuga_core && rm -rf build && \
    ANUGA_VERSION="0.0.0+g${ANUGA_CORE_SHA}" \
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
# NOTE the extras MUST match the cpu target's line exactly. TASK-2749: this
# line read [sim,platform] from the TASK-2671 unification (2026-08-07) until
# 2026-08-11, so every GPU run silently produced NO playback store —
# export_playback_store() degrades to {"status": "skipped_no_zarr"} and the
# simulation still SUCCEEDS, so the only trace was a CloudWatch warning while
# the customer just saw a completed run with nothing to play back. Both GPU
# targets are user-selectable on hydrata.com (anuga_compute_targets carries
# batch-gpu-a10g + batch-gpu-l40s), so this was reachable in production.
COPY --from=src /src/run_anuga /app/run_anuga_src
RUN pip install --no-cache-dir "/app/run_anuga_src[sim,platform,playback]"

# ---- playback-capability guard (TASK-2749) --------------------------------
# KEEP THIS AFTER THE LAST pip install OF THE STAGE, same placement rule as the
# substitution guard below. Probes the capability through the runtime's OWN
# predicate (run_anuga.playback_store.zarr_available), not a bare `import
# zarr`, because that predicate is what export_playback_store() actually
# branches on — so this fails for exactly the reason the runtime would skip.
# Its purpose is to turn a future extras-list edit into a RED BUILD instead of
# a fleet of successful runs with no playback data.
RUN python -c "import run_anuga.playback_store as p; assert p.zarr_available(), \
'TASK-2749: playback extra missing from this image — export_playback_store() would silently skip'"

# ---- substitution guard (TASK-2700) — same contract as the cpu target ------
# This is the COSTLY case: a PyPI wheel landing here replaces the nvc +
# OpenMP-target-offload engine with a generic CPU build. The offload-linkage
# check above CANNOT catch that — it inspects whichever sw_domain_gpu_ext it
# finds, and a wholesale package swap removes the module rather than mislinking
# it. This guard checks the package's build provenance instead.
RUN python /app/run_anuga_src/batch/assert-anuga-provenance.py \
        /app/anuga_core "${ANUGA_CORE_SHA}"

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
