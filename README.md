# run_anuga

Run [ANUGA](https://github.com/anuga-community/anuga_core) flood simulations from Hydrata scenario packages.

> README current as of 2026-05-28 (commit `4493a9b`).

## Quick Start

Run the bundled example (a 200x200m Australian floodplain with uniform rainfall):

```bash
# 1. Clone and install core package
git clone https://github.com/Hydrata/run_anuga.git
cd run_anuga
pip install .

# 2. Validate the example (no heavy deps needed)
run-anuga validate examples/australian_floodplain/
run-anuga info examples/australian_floodplain/

# 3. Install simulation dependencies (see "System Dependencies" below first)
pip install numpy setuptools
pip install --no-build-isolation --no-binary GDAL "GDAL==$(gdal-config --version)"
pip install ".[sim]"
pip install anuga mpi4py matplotlib scipy triangle netCDF4 pymetis

# 4. Run the simulation
run-anuga run examples/australian_floodplain/

# 5. Post-process SWW to GeoTIFFs
run-anuga post-process examples/australian_floodplain/
```

A second, smaller worked example is at `examples/merewether/` (Newcastle, NSW; mixed inflow and friction inputs).

## System Dependencies

The `[sim]` and `[full]` extras require native C libraries. Install these before `pip install`:

**Debian / Ubuntu:**

```bash
sudo apt-get install build-essential gfortran \
    libopenmpi-dev openmpi-bin \
    gdal-bin libgdal-dev libproj-dev libgeos-dev \
    libhdf5-dev libnetcdf-dev
```

**GDAL version pinning:** The GDAL Python bindings must match your system's libgdal version. Install numpy first (GDAL needs it at build time for `gdal_array` support), then build GDAL from source:

```bash
pip install numpy setuptools
pip install --no-build-isolation --no-binary GDAL "GDAL==$(gdal-config --version)"
pip install "run_anuga[sim]"
```

**ANUGA undeclared dependencies:** The `anuga` package (v3.3.0+, which requires `numpy>=2.0`) only declares `numpy` in its metadata, but actually requires several additional packages at import time. Install them explicitly:

```bash
pip install anuga mpi4py matplotlib scipy triangle netCDF4 pymetis
```

## Package Format

A scenario package is a directory (or zip) with this layout:

```
package/
  scenario.json          # simulation configuration
  inputs/
    dem.tif              # elevation raster (GeoTIFF, projected CRS)
    boundary.geojson     # domain boundary lines
    inflow.geojson       # optional surface inflow polygons
    rainfall.geojson     # optional rainfall catchment polygons
    friction.geojson     # optional friction zones (polygon)
    friction.tif         # optional friction zones (raster; takes precedence)
    structure.geojson    # optional building footprints
    mesh_region.geojson  # optional mesh refinement regions
  outputs_<project>_<scenario>_<run>/   # created at runtime
    run_*.sww            # ANUGA SWW output
    *_depth_max.tif      # max depth GeoTIFF
    *_velocity_max.tif   # max velocity GeoTIFF
```

## scenario.json Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `format_version` | string | no | Schema version (`"1.0"`) |
| `id` | int | yes | Scenario ID |
| `project` | int | yes | Project ID |
| `epsg` | string | yes | CRS code (e.g. `"EPSG:28355"`) |
| `duration` | int | yes | Simulation duration (seconds) |
| `boundary` | string | yes | Boundary GeoJSON filename. Each feature's `properties.boundary` may be `"Reflective"`, `"Transmissive"`, or `"Time"` (timeseries supplied via `properties.data`) |
| `elevation` | string | no | Elevation raster filename |
| `inflow` | string | no | Surface-inflow GeoJSON filename |
| `rainfall` | string | no | Rainfall catchment GeoJSON filename (separate from `inflow`) |
| `friction` | string | no | Friction GeoJSON filename (polygon zones) |
| `friction_raster` | string | no | Friction raster filename (GeoTIFF; takes precedence over `friction` when both are present) |
| `structure` | string | no | Structure GeoJSON filename |
| `mesh_region` | string | no | Mesh region GeoJSON filename |
| `resolution` | float | no | Base mesh resolution (metres) |
| `simplify_mesh` | bool | no | Use adaptive mesher |
| `model_start` | string | no | Start time (ISO 8601) |

See `run_anuga/config.py` for the Pydantic model with full validation.

## Installation

```bash
# Core only: config parsing, validation, CLI (no geo deps)
pip install run_anuga

# Pure-Python sim deps (numpy>=2, pandas>=2, shapely>=2, dill, psutil).
# Wheels available on bare CI runners; no apt packages needed.
pip install "run_anuga[sim-light]"

# Full simulation dependencies (sim-light + GDAL + rasterio).
# Requires system deps, see "System Dependencies" above.
pip install numpy setuptools
pip install --no-build-isolation --no-binary GDAL "GDAL==$(gdal-config --version)"
pip install "run_anuga[sim]"

# With visualisation (matplotlib, opencv)
pip install "run_anuga[viz]"

# With platform integration (requests, boto3, pystac)
pip install "run_anuga[platform]"

# Everything (sim + viz + platform + anuga>=3.3 + celery + django)
pip install "run_anuga[full]"

# Development tools (pytest, ruff, jsonschema, requests; pulls in sim-light)
pip install "run_anuga[dev]"
```

## CLI Usage

```bash
# Validate a scenario package (core only, no heavy deps)
run-anuga validate /path/to/package/

# Show package info
run-anuga info /path/to/package/

# Run a simulation (requires [sim] + anuga)
run-anuga run /path/to/package/

# Run with Hydrata server authentication
run-anuga run /path/to/package/ --username user@example.com --password secret

# Run with checkpointing (restart from a previous checkpoint)
run-anuga run /path/to/package/ --batch-number 2 --checkpoint-time 1800

# Post-process SWW to GeoTIFFs (requires [sim] + anuga)
run-anuga post-process /path/to/package/

# Generate video from result TIFFs (requires [viz])
run-anuga viz /path/to/outputs/ depth

# Upload results to S3 STAC catalog (requires [platform])
run-anuga upload /path/to/outputs/ --bucket my-bucket

# Run + hand results back to a Hydrata control server (requires [sim,platform] + anuga)
run-anuga run-and-report /path/to/package/ --result-bucket my-bucket
```

## Result-zip output contract

`run-anuga run-and-report` (and the AWS Batch entrypoint that wraps it) ends every
successful run by zipping the package directory, uploading the zip to
`s3://<RESULT_S3_BUCKET>/<PROJECT>_<SCENARIO>_<RUN>_results.zip`, and POSTing the
key back to the control server via `POST /api/v2/anuga/runs/<run_id>/process-result/`
(field name `result_package_key`).

The receiver extracts the zip and reads the canonical TIF artefacts produced by
`run_anuga.run_utils.post_process_sww`:

| File pattern | Quantity | Notes |
|--------------|----------|-------|
| `outputs_<project>_<scenario>_<run>/*_depth_max.tif` | maximum water depth (m) | |
| `outputs_<project>_<scenario>_<run>/*_velocity_max.tif` | maximum velocity (m/s) | |
| `outputs_<project>_<scenario>_<run>/*_dIV_max.tif` | maximum depth-integrated velocity | |
| `outputs_<project>_<scenario>_<run>/*_stage_max.tif` | maximum stage | optional |
| `outputs_<project>_<scenario>_<run>/run_*.sww` | raw ANUGA output | retained for re-processing |

The zip EXCLUDES `package.zip` (the input the entrypoint downloaded), any
embedded `run_anuga/` source tree, and the result zip itself.

The receiver-side spec for which TIFs become published layers lives in
`gn_anuga.services.RESULT_LAYER_SPECS`; this contract is the one residual
coupling between `run_anuga` and the Django host. The POST field name is owned
by `run_anuga._handoff.RESULT_PACKAGE_KEY_FIELD` so the sender and receiver
cannot drift (TASK-1158 / F0 wedge class).

Required env for `run-and-report`:

| Env var | Purpose |
|---------|---------|
| `HYDRATA_INTERNAL_COMPUTE_TOKEN` | Raw `X-Internal-Token` value for the control server |
| `RESULT_S3_BUCKET` | S3 bucket for the result zip (overridable via `--result-bucket`) |

`scenario.json` must carry `run_id`, `project`, `id`, and `control_server`.

## Python API

```python
from run_anuga.config import ScenarioConfig

# Parse and validate a scenario package (core only, no geo deps)
config = ScenarioConfig.from_package("/path/to/package")
print(config.run_label)   # "run_42_1_7"
print(config.duration)    # 3600
print(config.epsg)        # "EPSG:28355"

# Run a simulation with logging callback
from run_anuga.run import run_sim
from run_anuga.callbacks import LoggingCallback

run_sim("/path/to/package", callback=LoggingCallback())
```

Callback interface (see `run_anuga/callbacks.py`):

```python
class MyCallback:
    def on_status(self, status: str, **kwargs) -> None: ...
    def on_progress(self, pct: float, eta_seconds: int | None = None) -> None: ...
```

`on_progress` is the canonical way to report progress; the legacy pattern of encoding percentage in `on_status` is replaced. The `RUN_ANUGA_FINALIZE_TIMEOUT_SECONDS` env var (default `30`) controls the watchdog around `MPI.Finalize()`, used by celery workers to avoid hangs on shutdown.

## Defaults

All simulation constants are defined in `run_anuga/defaults.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `BUILDING_BURN_HEIGHT_M` | 5.0 | Height added to DEM for buildings |
| `BUILDING_MANNINGS_N` | 10.0 | Manning's n for building footprints |
| `DEFAULT_MANNINGS_N` | 0.04 | Default Manning's roughness |
| `RAINFALL_FACTOR` | 1.0e-6 | mm/hr to m/s conversion factor |
| `MINIMUM_STORABLE_HEIGHT_M` | 0.005 | Min depth stored in SWW |
| `MIN_ALLOWED_HEIGHT_M` | 1.0e-5 | Min depth for velocity extrapolation |
| `MAX_YIELDSTEPS` | 100 | Max yield steps per simulation |
| `MIN_YIELDSTEP_S` | 60 | Min yield interval (seconds) |
| `MAX_YIELDSTEP_S` | 1800 | Max yield interval (seconds) |
| `MAX_TRIANGLE_AREA` | 10_000_000 | Max triangle area for mesher |
| `K_NEAREST_NEIGHBOURS` | 3 | Neighbours for GeoTIFF interpolation |

## AWS Batch

The `batch/` directory ships a Dockerfile and entrypoint script for running ANUGA on AWS Batch. The container downloads a scenario package zip from S3, runs the simulation under `mpirun`, zips results, uploads them back to S3, and POSTs the result key to a Hydrata control server using V2 internal-token auth.

Required env vars (set on the Batch job definition):

| Variable | Description |
|----------|-------------|
| `PACKAGE_S3_BUCKET` | S3 bucket containing the package zip |
| `PACKAGE_S3_KEY` | S3 key of the package zip |
| `RESULT_S3_BUCKET` | S3 bucket for result uploads |
| `CONTROL_SERVER` | Hydrata control server base URL (e.g. `https://hydrata.com/`) |
| `PROJECT_ID` | Hydrata project ID |
| `SCENARIO_ID` | Hydrata scenario ID |
| `RUN_ID` | Hydrata run ID |
| `HYDRATA_INTERNAL_COMPUTE_TOKEN` | Shared secret for V2 `IsInternalComputeCaller`. Raw token sent in the `X-Internal-Token` header, NOT a Bearer token |

Optional: `CPUS` (default: `nproc`).

Notes:
- Auth is token-only. Legacy BasicAuth env vars (`COMPUTE_USERNAME` / `COMPUTE_PASSWORD`) are not read.
- The entrypoint exports `OMPI_ALLOW_RUN_AS_ROOT=1` so `mpirun` runs inside the single-purpose container.
- No SIGTERM trap, no checkpoint resume; operator accepts spot-loss.
- On failure, an `EXIT` trap POSTs to `/api/v2/anuga/runs/<RUN_ID>/error/` so the run does not wedge in `COMPUTING`.

Source: `batch/Dockerfile`, `batch/entrypoint.sh`.

## Testing

```bash
# Unit tests (no ANUGA required)
pytest tests/ -v --ignore=tests/test_integration.py

# Integration tests (requires ANUGA + MPI)
pytest tests/test_integration.py -v

# Docker-based README validation (tests install + CLI from scratch)
bash test-docker/test_readme.sh
```

See `docs/shape-variance-audit.md` for the GeoJSON geometry-reading hardening audit (TASK-1115 W0).

## License

MIT
