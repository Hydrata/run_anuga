# run_anuga

Run [ANUGA](https://github.com/anuga-community/anuga_core) flood simulations from Hydrata scenario packages.

## Package Format

A scenario package is a directory (or zip) with this layout:

```
package/
  scenario.json          # simulation configuration
  inputs/
    dem.tif              # elevation raster (GeoTIFF, projected CRS)
    boundary.geojson     # domain boundary lines
    inflow.geojson       # rainfall / surface inflow polygons
    friction.geojson     # optional friction zones
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
| `boundary` | string | yes | Boundary GeoJSON filename |
| `elevation` | string | no | Elevation raster filename |
| `inflow` | string | no | Inflow GeoJSON filename |
| `friction` | string | no | Friction GeoJSON filename |
| `structure` | string | no | Structure GeoJSON filename |
| `mesh_region` | string | no | Mesh region GeoJSON filename |
| `resolution` | float | no | Base mesh resolution (metres) |
| `simplify_mesh` | bool | no | Use adaptive mesher |
| `model_start` | string | no | Start time (ISO 8601) |

See `run_anuga/config.py` for the Pydantic model with full validation.

## Installation

```bash
# Core only — config parsing, validation, CLI (no geo deps)
pip install run_anuga

# With simulation dependencies (GDAL, numpy, shapely, etc.)
pip install "run_anuga[sim]"

# With visualisation (matplotlib, opencv)
pip install "run_anuga[viz]"

# With platform integration (requests, boto3, pystac)
pip install "run_anuga[platform]"

# Everything (sim + viz + platform + anuga + celery + django)
pip install "run_anuga[full]"

# Development tools (pytest, ruff)
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

# Post-process SWW to GeoTIFFs (requires [sim])
run-anuga post-process /path/to/package/

# Generate video from result TIFFs (requires [viz])
run-anuga viz /path/to/outputs/ depth

# Upload results to S3 STAC catalog (requires [platform])
run-anuga upload /path/to/outputs/ --bucket my-bucket
```

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

## Testing

```bash
# Unit tests (no ANUGA required)
pytest tests/ -v --ignore=tests/test_integration.py

# Integration tests (requires ANUGA + MPI)
pytest tests/test_integration.py -v
```

## License

MIT
