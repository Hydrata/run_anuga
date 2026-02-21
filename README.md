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

See `run_anuga/schema.py` for the full JSON Schema.

## Installation

```bash
# From the repo root
pip install -e .

# With ANUGA (for running simulations)
pip install -e ".[anuga]"

# With development tools
pip install -e ".[dev]"
```

System dependencies: `gdal-bin`, `libgdal-dev` (for the GDAL Python bindings).

## CLI Usage

```bash
# Run a simulation package
run-anuga --package_dir /path/to/package/

# With Hydrata server authentication
run-anuga user@example.com password --package_dir /path/to/package/

# With checkpointing (restart from a previous checkpoint)
run-anuga --package_dir /path/to/package/ --batch_number 2 --checkpoint_time 1800
```

## Python API

```python
from run_anuga.run import run_sim
from run_anuga.run_utils import setup_input_data

# Parse a package without running
input_data = setup_input_data("/path/to/package")
print(input_data['scenario_config']['duration'])

# Run a simulation
run_sim("/path/to/package")
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
| `MAX_TRIANGLE_AREA` | 10,000,000 | Max triangle area for mesher |
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
