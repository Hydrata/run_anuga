# Merewether e2e: Implementation Notes

## What Was Done (2026-02-25)

### Goal
Fix the Merewether benchmark numerical instability and validate against 5 ARR field
observation points within ±0.3m. Full plan in `.claude/plans/witty-juggling-lerdorf.md`.

### Root Cause (Confirmed)
`method=Mannings` buildings apply n=10 friction but leave triangles **inside** buildings
in the mesh. The constant 19.7 m³/s inlet forces water through inter-building gaps →
supercritical jets → CFL timestep collapse → implied_max_speed reaches 450 m/s at t≈720s.

### Fix Applied: Elevation Burn (Solution A from merewether-solutions.md)
Pre-process `dem.tif` to raise building footprints by +3.0m using `gdal_rasterize -add`.
Building interiors become dry elevated ground (h≈0, max_speed≈0 → no CFL constraint).
Matches original ANUGA benchmark `house_height=3.0` in `runMerewether.py`.

Sequence in `main()`:
1. `make_dem()` — fresh topography1.asc → dem.tif
2. `make_structures()` — write structure.geojson WITH 57 building polygons
3. `burn_buildings_into_dem()` — gdal_rasterize burns them into dem.tif
4. `clear_structure_methods()` — empty structure.geojson (no n=10 friction on top of bumps)

### Results
- outcome: completed
- wall_time: 273s (target <300s — PRIMARY MET)
- stable: true (max_implied=14.4 m/s < 20 m/s threshold)
- validation: 5/5 points pass within ±0.3m
  - Mean bias: +0.058 m (slight over-prediction)
  - RMSE: 0.154 m

### All Code Changes

**`scripts/prepare_merewether_scenario.py`** (modified)
- Added `BURN_HEIGHT_M = 3.0`
- Added `burn_buildings_into_dem()` using `gdal_rasterize -add -burn 3.0`
- Added `clear_structure_methods()` — empties structure.geojson post-burn
- Updated `main()` call order

**`run_anuga/run.py`** (modified)
- Added `domain.optimise_dry_cells = True` after `distribute()` — free speedup
- Wrapped `post_process_sww()` in try/except — TIF output non-fatal

**`run_anuga/diagnostics.py`** (modified)
- Fix false instability flag: skip implied speed calc when n_wet==0
- Previously at t=0: `CFL * inradius / 1e-12 = 178e9 m/s` → `stable: false`

**`examples/merewether/scenario.json`** (new)
- resolution=2.0, duration=1000, structure=null, model_start=null

**`examples/merewether/validation/validate.py`** (new)
- Reads SWW via netCDF4, finds nearest vertex, compares peak stage vs field
- Exit 0 = all pass, Exit 1 = any fail

### Setup Issues Encountered (dev-env specific, not product bugs)

1. **Two editable installs**: `/opt/hydrata/run_anuga/` (old, for `/opt/venv/hydrata/`) and
   `/home/dave/hydrata/run_anuga/` (new). Always use `/home/dave/anuga_venv/bin/run-anuga`.

2. **Missing deps in anuga_venv**: Had to install `shapely`, `pandas`, `psutil`, `gdal==3.8.4`,
   `netCDF4`, `scipy` post-initial venv setup.

3. **numpy/gdal conflict**: `gdal==3.8.4` wheel compiled for numpy 1.x; ANUGA requires
   numpy ≥2.0. The `_gdal_array.so` fails to import → TIF output skipped. Simulation core
   is unaffected (reads DEM fine via `gdal.Open()`, only `WriteArray` needs gdal_array).

4. **model_start=null required**: Setting `model_start: "2007-06-08T..."` causes
   `domain.set_starttime(unix_epoch)` → `domain.time ≈ 1.18e9 s` →
   `finaltime=1000 < current_time` → zero evolve iterations, no simulation.
   Root fix tracked as medium-priority next step: compute `finaltime = starttime + duration`.

5. **examples/merewether/ is untracked**: Directory was accidentally deleted mid-session.
   Recreate with `python scripts/prepare_merewether_scenario.py` then add scenario.json.

### Running the Benchmark

```bash
# One-time setup (requires /home/dave/hydrata/anuga_core/)
python scripts/prepare_merewether_scenario.py

# Run simulation
timeout 360 /home/dave/anuga_venv/bin/run-anuga run examples/merewether/

# Validate
/home/dave/anuga_venv/bin/python examples/merewether/validation/validate.py

# Read diagnostics
python3 -c "import json; d=json.load(open('examples/merewether/outputs_1_1_1/run_summary_1.json')); print(d['run']['outcome'], d['stability']['stable'], d['run']['total_wall_time_s'])"
```

### Next Steps (Priority Order)

1. **Fix numpy/gdal conflict** — need TIF output locally for visualization
2. **Stretch goal: 3m resolution** — delete mesh, change resolution to 3.0, re-run+validate
   Expected: ~25–40s wall time, likely still validates. If passes → can run on CI.
3. **Formal e2e test** — `tests/test_e2e_merewether.py` (marked `e2e`) that runs full cycle
4. **model_start finaltime fix** — `run.py`: `finaltime = domain.starttime + duration`
5. **4m resolution experiment** — explore sub-15s for CI gating
