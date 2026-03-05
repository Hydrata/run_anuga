# Mesh Experiment Implementation Progress

Started: 2026-03-04

## Plan Summary
- Add Gmsh and JIGSAW as alternative meshing libraries
- 4 new scenarios: gmsh/flat, gmsh/proximity, jigsaw/flat, jigsaw/topo
- Compare against 3 existing Triangle baselines
- Integrate into run.py with minimal changes

## Progress

### Step 1: Install dependencies
- [x] Install gmsh (4.15.1), cmake (4.2.3), jigsawpy (1.1.0 from source) in anuga_venv
- [x] Verify imports

### Step 2: Create mesher modules
- [x] run_anuga/meshers/__init__.py — MeshResult dataclass + dispatch
- [x] run_anuga/meshers/gmsh_mesher.py — flat + proximity modes
- [x] run_anuga/meshers/jigsaw_mesher.py — flat + topo modes

### Step 3: Add mesher dispatch to run.py
- [x] Insert mesher dispatch before simplify_mesh block (13 lines)

### Step 4: Create experiment directories
- [x] gmsh/flat/ scenario + inputs
- [x] gmsh/proximity/ scenario + inputs
- [x] jigsaw/flat/ scenario + inputs
- [x] jigsaw/topo/ scenario + inputs

### Step 5: Update experiment.json
- [x] Added gmsh and jigsaw to meshers list

### Step 6: Run experiments
- [x] Dry run verification — 7 scenarios found
- [x] gmsh/flat — completed 29s, 75760 tri, 2/5 pass
- [x] gmsh/proximity — completed 46s, 47745 tri, 4/5 pass
- [x] jigsaw/flat — completed 24s, 75392 tri, 3/5 pass
- [x] jigsaw/topo — completed 65s (no-MPI, segfaults with MPI), 21118 tri, 1/5 pass
- [x] triangle/burn — completed 59s, ~104K tri, 5/5 pass (baseline)
- [x] triangle/holes — timeout at 300s (CFL instability at ~72%)
- [x] triangle/mannings — timeout at 300s (CFL instability, n=10 Manning's)

### Step 7: Collect results and validate
- [x] collect_results.py run — HTML and CSV reports generated
- [x] Per-point validation analysis complete

## Issues / Notes

### Fixed: jigsawpy not on PyPI
- Built from source: `git clone https://github.com/dengwirda/jigsaw-python.git`
- Needs cmake on PATH + gcc/g++: `PATH="/home/dave/anuga_venv/bin:$PATH" python build.py`
- Then `pip install .`

### Fixed: JIGSAW hfun_scal default is "relative"
- Must set `jig.hfun_scal = "absolute"` for absolute metre sizing
- Without it, hfun_hmax=2.0 means 200% of domain characteristic length → 2 triangles

### Fixed: JIGSAW ygrid must be ascending
- rasterio rows go top-to-bottom (descending y), must flip for JIGSAW
- `ys = ys[::-1]; size_grid = size_grid[::-1, :]`

### Fixed: input_data has no 'package_dir' key
- Mesher used `input_data["package_dir"]` but that key doesn't exist
- Changed to use `input_data["boundary_filename"]`, `input_data["elevation_filename"]`,
  `input_data.get("structure_filename")`

### Fixed: numpy.bool_ not JSON serializable
- `diagnostics.py:_write_summary()` → `json.dump()` fails on `np.bool_`
- Added `_json_default()` handler for numpy types

### Gmsh flat baseline
- 38246 vertices, 75760 triangles (total before MPI split)
- After MPI 8-way split: 9985 triangles on rank 0
- 29s wall time, 936m³ final volume, 3.85 m/s max speed
- Stable simulation (max_implied_speed=22.7 m/s triggers `stable=false` at 20 threshold)

### Mesher prototype results
- Gmsh flat: 38K pts / 76K tri (uniform 2m)
- Gmsh proximity: 25K pts / 48K tri (adaptive 1-4m near buildings)
- JIGSAW flat: 38K pts / 75K tri (uniform 2m)
- JIGSAW topo: 12K pts / 21K tri (slope-adaptive 1-4m)

## Key Findings

### Triangle count vs validation accuracy
- Triangle (ANUGA default) produces ~104K triangles at 2m resolution → 5/5 pass, RMSE=0.15
- Gmsh flat produces ~76K triangles at 2m resolution → 2/5 pass, RMSE=0.46
- JIGSAW flat produces ~75K triangles at 2m resolution → 3/5 pass, RMSE=0.34
- Fewer triangles = worse validation despite better mesh quality (min angle)

### Mesh quality vs triangle count trade-off
- Gmsh flat: min_angle=53° (excellent) but 27% fewer triangles
- JIGSAW flat: min_angle=43° (good) but 28% fewer triangles
- Triangle burn: min_angle=28° (worst) but most triangles, best validation
- Triangle's min_triangle_angle=28° constraint lets it fill space more densely

### Adaptive sizing
- Gmsh proximity (4/5, RMSE=0.76): refined near buildings helps pts 0,1,4
  but pt3 near building gap still fails (+1.67m)
- JIGSAW topo (1/5, RMSE=2.30): too coarse — 21K tri insufficient for Merewether

### MPI compatibility
- JIGSAW topo mesh segfaults in ANUGA's distribute() with MPI
  - Likely mesh topology confuses pymetis partitioner
  - Works fine in single process (64.6s)
- All other meshers work with MPI-8

### Building treatment
- Triangle holes + mannings: CFL instability, timeout (known issue)
- Gmsh proximity (holes): completes fine, 4/5 pass
- JIGSAW topo (holes): completes but poor resolution

### Reports
- HTML: experiments/merewether/experiment_report.html
- CSV: experiments/merewether/results.csv
