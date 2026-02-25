# Merewether Simulation: Timestep Slowdown Analysis

## Symptom: Numerical Instability (NOT merely slowness)

Merewether 1000s simulation (100,530 triangles, 2m resolution, 57 buildings as Mannings):

| Sim step | Wall time (s) | ×base | Implied dt (ms) | Implied max_speed (m/s) |
|----------|--------------|-------|-----------------|------------------------|
| t=60–120s | 16 | 1.0 (baseline) | 125 | 2.0 |
| t=120–600s | 15–18 | ~1× | 111–133 | 1.9–2.2 |
| t=600–660s | 50 | 3.1× | 40 | **6.2** |
| t=660–720s | 254 | 15.9× | 7.9 | **31.8** |
| t=720–780s | >3600 | >225× | <0.6 | **>450** (unphysical) |

A velocity of 450 m/s is physically impossible for a flood. This is **numerical
instability** beginning at t≈660s, not merely a slow simulation. Velocities grow
unboundedly; the simulation will never complete in this configuration.

---

## Root Cause: The CFL Timestep Formula

**Source:** `anuga_core/anuga/shallow_water/sw_domain_openmp.c:547`

```c
double edge_timestep = D->radii[k] * 1.0 / fmax(max_speed_local, epsilon);
local_timestep = fmin(local_timestep, edge_timestep);
```

The adaptive internal timestep is:

```
dt = CFL * min_over_all_triangles( inradius[k] / max_speed_local[k] )
```

Where:
- `inradius[k]` = inscribed circle radius of triangle k (≈ area / semi-perimeter)
- `max_speed_local` = √(g·h) + |u| (wave speed + flow velocity)
- `CFL` = Courant number (DE0 default = **0.9**; DE1/DE2 = **1.0**)

The formula shows the two levers: **geometry** (inradius) and **physics** (velocity).

---

## What `method=Mannings` Actually Does

**Source:** `run_anuga/run_utils.py:458–459`

```python
if structure.get('properties').get('method') == 'Mannings':
    continue  # NOT added as interior hole
```

Critical: `method=Mannings` **does NOT create mesh holes**. Buildings are mesh polygons
with n=10.0 friction only. Triangles exist inside buildings.

| Method | Holes in mesh | Triangles inside building | Active in flux computation |
|--------|--------------|--------------------------|--------------------------|
| `Mannings` | No | Yes (n=10.0) | Yes |
| `Holes` | Yes | No | No |
| `Reflective` | Yes (reflective walls) | No | No |

---

## Why the Simulation Slows Down

**With `method=Mannings`**, the mesh is a uniform 2m triangulation over the whole
321×416m domain. Buildings have high friction (n=10.0) but still participate in
flux computation. This has two effects:

### Effect 1: High velocities in inter-building passages

As the flood fills the domain:
- Water enters building footprints but is heavily impeded by n=10.0
- Flow is deflected into the narrow passages between buildings
- Passages of 2–5m width with significant discharge → high local velocities
- High `max_speed_local` → small `dt` in those triangles

Example: inlet discharge 19.7 m³/s, passage width 3m, depth 0.5m:
- v ≈ 19.7/(3×0.5) ≈ 13 m/s in the passage (before friction equilibrium)
- dt = 0.9 × 0.6m / (√(9.81×0.5) + 13) ≈ 0.9 × 0.6 / 15.2 ≈ **0.036s**
- 60s yieldstep → **1,667 internal steps** (vs ~200 in the dry phase)

### Effect 2: Constraint-induced sliver triangles

The mesh has these polygon constraints that the Triangle mesh generator must honour:
1. 4-segment domain boundary
2. 95-vertex road friction polygon
3. 2-point inlet line
4. 57 building outlines (only as friction, not holes — but the mesh generator
   still inserts vertices where building edges intersect the background mesh)

Polygon vertices force the mesh to insert small triangles where constraint edges
cut through what would otherwise be larger elements. Min area observed: **0.62 m²**
vs max 2.0 m². For a non-equilateral triangle with area 0.62 m², inradius can be
as small as 0.05–0.1 m.

Sliver with inradius 0.05m at velocity 5 m/s:
- dt = 0.9 × 0.05 / (√(9.81×0.5) + 5) ≈ **0.006s**
- 60s yieldstep → **10,000 internal steps**

Both effects compound as the flood spreads into the building zone around t=700s.

---

## Flow Algorithm Comparison

**Source:** `anuga_core/anuga/shallow_water/shallow_water_domain.py:587–900`

| Algorithm | CFL | Timestepping | min_allowed_height | beta_w | optimise_dry_cells |
|-----------|-----|--------------|-------------------|--------|-------------------|
| DE0 (default) | 0.9 | Euler (1st order) | 1e-12 | 0.5 | False |
| DE1 | 1.0 | RK2 (2nd order) | **1e-5** | 1.0 | False |
| DE2 | 1.0 | RK3 (3rd order) | 1e-5 | 1.0 | False |
| DE0_7 | 0.9 | Euler | 1e-12 | 0.7 | False |
| DE1_7 | 1.0 | RK2 | 1e-12 | 0.75 | False |

Notes:
- DE1 has CFL=1.0 (11% larger dt) but RK2 does **2 flux evaluations per step** —
  likely slower overall despite larger timestep
- DE1's `min_allowed_height=1e-5` means thin water layers are treated as dry →
  fewer cells participate in flux computation → **may help more than the CFL gain**
- run_anuga currently sets **no flow algorithm** — defaults to DE0
- `optimise_dry_cells=False` in all algorithms — setting this True skips linear
  reconstruction for dry cells (useful early in simulation when domain is mostly dry)

---

## How the Original Benchmark Avoids This

**Source:** `anuga_core/validation_tests/case_studies/merewether/runMerewether.py:78`

```python
houses_as_holes = False  # KEY LINE
breaklines = project.holes  # buildings as breaklines, not holes
house_height = 3.0          # elevation raised 3m inside building footprints
```

The benchmark approach:
1. **Breaklines, not holes/Mannings**: Building edges are mesh breaklines. Triangles
   exist inside buildings, but there are no hole boundaries creating slivers at
   building corners.
2. **Elevation raised by 3m**: Inside building footprints, the DEM elevation is
   raised by 3m. Water is physically blocked by high ground, not by friction.
3. **No special friction**: Manning's n = 0.04 inside buildings (same as surroundings).
   The 3m elevation bump is the only mechanism.
4. **Consequence**: No high-velocity passages. Water flows around the high ground
   naturally. The wave speed √(g·h) in the elevated dry zone is zero — those
   triangles don't constrain the timestep.

**This is the key insight**: elevation bumps create truly dry cells at the building
locations. Dry cells have h≈0, so max_speed ≈ 0, so they don't constrain dt.
Mannings n=10.0 creates slow-moving wet cells which still constrain dt.

---

## Solution Options

### Option A: Elevation-burn buildings into DEM (Best, no code change)

Pre-process `dem.tif` to raise building footprints by `BUILDING_BURN_HEIGHT_M`
(currently 5.0m in defaults.py) before the simulation. `burn_structures_into_raster`
already exists in `run_utils.py:861` and is called in the mesher path (lines 197, 407).

The fix: call `burn_structures_into_raster` in the **standard ANUGA mesh path** as
well, when structure features have `method=Elevation` (new method) or adapt
`method=Mannings` to also burn elevation.

**Effect**: Building footprints become high ground in the DEM. No mesh modifications.
Building interior triangles are dry (h=0). Max_speed ≈ 0 in those triangles. CFL
constraint comes only from the active flood zone. **Matches benchmark exactly.**

### Option B: Coarser resolution (simplest, some accuracy loss)

Increase `resolution` from 2.0 to 4.0–5.0m. Inradius scales with triangle size:
- At 2m: inradius_min ≈ 0.35m for equilateral, ~0.05m for slivers
- At 4m: inradius_min ≈ 0.70m → ~2× larger → ~2× fewer internal steps

This roughly halves the number of triangles (25,000 vs 100,530) AND doubles the
minimum inradius: **~4× speedup** for the same physics. Accuracy loss: some
building edges not as precisely resolved.

### Option C: Add optimise_dry_cells=True to run.py

After domain creation, before evolve:
```python
domain.optimise_dry_cells = True
```
This skips linear reconstruction for dry cells (`sw_domain_openmp.c:1958`). Significant
speedup early in the simulation (most of domain dry), diminishing as the flood fills
the domain. No accuracy impact on the wet region.

### Option D: Use `method=Holes` with coarser building mesh region

With `method=Holes`, buildings are removed from the mesh. The building interiors
have zero triangles → no flux computation → those cells don't constrain dt.
But building polygon corners create slivers.

Mitigation: Add an `interior_region` for each building with coarser area (e.g., 4m)
to prevent the mesh generator creating tiny slivers immediately outside buildings.
Complex to implement for 57 buildings.

### Option E: Set flow_algorithm to DE1

```python
domain.set_flow_algorithm('DE1')
```
CFL=1.0 (vs 0.9, 11% larger dt) and `min_allowed_height=1e-5` (treats thin layers
as dry). The RK2 timestepping does 2 flux evaluations per step, so net effect on
wall time is mixed. Worth benchmarking.

---

## Recommended Approach

**Primary fix: Option A — burn building elevations into DEM.**

Implementation plan:
1. Add `method=Elevation` as a new structure method in `run_utils.py`
2. In the standard ANUGA mesh path in `run.py` (after domain creation, before evolve),
   if any structure has `method=Elevation`, call `burn_structures_into_raster` on the
   DEM and re-set the elevation quantity on the domain
3. OR: simpler — burn elevations into `dem.tif` at prepare-script time
   (`scripts/prepare_merewether_scenario.py`) and use no `structure.geojson` at all

**For Merewether specifically**: Update `prepare_merewether_scenario.py` to burn
building outlines (+3m) into `dem.tif`. Remove building polygons from `structure.geojson`
(or keep them with `method=Elevation` as documentation). This requires no run_anuga
source code changes and exactly matches the ANUGA benchmark methodology.

**Secondary fix: Option C** (`optimise_dry_cells=True`) — easy one-liner in run.py,
free speedup during the early (dry) phase, harmless when wet.

---

## Key Numbers Summary

| Scenario | Min inradius | Velocity | dt | Steps/yieldstep |
|----------|-------------|----------|-----|----------------|
| Early flood, normal triangle | 0.35m | 1 m/s | ~0.24s | ~250 |
| Late flood, narrow passage | 0.35m | 10 m/s | ~0.026s | ~2,300 |
| Sliver triangle, high velocity | 0.05m | 10 m/s | ~0.004s | ~15,000 |
| Elevation-bump building (dry) | 0.35m | 0 m/s | ∞ (skipped) | 0 |

The elevation approach removes ~10,000–30,000 building-footprint triangles from the
timestep constraint entirely (they are dry with h≈0 → max_speed≈0 → ignored by CFL).
