# Merewether: Solution Options Comparison

## Diagnosis Summary

The simulation goes numerically unstable at t≈660s. The CFL timestep collapses
exponentially (implied max_speed reaches 450+ m/s at t=720s), caused by unphysical
velocities growing in cells near building boundaries when using `method=Mannings`.

**Confirmed non-cause:** The mesh quality is excellent. Minimum inradius = 0.278m,
minimum angle = 28.0° (Triangle enforces this by design), maximum aspect ratio = 4.0,
spatially uniform across all 100 grid cells examined. No slivers, no hotspots.
The mesh cannot be improved to fix this problem.

---

## Mesh Quality Reference

| Metric | Value | Notes |
|--------|-------|-------|
| Total triangles | 100,530 | 321×416m domain at 2m resolution |
| Min inradius | 0.278m | At x=382446, y=6354499 |
| Median inradius | 0.476m | |
| Max inradius | 0.620m | |
| Min angle | 28.01° | Triangle enforces ≥28° throughout |
| Max aspect ratio | 4.0 | No slivers anywhere |
| Spatial variation | None | Min inradius uniform across all grid cells |

---

## Solutions

### A. Elevation-burn buildings into DEM — *Recommended*

**Mechanism:** Pre-process `dem.tif` to raise building footprints by BUILDING_BURN_HEIGHT_M
(5.0m, `defaults.py:9`). Buildings become high dry ground. No flow inside buildings.
No high-velocity passages. Matches original ANUGA benchmark exactly (`runMerewether.py`
uses house_height=3.0m with breaklines).

**Effect on CFL:** Building-footprint triangles have h≈0 → max_speed≈0 → the
`minimum_allowed_height` zeroes their momentum → they don't constrain CFL at all.

**Implementation options:**
1. At prepare-script time: burn buildings into `dem.tif` in `prepare_merewether_scenario.py`
   and remove `structure.geojson` entirely. No run_anuga source changes.
2. New `method=Elevation` in run_anuga: call `burn_structures_into_raster()` in the
   standard ANUGA mesh path (currently only called in the mesher path, `run.py:407`).
   Requires small source change.

**Trade-off:** Buildings are represented as terrain bumps, not solid walls. Very thin
water films can still flow over them (depth > BUILDING_BURN_HEIGHT_M). At 5m burn height
this is not a practical concern for typical flood depths.

**Expected runtime:** Minutes (original ANUGA benchmark at 1m resolution completes in
minutes on a single machine). Our 2m resolution version should be faster.

---

### B. Switch to `method=Holes` (Reflective walls)

**Mechanism:** Buildings become holes in the mesh with Reflective boundary conditions.
Water is physically blocked at building walls. No triangles inside buildings.

**Effect on CFL:** Removes ~10,000–20,000 building-interior triangles from computation.
BUT: all flow is forced around buildings through gaps → potentially higher gap velocities
than Mannings (which lets some water seep through slowly). May reduce instability or
worsen it.

**Implementation:** Change `structure.geojson` property from `"method": "Mannings"` to
`"method": "Holes"`. No source code changes needed.

**Trade-off:** Creates real mesh boundaries at building edges. These ANUGA boundary
triangles are subject to the reflective BC but are still normally-shaped (min angle 28°,
confirmed by mesh analysis). Gap velocities may be higher than Mannings approach.

**Risk:** Unknown — could improve or worsen the instability. Needs testing.

---

### C. Coarser mesh resolution (4–5m)

**Mechanism:** Larger triangles → larger inradius → allows same velocity with larger dt.
Also fewer triangles → less work per internal step.

**Inradius scaling:** At 4m, inradius_min ≈ 0.56m (2× current 0.278m).
At 5m, inradius_min ≈ 0.70m (2.5× current).

**Effect:** Approximately 2–2.5× larger dt for the same velocity → fewer internal steps.
Also ~4–6× fewer triangles. Total speedup factor: ~8–15×.

**BUT:** The instability is velocity-driven. At some later time, the same unphysical
velocities will appear. The simulation will still diverge — just later and slower.
Doesn't fix the root cause.

**Trade-off:** Resolution loss. Building edges less precisely represented. Mesh can't
resolve features smaller than 4–5m.

---

### D. `optimise_dry_cells = True`

**Mechanism:** Skip linear reconstruction in `distribute_to_vertices_and_edges` for
completely dry cells (`sw_domain_openmp.c:1958–1965`). Saves CPU in the early
(mostly-dry) phase.

**Effect:** Speedup proportional to the dry fraction of the domain. At t=0–300s
(domain mostly dry), meaningful speedup. At t=600s+ (domain mostly wet), negligible.
Has NO effect on the instability.

**Implementation:** Add `domain.optimise_dry_cells = True` after domain creation in
`run.py`. One line, no other changes.

**Trade-off:** None — purely a performance optimisation with no accuracy impact
on the wet region.

---

### E. Switch to DE1 algorithm

**Mechanism:** `domain.set_flow_algorithm('DE1')` sets:
- CFL = 1.0 (vs 0.9 for DE0) → 11% larger dt
- RK2 timestepping → 2 flux evaluations per dt step (2× more work)
- `minimum_allowed_height = 1e-5` (vs 1e-12 for DE0) → zeroes momentum in thin
  layers → may help near-dry cells early in simulation

**Effect on CFL:** 11% gain from CFL number is negated by 2× RK2 cost. Net effect
on wall time for stable region: roughly neutral or slightly slower. Does not address
the instability.

**`maximum_allowed_speed`:** Appears in algorithm defaults (`self.maximum_allowed_speed
= 0.0`) but is commented out in the C code (`sw_domain_openmp.c:625`). Not functional.

**Trade-off:** Higher-order accuracy in theory, but 2× per-step cost.

---

## Summary Table

| Solution | Fixes instability? | Runtime improvement | Accuracy | Complexity |
|----------|-------------------|--------------------|---------| -----------|
| **A. Elevation burn** | ✅ Yes | ~10× faster | Matches benchmark | Low |
| **B. method=Holes** | ❓ Unknown | Moderate (fewer cells) | Slightly different physics | None (JSON change only) |
| **C. Coarser mesh (4m)** | ❌ No | ~8–15× (until instability) | Lower | None (scenario.json) |
| **D. optimise_dry_cells** | ❌ No | Minor (early phase only) | Same | Tiny (one line) |
| **E. DE1 algorithm** | ❌ No | Neutral/negative | Higher order | Small |

---

## Recommended Test Plan

Run these variants in order of expected success:

1. **A1 (prepare-time elevation burn)** — burn buildings into DEM, no structure.geojson.
   Expected: stable, fast, accurate. Baseline for comparison.
2. **B (Holes)** — change method=Mannings → method=Holes, same resolution.
   Expected: faster than Mannings but uncertain re: instability.
3. **C (4m resolution)** — change scenario.json resolution 2.0→4.0.
   Expected: runs to completion but with accuracy loss.
4. **A1 + D** — elevation burn + optimise_dry_cells.
   Expected: fastest variant overall.

Validation: run `python examples/merewether/validation/validate.py` after each.
Solutions A1 and A1+D should both pass the ±0.3m tolerance.
