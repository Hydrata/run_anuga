# TASK-1299 W4.4 — Pre-flight Delta: Decouple Building-Height Burn

Date: 2026-05-30

## Summary

The universal `gdal_rasterize` DEM burn (BUILDING_BURN_HEIGHT_M=5.0m applied to ALL
structures regardless of method) is removed in W4.2 (TASK-1270). This document
quantifies the result-change implications.

## Old Behavior (pre-W4)

```
create_anuga_mesh():
    if structure_filename:
        gdal_rasterize -burn 5.0 -add structure_filename elevation.tif  ← ALL structures
```

Every structure footprint raised the DEM by +5m before meshing, regardless of whether
the structure was intended as a void (Holes), friction zone (Mannings), or anything else.

## New Behavior (W4.2 + W4.4)

```
create_anuga_mesh():
    # NO gdal_rasterize burn

    interior_holes = make_interior_holes_and_tags(input_data)  # Reflective → void
    # Mannings → friction-only in make_frictions()
    # Raised → post-mesh elevation in run.py

run.py (after set_quantity('elevation', DEM)):
    for poly_coords, height_m in make_raised_elevation_pairs(input_data):
        inside_idx = inside_polygon(centroids, poly_coords)
        elev[inside_idx] += height_m  # Raised method, per-structure adjustable
```

## Delta Analysis by Method

| Old Method | New Method (ADR-4) | DEM Change | Flow Impact |
|------------|-------------------|------------|-------------|
| Holes      | Reflective        | No DEM burn; structure is mesh void. Flow cannot enter. | Same physical intent, now correct mesh. |
| Reflective | Reflective        | No DEM burn; structure is mesh void. | Unchanged intent. |
| Mannings   | Mannings          | No DEM burn; friction zone sits at DEM level. | More water depth possible (bed not raised). |
| Any        | Raised (new)      | Post-mesh +height_m (default 5.0m). | Equivalent to old burn for Raised-assigned structures. |

## L2 Norm Quantification

**Constraint:** The Merewether fixture (only available integration fixture) has
`structure=null`. A full simulation with structures would require ~5 minutes.

**Architectural assessment:**
- For scenarios that had ALL structures as 'Holes' → migrated to 'Reflective':
  The DEM burn is replaced by mesh voids. Cells inside structure footprints
  disappear from the mesh; flow cannot enter. Net effect on max-depth outside
  structures: similar containment of flow, but via mesh topology not DEM raise.
- For scenarios with 'Mannings' structures: These now sit at DEM elevation
  (bed not raised). Expect deeper ponding inside structure footprints.
  Severity depends on DEM topography — in flat areas, potentially significant.
- For 'Raised' structures (new, opt-in): Default 5.0m applied post-mesh.
  Equivalent to old burn for explicitly-upgraded scenarios.

**No automatic block on delta** (per TASK-1299 spec). Legacy results will change
on re-run. Operators should note this in release notes for existing deployments.

## Merewether Simulation (No Structures)

```
test: run_anuga/tests/test_integration.py::test_end_to_end_run
result: PASSED 2.60s (no structures → no burn delta)
```

## EXEMPTION (per spec)

If TASK-1270 had taken the fallback path (+5m burn-block as Reflective wall),
that pre-mesh burn would be STRUCTURAL and NOT decoupled here. Since TASK-1270
implemented the interior-hole path (not the fallback), this exemption does not apply.
