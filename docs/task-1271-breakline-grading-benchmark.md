# TASK-1271 W4.3 — Breakline Distance-Graded Sizing Benchmark

Date: 2026-05-30  
Domain: 200m x 200m square, EPSG:28355, resolution=20m  
Breakline: 180m vertical line at x=321100, h_near=2.0m

## Buffer Ring Configuration

With h_near=2.0m and h_far=20.0m, the doubling series produces:

| Ring | Distance (m) | max_area (m²) | Target side (m) |
|------|-------------|---------------|-----------------|
| 0    | 2.0         | 1.73          | 2.0             |
| 1    | 4.0         | 6.93          | 4.0             |
| 2    | 8.0         | 27.71         | 8.0             |
| 3    | 16.0        | 110.85        | 16.0            |

## Mesh Quality

- **Baseline (no breakline):** 307 triangles, min_angle=28.2°
- **With breakline:** 3,282 triangles, min_angle=28.1°, slivers=0, no degenerate triangles

## Grading Assessment (Informational)

The 10.7x triangle count increase (307→3282) reflects the fine h_near=2m constraint
near the 80m breakline. This is expected — fine mesh near the breakline + coarser
away. The grading "works" in the sense that fine triangles concentrate near the line
and the overall mesh is valid (no slivers, no degenerate triangles).

Per operator 2026-05-29 scope guardrail: grading quality is INFORMATIONAL only.
"Crappy but coarse" is acceptable; Gmsh NOT triggered this epic. The hard floor
(no sub-CFL slivers) is met: sliver_count=0.

## Determinism

Two consecutive runs of create_anuga_mesh with identical inputs produce
identical triangle counts and QA metrics. Golden hash in test_golden_mesh.py
updated to reflect the mesh_geo_reference removal in TASK-1270.

## Notes

- ANUGA logs "Interior polygon ... is not fully inside bounding polygon" for
  rings that extend beyond the boundary — this is an ANUGA INFO-level note,
  not an error. ANUGA clips the region automatically.
- The `fail_if_polygons_outside=False` flag in create_mesh_from_regions handles
  this gracefully.
