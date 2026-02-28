# OOM and Memory Management in run_anuga / ANUGA

## What Happened (Towradgi session, 2026-02-27)

**NOT a reboot** — system ran continuously from Feb 18. Confirmed via `last -x reboot` and journalctl.

**OOM kill timeline:**
- `~00:22–00:28` — MPI simulation ranks killed by OOM in unlogged cascade events
- `00:18:34` — `python3` (one of our MPI ranks) invoked the OOM killer; Chrome (oom_score_adj=300) killed first
- `00:28:50` — second OOM event; Chrome killed again
- Simulation processes (oom_score_adj=200) had lower kill priority than Chrome (300)

**Root cause:** At 23:35, 5 × 8-rank MPI jobs were running simultaneously (40 python3 processes):
- 4 × 24h overnight simulations (de0_holes, sg_holes, de0_mann, sg_mann at 20m)
- 1 × 10m suite job (de0_elev_10m)

At OOM time, active+inactive anonymous memory = **62 GB** out of 64 GB total. Swap (2 GB) was fully consumed.

When the OOM killer fires on one MPI rank, OpenMPI detects the rank failure and kills all other ranks in that job — cascade kill per job. This explains why each job's log abruptly stops.

**Swap evidence now:** `SwapFree: 79 MB / 2048 MB` — nearly all swap still consumed by surviving processes.

---

## Memory Accounting: Why Each MPI Rank Uses So Much

Theoretical ANUGA array cost per triangle (~2.5 KB):

| Component | Bytes/triangle |
|---|---|
| 10 quantities × (vertex N×3 + centroid N + edge N×3 + gradients + updates) | ~1,600 |
| Mesh geometry (coords, neighbours, normals, areas, radii) | ~272 |
| Domain work arrays (flux, limiter, edge timestep) | ~80–100 |
| Python object overhead for Quantity instances | ~300–500 |
| **Total** | **~2.0–2.5 KB** |

For 10m mesh: 256,688 triangles / 8 ranks = 32,086 triangles/rank × 2.5 KB = **~78 MB ANUGA data per rank**.

**Actual measured RSS at OOM: 1.1–2.1 GB per rank.** That's 14–26× the array estimate.

The difference is dominated by:
1. **Python interpreter + loaded shared libraries** (~200–400 MB): numpy, scipy, OpenMPI Python bindings, shapely, GDAL, netCDF4 — all loaded by every rank
2. **Numpy heap fragmentation** (~100–300 MB): pymalloc arenas never released back to OS even after arrays are freed
3. **OpenMPI shared memory windows** (~50–200 MB): MPI_Win for one-sided communication, per-rank send/recv buffers
4. **SGS DEM loading** (for `_SG` flow algorithms): `set_subgrid_dem()` loads the DEM raster into memory per-rank to compute sub-grid tables

**Rule of thumb from empirical data:**
- 20m mesh (50k–147k triangles): ~1–2 GB/rank
- 10m mesh (256k triangles): ~2–3 GB/rank
- Add ~800 MB base per rank regardless of mesh size (Python overhead)

**Practical limit for this machine (64 GB, 8 ranks/job):**
- 20m jobs: safe to run 3–4 simultaneously (3 × 8 × 1.5 GB = 36 GB, OK)
- 10m jobs: safe to run 1–2 simultaneously (2 × 8 × 2.5 GB = 40 GB, OK; 3 = 60 GB, borderline)
- Never mix 10m jobs with multiple 20m jobs

---

## What run_anuga Does Now (since this issue)

Two improvements added to `run_anuga/run.py`:

### 1. Memory pressure warning escalation
In the evolve loop (every yieldstep), memory thresholds now trigger escalating log levels:
- ≥85%: `WARNING — memory pressure high`
- ≥92%: `CRITICAL — memory critically low, consider graceful exit`

### 2. SIGUSR1 graceful bail
Send `SIGUSR1` to the **rank-0 process** to request a clean stop at the next yieldstep.
Since checkpoint_step=1, a checkpoint was written at the previous yieldstep — restart from there with `--batch_number` and `--checkpoint_time`.

A bail flag file (`outputs/bail.flag`) is written so all MPI ranks detect the signal.

**How to use:**
```bash
# Find rank 0 PID
pgrep -a python3 | grep run_anuga   # or look in the run log

# Send soft-kill signal
kill -USR1 <rank0_pid>

# Check log for "bail signal received" message

# Restart from checkpoint
run-anuga run <scenario_dir> --batch_number 2 --checkpoint_time <t>
```

---

## Recommended Operational Practices

### Limit concurrent jobs by mesh resolution
```bash
# Script-level guard: wait until memory is available
while [ $(free -m | awk '/Mem:/{print $7}') -lt 20000 ]; do
    echo "Waiting for memory... ($(free -m | awk '/Mem:/{print $7}') MB available)"
    sleep 60
done
mpirun -n 8 run-anuga run <scenario> &
```

### Use ulimit / cgroup to prevent system-wide OOM
Wrap MPI jobs in a cgroup memory limit so they die cleanly rather than causing global OOM:
```bash
# Allow 20 GB per 8-rank job (2.5 GB/rank)
systemd-run --scope -p MemoryMax=20G mpirun -n 8 run-anuga run <scenario>
```
When the cgroup hits the limit it fires OOM within the scope, killing the contained job without affecting Chrome or other processes.

### Reduce MPI ranks for smaller meshes
More ranks = more Python interpreter copies = more RAM. For 20m mesh:
- 8 ranks: 1–2 GB × 8 = 8–16 GB
- 4 ranks: 1–2 GB × 4 = 4–8 GB, similar wall-time on this hardware
- Use 4 ranks for 20m, 8 ranks for 10m+

### Monitor memory in suite scripts
The `mpi_suite_outer.log` pattern should check `MemAvailable` before each job launch.

---

## ANUGA Memory Internals: What Could Be Improved

Source: code analysis of `anuga_core/anuga/abstract_2d_finite_volumes/quantity.py`,
`shallow_water_domain.py`, and `generic_domain.py`.

### Currently allocated per Quantity (quantity.py:78–81)
Every quantity (including static ones like `elevation`, `friction`, `x`, `y`) allocates:
- `vertex_values` (N×3 float64) — 24 bytes/triangle
- `centroid_values` (N float64) — 8 bytes/triangle
- `edge_values` (N×3 float64) — 24 bytes/triangle
- `x_gradient`, `y_gradient`, `phi` — 24 bytes/triangle
- `explicit_update`, `semi_implicit_update` — 16 bytes/triangle
- `centroid_backup_values` — 8 bytes/triangle (for RK stepping)

**Total: ~114 bytes/triangle × 10 quantities = 1,140 bytes/triangle** just for array data.

### Specific optimisation opportunities in anuga_core

**High impact, low difficulty:**

1. **Skip `edge_values` for static quantities** (`elevation`, `friction`, `x`, `y`, `height`)
   - They are read-only after initial setup; their edge values are computed at setup and never change
   - Saving: 24 bytes/triangle × 5 static quantities = **120 bytes/triangle**
   - File: `quantity.py:78–81` — add `if not self.is_static:` guard, or pass a flag at construction
   - Upstream PR candidate

2. **Lazy `centroid_backup_values` for Euler timestepping**
   - `centroid_backup_values` allocated unconditionally (quantity.py:103), but only used by RK2/RK3 stepping
   - Saving: 8 bytes/triangle × 3 conserved quantities = **24 bytes/triangle**
   - Only when `domain.timestepping_method == 'euler'`

**Medium impact, more refactoring:**

3. **Consolidate 3 domain work arrays into one** (shallow_water_domain.py:399–402)
   - `edge_flux_work`, `neigh_work`, `pressuregrad_work` (each N×3 float64) used sequentially not simultaneously
   - Saving: 48 bytes/edge (but edge count ≈ 3×triangles, so 144 bytes/triangle equivalent)
   - Requires careful audit of execution order

4. **Float32 for non-critical quantities**
   - `friction`, `height`, `xvelocity`, `yvelocity` could use float32 (7 significant digits vs 16)
   - Saving: half of those quantity's arrays ≈ **~200 bytes/triangle** for 5 quantities
   - Risk: numerical precision in friction/velocity calculations; needs testing
   - Probably not worth the risk vs. reward

**Low impact / not worth changing:**
- MPI send buffers are copies not views, but saving here is small vs Python overhead
- SWW I/O already uses minimal buffering (write-then-free per yieldstep) — no issue there

### Checkpoint / resume (already works!)
ANUGA already supports full checkpointing via `domain.set_checkpointing(checkpoint_step=1)`.
Each MPI rank writes a `domain_P{N}_{id}_{time}.pickle` file at every yieldstep.
Resume: `run-anuga run <dir> --batch_number 2 --checkpoint_time <t>`

The pickle includes all quantity arrays, mesh, and operator state. For a 10m mesh rank:
~32k triangles × 2.5 KB × 10 quantities ≈ ~800 MB pickle per rank. Large but functional.

**Improvement opportunity:** Custom `__getstate__` on Domain to exclude non-essential arrays
from the pickle (e.g., `edge_values` for static quantities, backup arrays when using Euler).
Could reduce pickle size by 30–50%.

---

## OpenMPI Buffer Pool Fix (MCA Tuning, 2026-02-28)

### The Problem

When running ANUGA with many MPI ranks (e.g. 24), memory grows linearly at ~62.5 MB/min/rank
(1.5 GB/min total at 24 ranks), exhausting 62 GB RAM + 50 GB swap within ~60 minutes.

This growth is NOT from ANUGA code — tracemalloc profiling showed < 1 MB Python-level growth,
and per-rank RSS was flat at 554 MB with 4 ranks over a 1h run.

### Root Cause

OpenMPI's ob1 PML (Point-to-point Management Layer) caches MPI request fragments in a free list
with **`pml_ob1_free_list_max = -1` (unlimited by default)**. The free list grows during transient
congestion but **never shrinks** during execution (only released at `MPI_Finalize()`).

ANUGA's ghost cell exchange (`Isend/Irecv/Waitall` per mesh neighbor) runs every internal timestep
(thousands per second). Any momentary congestion spike causes the free list to allocate more
fragments (in bursts of 64) which are cached permanently.

Secondary: `btl_vader_free_list_max = 512` allows the vader shared-memory BTL to cache up to
512 fragments per free list.

### Evidence

| Ranks | Peer connections | Per-rank growth | Python-level growth |
|-------|-----------------|-----------------|---------------------|
| 4     | 6               | 0 MB/min        | < 1 MB total        |
| 8     | 28              | 3 MB/min        | < 1 MB total        |
| 24    | 276             | 62.5 MB/min     | < 1 MB total        |

### The Fix

Cap the free lists with MCA parameters:

```bash
mpirun -np 24 \
  --mca pml_ob1_free_list_max 256 \
  --mca btl_vader_free_list_max 128 \
  --mca btl_vader_eager_limit 32768 \
  ...
```

| Parameter | Default | Tuned | Effect |
|-----------|---------|-------|--------|
| `pml_ob1_free_list_max` | **-1 (unlimited)** | 256 | Caps PML request fragment cache — the main fix |
| `btl_vader_free_list_max` | 512 | 128 | Caps vader BTL fragment cache |
| `btl_vader_eager_limit` | 4096 | 32768 | Keeps ANUGA's small ghost messages on fast eager path |

### Test Results

| Metric | Before (no tuning) | After (MCA tuning) |
|--------|--------------------|--------------------|
| Per-rank growth rate | 62.5 MB/min | ~2 MB/min |
| System memory at t+15 min | +46 GB | +1 GB |
| System memory at t+34 min | Swapping/stuck | +3 GB (stable) |
| **Reduction** | — | **97%** |

### ANUGA Communication Pattern

ANUGA communicates only with **mesh neighbours** (not all-to-all):
- `communicate_flux_timestep()`: 1× `Allreduce` (8 bytes, MIN) per internal timestep
- `communicate_ghosts_non_blocking()`: `Isend/Irecv` to each mesh neighbour per timestep
  - Message size: ~500–2500 bytes per neighbour (ghost centroids × 3 conserved quantities)
  - Typically 2–8 neighbours per rank

### Recommended Production Command

```bash
PYTHONMALLOC=malloc MALLOC_MMAP_THRESHOLD_=65536 MALLOC_TRIM_THRESHOLD_=65536 \
  mpirun -np 24 \
  -x PYTHONMALLOC -x MALLOC_MMAP_THRESHOLD_ -x MALLOC_TRIM_THRESHOLD_ \
  --mca pml_ob1_free_list_max 256 \
  --mca btl_vader_free_list_max 128 \
  --mca btl_vader_eager_limit 32768 \
  python3 your_script.py
```

### Additional env vars (marginal, but good practice)

- `PYTHONMALLOC=malloc`: Bypasses pymalloc, uses glibc for all allocations
- `MALLOC_MMAP_THRESHOLD_=65536`: Force mmap for > 64 KB allocs (immediate OS release on free)
- `MALLOC_TRIM_THRESHOLD_=65536`: Aggressive glibc heap trimming

---

## Memory at OOM Time (from kernel log, 2026-02-28 00:18:34)

```
active_anon:   42,075,032 kB  (~40 GB)
inactive_anon: 19,911,640 kB  (~19 GB)
free:             503,396 kB  (~0.5 GB)
swap used:      2,017,384 kB  (~2 GB — full)
```

Process table (selected from kernel OOM dump):
```
pid 4085334  mpirun         total_vm:1.16M pages  rss: 8.7 MB
pid 4085338  python3        total_vm:1.85M pages  rss_anon: 2.1 GB  ← rank 0
pid 4085339  python3        total_vm:1.81M pages  rss_anon: 2.0 GB  ← rank 1
pid 4085340  python3        total_vm:1.68M pages  rss_anon: 1.5 GB
pid 4085341  python3        total_vm:1.56M pages  rss_anon: 1.1 GB
pid 4085342  python3        total_vm:1.56M pages  rss_anon: 1.1 GB
pid 4085343  python3        total_vm:1.68M pages  rss_anon: 1.5 GB
pid 4085344  python3        total_vm:1.81M pages  rss_anon: 2.0 GB
pid 4085345  python3        total_vm:1.81M pages  rss_anon: 2.0 GB  ← rank 7
```

(This was the SURVIVING job at 00:18:34 — other jobs had already been cascade-killed.)
Virtual memory per rank (6–7 GB) >> RSS (1–2 GB) — indicative of large mmap regions
(OpenMPI shared memory, numpy mapped arrays) that are reserved but not fully faulted.
