# Stability fixtures (TASK-4316)

Saved production diagnostics that pin `run_anuga.diagnostics.classify_stability`
(`tests/test_diagnostics.py::TestStabilityClasses`). Both runs were written
with the 15-column CSV schema that predates TASK-4316, and both runs
reported themselves "unstable" because the implied speed divided by the step
that the engine had already clipped to land on the yield time.

| Run | Project / scenario | Model | Reported | Expected class |
|-----|--------------------|-------|----------|----------------|
| 1613 | 1135 / 508 | 4 h, yieldstep 144 s, 1 h rain, all-Transmissive edge | 407.9 m/s at 5,472 s (last step 7.88 ms) | `boundary_fed`; speed spikes `clip_artefact` (volume 1.41 M m3 at rain end, 5.10 M m3 at 4 h, no inflows) |
| 1608 | 1121 / 502 | 6 h, yieldstep 216 s, 1 h rain, Dirichlet + Transmissive | 1,122 m/s at 15,552 s = yield 72 (last step 2.16 ms) | `clip_artefact` |

Files per run:

- `run_<id>_diagnostics_1.csv`: the run's `run_diagnostics_1.csv`, unchanged
  apart from line endings.
- `run_<id>_summary_1.json`: the run's `run_summary_1.json` (schema 1).
- `run_<id>.log`: the run's `run_anuga_1.log`.

## Scrubbed (this repository is public)

Taken from each run's results zip (Results > Download) on 10-11-26. Both
projects belong to the operator's own accounts. Before committing, the
environment hostname and the storage bucket name and prefix in the
playback-export log line were replaced with `<redacted>`. Nothing else in the
files names a storage location, account, email address or user. Check any
change with:

```
grep -nEi 's3://|arn:aws|amazonaws|anuga-result-storage|@[a-z0-9.-]+\.[a-z]{2,}|[0-9]{12}|ip-10-' tests/fixtures/stability/run_*
```

which must print nothing.
