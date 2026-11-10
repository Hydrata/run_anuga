# Stability fixtures (TASK-4316)

Saved production diagnostics used to pin `run_anuga.diagnostics.classify_stability`.

## Pending

The real files for these two runs are not committed yet. Until they are,
`tests/test_diagnostics.py::TestStabilityClasses::test_stability_class_from_fixtures`
uses documented SYNTHETIC record sets that reproduce each run's signature:

| Run | Project / scenario | Expected class | Synthetic stand-in |
|-----|--------------------|----------------|--------------------|
| 1613 | 1135 / 508 | `boundary_fed`; its speed spikes `clip_artefact` | `_synthetic_1613_records()`: built from the run log's yield windows and volume figures |
| 1608 | 1121 / 502 | `clip_artefact` | `_synthetic_1608_records()`: 1,122 m/s at sim 15,552 s (yield 72) from a 3 ms clipped last step |

To add the real files, take from each run's results zip the
`run_diagnostics_1.csv`, `run_summary_1.json` and the run log, and commit them
here as `run_<id>_diagnostics_1.csv`, `run_<id>_summary_1.json` and
`run_<id>.log`.

## Scrub before committing (this repository is public)

Remove every S3 URI, bucket name, AWS account id, email address, user id and
hostname from all three files, then check with a grep that none remain, for
example:

```
grep -nEi 's3://|arn:aws|amazonaws|@[a-z0-9.-]+\.[a-z]{2,}|[0-9]{12}|hostname' tests/fixtures/stability/*
```

Confirm on the run rows that the projects belong to the operator's own
accounts before committing.
