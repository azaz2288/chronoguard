# Join and audit benchmark

Run from the repository root:

```sh
python -m benchmarks.join_audit --rows 8000 --entities 100
```

This generates deterministic synthetic facts and decisions, joins them, writes the joined CSV to a temporary directory, then independently audits it. It checks that every joined value is correct and that the auditor reports no violations. Elapsed time comes from `time.perf_counter`; peak Python allocations come from `tracemalloc`. The setup data and CSV-writing time are excluded from both timed sections. Results depend on the machine, Python build, OS, file cache and selected entity distribution. The audit remains a direct candidate scan per entity, not a sort/bisect mirror of the join implementation.

One local Windows 11 / Python 3.12.10 run on 2026-09-30, with 8,000 rows and 100 entities:

| Implementation | Join | Audit | Audit peak Python allocation |
|---|---:|---:|---:|
| Before per-entity grouping | 0.147 s | 3.744 s | 7.40 MiB |
| After per-entity grouping | 0.133 s | 0.478 s | 7.48 MiB |

These are single-run observations, not controlled benchmark medians. The improvement addresses cross-entity scanning; with only one entity, the independent audit remains quadratic in sample/fact count. The tool is appropriate for small to moderate research extracts; memory-bounded streaming joins/audits are still future work.
