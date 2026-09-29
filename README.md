# Chronoguard

Chronoguard is a standard-library Python CLI for preventing two common forms of time-series evaluation leakage: joining a feature before it was available, and training on labels whose observation window reaches into the test period. It is meant for forecasting, event-based classification, and quantitative research workflows where a random split or a naive timestamp join can produce optimistic results.

## Quick start

Requires Python 3.12+. From this repository root:

```powershell
python -m chronoguard join examples/samples.csv examples/facts.csv joined.csv
python -m chronoguard audit-join examples/samples.csv examples/facts.csv joined.csv
python -m chronoguard join examples/samples.csv examples/facts.csv fresh.csv --max-age-hours 48
python -m chronoguard audit-join examples/samples.csv examples/facts.csv fresh.csv --max-age-hours 48
python -m chronoguard split examples/events.csv plan.json --folds 2 --min-train-times 3 --test-times 1
python -m chronoguard audit examples/events.csv plan.json
python -m unittest discover -s tests -v
```

`join` matches the latest `available_at` for the same entity no later than `decision_time`. The output preserves sample order and explicitly marks unmatched samples. A fact's source/event timestamp is insufficient for this check: `available_at` must represent when the value could actually be used. Duplicate entity/availability pairs are rejected rather than resolved arbitrarily. Optional `--max-age-hours` rejects stale facts; a fact exactly on the age boundary is still valid. Use the same policy with `audit-join`. The audit independently scans eligible source facts and checks an existing joined CSV, including missing rows, stale matches and future-valued features.

`split` creates expanding walk-forward folds from distinct event start times. An event belongs in training only if `end_at + gap_hours <= first test start`. Earlier events failing this rule are listed as purged. Events sharing a start time stay in the same test block. The output JSON contains train/test/purged IDs per fold; `audit` checks overlap, unknown IDs, missing IDs, and label-horizon leakage in an existing plan.

Timestamps must be ISO 8601 with an explicit UTC offset, including `Z`. All output times are normalized to UTC. Commands return 0 on success, 1 when an audit finds violations, and 2 for invalid data or I/O errors. Existing output files are protected unless `--force` is provided. Output is written atomically.

## Limits and next steps

- This checks data availability and label horizons. It does not detect leakage inside a transformation that uses the full dataset, such as fitting a scaler before splitting.
- `audit` detects known leakage conditions, but cannot prove that an input's `available_at` or `end_at` accurately reflects the real data pipeline.
- Future work: multiple feature streams, explicit event/report-time lag policies, richer audit reports, and memory-bounded joins for large datasets. The independent audit uses a direct scan and is intentionally slower than the indexed join on large inputs.

This is a newly created portfolio project with genuine Git history. It is not a pre-existing repository eligible for the Feishu collection criteria supplied with this task.
