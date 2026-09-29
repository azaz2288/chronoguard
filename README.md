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
python -m chronoguard evaluate examples/events.csv plan.json examples/observations.csv report.json --feature signal
python -m chronoguard audit-evaluation examples/events.csv plan.json examples/observations.csv report.json
python -m unittest discover -s tests -v
```

`join` matches the latest `available_at` for the same entity no later than `decision_time`. The output preserves sample order and explicitly marks unmatched samples. A fact's source/event timestamp is insufficient for this check: `available_at` must represent when the value could actually be used. Duplicate entity/availability pairs are rejected rather than resolved arbitrarily. Optional `--max-age-hours` rejects stale facts; a fact exactly on the age boundary is still valid. Use the same policy with `audit-join`. The audit independently scans eligible source facts and checks an existing joined CSV, including missing rows, stale matches and future-valued features.

`split` creates expanding walk-forward folds from distinct event start times. An event belongs in training only if `end_at + gap_hours <= first test start`. Earlier events failing this rule are listed as purged. Events sharing a start time stay in the same test block. The output JSON contains train/test/purged IDs per fold; `audit` checks overlap, unknown IDs, missing IDs, and label-horizon leakage in an existing plan.

`evaluate` first audits that plan, then fits a standard-library ridge-regression model separately on each fold. Its input CSV has `sample_id`, numeric `target`, and one or more numeric `--feature` columns; a blank feature is imputed with that fold's training mean. Means and scales are fitted only on training rows, never on the test rows. The JSON report contains each fold's model state, out-of-sample predictions, MAE/RMSE/R² (null R² for a constant test target), a training-target-mean baseline for comparison, overall metrics, and SHA-256 fingerprints of the three inputs. `--alpha` sets positive ridge regularization (default 1).

Report schema version 2 also includes a paired squared-error comparison with the training-mean baseline. Negative `mean_loss_delta` favors the ridge model. For at least 10 held-out predictions, a seeded circular moving-block bootstrap supplies a descriptive 95% percentile interval (`--bootstrap-reps`, `--block-size`, `--seed`; defaults 1000, 1, 0). A block size of 1 resamples predictions independently; choose a larger block when neighboring outcomes are dependent. With fewer than 10 predictions the interval is null and its reason is recorded. The interval is not a formal p-value or a guarantee against model-selection bias. Reports from the earlier version 1 are not accepted by the version 2 auditor; regenerate them from source inputs.

`audit-evaluation` independently checks the saved report against those three input files. It re-audits fold membership, checks input hashes, recomputes training-only preprocessing statistics, verifies the saved ridge coefficients satisfy the regularized normal equations, and checks every test prediction and reported metric. It does not call the evaluation runner's fitting or metric functions. It regenerates the deterministic bootstrap comparison using the same bootstrap routine as the runner, so that routine is not independently verified by the audit. Exit status is 0 for a matching report, 1 for content violations, or 2 for malformed inputs.

Timestamps must be ISO 8601 with an explicit UTC offset, including `Z`. All output times are normalized to UTC. Commands return 0 on success, 1 when an audit finds violations, and 2 for invalid data or I/O errors. Existing output files are protected unless `--force` is provided. Output is written atomically.

## Limits and next steps

- `evaluate` performs its own imputation and normalization on training folds only, but it cannot detect leakage already baked into the provided feature CSV. Audit point-in-time joins upstream and avoid full-dataset feature fitting.
- `audit` detects known leakage conditions, but cannot prove that an input's `available_at` or `end_at` accurately reflects the real data pipeline.
- The ridge baseline is a reference implementation, not a trading recommendation or a claim of predictive advantage. Its normal-equation solver is intended for modest feature counts, not ill-conditioned wide datasets.
- The report audit verifies internal consistency, not that the input CSV itself was honestly collected, nor that another model with the same metrics would generalize. It also cannot prove original source files were immutable while evaluation ran.
- Future work: multiple feature streams, uncertainty intervals, richer public benchmark datasets, and memory-bounded joins for large datasets. The independent join audit uses a direct scan and is intentionally slower than the indexed join on large inputs.

This is a newly created portfolio project with genuine Git history. It is not a pre-existing repository eligible for the Feishu collection criteria supplied with this task.
