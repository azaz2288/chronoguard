# Findings

- The existing split plan explicitly partitions prior events into train/purged and test blocks, so an experiment runner can reuse it without inventing a second split policy.
- Existing `audit_plan` validates fold membership and label horizons; experiment output should invoke it before any fit.
- Existing point-in-time join and feature freshness audit are useful, but an experiment runner must not silently assume joined features were audited. Its contract should take an already materialized table and state that limitation.
- To prevent preprocessing leakage, per-fold numeric imputation and normalization must be estimated from training rows only; tests should make this observable.
- Existing report stores fold model state and per-sample predictions. An independent auditor can verify the ridge normal equations, train-only means/scales, targets, predictions, baseline, metrics and SHA-256 fingerprints without calling the experiment runner's fit or metric routines.
- Existing CI runs unit tests on Windows and Ubuntu, but does not yet build or install the distribution. Local Python 3.12 has pip and setuptools 80.1.0, so an offline wheel build can be verified before extending CI.
