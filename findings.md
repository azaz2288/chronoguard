# Findings

- The existing split plan explicitly partitions prior events into train/purged and test blocks, so an experiment runner can reuse it without inventing a second split policy.
- Existing `audit_plan` validates fold membership and label horizons; experiment output should invoke it before any fit.
- Existing point-in-time join and feature freshness audit are useful, but an experiment runner must not silently assume joined features were audited. Its contract should take an already materialized table and state that limitation.
- To prevent preprocessing leakage, per-fold numeric imputation and normalization must be estimated from training rows only; tests should make this observable.
