# Chronoguard flagship plan

## Objective
Turn Chronoguard from a useful small CLI into a reproducible, leakage-resistant time-series evaluation workbench. Prefer depth, audits, evidence and honest limitations over repo count. Do not call this a dozens-of-hours project until that work has actually happened.

## Milestones
1. Existing foundation: point-in-time join, purged walk-forward plans and independent join audit — complete.
2. Reproducible experiment runner: explicit sample/target data contract, train-only transforms, baseline model, per-fold predictions and metrics, input fingerprints — baseline and audited join-to-evaluation integration complete; immutable-input enforcement remains future work.
3. Stronger statistical evaluation: uncertainty intervals, multiple baseline comparisons, failure-mode fixtures and larger synthetic benchmarks — in progress; paired block-bootstrap comparison implemented, more robust comparison evidence still pending.
4. Artifact audit: independently verify predictions, fold membership, hashes, preprocessing state and reported metrics — complete; 21 local tests including 40 randomized small experiments and cross-platform CI passed at `583ec1979010c3c50359ae14873fc71bf008183b`.
5. Packaging and demonstration: installable package, realistic public dataset/example, performance characterization, cross-platform CI and release notes — in progress; wheel and console entry point passed cross-platform CI, synthetic end-to-end demo passes locally, realistic public data and performance work pending.

## Quality bar
- Each milestone has executable tests, failure cases, documentation and an honest limitation section.
- Never fabricate history, hours, benchmark results or data provenance.
- Publish increments to the existing public repository only after verification.

## Errors encountered
- A first lookup for `.github/workflows/test.yml` failed because the existing workflow is named `ci.yml`; located it with `rg --files .github` and continued.
- First local wheel smoke test ran from the repository root, so import resolved to source; reran from `work/` and confirmed the installed `site-packages` package. CI smoke step now changes to `$RUNNER_TEMP` before import.
