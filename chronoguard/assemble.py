"""Turn an audited point-in-time join and targets into model observations."""

from __future__ import annotations

import math
from pathlib import Path

from .common import InputError, read_csv
from .join import audit_join, load_facts, load_samples


OUTPUT_FIELDS = ("sample_id", "target", "signal")


def assemble(samples_path: Path, facts_path: Path, joined_path: Path, targets_path: Path,
             max_age_hours: int | None = None) -> list[dict[str, str]]:
    samples = load_samples(samples_path)
    facts = load_facts(facts_path)
    violations = audit_join(samples, facts, joined_path, max_age_hours)
    if violations:
        raise InputError("Joined features failed audit: " + "; ".join(violations[:5]))
    targets: dict[str, str] = {}
    for number, row in enumerate(read_csv(targets_path, ("sample_id", "target")), start=2):
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in targets:
            raise InputError(f"{targets_path}:{number}: empty or duplicate sample_id")
        try:
            target = float(row["target"])
        except ValueError as exc:
            raise InputError(f"{targets_path}:{number}: target must be numeric") from exc
        if not math.isfinite(target):
            raise InputError(f"{targets_path}:{number}: target must be finite")
        targets[sample_id] = row["target"].strip()
    sample_ids = {sample.sample_id for sample in samples}
    if set(targets) != sample_ids:
        missing = sorted(sample_ids - set(targets))
        extra = sorted(set(targets) - sample_ids)
        raise InputError(f"Targets do not match samples; missing={missing[:5]}, extra={extra[:5]}")
    joined = {row["sample_id"]: row for row in read_csv(joined_path, ("sample_id", "matched", "feature_value"))}
    result = []
    for sample in samples:
        fact_value = joined[sample.sample_id]["feature_value"] if joined[sample.sample_id]["matched"] == "1" else ""
        if fact_value.strip():
            try:
                value = float(fact_value)
            except ValueError as exc:
                raise InputError(f"Joined feature for {sample.sample_id!r} must be numeric") from exc
            if not math.isfinite(value):
                raise InputError(f"Joined feature for {sample.sample_id!r} must be finite")
        result.append({"sample_id": sample.sample_id, "target": targets[sample.sample_id], "signal": fact_value})
    return result
