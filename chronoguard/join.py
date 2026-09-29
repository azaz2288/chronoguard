"""Join each decision with only information available by that instant."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .common import InputError, format_time, parse_time, read_csv


@dataclass(frozen=True)
class Sample:
    sample_id: str
    entity: str
    decision_time: datetime


@dataclass(frozen=True)
class Fact:
    entity: str
    available_at: datetime
    value: str


def load_samples(path: Path) -> list[Sample]:
    samples = []
    seen = set()
    for number, row in enumerate(read_csv(path, ("sample_id", "entity", "decision_time")), start=2):
        sample_id, entity = row["sample_id"].strip(), row["entity"].strip()
        if not sample_id or not entity:
            raise InputError(f"{path}:{number}: sample_id and entity are required")
        if sample_id in seen:
            raise InputError(f"{path}:{number}: duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        samples.append(Sample(sample_id, entity, parse_time(row["decision_time"], f"{path}:{number} decision_time")))
    return samples


def load_facts(path: Path) -> list[Fact]:
    facts = []
    seen = set()
    for number, row in enumerate(read_csv(path, ("entity", "available_at", "value")), start=2):
        entity = row["entity"].strip()
        if not entity:
            raise InputError(f"{path}:{number}: entity is required")
        available = parse_time(row["available_at"], f"{path}:{number} available_at")
        key = entity, available
        if key in seen:
            raise InputError(f"{path}:{number}: duplicate entity/available_at pair")
        seen.add(key)
        facts.append(Fact(entity, available, row["value"]))
    return facts


def _max_age(max_age_hours: int | None) -> timedelta | None:
    if max_age_hours is None:
        return None
    if type(max_age_hours) is not int or not 0 <= max_age_hours <= 1_000_000:
        raise InputError("max-age-hours must be an integer between 0 and 1000000")
    return timedelta(hours=max_age_hours)


def point_in_time_join(samples: list[Sample], facts: list[Fact], max_age_hours: int | None = None) -> list[dict[str, str]]:
    max_age = _max_age(max_age_hours)
    by_entity: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        by_entity[fact.entity].append(fact)
    times = {}
    for entity, values in by_entity.items():
        values.sort(key=lambda fact: fact.available_at)
        times[entity] = [fact.available_at for fact in values]

    result = []
    for sample in samples:
        choices = by_entity.get(sample.entity, [])
        index = bisect_right(times.get(sample.entity, []), sample.decision_time) - 1
        fact = choices[index] if index >= 0 else None
        if fact is not None and max_age is not None and sample.decision_time - fact.available_at > max_age:
            fact = None
        result.append({
            "sample_id": sample.sample_id,
            "entity": sample.entity,
            "decision_time": format_time(sample.decision_time),
            "matched": "1" if fact is not None else "0",
            "feature_available_at": format_time(fact.available_at) if fact else "",
            "feature_value": fact.value if fact else "",
        })
    return result


OUTPUT_FIELDS = ("sample_id", "entity", "decision_time", "matched", "feature_available_at", "feature_value")


def audit_join(samples: list[Sample], facts: list[Fact], joined_path: Path, max_age_hours: int | None = None) -> list[str]:
    max_age = _max_age(max_age_hours)
    rows = read_csv(joined_path, OUTPUT_FIELDS)
    # Audit with a direct candidate scan per entity, independent of the
    # indexed join's sort/bisect selection path.
    facts_by_entity: dict[str, list[Fact]] = defaultdict(list)
    for fact in facts:
        facts_by_entity[fact.entity].append(fact)
    expected = {}
    for sample in samples:
        eligible = [fact for fact in facts_by_entity.get(sample.entity, [])
                    if fact.available_at <= sample.decision_time
                    and (max_age is None or sample.decision_time - fact.available_at <= max_age)]
        latest = max(eligible, key=lambda fact: fact.available_at, default=None)
        expected[sample.sample_id] = {
            "entity": sample.entity,
            "decision_time": format_time(sample.decision_time),
            "matched": "1" if latest else "0",
            "feature_available_at": format_time(latest.available_at) if latest else "",
            "feature_value": latest.value if latest else "",
        }
    observed = set()
    violations = []
    for number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"]
        if sample_id not in expected:
            violations.append(f"Row {number}: unknown sample_id {sample_id!r}")
            continue
        if sample_id in observed:
            violations.append(f"Row {number}: duplicate sample_id {sample_id!r}")
            continue
        observed.add(sample_id)
        correct = expected[sample_id]
        for field in OUTPUT_FIELDS[1:]:
            if row[field] != correct[field]:
                violations.append(f"Row {number}: {field} mismatch for {sample_id!r}; expected {correct[field]!r}")
    for sample_id in sorted(expected.keys() - observed):
        violations.append(f"Missing joined sample_id {sample_id!r}")
    return violations
