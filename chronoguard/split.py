"""Walk-forward plans with label-horizon purge and an independent audit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .common import InputError, format_time, parse_time, read_csv


@dataclass(frozen=True)
class Event:
    sample_id: str
    start_at: datetime
    end_at: datetime


def load_events(path: Path) -> list[Event]:
    events = []
    seen = set()
    for number, row in enumerate(read_csv(path, ("sample_id", "start_at", "end_at")), start=2):
        sample_id = row["sample_id"].strip()
        if not sample_id:
            raise InputError(f"{path}:{number}: sample_id is required")
        if sample_id in seen:
            raise InputError(f"{path}:{number}: duplicate sample_id {sample_id!r}")
        start = parse_time(row["start_at"], f"{path}:{number} start_at")
        end = parse_time(row["end_at"], f"{path}:{number} end_at")
        if end < start:
            raise InputError(f"{path}:{number}: end_at precedes start_at")
        seen.add(sample_id)
        events.append(Event(sample_id, start, end))
    return sorted(events, key=lambda item: (item.start_at, item.sample_id))


def make_plan(events: list[Event], folds: int, min_train_times: int, test_times: int, gap_hours: int) -> dict[str, Any]:
    if min(folds, min_train_times, test_times) < 1 or gap_hours < 0:
        raise InputError("folds, min-train-times and test-times must be positive; gap-hours must be nonnegative")
    unique_times = sorted({event.start_at for event in events})
    required = min_train_times + folds * test_times
    if len(unique_times) < required:
        raise InputError(f"Need at least {required} distinct start times, found {len(unique_times)}")
    gap = timedelta(hours=gap_hours)
    output = []
    for fold_number in range(folds):
        first = min_train_times + fold_number * test_times
        test_start = unique_times[first]
        test_window = set(unique_times[first:first + test_times])
        prior = [event for event in events if event.start_at < test_start]
        train = [event.sample_id for event in prior if event.end_at + gap <= test_start]
        purged = [event.sample_id for event in prior if event.end_at + gap > test_start]
        test = [event.sample_id for event in events if event.start_at in test_window]
        if not train:
            raise InputError(f"Fold {fold_number + 1} has no training events after purge")
        output.append({
            "fold": fold_number + 1,
            "test_start": format_time(test_start),
            "test_last_start": format_time(unique_times[first + test_times - 1]),
            "train_ids": train,
            "test_ids": test,
            "purged_ids": purged,
        })
    return {"version": 1, "gap_hours": gap_hours, "folds": output}


def audit_plan(events: list[Event], plan: Any) -> list[str]:
    if not isinstance(plan, dict) or plan.get("version") != 1 or type(plan.get("gap_hours")) is not int or plan["gap_hours"] < 0 or not isinstance(plan.get("folds"), list):
        raise InputError("Unsupported split plan format")
    lookup = {event.sample_id: event for event in events}
    gap = timedelta(hours=plan["gap_hours"])
    violations = []
    previous_end: datetime | None = None
    for number, fold in enumerate(plan["folds"], start=1):
        if not isinstance(fold, dict):
            raise InputError(f"Fold {number} is not an object")
        test_start = parse_time(fold.get("test_start"), f"Fold {number} test_start")
        test_last = parse_time(fold.get("test_last_start"), f"Fold {number} test_last_start")
        if test_last < test_start:
            violations.append(f"Fold {number}: test_last_start precedes test_start")
        if previous_end is not None and test_start <= previous_end:
            violations.append(f"Fold {number}: test windows overlap or are out of order")
        previous_end = test_last
        groups = {}
        for field in ("train_ids", "test_ids", "purged_ids"):
            ids = fold.get(field)
            if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
                raise InputError(f"Fold {number}: {field} must be a list of IDs")
            groups[field] = ids
            if len(ids) != len(set(ids)):
                violations.append(f"Fold {number}: duplicate IDs in {field}")
            for sample_id in ids:
                if sample_id not in lookup:
                    violations.append(f"Fold {number}: unknown ID {sample_id}")
        train, test, purged = (set(groups[name]) for name in ("train_ids", "test_ids", "purged_ids"))
        if not train:
            violations.append(f"Fold {number}: no training events")
        if not test:
            violations.append(f"Fold {number}: no test events")
        if train & test or train & purged or test & purged:
            violations.append(f"Fold {number}: IDs overlap between train, test, or purged groups")
        expected_test = {event.sample_id for event in events if test_start <= event.start_at <= test_last}
        expected_train = {event.sample_id for event in events if event.start_at < test_start and event.end_at + gap <= test_start}
        expected_purged = {event.sample_id for event in events if event.start_at < test_start and event.end_at + gap > test_start}
        for label, actual, expected in (("test", test, expected_test), ("train", train, expected_train), ("purged", purged, expected_purged)):
            missing = expected - actual
            extra = actual - expected
            if missing:
                violations.append(f"Fold {number}: missing {label} IDs: {', '.join(sorted(missing))}")
            if extra:
                violations.append(f"Fold {number}: unexpected {label} IDs: {', '.join(sorted(extra))}")
        for sample_id in sorted(train):
            event = lookup.get(sample_id)
            if event is not None and (event.start_at >= test_start or event.end_at + gap > test_start):
                violations.append(f"Fold {number}: training label leaks into test window: {sample_id}")
        for sample_id in sorted(purged):
            event = lookup.get(sample_id)
            if event is not None and event.start_at >= test_start:
                violations.append(f"Fold {number}: purged event starts in test or future: {sample_id}")
    return violations
