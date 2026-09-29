"""Reproducible real-data case study using NOAA-derived Seattle weather.

The source file has observation dates, not actual release timestamps. This
case study explicitly assumes yesterday's maximum temperature becomes usable
at midnight today. That is a modeling scenario, not verified NOAA latency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path


SOURCE_URL = (
    "https://raw.githubusercontent.com/vega/vega-datasets/"
    "ede9366badecc625cd6bcea5c4aaa055870c6ca6/data/seattle-weather.csv"
)
SOURCE_SHA256 = "0845078a290b48e3149ab8639966824110a251db4e06fc144c06ebb534af23be"


def run(*arguments: object) -> None:
    subprocess.run([sys.executable, "-m", "chronoguard", *(str(value) for value in arguments)], check=True)


def write_table(path: Path, fields: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def timestamp(day: date) -> str:
    return datetime.combine(day, time.min, timezone.utc).isoformat().replace("+00:00", "Z")


def prepare(source_bytes: bytes, root: Path) -> tuple[Path, Path, Path, Path, int]:
    reader = csv.DictReader(io.StringIO(source_bytes.decode("utf-8-sig")))
    if not reader.fieldnames or not {"date", "temp_max"}.issubset(reader.fieldnames):
        raise ValueError("Source CSV needs date and temp_max columns")
    records = []
    for row in reader:
        day = date.fromisoformat(row["date"])
        value = float(row["temp_max"])
        records.append((day, value))
    if len(records) < 550 or any(left[0] >= right[0] for left, right in zip(records, records[1:])):
        raise ValueError("Expected at least 550 chronologically ordered daily records")

    samples = []
    facts = []
    targets = []
    events = []
    for day, value in records:
        facts.append(("Seattle", timestamp(day + timedelta(days=1)), str(value)))
    for day, value in records[1:]:
        sample_id = day.isoformat()
        samples.append((sample_id, "Seattle", timestamp(day)))
        targets.append((sample_id, str(value)))
        events.append((sample_id, timestamp(day), timestamp(day + timedelta(days=1))))
    samples_path, facts_path, targets_path, events_path = (root / name for name in
                                                          ("samples.csv", "facts.csv", "targets.csv", "events.csv"))
    write_table(samples_path, ("sample_id", "entity", "decision_time"), samples)
    write_table(facts_path, ("entity", "available_at", "value"), facts)
    write_table(targets_path, ("sample_id", "target"), targets)
    write_table(events_path, ("sample_id", "start_at", "end_at"), events)
    return samples_path, facts_path, targets_path, events_path, len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the audited Seattle weather case study")
    parser.add_argument("--source", type=Path, help="local copy of the pinned CSV for offline use")
    parser.add_argument("--output-dir", type=Path, help="save input and report artifacts here; directory must not exist")
    args = parser.parse_args()
    if args.source:
        source_bytes = args.source.read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
            source_bytes = response.read()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Pinned source SHA-256 mismatch: {digest}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        root = args.output_dir
        temporary = None
    else:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
    try:
        samples, facts, targets, events, count = prepare(source_bytes, root)
        joined, observations, plan, report = (root / name for name in
                                              ("joined.csv", "observations.csv", "plan.json", "report.json"))
        run("join", samples, facts, joined, "--max-age-hours", 24)
        run("audit-join", samples, facts, joined, "--max-age-hours", 24)
        run("assemble", samples, facts, joined, targets, observations, "--max-age-hours", 24)
        run("split", events, plan, "--folds", 5, "--min-train-times", 365, "--test-times", 90)
        run("audit", events, plan)
        run("evaluate", events, plan, observations, report, "--feature", "signal",
            "--bootstrap-reps", 500, "--block-size", 7, "--seed", 23)
        run("audit-evaluation", events, plan, observations, report)
        result = json.loads(report.read_text(encoding="utf-8"))
        print(f"Source SHA-256: {digest}; daily samples: {count}")
        print("Held-out RMSE:", result["overall"]["rmse"])
        print("Train-mean baseline RMSE:", result["overall_baseline"]["rmse"])
        print("Paired loss-delta interval:", result["loss_comparison"]["interval_95"])
        if args.output_dir:
            print("Artifacts:", root)
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
