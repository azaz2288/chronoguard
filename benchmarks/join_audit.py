"""Reproducible synthetic join/audit benchmark; not a correctness test."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path

from chronoguard.common import write_csv
from chronoguard.join import OUTPUT_FIELDS, Fact, Sample, audit_join, point_in_time_join


def measure(operation):
    tracemalloc.start()
    start = time.perf_counter()
    result = operation()
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {"seconds": seconds, "peak_python_mib": peak / (1024 * 1024)}


def benchmark(rows: int, entities: int) -> dict[str, object]:
    if rows < 1 or entities < 1 or entities > rows:
        raise ValueError("Require rows >= entities >= 1")
    origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
    facts = []
    samples = []
    for index in range(rows):
        entity = f"E{index % entities}"
        available = origin + timedelta(hours=index)
        facts.append(Fact(entity, available, str(index)))
        samples.append(Sample(f"S{index}", entity, available + timedelta(minutes=30)))
    joined, join_stats = measure(lambda: point_in_time_join(samples, facts))
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "joined.csv"
        write_csv(path, OUTPUT_FIELDS, joined)
        issues, audit_stats = measure(lambda: audit_join(samples, facts, path))
    if issues or any(row["feature_value"] != str(index) for index, row in enumerate(joined)):
        raise RuntimeError(f"Benchmark failed its correctness check: {issues[:3]}")
    return {"rows": rows, "entities": entities, "python": platform.python_version(),
            "platform": platform.platform(), "join": join_stats, "audit": audit_stats,
            "audit_to_join_time_ratio": audit_stats["seconds"] / join_stats["seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure synthetic point-in-time join and independent audit")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--entities", type=int, default=100)
    parser.add_argument("--output", type=Path, help="optional JSON result path; must not exist")
    args = parser.parse_args()
    result = benchmark(args.rows, args.entities)
    content = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as target:
            target.write(content)
    print(content, end="")


if __name__ == "__main__":
    main()
