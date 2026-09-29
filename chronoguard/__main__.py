"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .common import InputError, check_output, read_json, write_csv, write_json
from .join import OUTPUT_FIELDS, audit_join, load_facts, load_samples, point_in_time_join
from .split import audit_plan, load_events, make_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit point-in-time joins and temporal validation splits")
    commands = parser.add_subparsers(dest="command", required=True)
    join = commands.add_parser("join", help="join only features available by each decision time")
    join.add_argument("samples", type=Path)
    join.add_argument("facts", type=Path)
    join.add_argument("output", type=Path)
    join.add_argument("--force", action="store_true")

    audit_join_cmd = commands.add_parser("audit-join", help="check an existing join against source samples and facts")
    audit_join_cmd.add_argument("samples", type=Path)
    audit_join_cmd.add_argument("facts", type=Path)
    audit_join_cmd.add_argument("joined", type=Path)

    split = commands.add_parser("split", help="create a purged walk-forward plan")
    split.add_argument("events", type=Path)
    split.add_argument("output", type=Path)
    split.add_argument("--folds", type=int, required=True)
    split.add_argument("--min-train-times", type=int, required=True)
    split.add_argument("--test-times", type=int, required=True)
    split.add_argument("--gap-hours", type=int, default=0)
    split.add_argument("--force", action="store_true")

    audit = commands.add_parser("audit", help="audit an existing plan for label leakage")
    audit.add_argument("events", type=Path)
    audit.add_argument("plan", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "join":
            check_output(args.output, (args.samples, args.facts), args.force)
            rows = point_in_time_join(load_samples(args.samples), load_facts(args.facts))
            write_csv(args.output, OUTPUT_FIELDS, rows)
            print(f"Wrote {len(rows)} samples; {sum(row['matched'] == '0' for row in rows)} had no available fact")
        elif args.command == "audit-join":
            violations = audit_join(load_samples(args.samples), load_facts(args.facts), args.joined)
            for violation in violations:
                print(violation)
            if violations:
                return 1
            print("OK: joined rows match information available at each decision")
        elif args.command == "split":
            check_output(args.output, (args.events,), args.force)
            events = load_events(args.events)
            plan = make_plan(events, args.folds, args.min_train_times, args.test_times, args.gap_hours)
            violations = audit_plan(events, plan)
            if violations:
                raise InputError("Internal split audit failed: " + "; ".join(violations))
            write_json(args.output, plan)
            print(f"Wrote {len(plan['folds'])} audited folds")
        else:
            violations = audit_plan(load_events(args.events), read_json(args.plan))
            for violation in violations:
                print(violation)
            if violations:
                return 1
            print("OK: no train/test label leakage found")
        return 0
    except InputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
