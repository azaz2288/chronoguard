import csv
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from chronoguard.common import InputError, read_csv
from chronoguard.join import Fact, Sample, audit_join, point_in_time_join
from chronoguard.split import Event, audit_plan, make_plan


def when(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 1, day, hour, tzinfo=timezone.utc)


def write_table(path: Path, fields: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(fields)
        writer.writerows(rows)


class JoinTests(unittest.TestCase):
    def test_publication_time_prevents_future_leakage(self):
        samples = [Sample("s1", "A", when(2)), Sample("s2", "A", when(3)), Sample("s3", "B", when(3))]
        facts = [Fact("A", when(1), "old"), Fact("A", when(3), "new"), Fact("B", when(4), "future")]
        rows = point_in_time_join(samples, facts)
        self.assertEqual([row["feature_value"] for row in rows], ["old", "new", ""])
        self.assertEqual(rows[2]["matched"], "0")

    def test_same_instant_is_available(self):
        row = point_in_time_join([Sample("x", "A", when(2))], [Fact("A", when(2), "ready")])[0]
        self.assertEqual(row["feature_value"], "ready")


class SplitTests(unittest.TestCase):
    def test_purges_label_reaching_test_window(self):
        events = [
            Event("a", when(1), when(1)),
            Event("b", when(2), when(4, 1)),
            Event("c", when(3), when(3)),
            Event("d", when(4), when(4)),
            Event("e", when(5), when(5)),
        ]
        plan = make_plan(events, folds=2, min_train_times=3, test_times=1, gap_hours=0)
        self.assertEqual(plan["folds"][0]["train_ids"], ["a", "c"])
        self.assertEqual(plan["folds"][0]["purged_ids"], ["b"])
        self.assertEqual(plan["folds"][0]["test_ids"], ["d"])
        self.assertEqual(audit_plan(events, plan), [])
        plan["folds"][0]["train_ids"].append("b")
        self.assertTrue(any("leaks" in issue for issue in audit_plan(events, plan)))

    def test_gap_purges_nearby_training_events(self):
        events = [Event("a", when(1), when(1)), Event("b", when(2), when(2)), Event("c", when(3), when(3))]
        plan = make_plan(events, folds=1, min_train_times=2, test_times=1, gap_hours=25)
        self.assertEqual(plan["folds"][0]["train_ids"], ["a"])
        self.assertEqual(plan["folds"][0]["purged_ids"], ["b"])

    def test_insufficient_distinct_times(self):
        events = [Event("a", when(1), when(1)), Event("b", when(1), when(1))]
        with self.assertRaisesRegex(InputError, "distinct start times"):
            make_plan(events, folds=1, min_train_times=1, test_times=1, gap_hours=0)

    def test_audit_finds_omitted_events(self):
        events = [Event("a", when(1), when(1)), Event("b", when(2), when(2)), Event("c", when(3), when(3))]
        plan = make_plan(events, folds=1, min_train_times=2, test_times=1, gap_hours=0)
        plan["folds"][0]["train_ids"].remove("b")
        self.assertTrue(any("missing train IDs: b" in issue for issue in audit_plan(events, plan)))
        plan["folds"][0]["test_ids"].clear()
        self.assertTrue(any("missing test IDs: c" in issue for issue in audit_plan(events, plan)))


class CliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-m", "chronoguard", *args], capture_output=True, text=True, check=False)

    def test_join_split_and_audit_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples.csv"
            facts = root / "facts.csv"
            events = root / "events.csv"
            joined = root / "joined.csv"
            plan = root / "plan.json"
            write_table(samples, ("sample_id", "entity", "decision_time"), [("x", "A", "2026-01-02T00:00:00Z")])
            write_table(facts, ("entity", "available_at", "value"), [("A", "2026-01-01T00:00:00Z", "1"), ("A", "2026-01-03T00:00:00Z", "2")])
            write_table(events, ("sample_id", "start_at", "end_at"), [("a", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"), ("b", "2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z"), ("c", "2026-01-03T00:00:00Z", "2026-01-03T00:00:00Z")])

            self.assertEqual(self.run_cli("join", str(samples), str(facts), str(joined)).returncode, 0)
            self.assertEqual(read_csv(joined, ("feature_value",))[0]["feature_value"], "1")
            self.assertEqual(self.run_cli("audit-join", str(samples), str(facts), str(joined)).returncode, 0)
            self.assertEqual(self.run_cli("split", str(events), str(plan), "--folds", "1", "--min-train-times", "2", "--test-times", "1").returncode, 0)
            self.assertEqual(self.run_cli("audit", str(events), str(plan)).returncode, 0)
            self.assertEqual(self.run_cli("join", str(samples), str(facts), str(joined)).returncode, 2)
            saved = json.loads(plan.read_text(encoding="utf-8"))
            saved["folds"][0]["train_ids"].append("c")
            plan.write_text(json.dumps(saved), encoding="utf-8")
            self.assertEqual(self.run_cli("audit", str(events), str(plan)).returncode, 1)

    def test_join_audit_detects_future_fact_and_missing_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            joined = root / "joined.csv"
            samples = [Sample("a", "A", when(2)), Sample("b", "B", when(2))]
            facts = [Fact("A", when(1), "old"), Fact("A", when(3), "future")]
            write_table(joined, ("sample_id", "entity", "decision_time", "matched", "feature_available_at", "feature_value"), [
                ("a", "A", "2026-01-02T00:00:00Z", "1", "2026-01-03T00:00:00Z", "future")
            ])
            issues = audit_join(samples, facts, joined)
            self.assertTrue(any("feature_available_at mismatch" in issue for issue in issues))
            self.assertTrue(any("Missing joined sample_id 'b'" in issue for issue in issues))

    def test_rejects_malformed_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            path.write_text("sample_id,entity,decision_time\nx,A\n", encoding="utf-8")
            with self.assertRaisesRegex(InputError, "different number of columns"):
                read_csv(path, ("sample_id", "entity", "decision_time"))


if __name__ == "__main__":
    unittest.main()
