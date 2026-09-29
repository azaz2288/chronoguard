import csv
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from chronoguard.common import InputError, read_csv
from chronoguard.audit_evaluation import audit_evaluation
from chronoguard.evaluate import Observation, evaluate, load_observations
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

    def test_age_policy_excludes_stale_fact_at_exact_boundary(self):
        samples = [Sample("edge", "A", when(2)), Sample("stale", "A", when(2, 1))]
        facts = [Fact("A", when(1), "old")]
        rows = point_in_time_join(samples, facts, max_age_hours=24)
        self.assertEqual([row["matched"] for row in rows], ["1", "0"])
        self.assertEqual(rows[1]["feature_value"], "")

    def test_invalid_age_policy(self):
        with self.assertRaisesRegex(InputError, "max-age-hours"):
            point_in_time_join([], [], max_age_hours=-1)
        with self.assertRaisesRegex(InputError, "max-age-hours"):
            point_in_time_join([], [], max_age_hours=1_000_001)


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

    def test_audit_rejects_empty_or_misnumbered_folds(self):
        events = [Event("a", when(1), when(1)), Event("b", when(2), when(2)), Event("c", when(3), when(3))]
        self.assertIn("Split plan has no folds", audit_plan(events, {"version": 1, "gap_hours": 0, "folds": []}))
        plan = make_plan(events, folds=1, min_train_times=2, test_times=1, gap_hours=0)
        plan["folds"][0]["fold"] = 2
        self.assertTrue(any("out of sequence" in issue for issue in audit_plan(events, plan)))


class EvaluationTests(unittest.TestCase):
    def fixture(self):
        events = [Event(str(index), when(index), when(index)) for index in range(1, 6)]
        plan = make_plan(events, folds=2, min_train_times=2, test_times=1, gap_hours=0)
        observations = {str(index): Observation(str(index), float(2 * index + 1), (float(index),)) for index in range(1, 6)}
        return events, plan, observations

    def test_train_only_statistics_and_out_of_sample_rows(self):
        events, plan, observations = self.fixture()
        result = evaluate(events, plan, observations, ["signal"], 0.1, {"events": "test"})
        self.assertEqual(result["overall"]["count"], 2)
        self.assertLess(result["overall"]["rmse"], result["overall_baseline"]["rmse"])
        self.assertEqual(result["folds"][0]["model"]["impute_means"], [1.5])
        self.assertEqual([row["sample_id"] for row in result["folds"][0]["predictions"]], ["3"])
        self.assertEqual(result["source_sha256"], {"events": "test"})
        altered = dict(observations)
        altered["3"] = Observation("3", 100000.0, (100000.0,))
        other = evaluate(events, plan, altered, ["signal"], 0.1, {})
        self.assertEqual(result["folds"][0]["model"], other["folds"][0]["model"])

    def test_missing_features_use_training_imputation(self):
        events, plan, observations = self.fixture()
        observations["1"] = Observation("1", 3.0, (None,))
        observations["3"] = Observation("3", 7.0, (None,))
        result = evaluate(events, plan, observations, ["signal"], 1.0, {})
        self.assertEqual(result["folds"][0]["model"]["impute_means"], [2.0])
        self.assertAlmostEqual(result["folds"][0]["predictions"][0]["prediction"], 4.0)

    def test_rejects_invalid_plan_alpha_and_observations(self):
        events, plan, observations = self.fixture()
        plan["folds"][0]["train_ids"].append("3")
        with self.assertRaisesRegex(InputError, "failed audit"):
            evaluate(events, plan, observations, ["signal"], 1.0, {})
        plan["folds"][0]["train_ids"].pop()
        with self.assertRaisesRegex(InputError, "no folds"):
            evaluate(events, {"version": 1, "gap_hours": 0, "folds": []}, observations, ["signal"], 1.0, {})
        with self.assertRaisesRegex(InputError, "alpha"):
            evaluate(events, plan, observations, ["signal"], 0.0, {})
        observations.pop("5")
        with self.assertRaisesRegex(InputError, "do not match"):
            evaluate(events, plan, observations, ["signal"], 1.0, {})

    def test_rejects_duplicate_ids_and_nonfinite_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.csv"
            write_table(path, ("sample_id", "target", "signal"), [("a", "1", "2"), ("a", "2", "3")])
            with self.assertRaisesRegex(InputError, "duplicate sample_id"):
                load_observations(path, ["signal"])
            write_table(path, ("sample_id", "target", "signal"), [("a", "1", "NaN")])
            with self.assertRaisesRegex(InputError, "non-finite"):
                load_observations(path, ["signal"])

    def test_independent_report_audit_finds_tampering(self):
        events, plan, observations = self.fixture()
        hashes = {"events": "a" * 64, "plan": "b" * 64, "observations": "c" * 64}
        report = evaluate(events, plan, observations, ["signal"], 0.1, hashes)
        self.assertEqual(audit_evaluation(events, plan, observations, report, hashes), [])

        changed = copy.deepcopy(report)
        changed["folds"][0]["predictions"][0]["prediction"] += 1
        self.assertTrue(any("prediction mismatch" in item for item in audit_evaluation(events, plan, observations, changed, hashes)))

        changed = copy.deepcopy(report)
        changed["folds"][0]["model"]["impute_means"][0] += 1
        self.assertTrue(any("imputation mean mismatch" in item for item in audit_evaluation(events, plan, observations, changed, hashes)))

        changed = copy.deepcopy(report)
        changed["folds"][0]["model"]["coefficients"][0] += 1
        self.assertTrue(any("normal equation" in item for item in audit_evaluation(events, plan, observations, changed, hashes)))

        changed = copy.deepcopy(report)
        changed["overall"]["rmse"] += 1
        self.assertTrue(any("Overall metrics.rmse mismatch" in item for item in audit_evaluation(events, plan, observations, changed, hashes)))

        changed = copy.deepcopy(report)
        changed["loss_comparison"]["mean_loss_delta"] += 1
        self.assertTrue(any("loss_comparison.mean_loss_delta mismatch" in item for item in audit_evaluation(events, plan, observations, changed, hashes)))

        changed = copy.deepcopy(report)
        changed["folds"][1]["model"]["alpha"] += 1
        self.assertTrue(any("alpha differs" in item for item in audit_evaluation(events, plan, observations, changed, hashes)))

        self.assertIn("Source SHA-256 fingerprints do not match current inputs",
                      audit_evaluation(events, plan, observations, report, {**hashes, "events": "changed"}))

    def test_report_audit_rejects_malformed_values(self):
        events, plan, observations = self.fixture()
        report = evaluate(events, plan, observations, ["signal"], 1.0, {})
        report["folds"][0]["model"]["alpha"] = float("nan")
        with self.assertRaisesRegex(InputError, "finite number"):
            audit_evaluation(events, plan, observations, report, {})


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

    def test_cli_age_policy_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples, facts, joined = (root / name for name in ("samples.csv", "facts.csv", "joined.csv"))
            write_table(samples, ("sample_id", "entity", "decision_time"), [("x", "A", "2026-01-03T01:00:00Z")])
            write_table(facts, ("entity", "available_at", "value"), [("A", "2026-01-01T00:00:00Z", "stale")])
            self.assertEqual(self.run_cli("join", str(samples), str(facts), str(joined), "--max-age-hours", "24").returncode, 0)
            self.assertEqual(read_csv(joined, ("matched",))[0]["matched"], "0")
            self.assertEqual(self.run_cli("audit-join", str(samples), str(facts), str(joined), "--max-age-hours", "24").returncode, 0)
            self.assertEqual(self.run_cli("audit-join", str(samples), str(facts), str(joined)).returncode, 1)
            self.assertEqual(self.run_cli("audit-join", str(samples), str(facts), str(joined), "--max-age-hours", "-1").returncode, 2)

    def test_rejects_malformed_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.csv"
            path.write_text("sample_id,entity,decision_time\nx,A\n", encoding="utf-8")
            with self.assertRaisesRegex(InputError, "different number of columns"):
                read_csv(path, ("sample_id", "entity", "decision_time"))

    def test_evaluate_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            events, observations, plan, report = (root / name for name in ("events.csv", "observations.csv", "plan.json", "report.json"))
            event_rows = [(str(index), f"2026-01-0{index}T00:00:00Z", f"2026-01-0{index}T00:00:00Z") for index in range(1, 6)]
            write_table(events, ("sample_id", "start_at", "end_at"), event_rows)
            write_table(observations, ("sample_id", "target", "signal"), [(str(index), str(2 * index + 1), str(index)) for index in range(1, 6)])
            self.assertEqual(self.run_cli("split", str(events), str(plan), "--folds", "2", "--min-train-times", "2", "--test-times", "1").returncode, 0)
            run = self.run_cli("evaluate", str(events), str(plan), str(observations), str(report), "--feature", "signal")
            self.assertEqual(run.returncode, 0, run.stderr)
            saved = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(saved["overall"]["count"], 2)
            self.assertEqual(len(saved["source_sha256"]["observations"]), 64)
            self.assertEqual(self.run_cli("audit-evaluation", str(events), str(plan), str(observations), str(report)).returncode, 0)
            saved["folds"][0]["predictions"][0]["prediction"] += 10
            report.write_text(json.dumps(saved), encoding="utf-8")
            self.assertEqual(self.run_cli("audit-evaluation", str(events), str(plan), str(observations), str(report)).returncode, 1)
            self.assertEqual(self.run_cli("evaluate", str(events), str(plan), str(observations), str(report), "--feature", "signal").returncode, 2)


if __name__ == "__main__":
    unittest.main()
