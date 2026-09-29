"""Deterministic randomized cross-checks for report generation and audit."""

import copy
import random
import unittest

from chronoguard.audit_evaluation import audit_evaluation
from chronoguard.evaluate import Observation, evaluate
from chronoguard.split import Event, make_plan
from datetime import datetime, timedelta, timezone


class EvaluationProperties(unittest.TestCase):
    def test_diverse_small_experiments_round_trip_and_tamper(self):
        randomizer = random.Random(20260929)
        origin = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for case in range(40):
            with self.subTest(case=case):
                events = []
                observations = {}
                count = randomizer.randrange(6, 13)
                for index in range(count):
                    identifier = f"s{index}"
                    start = origin + timedelta(days=index)
                    end = start + timedelta(hours=randomizer.choice([0, 1, 12, 24, 36]))
                    events.append(Event(identifier, start, end))
                    first = randomizer.uniform(-5, 5)
                    second = randomizer.uniform(-3, 3) if index % 4 else None
                    target = 2 * first + (second or 0) + randomizer.uniform(-0.5, 0.5)
                    observations[identifier] = Observation(identifier, target, (first, second))
                plan = make_plan(events, folds=2, min_train_times=4, test_times=1, gap_hours=0)
                report = evaluate(events, plan, observations, ["first", "second"], 0.5, {"events": "a", "plan": "b", "observations": "c"})
                self.assertEqual(audit_evaluation(events, plan, observations, report, report["source_sha256"]), [])
                changed = copy.deepcopy(report)
                changed["folds"][0]["predictions"][0]["target"] += 0.1
                self.assertTrue(any("target mismatch" in issue for issue in
                                    audit_evaluation(events, plan, observations, changed, report["source_sha256"])))


if __name__ == "__main__":
    unittest.main()
