import csv
import io
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from chronoguard.common import read_csv
from examples.run_seattle_weather import prepare


class SeattleExampleTests(unittest.TestCase):
    def test_lag_mapping_is_explicit_and_has_no_same_day_fact(self):
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(("date", "temp_max"))
        first = date(2020, 1, 1)
        for index in range(551):
            writer.writerow(((first + timedelta(days=index)).isoformat(), str(index)))
        with tempfile.TemporaryDirectory() as temporary:
            samples, facts, targets, events, count = prepare(stream.getvalue().encode(), Path(temporary))
            self.assertEqual(count, 550)
            first_sample = read_csv(samples, ("sample_id", "decision_time"))[0]
            first_fact = read_csv(facts, ("available_at", "value"))[0]
            first_target = read_csv(targets, ("sample_id", "target"))[0]
            first_event = read_csv(events, ("sample_id", "start_at", "end_at"))[0]
            self.assertEqual(first_sample["decision_time"], first_fact["available_at"])
            self.assertEqual(first_fact["value"], "0.0")
            self.assertEqual(first_target["target"], "1.0")
            self.assertEqual(first_event["start_at"], first_sample["decision_time"])


if __name__ == "__main__":
    unittest.main()
