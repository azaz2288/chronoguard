import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from chronoguard.assemble import assemble
from chronoguard.common import InputError, read_csv, write_csv
from chronoguard.join import OUTPUT_FIELDS, load_facts, load_samples, point_in_time_join


def table(path: Path, fields: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(fields)
        writer.writerows(rows)


class AssembleTests(unittest.TestCase):
    def fixture(self, root: Path):
        samples, facts, joined, targets, output = (root / name for name in
                                                    ("samples.csv", "facts.csv", "joined.csv", "targets.csv", "observations.csv"))
        table(samples, ("sample_id", "entity", "decision_time"), [
            ("a", "X", "2026-01-02T00:00:00Z"), ("b", "Y", "2026-01-02T00:00:00Z")])
        table(facts, ("entity", "available_at", "value"), [
            ("X", "2026-01-01T00:00:00Z", "2.5"), ("Y", "2026-01-03T00:00:00Z", "10")])
        write_csv(joined, OUTPUT_FIELDS, point_in_time_join(load_samples(samples), load_facts(facts)))
        table(targets, ("sample_id", "target"), [("a", "4"), ("b", "6")])
        return samples, facts, joined, targets, output

    def test_clean_join_becomes_observations_and_cli_works(self):
        with tempfile.TemporaryDirectory() as temporary:
            sources = self.fixture(Path(temporary))
            rows = assemble(*sources[:4])
            self.assertEqual(rows, [
                {"sample_id": "a", "target": "4", "signal": "2.5"},
                {"sample_id": "b", "target": "6", "signal": ""},
            ])
            result = subprocess.run([sys.executable, "-m", "chronoguard", "assemble", *(str(path) for path in sources)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(read_csv(sources[4], ("sample_id", "target", "signal")), rows)

    def test_rejects_tampered_join_before_materializing(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, facts, joined, targets, output = self.fixture(Path(temporary))
            rows = read_csv(joined, OUTPUT_FIELDS)
            rows[0]["feature_value"] = "999"
            write_csv(joined, OUTPUT_FIELDS, rows)
            with self.assertRaisesRegex(InputError, "failed audit"):
                assemble(samples, facts, joined, targets)
            self.assertFalse(output.exists())

    def test_rejects_missing_targets_and_non_numeric_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            samples, facts, joined, targets, _ = self.fixture(Path(temporary))
            table(targets, ("sample_id", "target"), [("a", "4")])
            with self.assertRaisesRegex(InputError, "Targets do not match"):
                assemble(samples, facts, joined, targets)
            table(targets, ("sample_id", "target"), [("a", "4"), ("b", "6")])
            table(facts, ("entity", "available_at", "value"), [("X", "2026-01-01T00:00:00Z", "hello")])
            write_csv(joined, OUTPUT_FIELDS, point_in_time_join(load_samples(samples), load_facts(facts)))
            with self.assertRaisesRegex(InputError, "must be numeric"):
                assemble(samples, facts, joined, targets)


if __name__ == "__main__":
    unittest.main()
