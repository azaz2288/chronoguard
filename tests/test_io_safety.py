import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from chronoguard.common import InputError, check_output, read_json, write_csv, write_json
from chronoguard.__main__ import main


class OutputSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'result.json'

    def test_json_late_competing_output_not_overwritten(self):
        check_output(self.path, (), False)
        original_dump = json.dump

        def competing_writer(value, stream, **kwargs):
            self.path.write_text('other process result')
            return original_dump(value, stream, **kwargs)

        with patch('chronoguard.common.json.dump', side_effect=competing_writer):
            with self.assertRaises(InputError):
                write_json(self.path, {'own': 'result'})
        self.assertEqual(self.path.read_text(), 'other process result')
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['result.json'])

    def test_csv_late_competing_output_not_overwritten(self):
        class CompetingRows:
            def __iter__(inner):
                self.path.write_text('other CSV')
                yield {'value': 'mine'}

        with self.assertRaises(InputError):
            write_csv(self.path, ('value',), CompetingRows())
        self.assertEqual(self.path.read_text(), 'other CSV')
        self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_explicit_force_replaces_output(self):
        self.path.write_text('previous')
        write_json(self.path, {'new': True}, force=True)
        self.assertEqual(read_json(self.path), {'new': True})
        write_csv(self.path, ('value',), [{'value': 'new'}], force=True)
        self.assertEqual(self.path.read_text(), 'value\nnew\n')

    def test_symlink_conflict_not_followed_or_replaced(self):
        target = self.root / 'missing.json'
        try:
            self.path.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest('symlink creation unavailable')
        with self.assertRaises(InputError):
            write_json(self.path, {})
        self.assertTrue(self.path.is_symlink())
        self.assertFalse(target.exists())

    def test_json_rejects_ambiguous_and_nonfinite_content(self):
        for content in ('{"version":1,"version":2}', '{"folds":[{"train_ids":[],"train_ids":[1]}]}',
                        '{"metric":NaN}', '[Infinity]', '[-Infinity]', '[1e400]'):
            self.path.write_text(content)
            with self.subTest(content=content), self.assertRaises(InputError):
                read_json(self.path)

    def test_failed_serialization_leaves_old_output_and_no_temp(self):
        self.path.write_text('original')
        for invalid in ({'value': float('nan')}, {'value': float('inf')}, {'value': object()}):
            with self.subTest(invalid=invalid), self.assertRaises(InputError):
                write_json(self.path, invalid, force=True)
            self.assertEqual(self.path.read_text(), 'original')
            self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_output_link_failure_is_fail_closed(self):
        with patch('chronoguard.common.os.link', side_effect=OSError('unsupported links')):
            with self.assertRaises(InputError):
                write_json(self.path, {'valid': 1})
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_cli_force_reaches_publish_step(self):
        samples, facts = (self.root / name for name in ('samples.csv', 'facts.csv'))
        samples.write_text('sample_id,entity,decision_time\na,A,2026-01-02T00:00:00Z\n')
        facts.write_text('entity,available_at,value\nA,2026-01-01T00:00:00Z,1\n')
        self.path.write_text('old')
        self.assertEqual(main(['join', str(samples), str(facts), str(self.path), '--force']), 0)
        self.assertIn('sample_id,', self.path.read_text())

    def test_concurrent_publish_has_exactly_one_complete_winner(self):
        def publish(index):
            try:
                write_json(self.path, {'writer': index, 'payload': str(index) * 10000})
                return index
            except InputError:
                return None

        with ThreadPoolExecutor(max_workers=8) as workers:
            winners = [index for index in workers.map(publish, range(8)) if index is not None]
        self.assertEqual(len(winners), 1)
        self.assertEqual(read_json(self.path), {'writer': winners[0], 'payload': str(winners[0]) * 10000})
        self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_fsync_failure_does_not_publish_partial_output(self):
        with patch('chronoguard.common.os.fsync', side_effect=OSError('disk failure')):
            with self.assertRaises(InputError):
                write_json(self.path, {'value': 1})
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
