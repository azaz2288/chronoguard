"""Synthetic race regressions; also executable against an installed wheel."""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chronoguard.__main__ import main
from chronoguard.evaluate import load_observations
from chronoguard.common import InputError, InputSnapshot, read_csv, read_json


class SnapshotCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events, self.plan, self.observations, self.report = (
            self.root / name for name in ('events.csv', 'plan.json', 'obs.csv', 'report.json'))
        self.events.write_text('sample_id,start_at,end_at\n' + ''.join(
            f'{i},2026-01-0{i}T00:00:00Z,2026-01-0{i}T00:00:00Z\n' for i in range(1, 6)), encoding='utf-8')
        self.observations.write_text('sample_id,target,signal\n' + ''.join(
            f'{i},{i * 2 + 1},{i}\n' for i in range(1, 6)), encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['split', str(self.events), str(self.plan), '--folds', '2',
                                   '--min-train-times', '2', '--test-times', '1']), 0)

    def evaluate_args(self):
        return ['evaluate', str(self.events), str(self.plan), str(self.observations),
                str(self.report), '--feature', 'signal', '--bootstrap-reps', '100']

    def audit_args(self):
        return ['audit-evaluation', str(self.events), str(self.plan), str(self.observations), str(self.report)]

    def test_evaluation_hashes_match_parsed_bytes_after_path_changes(self):
        originals = {name: path.read_bytes() for name, path in (
            ('events', self.events), ('plan', self.plan), ('observations', self.observations))}

        def mutate_after_parse(source, features):
            result = load_observations(source, features)
            for path in (self.events, self.plan, self.observations):
                path.write_bytes(b'replaced after parsing')
            return result

        with patch('chronoguard.__main__.load_observations', side_effect=mutate_after_parse), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.evaluate_args()), 0)
        report = json.loads(self.report.read_text(encoding='utf-8'))
        self.assertEqual(report['source_sha256'], {name: hashlib.sha256(data).hexdigest()
                                                   for name, data in originals.items()})
        for name, path in (('events', self.events), ('plan', self.plan), ('observations', self.observations)):
            path.write_bytes(originals[name])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.audit_args()), 0)

    def test_auditor_hashes_match_its_parsed_bytes_after_path_changes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.evaluate_args()), 0)

        def mutate_after_parse(source, features):
            result = load_observations(source, features)
            for path in (self.events, self.plan, self.observations):
                path.unlink()
                path.write_bytes(b'replaced after audit parsing')
            return result

        with patch('chronoguard.__main__.load_observations', side_effect=mutate_after_parse), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.audit_args()), 0)

    def test_capture_failure_preserves_existing_report_even_with_force(self):
        self.report.write_bytes(b'original report')
        with patch('chronoguard.__main__.InputSnapshot.capture', side_effect=InputError('changed')), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(self.evaluate_args() + ['--force']), 2)
        self.assertEqual(self.report.read_bytes(), b'original report')

    def test_changed_source_after_evaluation_is_audit_violation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.evaluate_args()), 0)
        self.observations.write_bytes(self.observations.read_bytes().replace(b'1,3,1', b'1,9,1'))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(self.audit_args()), 1)

    def test_real_subprocess_evaluate_and_audit(self):
        for args in (self.evaluate_args(), self.audit_args()):
            result = subprocess.run([sys.executable, '-m', 'chronoguard', *args],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(self.report.read_text(encoding='utf-8'))
        self.assertEqual(report['source_sha256']['observations'], hashlib.sha256(self.observations.read_bytes()).hexdigest())


class SnapshotCaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'input.csv'
        self.path.write_bytes(b'header\noriginal\n')

    def capture_with_mutation(self, action):
        original_open = Path.open
        path = self.path

        class Reader:
            def __enter__(inner):
                inner.source = original_open(path, 'rb')
                return inner

            def __exit__(inner, *args):
                inner.source.close()

            def fileno(inner):
                return inner.source.fileno()

            def read(inner, size):
                data = inner.source.read(size)
                action()
                return data

        def opened(candidate, *args, **kwargs):
            return Reader() if candidate == path and args == ('rb',) else original_open(candidate, *args, **kwargs)

        with patch.object(Path, 'open', opened):
            return InputSnapshot.capture(path)

    def test_growth_and_truncation_during_capture_rejected(self):
        for value in (b'x' * 10000, b''):
            self.path.write_bytes(b'header\noriginal\n')
            with self.subTest(length=len(value)), self.assertRaisesRegex(InputError, 'changed during capture'):
                self.capture_with_mutation(lambda: self.path.write_bytes(value))

    def test_same_size_inplace_change_rejected(self):
        def change():
            before = self.path.stat()
            self.path.write_bytes(b'header\nmodified\n')
            os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns + 2_000_000_000))

        with self.assertRaisesRegex(InputError, 'changed during capture'):
            self.capture_with_mutation(change)

    def test_path_replacement_during_capture_rejected(self):
        # Windows cannot unlink an open file; emulate only the *path* stat of a
        # replacement there, while POSIX exercises an actual rename-over-open.
        if os.name == 'nt':
            alternative = self.path.with_suffix('.replacement')
            alternative.write_bytes(self.path.read_bytes())
            other_stat = alternative.stat()
            original_stat = Path.stat
            with patch.object(Path, 'stat', lambda path, *a, **k: other_stat if path == self.path else original_stat(path, *a, **k)):
                with self.assertRaisesRegex(InputError, 'changed during capture'):
                    InputSnapshot.capture(self.path)
        else:
            def replace():
                alternative = self.path.with_suffix('.replacement')
                alternative.write_bytes(self.path.read_bytes())
                alternative.replace(self.path)

            with self.assertRaisesRegex(InputError, 'changed during capture'):
                self.capture_with_mutation(replace)

    def test_snapshot_remains_parseable_after_original_deleted(self):
        captured = InputSnapshot.capture(self.path)
        self.path.unlink()
        self.assertEqual(read_csv(captured, ('header',)), [{'header': 'original'}])
        self.assertEqual(captured.sha256, hashlib.sha256(captured.data).hexdigest())
        with self.assertRaises(AttributeError):
            captured.data = b'changed'

    def test_csv_bom_crlf_quoted_newlines_and_nonascii(self):
        self.path.write_bytes('\ufeffid,value\r\nx,"你好\r\n世界"\r\n'.encode('utf-8'))
        captured = InputSnapshot.capture(self.path)
        self.assertEqual(read_csv(captured, ('id', 'value')), read_csv(self.path, ('id', 'value')))
        self.assertEqual(read_csv(captured, ('id', 'value'))[0]['value'], '你好\r\n世界')

    def test_snapshot_json_retains_strict_validation(self):
        for data in (b'{"a":1,"a":2}', b'{"a":NaN}', b'[1e400]', b'\xff'):
            self.path.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(InputError):
                read_json(InputSnapshot.capture(self.path))
        self.path.write_bytes(b'{"valid":[1,2]}')
        self.assertEqual(read_json(InputSnapshot.capture(self.path)), {'valid': [1, 2]})

    def test_missing_nonregular_and_read_error_rejected(self):
        with self.assertRaises(InputError):
            InputSnapshot.capture(self.path.with_suffix('.missing'))
        with self.assertRaises(InputError):
            InputSnapshot.capture(self.path.parent)
        with patch.object(Path, 'open', side_effect=OSError('read failure')):
            with self.assertRaisesRegex(InputError, 'Cannot capture'):
                InputSnapshot.capture(self.path)

    def test_invalid_snapshot_csv_rejected(self):
        for data in (b'\xff', b'id,id\na,b\n', b'id,value\na\n'):
            self.path.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(InputError):
                read_csv(InputSnapshot.capture(self.path), ('id',))

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'POSIX FIFO only')
    def test_fifo_rejected_before_blocking_open(self):
        fifo = self.path.with_suffix('.fifo')
        os.mkfifo(fifo)
        original_open = Path.open

        def forbidden(candidate, *args, **kwargs):
            if candidate == fifo:
                self.fail('attempted to open FIFO')
            return original_open(candidate, *args, **kwargs)

        with patch.object(Path, 'open', forbidden), self.assertRaisesRegex(InputError, 'regular file'):
            InputSnapshot.capture(fifo)


if __name__ == '__main__':
    unittest.main()
