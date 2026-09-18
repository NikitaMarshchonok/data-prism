from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import uuid

from src.drift_store import DriftStore
from src.runtime_backup import BackupError, create_backup, restore_backup, verify_backup
from vibedash.analysis_jobs import AnalysisJobStore
from vibedash.decision_cases import DecisionCaseStore
from vibedash.pilot_metrics import build_pilot_report, scope_token


class RuntimeBackupTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.state = self.root / 'runtime'
        self.backup = self.root / 'snapshot'
        self.restored = self.root / 'recovered'
        self.scope = uuid.uuid4().hex
        self.session_id = str(uuid.uuid4())
        self.database = self.state / 'jobs' / 'analysis_jobs.sqlite3'
        store = AnalysisJobStore(self.database)
        self.job = store.create(self.scope, {
            'filename': 'synthetic.csv', 'prompt': 'Synthetic recovery review',
        }, pilot_scope_token=scope_token(self.scope, 'synthetic-key'))
        store.claim(self.job['id'])
        store.complete(self.job['id'], self.session_id)
        self.case = DecisionCaseStore(self.database).create(
            self.scope, self.job['id'], priority=1, owner='Synthetic role',
            decision='Verify recovery', success_metric='Readable case',
            target_outcome='Same evidence after restore', review_date='2026-12-01',
            evidence_snapshot={
                'contract': 'decision-case-source-v1', 'analysis_job_id': self.job['id'],
                'priority': {'number': 1, 'title': 'Synthetic trend', 'finding': 'Test evidence',
                             'confidence': 'low', 'evidence': ['Synthetic data only']},
            },
        )
        drift = DriftStore(self.state / 'drift' / 'drift_history.sqlite3', 'synthetic-scope')
        drift.record_run({'status': 'critical', 'baseline_created_at': '2026-01-01',
                          'summary': 'Synthetic drift', 'feature_drift': []}, batch_id='synthetic-batch')
        self.upload = self.state / 'uploads' / ('vibedash-' + 'b' * 32 + '.csv')
        self.upload.parent.mkdir()
        self.upload.write_text('value\n10\n20\n', encoding='utf-8')
        # Old mtime must stay old: recovery must not reset artifact retention.
        self.old_mtime = 1700000000000000000
        os.utime(self.upload, ns=(self.old_mtime, self.old_mtime))
        session = self.state / 'sessions' / 'vibedash' / f'{self.session_id}.json'
        session.parent.mkdir(parents=True)
        session.write_text(json.dumps({
            'analysis_scope_id': self.scope,
            'viz_spec': {'title': 'Recovery test'}, 'dashboard_data': {},
            'filename': 'synthetic.csv', 'prompt': 'Synthetic recovery review',
        }), encoding='utf-8')
        self.baseline = self.state / 'baselines' / 'synthetic.json'
        self.baseline.parent.mkdir()
        self.baseline.write_text('{"synthetic": true}', encoding='utf-8')

    def snapshot(self):
        return create_backup(self.state, self.backup, offline=True, revision='050fa75')

    def manifest(self):
        return json.loads((self.backup / 'manifest.json').read_text(encoding='utf-8'))

    def replace_manifest(self, value):
        (self.backup / 'manifest.json').write_text(json.dumps(value), encoding='utf-8')

    def test_round_trip_preserves_jobs_cases_metrics_drift_and_files(self):
        result = self.snapshot()
        self.assertEqual(result, verify_backup(self.backup))
        self.assertEqual(result, restore_backup(self.backup, self.restored, offline=True))
        database = self.restored / 'jobs' / 'analysis_jobs.sqlite3'
        self.assertEqual(AnalysisJobStore(database).get(self.job['id'], self.scope)['status'], 'completed')
        self.assertEqual(DecisionCaseStore(database).get(self.case['id'], self.scope)['evidence_snapshot'], self.case['evidence_snapshot'])
        self.assertEqual(build_pilot_report(database)['cohorts']['upload']['analyses_with_decision'], 1)
        self.assertEqual(len(DriftStore(self.restored / 'drift' / 'drift_history.sqlite3', 'synthetic-scope').list_alerts()), 1)
        restored_upload = self.restored / self.upload.relative_to(self.state)
        self.assertEqual(restored_upload.read_bytes(), self.upload.read_bytes())
        self.assertEqual(restored_upload.stat().st_mtime_ns, self.old_mtime)
        self.assertEqual((self.restored / 'baselines' / 'synthetic.json').read_bytes(), self.baseline.read_bytes())
        self.assertNotIn('Synthetic role', json.dumps(result))

    def test_no_offline_acknowledgment_means_no_output(self):
        with self.assertRaises(BackupError):
            create_backup(self.state, self.backup)
        self.assertFalse(self.backup.exists())
        self.snapshot()
        with self.assertRaises(BackupError):
            restore_backup(self.backup, self.restored)
        self.assertFalse(self.restored.exists())

    def test_never_overwrites_existing_even_empty_destinations(self):
        self.snapshot()
        before = (self.backup / 'manifest.json').read_bytes()
        with self.assertRaises(BackupError):
            self.snapshot()
        self.assertEqual((self.backup / 'manifest.json').read_bytes(), before)
        self.restored.mkdir()
        with self.assertRaises(BackupError):
            restore_backup(self.backup, self.restored, offline=True)
        self.assertEqual(list(self.restored.iterdir()), [])
        with self.assertRaises(BackupError):
            restore_backup(self.backup, self.state, offline=True)
        self.assertIsNotNone(AnalysisJobStore(self.database).get(self.job['id']))

    def test_refuses_nested_paths_and_legacy_repository_layout(self):
        for destination in (self.state, self.state / 'snapshot'):
            with self.assertRaises(BackupError):
                create_backup(self.state, destination, offline=True)
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaises(BackupError):
            create_backup(empty, self.backup, offline=True)
        with self.assertRaises(BackupError):
            create_backup(self.root, self.backup, offline=True)

    def test_active_jobs_block_snapshot_without_mutating_them(self):
        store = AnalysisJobStore(self.database)
        active = store.create(self.scope, {})
        for state in ('queued', 'running'):
            with self.assertRaises(BackupError):
                self.snapshot()
            self.assertEqual(store.get(active['id'])['status'], state)
            store.claim(active['id'])
        self.assertFalse(self.backup.exists())

    def test_sqlite_backup_includes_committed_wal_pages(self):
        with closing(sqlite3.connect(self.database)) as keeper:
            keeper.execute('PRAGMA journal_mode = WAL')
            keeper.execute('PRAGMA wal_autocheckpoint = 0')
            keeper.execute('CREATE TABLE recovery_probe (value TEXT)')
            keeper.execute("INSERT INTO recovery_probe VALUES ('committed WAL content')")
            keeper.commit()
            self.assertGreater(Path(str(self.database) + '-wal').stat().st_size, 0)
            self.snapshot()
        self.assertFalse(any(name.endswith(('-wal', '-shm')) for name in self.manifest()['files']))
        restored_db = self.backup / 'state' / 'jobs' / 'analysis_jobs.sqlite3'
        with closing(sqlite3.connect(restored_db)) as connection:
            self.assertEqual(connection.execute('SELECT value FROM recovery_probe').fetchone()[0], 'committed WAL content')

    def test_fifo_manifest_is_rejected_without_blocking(self):
        self.snapshot()
        manifest = self.backup / 'manifest.json'
        manifest.unlink()
        os.mkfifo(manifest)
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'runtime_backup.py'),
                   'verify', '--backup', str(self.backup)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn('Traceback', result.stderr)

    def test_cli_rejects_nul_and_symlink_loop_without_echoing_path(self):
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'runtime_backup.py')]
        nul = 'bad\x00path'
        # execve itself rejects NUL-bearing argv values, so exercise the same
        # path validation used by the CLI without attempting to spawn a child.
        from src.runtime_backup import _path
        with self.assertRaises(BackupError) as caught:
            _path(nul)
        self.assertNotIn('bad', str(caught.exception))
        loop_a, loop_b = self.root / 'loop-a', self.root / 'loop-b'
        loop_a.symlink_to(loop_b)
        loop_b.symlink_to(loop_a)
        result = subprocess.run(command + ['verify', '--backup', str(loop_a)], capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(str(loop_a), result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_locked_sqlite_fails_with_bounded_timeout(self):
        # Hold a real SQLite EXCLUSIVE lock in another process.  This exercises
        # sqlite's busy timeout rather than a mock that fails before connecting.
        lock_script = (
            "import sqlite3,sys,time; "
            "c=sqlite3.connect(sys.argv[1]); c.execute('PRAGMA journal_mode=DELETE'); "
            "c.commit(); c.execute('BEGIN EXCLUSIVE'); "
            "open(sys.argv[2],'w').close(); time.sleep(3)"
        )
        lock_ready = self.root / 'lock-ready'
        locker = subprocess.Popen([sys.executable, '-c', lock_script,
                                   str(self.database), str(lock_ready)])
        self.addCleanup(lambda: (locker.kill(), locker.wait()) if locker.poll() is None else None)
        deadline = time.monotonic() + 2
        while not lock_ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(lock_ready.exists())
        with patch('src.runtime_backup.SQLITE_TIMEOUT_SECONDS', 0.2):
            started = time.monotonic()
            with self.assertRaises((BackupError, sqlite3.Error)):
                self.snapshot()
            self.assertLess(time.monotonic() - started, 2.0)
        self.assertFalse((self.backup / 'manifest.json').exists())

    def test_snapshot_progress_limit_is_enforced_by_real_backup_api(self):
        from src.runtime_backup import _snapshot_database

        source = self.root / 'large.sqlite3'
        with closing(sqlite3.connect(source)) as connection:
            connection.execute('PRAGMA page_size=1024')
            connection.execute('VACUUM')
            connection.execute('CREATE TABLE payload (value BLOB)')
            connection.executemany('INSERT INTO payload VALUES (?)', [(b'x' * 1024,) for _ in range(400)])
            connection.commit()
        destination = self.root / 'limited.sqlite3'
        with patch('src.runtime_backup.MAX_BYTES', 0):
            with self.assertRaises(BackupError):
                _snapshot_database(source, destination)
        self.assertTrue(destination.exists())

    def test_committed_wal_mutation_during_snapshot_fails_without_manifest(self):
        with closing(sqlite3.connect(self.database)) as keeper:
            keeper.execute('PRAGMA journal_mode = WAL')
            keeper.execute('PRAGMA wal_autocheckpoint = 0')
            keeper.execute('CREATE TABLE wal_mutation_probe (value TEXT)')
            keeper.execute("INSERT INTO wal_mutation_probe VALUES ('before')")
            keeper.commit()
            self.assertGreater(Path(str(self.database) + '-wal').stat().st_size, 0)
            import src.runtime_backup as backup_module
            original_snapshot = backup_module._snapshot_database
            mutated = False

            def snapshot_then_mutate(source, destination):
                nonlocal mutated
                original_snapshot(source, destination)
                if not mutated:
                    keeper.execute("INSERT INTO wal_mutation_probe VALUES ('during')")
                    keeper.commit()
                    mutated = True

            with patch('src.runtime_backup._snapshot_database', side_effect=snapshot_then_mutate):
                with self.assertRaises(BackupError):
                    self.snapshot()
        self.assertFalse((self.backup / 'manifest.json').exists())
        with closing(sqlite3.connect(self.database)) as connection:
            values = [row[0] for row in connection.execute('SELECT value FROM wal_mutation_probe ORDER BY rowid')]
        self.assertEqual(values, ['before', 'during'])

    def test_restore_failure_leaves_no_success_and_retry_uses_new_directory(self):
        self.snapshot()
        manifest_before = (self.backup / 'manifest.json').read_bytes()
        payload_before = {
            path.relative_to(self.backup).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (self.backup / 'state').rglob('*') if path.is_file()
        }
        source_before = {
            path.relative_to(self.state).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.state.rglob('*') if path.is_file() and not path.is_symlink()
        }
        original_copy = __import__('src.runtime_backup', fromlist=['_copy_file'])._copy_file
        calls = 0

        def fail_once(source, destination, expected=None):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('simulated full disk')
            return original_copy(source, destination, expected)

        with patch('src.runtime_backup._copy_file', side_effect=fail_once):
            with self.assertRaises(OSError):
                restore_backup(self.backup, self.restored, offline=True)
        self.assertEqual((self.backup / 'manifest.json').read_bytes(), manifest_before)
        self.assertEqual({
            path.relative_to(self.backup).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (self.backup / 'state').rglob('*') if path.is_file()
        }, payload_before)
        self.assertEqual({
            path.relative_to(self.state).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.state.rglob('*') if path.is_file() and not path.is_symlink()
        }, source_before)
        partial_before = {path.relative_to(self.restored).as_posix(): path.read_bytes()
                          for path in self.restored.rglob('*') if path.is_file()}
        with self.assertRaises(BackupError):
            restore_backup(self.backup, self.restored, offline=True)
        self.assertEqual({path.relative_to(self.restored).as_posix(): path.read_bytes()
                          for path in self.restored.rglob('*') if path.is_file()}, partial_before)
        self.assertEqual(verify_backup(self.backup)['contract'], 'data-prism-runtime-backup-v1')
        retry = self.root / 'recovered-retry'
        self.assertEqual(restore_backup(self.backup, retry, offline=True)['contract'], 'data-prism-runtime-backup-v1')

    def test_corrupt_source_database_is_rejected(self):
        self.database.write_bytes(b'not a database')
        with self.assertRaises((BackupError, sqlite3.Error)):
            self.snapshot()
        self.assertFalse(self.backup.exists())

    def test_corrupt_payload_rejected_before_destination_is_created(self):
        self.snapshot()
        (self.backup / 'state' / 'baselines' / 'synthetic.json').write_text('changed')
        with self.assertRaises(BackupError):
            restore_backup(self.backup, self.restored, offline=True)
        self.assertFalse(self.restored.exists())

    def test_extra_files_and_sqlite_sidecars_are_rejected(self):
        self.snapshot()
        extra = self.backup / 'state' / 'jobs' / 'analysis_jobs.sqlite3-wal'
        extra.write_bytes(b'unexpected')
        with self.assertRaises(BackupError):
            verify_backup(self.backup)
        extra.unlink()
        extra = self.backup / 'state' / 'uploads' / 'extra.txt'
        extra.write_bytes(b'unexpected')
        with self.assertRaises(BackupError):
            verify_backup(self.backup)

    def test_manifest_top_level_enumeration_stops_at_small_bound(self):
        self.snapshot()
        for index in range(10):
            (self.backup / ('junk' + str(index))).write_text('unexpected')
        import src.runtime_backup as backup_module
        original_scandir = os.scandir

        class TrackingEntries:
            def __init__(self, entries):
                self.entries = entries
                self.closed = False
                self.count = 0

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.close()

            def close(self):
                self.closed = True
                self.entries.close()

            def __iter__(self):
                return self

            def __next__(self):
                entry = next(self.entries)
                self.count += 1
                return entry

        tracked = []

        def bounded_scandir(path):
            if Path(path) == self.backup:
                entries = TrackingEntries(original_scandir(path))
                tracked.append(entries)
                return entries
            return original_scandir(path)

        with patch.object(backup_module.os, 'scandir', bounded_scandir), \
                patch.object(backup_module.os, 'listdir', side_effect=AssertionError('os.listdir forbidden')):
            with self.assertRaises(BackupError):
                verify_backup(self.backup)
        self.assertEqual(len(tracked), 1)
        self.assertTrue(tracked[0].closed)
        self.assertEqual(tracked[0].count, 3)

    def test_runtime_inventory_enumeration_stops_at_bound_and_closes(self):
        import src.runtime_backup as backup_module
        original_scandir = os.scandir
        tracked = []

        class TrackingEntries:
            def __init__(self, entries):
                self.entries, self.closed, self.count = entries, False, 0

            def __iter__(self):
                return self

            def __next__(self):
                entry = next(self.entries)
                self.count += 1
                return entry

            def close(self):
                self.closed = True
                self.entries.close()

        def bounded_scandir(path):
            entries = TrackingEntries(original_scandir(path))
            tracked.append(entries)
            return entries

        with patch.object(backup_module.os, 'scandir', bounded_scandir), \
                patch.object(backup_module.os, 'listdir', side_effect=AssertionError('os.listdir forbidden')), \
                patch.object(backup_module, 'MAX_FILES', 2):
            with self.assertRaises(BackupError):
                self.snapshot()
        self.assertLessEqual(sum(item.count for item in tracked), 5)
        self.assertTrue(tracked)
        self.assertTrue(all(item.closed for item in tracked))

    def test_path_traversal_manifest_cannot_write_outside_destination(self):
        self.snapshot()
        manifest = self.manifest()
        example = next(iter(manifest['files'].values()))
        for name in ('../outside', '/tmp/outside', 'jobs/../../outside', 'jobs\\outside', 'jobs/.env'):
            with self.subTest(name=name):
                forged = {**manifest, 'files': {**manifest['files'], name: example}}
                self.replace_manifest(forged)
                with self.assertRaises(BackupError):
                    restore_backup(self.backup, self.restored, offline=True)
                self.assertFalse(self.restored.exists())
        self.assertFalse((self.root / 'outside').exists())

    def test_symlinks_and_hardlinks_are_rejected(self):
        link = self.state / 'uploads' / 'link'
        link.symlink_to(self.baseline)
        with self.assertRaises(BackupError):
            self.snapshot()
        link.unlink()
        os.link(self.baseline, link)
        with self.assertRaises(BackupError):
            self.snapshot()
        link.unlink()
        self.snapshot()
        payload_file = self.backup / 'state' / 'baselines' / 'synthetic.json'
        payload_file.unlink()
        payload_file.symlink_to(self.baseline)
        with self.assertRaises(BackupError):
            restore_backup(self.backup, self.restored, offline=True)
        self.assertFalse(self.restored.exists())

    def test_sqlite_sidecar_symlink_is_rejected_without_following_it(self):
        sidecar = self.state / 'jobs' / 'analysis_jobs.sqlite3-wal'
        sidecar.symlink_to(self.baseline)
        with self.assertRaises(BackupError):
            self.snapshot()
        self.assertFalse(self.backup.exists())

    def test_orphan_sqlite_sidecars_are_rejected(self):
        (self.state / 'drift' / 'drift_history.sqlite3').unlink()
        for suffix in ('-wal', '-shm'):
            with self.subTest(suffix=suffix):
                orphan = self.state / 'drift' / ('drift_history.sqlite3' + suffix)
                orphan.write_bytes(b'orphan')
                output = self.root / ('snapshot-' + suffix[1:])
                with self.assertRaises(BackupError):
                    create_backup(self.state, output, offline=True)
                self.assertFalse(output.exists())
                orphan.unlink()

    def test_secrets_and_unknown_top_level_entries_fail_closed(self):
        secret = self.state / '.env'
        secret.write_text('PRIVATE_SENTINEL=do-not-copy')
        with self.assertRaises(BackupError) as caught:
            self.snapshot()
        self.assertNotIn('PRIVATE_SENTINEL', str(caught.exception))
        self.assertFalse(self.backup.exists())

    def test_partial_copy_has_no_completion_manifest(self):
        with patch('src.runtime_backup._copy_file', side_effect=OSError('simulated full disk')):
            with self.assertRaises(OSError):
                self.snapshot()
        self.assertFalse((self.backup / 'manifest.json').exists())
        with self.assertRaises(BackupError):
            verify_backup(self.backup)
        self.assertIsNotNone(AnalysisJobStore(self.database).get(self.job['id']))

    def test_detects_source_changes_during_copy(self):
        from src.runtime_backup import _copy_file

        def changing_copy(source, destination, expected=None):
            _copy_file(source, destination, expected)
            self.baseline.write_text('{"changed": true}')

        with patch('src.runtime_backup._copy_file', side_effect=changing_copy):
            with self.assertRaises(BackupError):
                self.snapshot()
        self.assertFalse((self.backup / 'manifest.json').exists())

    def test_limits_and_unsupported_contracts_are_rejected(self):
        with patch('src.runtime_backup.MAX_FILES', 1):
            with self.assertRaises(BackupError):
                self.snapshot()
        self.assertFalse(self.backup.exists())
        self.snapshot()
        manifest = self.manifest()
        self.replace_manifest({**manifest, 'contract': 'future-version'})
        with self.assertRaises(BackupError):
            verify_backup(self.backup)
        self.replace_manifest(manifest)
        with patch('src.runtime_backup.MAX_MANIFEST_BYTES', 1):
            with self.assertRaises(BackupError):
                verify_backup(self.backup)
        with patch('src.runtime_backup.MAX_BYTES', 1):
            with self.assertRaises(BackupError):
                verify_backup(self.backup)

    def test_duplicate_manifest_keys_and_invalid_metadata_are_rejected(self):
        self.snapshot()
        manifest = self.manifest()
        path = self.backup / 'manifest.json'
        path.write_text('{"contract": "a", "contract": "b"}')
        with self.assertRaises(BackupError):
            verify_backup(self.backup)
        record = next(iter(manifest['files'].values()))
        record['mtime_ns'] = 'not a number'
        self.replace_manifest(manifest)
        with self.assertRaises(BackupError):
            verify_backup(self.backup)

    @unittest.skipIf(os.name != 'posix', 'POSIX permission contract')
    def test_private_permissions(self):
        self.snapshot()
        restore_backup(self.backup, self.restored, offline=True)
        for root in (self.backup, self.restored):
            for path in [root, *root.rglob('*')]:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o077, 0)

    def test_database_integrity_is_checked_even_when_digest_matches(self):
        self.snapshot()
        name = 'jobs/analysis_jobs.sqlite3'
        payload = self.backup / 'state' / name
        payload.write_bytes(b'corrupt database')
        manifest = self.manifest()
        manifest['files'][name].update(size=payload.stat().st_size, sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
        self.replace_manifest(manifest)
        with self.assertRaises((BackupError, sqlite3.Error)):
            restore_backup(self.backup, self.restored, offline=True)
        self.assertFalse(self.restored.exists())

    def test_cli_backup_verify_restore_and_error_exit_codes(self):
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, str(root / 'runtime_backup.py')]
        for arguments in (
            ['backup', '--state-dir', str(self.state), '--output', str(self.backup), '--offline'],
            ['verify', '--backup', str(self.backup)],
            ['restore', '--backup', str(self.backup), '--destination', str(self.restored), '--offline'],
        ):
            result = subprocess.run(command + arguments, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'], 'ok')
            self.assertNotIn(self.scope, result.stdout)
        result = subprocess.run(command + arguments, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Destination already exists', result.stderr)

    def test_recovered_routes_work_in_a_fresh_process_with_same_signed_cookie(self):
        from flask import Flask
        from flask.sessions import SecureCookieSessionInterface

        secret = 'synthetic-recovery-test-key-not-a-real-secret'
        original_app = Flask(__name__)
        original_app.secret_key = secret
        cookie = SecureCookieSessionInterface().get_signing_serializer(original_app).dumps({
            'vibedash_analysis_scope_id': self.scope,
            'vibedash_decision_csrf_token': 'a' * 64,
        })
        self.snapshot()
        restore_backup(self.backup, self.restored, offline=True)
        # Independent process: no parent Flask state, dispatcher, or DB connection.
        root = Path(__file__).resolve().parents[1]
        script = '''
import json, sys
import os
from pathlib import Path
import web_app
app = web_app.app
owner = app.test_client()
owner.set_cookie(app.config['SESSION_COOKIE_NAME'], sys.argv[1])
expired_upload = Path(os.environ['DATA_PRISM_STATE_DIR']) / 'uploads' / ('vibedash-' + 'b' * 32 + '.csv')
assert expired_upload.exists()
assert owner.get('/vibedash/history').status_code == 200
assert not expired_upload.exists()
assert owner.get('/vibedash/jobs/' + sys.argv[2] + '/result').status_code == 200
case_url = '/vibedash/decisions/' + sys.argv[3]
assert owner.get(case_url).status_code == 200
assert app.test_client().get(case_url).status_code == 404
app.secret_key = 'different-synthetic-key'
assert owner.get(case_url).status_code == 404
print(json.dumps({'restored_routes': 'ok'}))
'''
        result = subprocess.run(
            [sys.executable, '-c', script, cookie, self.job['id'], self.case['id']],
            cwd=root, capture_output=True, text=True, timeout=45,
            env={**os.environ, 'DATA_PRISM_STATE_DIR': str(self.restored),
                 'FLASK_SECRET_KEY': secret, 'LOG_LEVEL': 'ERROR'},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['restored_routes'], 'ok')


if __name__ == '__main__':
    unittest.main()
