"""Prepared server storage: scopes, fencing, retention, continuity and recovery."""
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from src.forecast_artifact import _digest, create_artifact, predict_artifact
from src.forecasting import ForecastTask, forecast_series
from src.runtime_backup import create_backup, restore_backup
from src.runtime_state import initialize_runtime_state
from tests.test_forecasting import frame
from vibedash.analysis_jobs import AnalysisJobStore, ScopeClosedError, ScopeHasActiveJobsError
from vibedash.forecast_models import (
    ForecastModelStore, ForecastModelCapacityError, ForecastModelStorageUnavailable,
)


OWNER = 'a' * 32
FOREIGN = 'b' * 32
NOW = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)


def resign(artifact):
    artifact['sha256'] = _digest({key: value for key, value in artifact.items() if key != 'sha256'})
    return artifact


class ForecastModelStoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        task = ForecastTask('demand', 'date', 7)
        data = frame()
        cls.artifact = create_artifact(data, task, forecast_series(data, task))

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = self.root / 'runtime'
        self.state.mkdir()
        self.identity = initialize_runtime_state(self.state)
        self.config = {'expected_state_id': self.identity['state_id'],
                       'deployment_profile': 'production', 'state_durability': 'persistent'}
        self.clock = patch('vibedash.forecast_models._now', return_value=NOW)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.store = ForecastModelStore(self.state, **self.config)
        self.model_id = self.artifact['model_id']

    def register(self, owner=OWNER, artifact=None, days=7):
        return self.store.register(owner, self.artifact if artifact is None else artifact, retention_days=days)

    def changed_model(self):
        artifact = copy.deepcopy(self.artifact)
        artifact['model_id'] = 'c' * 32
        return resign(artifact)

    def test_ordinary_analysis_store_does_not_initialize_or_enable_registry(self):
        path = self.root / 'ordinary.sqlite3'
        AnalysisJobStore(path)
        with sqlite3.connect(path) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM sqlite_master WHERE name='forecast_models'").fetchone()[0], 0)

    def test_ephemeral_demo_missing_and_wrong_identity_cannot_initialize_registry(self):
        for config in (self.config | {'deployment_profile': 'demo'},
                       self.config | {'state_durability': 'ephemeral'},
                       self.config | {'expected_state_id': '0' * 64},
                       self.config | {'expected_state_id': None}):
            target = self.root / 'not-configured'
            with self.assertRaises(ForecastModelStorageUnavailable): ForecastModelStore(target, **config)
            self.assertFalse(target.exists())
            with self.assertRaises(ForecastModelStorageUnavailable): ForecastModelStore(self.state, **config)

    def test_saved_model_is_scoped_and_roundtrip_parameters_do_not_refit(self):
        saved = self.register()
        self.assertTrue(saved['created'])
        self.assertEqual(self.store.get(OWNER, self.model_id), self.artifact)
        with patch('src.forecast_artifact._fit_lag_model', side_effect=AssertionError('no fit')):
            report = self.store.predict(OWNER, self.model_id)
        self.assertEqual(report['predictions'], predict_artifact(self.artifact)['predictions'])
        self.assertFalse(report['parameters_refitted'])
        self.assertFalse(report['method_reselected'])
        self.assertEqual(report['quality_on_new_outcomes'], 'unknown')
        self.assertEqual(report['production_readiness'], 'not_established')

    def test_foreign_model_id_cannot_read_predict_list_or_delete(self):
        self.register()
        self.assertIsNone(self.store.get(FOREIGN, self.model_id))
        self.assertEqual(self.store.list_for_scope(FOREIGN), [])
        self.assertFalse(self.store.delete(FOREIGN, self.model_id))
        with patch('vibedash.forecast_models.predict_artifact') as predict:
            with self.assertRaises(ValueError): self.store.predict(FOREIGN, self.model_id)
            predict.assert_not_called()
        self.assertEqual(self.store.get(OWNER, self.model_id), self.artifact)

    def test_same_model_id_in_two_scopes_is_independently_owned(self):
        self.register()
        self.register(FOREIGN)
        self.assertTrue(self.store.delete(OWNER, self.model_id))
        self.assertIsNone(self.store.get(OWNER, self.model_id))
        self.assertEqual(self.store.get(FOREIGN, self.model_id), self.artifact)

    def test_invalid_scope_ids_and_limits_do_not_query_unscoped_data(self):
        self.register()
        for scope in (None, '', 'a', "' OR 1=1 --", True):
            with self.assertRaises(ValueError): self.store.get(scope, self.model_id)
            with self.assertRaises(ValueError): self.store.list_for_scope(scope)
            with self.assertRaises(ValueError): self.store.register(scope, self.artifact, retention_days=7)
        for limit in (0, 11, True, 1.5):
            with self.assertRaises(ValueError): self.store.list_for_scope(OWNER, limit=limit)
        self.assertEqual(len(self.store.list_for_scope(OWNER)), 1)

    def test_registration_is_immutable_and_duplicate_never_refreshes_ttl(self):
        original = self.register()
        with patch('vibedash.forecast_models._now', return_value=NOW + timedelta(days=3)):
            repeated = self.register(days=30)
        self.assertFalse(repeated['created'])
        self.assertEqual(repeated['created_at'], original['created_at'])
        self.assertEqual(repeated['expires_at'], original['expires_at'])
        changed = copy.deepcopy(self.artifact)
        changed['training']['tail'][-1] += 1
        resign(changed)
        with self.assertRaises(ValueError): self.register(artifact=changed)
        self.assertEqual(self.store.get(OWNER, self.model_id), self.artifact)

    def test_corrupt_artifact_and_invalid_retention_never_save(self):
        changed = copy.deepcopy(self.artifact)
        changed['sha256'] = '0' * 64
        with self.assertRaises(ValueError): self.register(artifact=changed)
        for days in (0, 31, True, '7', 1.5, None):
            with self.assertRaises(ValueError): self.register(days=days)
        self.assertEqual(self.store.list_for_scope(OWNER), [])

    def test_expiry_denies_read_and_predict_and_explicit_purge_is_scoped(self):
        saved = self.register(days=1)
        self.register(FOREIGN, days=1)
        at_expiry = datetime.fromisoformat(saved['expires_at'])
        with patch('vibedash.forecast_models._now', return_value=at_expiry):
            self.assertIsNone(self.store.get(OWNER, self.model_id))
            self.assertEqual(self.store.list_for_scope(OWNER), [])
            with self.assertRaises(ValueError): self.store.predict(OWNER, self.model_id)
            with self.assertRaises(ValueError): self.register()
            self.assertEqual(self.store.purge_expired(OWNER), 1)
            self.assertEqual(self.store.purge_expired(OWNER), 0)
            self.assertTrue(self.register()['created'])
        with sqlite3.connect(self.store.database_path) as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM forecast_models WHERE scope_id=?', (FOREIGN,)).fetchone()[0], 1)

    def test_scope_global_and_byte_quotas_never_evict_existing_models(self):
        self.register()
        for constant, value, owner, artifact in (
            ('MAX_MODELS_PER_SCOPE', 1, OWNER, self.changed_model()),
            ('MAX_MODELS_TOTAL', 1, FOREIGN, self.artifact),
            ('MAX_BYTES_PER_SCOPE', 1, OWNER, self.changed_model()),
            ('MAX_BYTES_TOTAL', 1, FOREIGN, self.artifact),
        ):
            with patch('vibedash.forecast_models.' + constant, value):
                with self.assertRaises(ForecastModelCapacityError): self.register(owner, artifact)
        self.assertEqual(self.store.get(OWNER, self.model_id), self.artifact)
        self.assertEqual(self.store.list_for_scope(FOREIGN), [])

    def test_summaries_do_not_disclose_model_parameters_policy_or_training_tail(self):
        self.register()
        summary = self.store.list_for_scope(OWNER)[0]
        self.assertEqual(set(summary), {'contract', 'model_id', 'model_sha256', 'created_at',
                                        'expires_at', 'artifact_bytes', 'production_readiness'})
        self.assertNotIn('training', summary)
        self.assertNotIn('decision_policy', summary)

    def test_mount_identity_is_rechecked_for_every_operation_without_recreation(self):
        self.register()
        marker = self.state / 'identity' / 'runtime_state.json'
        original = marker.read_text()
        changed = json.loads(original)
        changed['state_id'] = '0' * 64
        marker.write_text(json.dumps(changed))
        operations = [lambda: self.store.get(OWNER, self.model_id),
                      lambda: self.store.list_for_scope(OWNER), self.register,
                      lambda: self.store.predict(OWNER, self.model_id),
                      lambda: self.store.delete(OWNER, self.model_id),
                      lambda: self.store.purge_expired(OWNER)]
        for operation in operations:
            with self.assertRaises(ForecastModelStorageUnavailable): operation()
        marker.write_text(original)
        self.assertEqual(self.store.get(OWNER, self.model_id), self.artifact)

    def test_missing_database_and_symbolic_link_mount_fail_closed(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.state, target_is_directory=True)
        with self.assertRaises(ForecastModelStorageUnavailable): ForecastModelStore(alias, **self.config)
        self.store.database_path.unlink()
        with self.assertRaises(ForecastModelStorageUnavailable): self.register()
        self.assertFalse(self.store.database_path.exists())

    def test_stored_identity_size_contract_and_retention_corruption_are_not_served(self):
        self.register()
        for field, value in (('artifact_sha256', '0' * 64), ('artifact_bytes', 1),
                             ('contract', 'unknown'), ('expires_at', '2099-01-01T00:00:00.000000+00:00')):
            with self.subTest(field=field):
                with sqlite3.connect(self.store.database_path) as connection:
                    previous = connection.execute(f'SELECT {field} FROM forecast_models').fetchone()[0]
                    connection.execute(f'UPDATE forecast_models SET {field}=?', (value,))
                with self.assertRaises(ValueError): self.store.get(OWNER, self.model_id)
                with sqlite3.connect(self.store.database_path) as connection:
                    connection.execute(f'UPDATE forecast_models SET {field}=?', (previous,))

    def test_account_scope_erasure_removes_models_atomically_but_keeps_foreign(self):
        self.register()
        self.register(FOREIGN)
        self.store.jobs.erase_scope_if_idle(OWNER)
        with sqlite3.connect(self.store.database_path) as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM forecast_models WHERE scope_id=?', (OWNER,)).fetchone()[0], 0)
        for operation in (self.register, lambda: self.store.get(OWNER, self.model_id),
                          lambda: self.store.list_for_scope(OWNER),
                          lambda: self.store.predict(OWNER, self.model_id),
                          lambda: self.store.delete(OWNER, self.model_id)):
            with self.assertRaises(ScopeClosedError): operation()
        self.assertEqual(self.store.get(FOREIGN, self.model_id), self.artifact)

    def test_active_jobs_and_cleanup_failure_leave_models_and_fence_unchanged(self):
        self.register()
        job = self.store.jobs.create(OWNER, {})
        with self.assertRaises(ScopeHasActiveJobsError): self.store.jobs.erase_scope_if_idle(OWNER)
        self.store.jobs.fail(job['id'])
        def fail(_jobs):
            raise OSError('simulated cleanup failure')
        with self.assertRaises(OSError): self.store.jobs.erase_scope_if_idle(OWNER, artifact_cleanup=fail)
        self.assertEqual(self.store.get(OWNER, self.model_id), self.artifact)
        self.assertFalse(self.register()['created'])

    def test_concurrent_registration_and_erasure_cannot_resurrect_models(self):
        inside, release = threading.Event(), threading.Event()
        original = self.store._summary
        def pause(row):
            inside.set()
            if not release.wait(timeout=5):
                raise AssertionError('test synchronization failed')
            return original(row)
        with ThreadPoolExecutor(max_workers=2) as pool, patch.object(self.store, '_summary', side_effect=pause):
            saved = pool.submit(self.register)
            self.assertTrue(inside.wait(timeout=5))
            erased = pool.submit(self.store.jobs.erase_scope_if_idle, OWNER)
            release.set()
            self.assertTrue(saved.result(timeout=5)['created'])
            erased.result(timeout=5)
        with self.assertRaises(ScopeClosedError): self.register()
        with sqlite3.connect(self.store.database_path) as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM forecast_models').fetchone()[0], 0)

    def test_reopen_and_offline_backup_restore_preserve_model_and_original_expiry(self):
        saved = self.register(days=3)
        reopened = ForecastModelStore(self.state, **self.config)
        self.assertEqual(reopened.get(OWNER, self.model_id), self.artifact)
        backup, restored = self.root / 'backup', self.root / 'restored'
        create_backup(self.state, backup, offline=True, revision='fcb00ec')
        restore_backup(backup, restored, offline=True)
        recovered = ForecastModelStore(restored, **self.config)
        self.assertEqual(recovered.get(OWNER, self.model_id), self.artifact)
        self.assertEqual(recovered.list_for_scope(OWNER)[0]['expires_at'], saved['expires_at'])
        self.assertEqual(recovered.predict(OWNER, self.model_id)['predictions'], reopened.predict(OWNER, self.model_id)['predictions'])


if __name__ == '__main__':
    unittest.main()
