"""Exercise explicit, authenticated model retention across the real web/worker path."""
from datetime import datetime, timedelta
import os
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

import web_app
from src.forecast_artifact import predict_artifact, validate_artifact
from src.runtime_state import initialize_runtime_state
from src.semantic_contract import validate_semantic_contract
from tests import test_accounts as accounts_fixture
from tests import test_vibedash_forecast_routes as forecast_fixture
from tests.test_forecast_tasks import confirmed, daily_frame
from vibedash.accounts import account_scope_id
from vibedash.analysis_jobs import AnalysisJobStore
from vibedash.forecast_models import RETENTION_CONTRACT
from vibedash.routes import analysis_job_dispatcher, _process_analysis_job


class ForecastModelWebTests(unittest.TestCase):
    fixture = accounts_fixture.AccountRouteIntegrationTests
    PASSWORD = fixture.PASSWORD
    _restore_app = fixture._restore_app
    _csrf = staticmethod(fixture._csrf)
    _register = fixture._register
    request_data = forecast_fixture.VibeDashForecastRouteTests.request_data

    def setUp(self):
        self.fixture.setUp(self)
        self.root = Path(self.directory.name)
        identity = initialize_runtime_state(self.root)
        config = patch.dict(web_app.app.config, {
            'VIBEDASH_MODEL_STORAGE_ENABLED': True,
            'DATA_PRISM_STATE_DIRECTORY_CONFIGURED': True,
            'DATA_PRISM_STATE_DIR': self.root,
            'DATA_PRISM_DEPLOYMENT_PROFILE': 'production',
            'DATA_PRISM_STATE_DURABILITY': 'persistent',
            'DATA_PRISM_EXPECTED_STATE_ID': identity['state_id'],
        })
        config.start()
        self.addCleanup(config.stop)
        environment = patch.dict(os.environ, {'DATA_PRISM_STATE_DIR': str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)
        self.owner = web_app.app.test_client()
        self.assertEqual(self._register(self.owner).status_code, 302)
        with self.owner.session_transaction() as session:
            self.scope = account_scope_id(session['vibedash_account_id'], web_app.app.secret_key)

    def _queue(self, **fields):
        response = self.owner.post('/vibedash/jobs', data=self.request_data(self.owner, **fields))
        self.assertEqual(response.status_code, 202, response.get_json())
        return response.get_json()['job_id']

    def _complete(self, job_id):
        analysis_job_dispatcher._run(web_app.app, job_id, _process_analysis_job)
        return AnalysisJobStore(self.job_path).get(job_id)

    def _rows(self):
        with sqlite3.connect(self.job_path) as connection:
            if not connection.execute("SELECT 1 FROM sqlite_master WHERE name='forecast_models'").fetchone():
                return []
            return connection.execute('SELECT scope_id, model_id, created_at, expires_at FROM forecast_models').fetchall()

    def _export(self, client, model_id, *, csrf=None):
        if csrf is None:
            client.get('/vibedash/')
            with client.session_transaction() as session:
                csrf = session['vibedash_auth_csrf_token']
        return client.post(f'/vibedash/account/models/{model_id}/export.json', data={'csrf_token': csrf})

    def test_landing_defaults_to_no_retention_and_hides_control_when_storage_is_unavailable(self):
        html = self.owner.get('/vibedash/').get_data(as_text=True)
        self.assertIn('<option value="" selected>Do not save</option>', html)
        self.assertIn('aria-describedby="forecast-model-storage-help"', html)
        for settings in ({'VIBEDASH_MODEL_STORAGE_ENABLED': False},
                         {'DATA_PRISM_STATE_DURABILITY': 'ephemeral'},
                         {'DATA_PRISM_EXPECTED_STATE_ID': '0' * 64},
                         {'DATA_PRISM_STATE_DIRECTORY_CONFIGURED': False},
                         {'VIBEDASH_JOB_STORE_PATH': str(self.root / 'wrong.sqlite3')}):
            with patch.dict(web_app.app.config, settings):
                response = self.owner.get('/vibedash/')
                self.assertNotIn('name="forecast_model_retention_days"', response.get_data(as_text=True))
                response = self.owner.post('/vibedash/jobs', data=self.request_data(
                    self.owner, forecast_model_retention_days='7'))
                self.assertEqual(response.status_code, 503)
        self.assertEqual(list((self.root / 'uploads').iterdir()), [])
        self.assertEqual(self._rows(), [])

    def test_guest_invalid_duplicate_and_evidence_retention_requests_do_not_queue_or_upload(self):
        guest = web_app.app.test_client()
        html = guest.get('/vibedash/').get_data(as_text=True)
        self.assertNotIn('name="forecast_model_retention_days"', html)
        response = guest.post('/vibedash/jobs', data=self.request_data(guest, forecast_model_retention_days='7'))
        self.assertEqual(response.status_code, 401)
        for value in ('0', '31', '1.0', ' 7', ['1', '7']):
            response = self.owner.post('/vibedash/jobs', data=self.request_data(self.owner, forecast_model_retention_days=value))
            self.assertEqual(response.status_code, 422)
        response = self.owner.post('/vibedash/jobs', data=self.request_data(
            self.owner, analysis_mode='evidence', forecast_model_retention_days='7'))
        self.assertEqual(response.status_code, 422)
        response = self.owner.post('/vibedash/jobs', data=self.request_data(
            self.owner, csrf_token='', forecast_model_retention_days='7'))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(AnalysisJobStore(self.job_path).list_for_scope(self.scope), [])
        self.assertEqual(list((self.root / 'uploads').iterdir()), [])

    def test_unsupported_units_require_evaluation_without_saving(self):
        frame = daily_frame()
        contract = confirmed(frame)
        next(column for column in contract['columns'] if column['is_target'])['unit'] = 'currency'
        contract = validate_semantic_contract(frame, contract)
        response = self.owner.post('/vibedash/jobs', data=self.request_data(
            self.owner, frame=frame, contract=contract, forecast_model_retention_days='7'))
        self.assertEqual(response.status_code, 422)
        self.assertIn('only number and count', response.get_json()['error'])
        self.assertEqual(list((self.root / 'uploads').iterdir()), [])

    def test_no_consent_never_builds_or_retains_a_model(self):
        job_id = self._queue(forecast_model_retention_days='')
        with patch('vibedash.routes.create_artifact') as create:
            job = self._complete(job_id)
        create.assert_not_called()
        self.assertEqual(job['status'], 'completed')
        self.assertNotIn('forecast_model_retention', job['payload'])
        self.assertEqual(self._rows(), [])

    def test_explicit_consent_completes_owned_job_and_exports_verified_frozen_model(self):
        job_id = self._queue(forecast_model_retention_days='7', scope_id='f' * 32, model_id='f' * 32)
        self.assertEqual(self._rows(), [])
        job = self._complete(job_id)
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['payload']['forecast_model_retention'], {'contract': RETENTION_CONTRACT, 'days': 7})
        [(scope, model_id, created, expiry)] = self._rows()
        self.assertEqual((scope, model_id), (self.scope, job_id))
        self.assertEqual(datetime.fromisoformat(expiry) - datetime.fromisoformat(created), timedelta(days=7))
        result = self.owner.get(f'/vibedash/jobs/{job_id}/result').get_data(as_text=True)
        self.assertIn('Model saved to your account', result)
        account = self.owner.get('/vibedash/account').get_data(as_text=True)
        self.assertIn(f'/vibedash/account/models/{job_id}/export.json', account)
        self.assertEqual(self._export(self.owner, job_id, csrf='bad').status_code, 400)
        exported = self._export(self.owner, job_id)
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(exported.headers['Cache-Control'], 'private, no-store')
        self.assertEqual(exported.headers['X-Content-Type-Options'], 'nosniff')
        self.assertIn('.forecast-model.json', exported.headers['Content-Disposition'])
        model = exported.get_json()
        validate_artifact(model)
        self.assertEqual(predict_artifact(model)['quality_on_new_outcomes'], 'unknown')
        self.assertNotIn(model['sha256'], result)
        self.assertNotIn(model['sha256'], str(job['manifest']))

        stranger = web_app.app.test_client()
        self.assertEqual(self._register(stranger, 'stranger@example.com').status_code, 302)
        self.assertEqual(self._export(stranger, job_id).status_code, 404)
        self.assertNotIn(job_id, stranger.get('/vibedash/account').get_data(as_text=True))
        self.assertEqual(self._export(web_app.app.test_client(), job_id).status_code, 404)
        self.assertEqual(self.owner.get(f'/vibedash/account/models/{job_id}/export.json').status_code, 405)

    def test_failed_optional_save_rolls_back_insert_but_keeps_completed_report(self):
        job_id = self._queue(forecast_model_retention_days='7')
        with patch('vibedash.forecast_models.ForecastModelStore._summary', side_effect=RuntimeError('write interrupted')):
            job = self._complete(job_id)
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(self._rows(), [])
        self.assertIn('No saved model is available', self.owner.get(f'/vibedash/jobs/{job_id}/result').get_data(as_text=True))

    def test_job_failed_before_completion_cannot_leave_an_orphan_model(self):
        job_id = self._queue(forecast_model_retention_days='7')
        complete = AnalysisJobStore.complete

        def interrupt(store, current_job_id, *args, **kwargs):
            self.assertTrue(store.fail(current_job_id, 'worker_interrupted'))
            return complete(store, current_job_id, *args, **kwargs)

        with patch.object(AnalysisJobStore, 'complete', interrupt):
            job = self._complete(job_id)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(self._rows(), [])

    def test_withdrawn_deployment_gate_keeps_report_without_saving(self):
        job_id = self._queue(forecast_model_retention_days='7')
        with patch.dict(web_app.app.config, VIBEDASH_MODEL_STORAGE_ENABLED=False):
            job = self._complete(job_id)
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(self._rows(), [])

    def test_completion_failure_after_model_insert_rolls_back_both(self):
        job_id = self._queue(forecast_model_retention_days='7')
        complete = AnalysisJobStore.complete

        def interrupt(store, current_job_id, *args, on_complete, **kwargs):
            def fail_after_write(connection):
                on_complete(connection)
                row = connection.execute('SELECT model_id FROM forecast_models WHERE model_id=?',
                                         (current_job_id,)).fetchone()
                self.assertIsNotNone(row)
                raise RuntimeError('simulated completion failure')
            return complete(store, current_job_id, *args, on_complete=fail_after_write, **kwargs)

        with patch.object(AnalysisJobStore, 'complete', interrupt):
            job = self._complete(job_id)
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(self._rows(), [])

    def test_model_outlives_job_history_and_purge_removes_expired_rows_even_when_disabled(self):
        job_id = self._queue(forecast_model_retention_days='7')
        self.assertEqual(self._complete(job_id)['status'], 'completed')
        with sqlite3.connect(self.job_path) as connection:
            connection.execute("UPDATE analysis_jobs SET updated_at='2000-01-01T00:00:00+00:00' WHERE id=?", (job_id,))
        self.assertEqual(self._export(self.owner, job_id).status_code, 200)
        self.assertIsNone(AnalysisJobStore(self.job_path).get(job_id))
        with sqlite3.connect(self.job_path) as connection:
            connection.execute("UPDATE forecast_models SET expires_at='2000-01-01T00:00:00+00:00' WHERE model_id=?", (job_id,))
        with patch.dict(web_app.app.config, VIBEDASH_MODEL_STORAGE_ENABLED=False):
            self.assertEqual(self._export(self.owner, job_id).status_code, 404)
        self.assertEqual(self._rows(), [])
        self.assertEqual(self._export(self.owner, job_id).status_code, 404)
