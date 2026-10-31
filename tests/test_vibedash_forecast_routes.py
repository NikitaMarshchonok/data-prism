from io import BytesIO
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import web_app
from tests.test_forecast_tasks import FORM, confirmed, daily_frame
from tests import test_vibedash_job_routes as job_fixture
from vibedash.analysis_jobs import AnalysisJobStore


class VibeDashForecastRouteTests(unittest.TestCase):
    setUp = job_fixture.VibeDashJobRouteTests.setUp
    tearDown = job_fixture.VibeDashJobRouteTests.tearDown

    def request_data(self, client, frame=None, contract=None, **overrides):
        frame = daily_frame() if frame is None else frame
        client.get('/vibedash/')
        with client.session_transaction() as browser:
            token = browser['vibedash_decision_csrf_token']
        return dict(FORM, prompt='Evaluate daily cnt forecast', csrf_token=token,
                    semantic_contract=json.dumps(confirmed(frame) if contract is None else contract),
                    datafile=(BytesIO(frame.to_csv(index=False).encode()), 'daily.csv')) | overrides

    @patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=True)
    def test_missing_confirmation_csrf_or_target_creates_no_job_and_cleans_input(self, submit):
        with web_app.app.test_client() as client:
            for changes, expected in (({'csrf_token': ''}, 400), ({'forecast_available': ''}, 422),
                                      ({'forecast_horizon': '0'}, 422)):
                response = client.post('/vibedash/jobs', data=self.request_data(client, **changes))
                self.assertEqual(response.status_code, expected)
                self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])
            contract = confirmed(daily_frame())
            for column in contract['columns']: column['is_target'] = False
            response = client.post('/vibedash/jobs', data=self.request_data(client, contract=contract))
            self.assertEqual(response.status_code, 422)
        submit.assert_not_called()

    @patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=True)
    def test_gaps_and_preview_truncation_are_rejected_before_queue(self, submit):
        with web_app.app.test_client() as client:
            data = daily_frame().drop(index=30)
            response = client.post('/vibedash/jobs', data=self.request_data(client, frame=data))
            self.assertEqual(response.status_code, 422)
            with patch.dict(web_app.app.config, MAX_ROWS_PREVIEW=100):
                response = client.post('/vibedash/jobs', data=self.request_data(client))
                self.assertEqual(response.status_code, 422)
        submit.assert_not_called()

    @patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=True)
    def test_queued_task_binds_only_server_validated_settings(self, submit):
        with web_app.app.test_client() as client:
            response = client.post('/vibedash/jobs', data=self.request_data(client))
            self.assertEqual(response.status_code, 202)
            with client.session_transaction() as browser:
                scope = browser['vibedash_analysis_scope_id']
            store = AnalysisJobStore(web_app.app.config['VIBEDASH_JOB_STORE_PATH'])
            job = store.get(response.get_json()['job_id'], scope)
        self.assertEqual(job['payload']['analysis_kind'], 'daily_forecast')
        self.assertEqual(job['payload']['forecast_task']['settings']['target_column'], 'cnt')
        self.assertTrue(job['payload']['forecast_task']['availability_confirmed'])

    def test_landing_exposes_explicit_task_and_reconfirms_availability_for_new_input(self):
        with web_app.app.test_client() as client:
            html = client.get('/vibedash/').get_data(as_text=True)
        for field in ('analysis_mode', 'forecast_horizon', 'forecast_period', 'forecast_available'):
            self.assertIn('name="' + field + '"', html)
        self.assertIn('id="forecast-settings" class="semantic-primary-fields" hidden disabled', html)
        render_semantics = html.split('function renderSemanticContract(proposal)', 1)[1].split(
            'function serializeSemanticContract()', 1)[0]
        self.assertIn('forecastAvailable.checked = false', render_semantics)
        self.assertIn('semanticConfirmed.checked = false', render_semantics)

    def test_forecast_completes_through_owned_background_workflow_and_offline_export(self):
        with patch.dict(os.environ, DATA_PRISM_STATE_DIR=self.temporary_directory.name):
            with web_app.app.test_client() as owner:
                queued = owner.post('/vibedash/jobs', data=self.request_data(owner))
                self.assertEqual(queued.status_code, 202)
                url = queued.get_json()['status_url']
                for _ in range(150):
                    status = owner.get(url).get_json()
                    if status['status'] in {'completed', 'failed'}: break
                    time.sleep(.05)
                self.assertEqual(status['status'], 'completed', status)
                page = owner.get(status['result_url'])
                html = page.get_data(as_text=True)
                self.assertIn('Daily forecast evaluation', html)
                self.assertIn('Selected method MAE', html)
                self.assertIn('not approved for operational use', html.lower())
                manifest = owner.get(status['manifest_url']).get_json()
                self.assertIn('forecast', manifest)
                self.assertEqual(manifest['forecast']['confirmed_task']['settings']['horizon'], 7)
                self.assertNotIn('predictions', json.dumps(manifest))
                with owner.session_transaction() as browser:
                    scope = browser['vibedash_analysis_scope_id']
                store = AnalysisJobStore(web_app.app.config['VIBEDASH_JOB_STORE_PATH'])
                job = store.get(queued.get_json()['job_id'], scope)
                download = owner.get('/vibedash/export/' + job['session_id'])
                self.assertEqual(download.status_code, 200)
                self.assertIn('Daily forecast evaluation', download.get_data(as_text=True))
                download.close()
            with web_app.app.test_client() as stranger:
                self.assertEqual(stranger.get(url + '/result').status_code, 404)
                self.assertEqual(stranger.get(url + '/manifest').status_code, 404)
                response = stranger.get('/vibedash/export/' + job['session_id'])
                self.assertEqual(response.status_code, 302)


if __name__ == '__main__': unittest.main()
