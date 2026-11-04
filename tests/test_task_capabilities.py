"""Task availability is not domain expertise, data approval or permission to fit."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import web_app
from tests import test_vibedash_job_routes as job_fixture
from tests import test_vibedash_forecast_routes as forecast_fixture
from tests.test_forecast_policy import POLICY_FORM
from vibedash import routes
from vibedash.analysis_jobs import AnalysisJobStore
from vibedash.task_capabilities import (
    TaskAdmissionError, capability_catalog, assess_task_request,
    resolve_analysis_task, validate_stored_task_routing,
)


class TaskCapabilityTests(unittest.TestCase):
    def test_catalogue_has_bounded_workflows_without_industry_eligibility_or_professional_claim(self):
        catalog = capability_catalog()
        self.assertTrue(catalog['industry_independent'])
        self.assertFalse(catalog['automatic_business_understanding'])
        self.assertEqual(catalog['professional_equivalence'], 'not_established')
        available = {task['task_type'] for task in catalog['tasks'] if task['status'] == 'available'}
        self.assertEqual(available, {'descriptive_analysis', 'daily_forecast', 'period_comparison'})
        catalog['tasks'][0]['requirements'].append('tampered')
        self.assertNotIn('tampered', json.dumps(capability_catalog()))

    def test_absent_task_requests_clarification_without_automatic_execution(self):
        for value in (None, '', '  '):
            result = assess_task_request(value)
            self.assertEqual(result['status'], 'needs_clarification')
            self.assertFalse(result['ready_for_execution'])
            self.assertTrue(result['questions'])

    def test_supported_type_still_requires_dataset_review(self):
        for task_type in ('descriptive_analysis', 'daily_forecast', 'period_comparison'):
            result = assess_task_request(task_type)
            self.assertEqual(result['status'], 'requires_dataset_review')
            self.assertFalse(result['ready_for_execution'])
            self.assertTrue(result['requirements'])

    def test_unknown_or_exploratory_task_is_not_silently_replaced(self):
        for value in ('classification', 'regression', 'segmentation', 'anomaly_detection',
                      'causal_analysis', 'healthcare', 'banking', '<script>private</script>', [], {}, 1):
            result = assess_task_request(value)
            self.assertEqual(result['status'], 'not_supported')
            self.assertFalse(result['ready_for_execution'])
            self.assertIsNone(result['workflow'])
        for workflow in ('unknown', {}, []):
            self.assertEqual(assess_task_request('daily_forecast', workflow=workflow)['status'], 'unsupported_workflow')

    def test_explicit_task_can_select_mode_but_conflicts_are_rejected(self):
        mode, routing = resolve_analysis_task({'task_type': 'daily_forecast'})
        self.assertEqual(mode, 'forecast')
        self.assertEqual(routing['intent_source'], 'explicit_task_type')
        for form in ({'task_type': ''}, {'task_type': 'classification'}, {'task_type': 'period_comparison'},
                     {'task_type': 'daily_forecast', 'analysis_mode': 'evidence'},
                     {'task_type': 'descriptive_analysis', 'analysis_mode': 'forecast'}):
            with self.subTest(form=form), self.assertRaises(TaskAdmissionError):
                resolve_analysis_task(form)

    def test_legacy_modes_keep_their_meaning_without_claiming_inferred_intent(self):
        for form, expected in (({}, 'descriptive_analysis'), ({'analysis_mode': 'forecast'}, 'daily_forecast')):
            _, routing = resolve_analysis_task(form)
            self.assertEqual(routing['task_type'], expected)
            self.assertEqual(routing['intent_source'], 'legacy_analysis_mode')
        for mode in ('unknown', [], None):
            with self.assertRaises(TaskAdmissionError): resolve_analysis_task({'analysis_mode': mode})

    def test_worker_rechecks_type_workflow_and_execution_binding(self):
        _, routing = resolve_analysis_task({'task_type': 'daily_forecast'})
        payload = {'task_routing': routing, 'analysis_kind': 'daily_forecast',
                   'forecast_task': {'decision_policy': {}, 'decision_policy_confirmed': True}}
        validate_stored_task_routing(payload)
        validate_stored_task_routing({})  # old stored jobs remain readable
        for key, value in (('task_type', 'classification'), ('workflow', 'period_comparison_jobs'),
                           ('contract', 'unknown'), ('intent_source', [])):
            changed = copy.deepcopy(payload)
            changed['task_routing'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): validate_stored_task_routing(changed)
        for changed in (payload | {'analysis_kind': None}, payload | {'forecast_task': None},
                        payload | {'analysis_kind': []}, payload | {'task_routing': []}):
            with self.assertRaises(ValueError): validate_stored_task_routing(changed)


class TaskCapabilityRouteTests(unittest.TestCase):
    setUp = job_fixture.VibeDashJobRouteTests.setUp
    tearDown = job_fixture.VibeDashJobRouteTests.tearDown
    request_data = forecast_fixture.VibeDashForecastRouteTests.request_data

    def test_public_catalogue_and_guidance_do_not_load_data_or_queue_work(self):
        with patch('vibedash.routes._load_vibedash_csv') as load, \
                patch('vibedash.routes.analysis_job_dispatcher.submit') as submit:
            with web_app.app.test_client() as client:
                catalog = client.get('/vibedash/task-capabilities')
                self.assertEqual(catalog.status_code, 200)
                self.assertEqual(catalog.get_json(), capability_catalog())
                self.assertEqual(client.get('/vibedash/task-intake').get_json()['status'], 'needs_clarification')
                for task in ('daily_forecast', 'classification'):
                    result = client.get('/vibedash/task-intake', query_string={'task_type': task}).get_json()
                    self.assertFalse(result['ready_for_execution'])
                result = client.get('/vibedash/task-intake', query_string={
                    'task_type': 'period_comparison', 'workflow': 'analysis_jobs'}).get_json()
                self.assertEqual(result['status'], 'workflow_mismatch')
            load.assert_not_called()
            submit.assert_not_called()
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])

    def test_unsupported_or_conflicting_explicit_request_is_rejected_before_upload(self):
        with patch('vibedash.routes._load_vibedash_csv') as load, \
                patch('vibedash.routes.analysis_job_dispatcher.submit') as submit:
            with web_app.app.test_client() as client:
                for fields in ({'task_type': 'classification'}, {'task_type': 'period_comparison'},
                               {'task_type': ''}, {'task_type': 'daily_forecast', 'analysis_mode': 'evidence'}):
                    response = client.post('/vibedash/jobs', data={'prompt': 'Do the requested work'} | fields)
                    self.assertEqual(response.status_code, 422)
                    self.assertIn('task_intake', response.get_json())
                    self.assertFalse(response.get_json()['task_intake']['ready_for_execution'])
                response = client.post('/vibedash/preview', data={'task_type': 'classification'})
                self.assertEqual(response.status_code, 302)
            load.assert_not_called()
            submit.assert_not_called()
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])

    @patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=True)
    def test_explicit_forecast_without_legacy_mode_still_requires_csrf_and_confirmed_data(self, submit):
        with web_app.app.test_client() as client:
            data = self.request_data(client, task_type='daily_forecast', csrf_token='')
            data.pop('analysis_mode')
            self.assertEqual(client.post('/vibedash/jobs', data=data).status_code, 400)
            submit.assert_not_called()
            data = self.request_data(client, task_type='daily_forecast')
            data.pop('analysis_mode')
            self.assertEqual(client.post('/vibedash/jobs', data=data).status_code, 422)
            submit.assert_not_called()
            data = self.request_data(client, task_type='daily_forecast', **POLICY_FORM)
            data.pop('analysis_mode')
            response = client.post('/vibedash/jobs', data=data)
            self.assertEqual(response.status_code, 202)
            with client.session_transaction() as browser:
                scope = browser['vibedash_analysis_scope_id']
            job = AnalysisJobStore(web_app.app.config['VIBEDASH_JOB_STORE_PATH']).get(response.get_json()['job_id'], scope)
        self.assertEqual(job['payload']['analysis_kind'], 'daily_forecast')
        self.assertEqual(job['payload']['task_routing']['task_type'], 'daily_forecast')
        self.assertEqual(job['payload']['forecast_task']['settings']['target_column'], 'cnt')

    def test_mutated_worker_routing_fails_before_csv_processing(self):
        _, routing = resolve_analysis_task({'task_type': 'daily_forecast'})
        payload = {'task_routing': routing | {'task_type': 'descriptive_analysis'},
                   'analysis_kind': 'daily_forecast', 'forecast_task': {}}
        with web_app.app.app_context(), patch('vibedash.routes._load_vibedash_csv') as load:
            with self.assertRaises(ValueError): routes._build_dashboard_session(payload)
            load.assert_not_called()

    def test_mutated_execution_kind_cannot_switch_to_the_comparison_executor(self):
        _, routing = resolve_analysis_task({'task_type': 'daily_forecast'})
        payload = {'task_routing': routing, 'analysis_kind': 'period_comparison', 'forecast_task': {}}
        with web_app.app.app_context(), patch('vibedash.routes._session_persist_scope'), \
                patch('vibedash.routes._build_period_comparison_session') as comparison, \
                patch('vibedash.routes._build_dashboard_session') as dashboard:
            with self.assertRaises(ValueError):
                routes._process_analysis_job({'id': 'test', 'scope_id': 'guest:test', 'payload': payload})
            comparison.assert_not_called()
            dashboard.assert_not_called()


if __name__ == '__main__':
    unittest.main()
