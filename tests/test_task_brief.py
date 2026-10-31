"""Explicit context is a request contract, not inferred business understanding."""
import copy
import json
import os
import sqlite3
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import web_app
from tests import test_vibedash_job_routes as job_fixture
from tests import test_vibedash_forecast_routes as forecast_fixture
from tests.test_forecast_policy import POLICY_FORM
from vibedash import routes
from vibedash.analysis_jobs import AnalysisJobStore
from vibedash.task_brief import (
    CONTRACT, MAX_REVIEW_BYTES, TaskBriefError, bind_task_brief, parse_review_json,
    prepare_task_brief, review_task_brief, task_brief_audit_view, validate_stored_task_brief,
)


BRIEF = {'task_type': 'descriptive_analysis', 'question': 'Describe daily demand.',
         'observation_unit': 'One completed day for one location',
         'deliverable': 'evidence_summary', 'confirmed': True}
BRIEF_FORM = {'task_observation_unit': BRIEF['observation_unit'],
              'task_deliverable': 'forecast_evaluation', 'task_brief_confirmed': 'yes'}


class TaskBriefTests(unittest.TestCase):
    def test_missing_context_produces_field_questions_not_an_execution_approval(self):
        result = review_task_brief({'task_type': 'descriptive_analysis'})
        self.assertEqual(result['status'], 'needs_clarification')
        self.assertFalse(result['ready_for_execution'])
        self.assertEqual({item['field'] for item in result['questions']},
                         {'question', 'observation_unit', 'deliverable', 'confirmed'})

    def test_complete_explicit_context_only_admits_dataset_review(self):
        for task_type, deliverable in (('descriptive_analysis', 'evidence_summary'),
                                       ('daily_forecast', 'forecast_evaluation')):
            result = review_task_brief(BRIEF | {'task_type': task_type, 'deliverable': deliverable})
            self.assertEqual(result['status'], 'ready_for_dataset_review')
            self.assertFalse(result['ready_for_execution'])
            self.assertEqual(result['business_understanding'], 'not_independently_verified')
            self.assertEqual(result['interpretation_source'], 'explicit_user_context')
            self.assertNotIn(BRIEF['question'], json.dumps(result))
            self.assertNotIn(BRIEF['observation_unit'], json.dumps(result))

    def test_unsupported_requests_and_wrong_workflow_are_never_replaced(self):
        for task_type, status in (('classification', 'not_supported'),
                                  ('period_comparison', 'workflow_mismatch'), (None, 'needs_clarification')):
            result = review_task_brief(BRIEF | {'task_type': task_type})
            self.assertEqual(result['status'], status)
            self.assertFalse(result['ready_for_execution'])

    def test_wrong_deliverable_requires_explicit_correction(self):
        result = review_task_brief(BRIEF | {'deliverable': 'trained_production_model'})
        self.assertEqual(result['status'], 'needs_clarification')
        self.assertEqual(result['questions'][0]['field'], 'deliverable')

    def test_invalid_types_sizes_and_unknown_fields_are_rejected(self):
        for changed in ({'question': 'a' * 4001}, {'question': '\ud800'},
                        {'observation_unit': 'a' * 241}, {'observation_unit': []},
                        {'deliverable': {}}, {'confirmed': 'yes'}, {'confirmed': 1}, {'industry': 'banking'}):
            with self.subTest(changed=changed), self.assertRaises(TaskBriefError):
                review_task_brief(BRIEF | changed)
        with self.assertRaises(TaskBriefError): review_task_brief([])

    def test_strict_json_rejects_duplicates_nonfinite_nesting_and_invalid_encoding(self):
        self.assertEqual(parse_review_json(json.dumps(BRIEF).encode()), BRIEF)
        for raw in (b'{"confirmed":false,"confirmed":true}', b'{"confirmed":NaN}',
                    b'\xff', b'[' * 2000, b' ' * (MAX_REVIEW_BYTES + 1)):
            with self.subTest(raw=raw[:30]), self.assertRaises(TaskBriefError): parse_review_json(raw)

    def test_legacy_form_remains_opt_in_but_partial_briefs_cannot_bypass_confirmation(self):
        self.assertIsNone(prepare_task_brief({}, task_type=BRIEF['task_type'], question=BRIEF['question']))
        for form in ({'task_observation_unit': ''}, {'task_brief_confirmed': 'yes'},
                     {'task_deliverable': 'evidence_summary'}):
            with self.assertRaises(TaskBriefError):
                prepare_task_brief(form, task_type=BRIEF['task_type'], question=BRIEF['question'])
        prepared = prepare_task_brief(BRIEF_FORM | {'task_deliverable': 'evidence_summary'},
                                     task_type=BRIEF['task_type'], question='  ' + BRIEF['question'] + '  ')
        self.assertEqual(prepared, BRIEF)

    def test_binding_detects_changed_question_mode_semantics_context_hash_or_extra_fields(self):
        brief = bind_task_brief(BRIEF, 'a' * 64)
        arguments = {'task_type': BRIEF['task_type'], 'question': BRIEF['question'], 'semantics_sha256': 'a' * 64}
        self.assertEqual(validate_stored_task_brief(brief, **arguments), brief)
        for key, value in (('question', 'Another decision'), ('task_type', 'daily_forecast'),
                           ('semantics_sha256', 'b' * 64)):
            with self.subTest(key=key), self.assertRaises(TaskBriefError):
                validate_stored_task_brief(brief, **(arguments | {key: value}))
        for key, value in (('observation_unit', 'A different unit'), ('confirmed', False),
                           ('contract', 'old'), ('sha256', 'f' * 64), ('extra', True)):
            with self.subTest(key=key), self.assertRaises(TaskBriefError):
                validate_stored_task_brief(brief | {key: value}, **arguments)
        with self.assertRaises(TaskBriefError): bind_task_brief(BRIEF, 'invalid')

    def test_audit_does_not_include_free_text(self):
        audit = task_brief_audit_view(bind_task_brief(BRIEF, 'a' * 64))
        self.assertEqual(audit['contract'], CONTRACT)
        self.assertNotIn('question', audit)
        self.assertNotIn('observation_unit', audit)
        self.assertEqual(len(audit['sha256']), 64)


class TaskBriefRouteTests(unittest.TestCase):
    setUp = job_fixture.VibeDashJobRouteTests.setUp
    tearDown = job_fixture.VibeDashJobRouteTests.tearDown
    request_data = forecast_fixture.VibeDashForecastRouteTests.request_data

    def token(self, client):
        client.get('/vibedash/')
        with client.session_transaction() as browser:
            return browser['vibedash_decision_csrf_token']

    def test_review_is_bounded_csrf_protected_metadata_only_and_does_not_train_store_or_echo(self):
        with patch('vibedash.routes._load_vibedash_csv') as load, \
                patch('vibedash.routes.analysis_job_dispatcher.submit') as submit:
            with web_app.app.test_client() as client:
                self.assertEqual(client.post('/vibedash/task-brief', json=BRIEF).status_code, 400)
                headers = {'X-CSRF-Token': self.token(client)}
                self.assertEqual(client.post('/vibedash/task-brief', data=BRIEF, headers=headers).status_code, 415)
                response = client.post('/vibedash/task-brief', json=BRIEF, headers=headers)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()['status'], 'ready_for_dataset_review')
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                self.assertNotIn(BRIEF['question'], response.get_data(as_text=True))
                self.assertEqual(client.post('/vibedash/task-brief', headers=headers,
                    content_type='application/json', data=b' ' * (MAX_REVIEW_BYTES + 1)).status_code, 413)
                for raw in (b'{"confirmed":false,"confirmed":true}', b'[]', b'{"question":NaN}'):
                    self.assertEqual(client.post('/vibedash/task-brief', headers=headers,
                        content_type='application/json', data=raw).status_code, 422)
                response = client.post('/vibedash/task-brief', json={'task_type': 'daily_forecast'}, headers=headers)
                self.assertEqual(response.get_json()['status'], 'needs_clarification')
            load.assert_not_called()
            submit.assert_not_called()
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])
        with sqlite3.connect(web_app.app.config['VIBEDASH_JOB_STORE_PATH']) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM analysis_jobs').fetchone()[0], 0)

    def test_partial_job_brief_is_rejected_before_loading_or_copying_source(self):
        with patch('vibedash.routes._load_vibedash_csv') as load, \
                patch('vibedash.routes.analysis_job_dispatcher.submit') as submit:
            with web_app.app.test_client() as client:
                data = self.request_data(client, task_observation_unit='One day')
                response = client.post('/vibedash/jobs', data=data)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.get_json()['task_brief']['status'], 'needs_clarification')
            load.assert_not_called()
            submit.assert_not_called()
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])

    @patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=True)
    def test_job_binds_context_and_worker_rejects_changes_before_spec_generation_or_fit(self, submit):
        with web_app.app.test_client() as client:
            response = client.post('/vibedash/jobs', data=self.request_data(client, **(POLICY_FORM | BRIEF_FORM)))
            self.assertEqual(response.status_code, 202, response.get_json())
            with client.session_transaction() as browser: scope = browser['vibedash_analysis_scope_id']
            job = AnalysisJobStore(web_app.app.config['VIBEDASH_JOB_STORE_PATH']).get(response.get_json()['job_id'], scope)
        brief = job['payload']['task_brief']
        self.assertEqual(brief['semantics_sha256'], job['payload']['semantic_contract']['sha256'])
        self.assertEqual(brief['task_type'], 'daily_forecast')
        changed_semantics = copy.deepcopy(job['payload']['semantic_contract'])
        next(column for column in changed_semantics['columns'] if column['name'] == 'cnt')['unit'] = 'number'
        for field, value in (('prompt', 'A changed question'), ('task_brief', None),
                             ('task_brief', brief | {'observation_unit': 'One customer'}),
                             ('semantic_contract', changed_semantics)):
            payload = copy.deepcopy(job['payload']) | {field: value}
            with web_app.app.app_context(), patch('vibedash.routes.parse_prompt_to_viz_spec') as spec, \
                    patch('vibedash.routes.bridge_generate_dashboard_data') as generate, \
                    patch('vibedash.routes.run_confirmed_forecast') as fit:
                with self.assertRaises(TaskBriefError): routes._build_dashboard_session(payload)
                spec.assert_not_called()
                generate.assert_not_called()
                fit.assert_not_called()

    @patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=True)
    def test_descriptive_brief_requires_csrf_and_keeps_the_selected_evidence_executor(self, submit):
        fields = BRIEF_FORM | {'analysis_mode': 'evidence', 'task_type': 'descriptive_analysis',
                               'task_deliverable': 'evidence_summary'}
        with web_app.app.test_client() as client:
            rejected = client.post('/vibedash/jobs', data=self.request_data(client, **(fields | {'csrf_token': ''})))
            self.assertEqual(rejected.status_code, 400)
            submit.assert_not_called()
            response = client.post('/vibedash/jobs', data=self.request_data(client, **fields))
            self.assertEqual(response.status_code, 202, response.get_json())
            with client.session_transaction() as browser: scope = browser['vibedash_analysis_scope_id']
            payload = AnalysisJobStore(web_app.app.config['VIBEDASH_JOB_STORE_PATH']).get(response.get_json()['job_id'], scope)['payload']
        self.assertEqual(payload['task_brief']['task_type'], 'descriptive_analysis')
        self.assertNotIn('forecast_task', payload)

    def test_completed_owned_result_and_export_show_escaped_context_aggregate_manifest_omits_it(self):
        unit = '<script>private row description</script>'
        with patch.dict(os.environ, DATA_PRISM_STATE_DIR=self.temporary_directory.name):
            with web_app.app.test_client() as client:
                data = self.request_data(client, **(POLICY_FORM | BRIEF_FORM | {'task_observation_unit': unit}))
                queued = client.post('/vibedash/jobs', data=data)
                self.assertEqual(queued.status_code, 202, queued.get_json())
                for _ in range(150):
                    status = client.get(queued.get_json()['status_url']).get_json()
                    if status['status'] in {'completed', 'failed'}: break
                    time.sleep(.05)
                self.assertEqual(status['status'], 'completed', status)
                html = client.get(status['result_url']).get_data(as_text=True)
                self.assertIn('Confirmed task context', html)
                self.assertIn('&lt;script&gt;private row description&lt;/script&gt;', html)
                self.assertNotIn(unit, html)
                manifest = client.get(status['manifest_url']).get_json()
                self.assertIn('task_brief', manifest)
                self.assertNotIn('private row description', json.dumps(manifest))
                with client.session_transaction() as browser: scope = browser['vibedash_analysis_scope_id']
                job = AnalysisJobStore(web_app.app.config['VIBEDASH_JOB_STORE_PATH']).get(queued.get_json()['job_id'], scope)
                export = client.get('/vibedash/export/' + job['session_id'])
                self.assertIn('Confirmed task context', export.get_data(as_text=True))
                export.close()

    def test_landing_guides_context_and_resets_confirmation_on_question_or_row_meaning_changes(self):
        with web_app.app.test_client() as client: html = client.get('/vibedash/').get_data(as_text=True)
        for field in ('task_type', 'task_observation_unit', 'task_deliverable', 'task_brief_confirmed'):
            self.assertIn('name="' + field + '"', html)
        self.assertIn('What does one row represent?', html)
        self.assertIn('Planned result', html)
        self.assertIn("analysisForm.addEventListener('input'", html)
        self.assertIn('invalidateTaskConfirmation(true);', html)
        self.assertIn('revision !== taskRevision', html)
        self.assertIn('selectedFile !== fileInput.files[0]', html)
        self.assertIn('showTaskQuestions', html)
        self.assertIn('fetch(taskBriefUrl', html)


if __name__ == '__main__': unittest.main()
