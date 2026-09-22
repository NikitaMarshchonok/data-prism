import re
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import web_app
from vibedash.analysis_jobs import AnalysisJobStore
from vibedash.pilot_metrics import build_pilot_report


class PilotRouteTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'jobs.sqlite3'
        config = patch.dict(web_app.app.config, {
            'TESTING': True, 'UPLOAD_FOLDER': str(Path(temporary.name) / 'uploads'),
            'VIBEDASH_JOB_STORE_PATH': str(self.path),
        })
        config.start()
        self.addCleanup(config.stop)
        dispatcher = patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=True)
        dispatcher.start()
        self.addCleanup(dispatcher.stop)
        self.client = web_app.app.test_client()
        self.landing = self.client.get('/vibedash/').get_data(as_text=True)
        self.csrf = re.search(r'name="csrf_token" value="([a-f0-9]{64})"', self.landing).group(1)

    def create_job(self, **values):
        form = {'prompt': 'Review SaaS growth', 'demo_dataset': 'saas_growth',
                'pilot_metrics': 'yes', 'csrf_token': self.csrf}
        form.update(values)
        return self.client.post('/vibedash/jobs', data=form)

    def complete(self, job_id):
        store = AnalysisJobStore(self.path)
        store.claim(job_id)
        store.complete(job_id, str(uuid.uuid4()))

    def report(self):
        return build_pilot_report(self.path)['cohorts']['demo']

    def feedback_form(self):
        return {
            'csrf_token': self.csrf,
            'usefulness': 'useful',
            'blocker': 'none',
            'perceived_time_saved': '15_to_30_minutes',
            'next_cycle_intent': 'yes',
        }

    def test_onboarding_and_opt_in_are_visible_and_unchecked(self):
        self.assertIn('Weekly SaaS review', self.landing)
        inputs = re.findall(r'<input[^>]+name="pilot_metrics"[^>]*>', self.landing)
        self.assertEqual(len(inputs), 2)
        self.assertTrue(all('checked' not in element for element in inputs))
        self.assertIn('pseudonymous', self.landing)
        self.assertIn('not a secure enterprise workspace', self.landing)

    def test_consent_requires_valid_csrf_but_analysis_does_not_require_consent(self):
        rejected = self.create_job(csrf_token='wrong')
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(self.report()['accepted_analyses'], 0)
        plain = self.create_job(pilot_metrics='', csrf_token='')
        opted = self.create_job()
        self.assertEqual(plain.status_code, 202)
        self.assertEqual(opted.status_code, 202)
        self.assertEqual(self.report()['accepted_analyses'], 1)
        for _ in range(3):
            self.client.get(opted.get_json()['status_url'])
        self.assertEqual(self.report()['accepted_analyses'], 1)

    def test_feedback_is_scoped_csrf_protected_and_requires_completion(self):
        job_id = self.create_job().get_json()['job_id']
        url = f'/vibedash/jobs/{job_id}/feedback'
        form = self.feedback_form()
        self.assertEqual(self.client.post(url, data=form).status_code, 409)
        self.complete(job_id)
        self.assertEqual(self.client.post(url, data={**form, 'csrf_token': 'bad'}).status_code, 400)
        stranger = web_app.app.test_client()
        with stranger.session_transaction() as browser:
            browser['vibedash_decision_csrf_token'] = self.csrf
        self.assertEqual(stranger.post(url, data=form).status_code, 404)
        self.assertEqual(self.client.post(url, data=form).status_code, 303)
        self.assertEqual(self.report()['feedback_responses'], 1)
        self.assertEqual(self.report()['value_feedback_responses'], 1)
        invalid = self.client.post(url, data={**form, 'usefulness': 'free text'})
        self.assertEqual(invalid.status_code, 303)
        self.assertEqual(self.report()['feedback_responses'], 1)
        invalid_time = self.client.post(
            url, data={**form, 'perceived_time_saved': 'two hours exactly'}
        )
        self.assertEqual(invalid_time.status_code, 303)
        self.assertEqual(self.report()['perceived_time_saved']['15_to_30_minutes'], 1)
        missing_intent = dict(form)
        missing_intent.pop('next_cycle_intent')
        self.assertEqual(self.client.post(url, data=missing_intent).status_code, 303)
        self.assertEqual(self.report()['next_cycle_intent']['yes'], 1)

    @patch('vibedash.routes.load_session_data')
    def test_feedback_render_and_withdrawal_leave_analysis_available(self, load_session):
        load_session.return_value = {
            'viz_spec': {'title': 'SaaS review'}, 'dashboard_data': {},
            'filename': 'demo.csv', 'prompt': 'Review SaaS growth',
        }
        job_id = self.create_job().get_json()['job_id']
        self.complete(job_id)
        result_url = f'/vibedash/jobs/{job_id}/result'
        page = self.client.get(result_url).get_data(as_text=True)
        self.assertIn('pilot-feedback-heading', page)
        self.assertIn('name="perceived_time_saved"', page)
        self.assertIn('name="next_cycle_intent"', page)
        response = self.client.post(f'/vibedash/jobs/{job_id}/feedback', data={
            **self.feedback_form(),
        }, follow_redirects=True)
        self.assertIn('Your feedback was saved', response.get_data(as_text=True))
        self.assertEqual(self.client.post('/vibedash/pilot/forget').status_code, 400)
        self.assertEqual(self.client.post('/vibedash/pilot/forget', data={'csrf_token': self.csrf}).status_code, 303)
        result = self.client.get(result_url)
        self.assertEqual(result.status_code, 200)
        self.assertNotIn('pilot-feedback-heading', result.get_data(as_text=True))
        self.assertEqual(self.report()['accepted_analyses'], 0)
        self.assertIsNotNone(AnalysisJobStore(self.path).get(job_id))


if __name__ == '__main__':
    unittest.main()
