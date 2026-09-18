import unittest
import sqlite3
from collections import UserDict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import web_app

from vibedash.analysis_jobs import AnalysisJobStore
from vibedash.comparison_report import build_comparison_decision_guidance


class ComparisonReportExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        self.previous = {
            key: web_app.app.config.get(key)
            for key in ('UPLOAD_FOLDER', 'VIBEDASH_JOB_STORE_PATH')
        }
        web_app.app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=str(root / 'uploads'),
            VIBEDASH_JOB_STORE_PATH=str(root / 'jobs.sqlite3'),
        )
        Path(web_app.app.config['UPLOAD_FOLDER']).mkdir()

    def tearDown(self):
        web_app.app.config.update(self.previous)
        self.tmp.cleanup()

    def _job(self, client, *, status='completed', comparison=True, session_id=True):
        client.get('/vibedash/')
        with client.session_transaction() as browser_session:
            scope_id = browser_session['vibedash_analysis_scope_id']
        store = AnalysisJobStore(web_app.app.config['VIBEDASH_JOB_STORE_PATH'])
        payload = {'analysis_kind': 'period_comparison'} if comparison else {'filename': 'data.csv'}
        job = store.create(scope_id, payload)
        if status == 'running':
            store.claim(job['id'])
        elif status == 'failed':
            store.fail(job['id'])
        elif status == 'completed':
            store.claim(job['id'])
            store.complete(
                job['id'],
                '12345678-1234-4234-8234-123456789012' if session_id else '12345678-1234-4234-8234-123456789012',
                {'comparison': {'status': 'stable'}} if comparison else {'analysis_contract': 'other'},
            )
            if not session_id:
                # The route contract treats a completed record without a
                # retained session as not ready; avoid using a client id.
                with sqlite3.connect(web_app.app.config['VIBEDASH_JOB_STORE_PATH']) as connection:
                    connection.execute('UPDATE analysis_jobs SET session_id = NULL WHERE id = ?', (job['id'],))
        return job['id']

    @staticmethod
    def _report():
        return {
            'contract': 'period-comparison-v1',
            'status': 'review_required',
            'summary': 'Aggregate summary',
            'input': {'baseline': {'rows': 20, 'columns': 2}, 'current': {'rows': 22, 'columns': 2}},
            'schema_changes': {'new_columns': ['secret-column'], 'missing_columns': [], 'type_changes': []},
            'numeric_metrics': {'eligible_count': 2, 'tested_count': 2, 'significant_count': 1, 'metrics': []},
            'distribution_drift': {'signals': [{'feature': 'secret-feature'}]},
            'limitations': ['Observational and non-causal.'],
        }

    def test_guidance_is_bounded_aggregate_only(self):
        guidance = build_comparison_decision_guidance(self._report())
        self.assertEqual(guidance['contract'], 'comparison-decision-guidance-v1')
        self.assertEqual(guidance['evidence_snapshot']['schema_changes'], 1)
        self.assertLessEqual(len(guidance['questions']), 5)
        self.assertNotIn('secret-column', str(guidance))
        self.assertNotIn('secret-feature', str(guidance))

    def test_guidance_caps_sum_of_schema_change_lists(self):
        report = self._report()
        report['schema_changes'] = {
            'new_columns': [None] * 100_000,
            'missing_columns': [None] * 100_000,
            'type_changes': [None] * 100_000,
        }
        guidance = build_comparison_decision_guidance(report)
        self.assertEqual(guidance['evidence_snapshot']['schema_changes'], 100_000)

    def test_guidance_bounds_huge_and_nonfinite_counts(self):
        report = UserDict(self._report())
        report['input']['baseline']['rows'] = float('inf')
        report['input']['current']['rows'] = 10 ** 1000
        report['numeric_metrics']['eligible_count'] = float('nan')
        report['numeric_metrics']['tested_count'] = -10
        report['numeric_metrics']['significant_count'] = '999999999999999999'
        guidance = build_comparison_decision_guidance(report)
        snapshot = guidance['evidence_snapshot']
        self.assertEqual(snapshot['baseline_rows'], 0)
        self.assertEqual(snapshot['current_rows'], 100_000)
        self.assertEqual(snapshot['eligible_metrics'], 0)
        self.assertEqual(snapshot['tested_metrics'], 0)
        self.assertEqual(snapshot['significant_metrics'], 100_000)

    def test_guidance_handles_malformed_status_deterministically(self):
        class UnhashableStatus(str):
            __hash__ = None

        for malformed_status in ([], {}, float('nan'), float('inf'), UnhashableStatus('stable')):
            report = self._report()
            report['status'] = malformed_status
            guidance = build_comparison_decision_guidance(report)
            expected = 'stable' if isinstance(malformed_status, UnhashableStatus) else 'unknown'
            self.assertEqual(guidance['evidence_snapshot']['status'], expected)

    def test_owner_gets_standalone_download_with_safe_headers(self):
        with web_app.app.test_client() as client:
            job_id = self._job(client)
            with patch('vibedash.routes.load_session_data', return_value={
                'analysis_kind': 'period_comparison',
                'report': self._report(),
                'baseline_filename': '<private>.csv',
                'current_filename': 'current.csv',
                'baseline_label': '<Before>',
                'current_label': 'After',
            }):
                response = client.get(f'/vibedash/jobs/{job_id}/comparison-report.html')
        self.assertEqual(response.status_code, 200)
        self.assertRegex(response.headers['Content-Disposition'], r'data-prism-comparison-[0-9a-f]{12}\.html')
        self.assertEqual(response.headers['Cache-Control'], 'private, no-store')
        self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
        self.assertEqual(response.headers['X-Frame-Options'], 'DENY')
        self.assertIn("default-src 'none'", response.headers['Content-Security-Policy'])
        self.assertIn("script-src 'none'", response.headers['Content-Security-Policy'])
        self.assertIn("frame-ancestors 'none'", response.headers['Content-Security-Policy'])
        self.assertNotIn("navigate-to", response.headers['Content-Security-Policy'])
        self.assertIn('Referrer-Policy', response.headers)
        html = response.get_data(as_text=True)
        self.assertIn('<style>', html)
        self.assertIn('@media print', html)
        self.assertNotIn('fonts.googleapis.com', html)
        self.assertNotIn('<script', html.lower())
        self.assertNotIn('<form', html.lower())
        self.assertNotIn('href="/vibedash/jobs/', html)
        self.assertNotIn('Open audit manifest', html)
        self.assertNotIn('<private>.csv', html)
        self.assertIn('&lt;private&gt;.csv', html)

    def test_status_and_result_expose_the_scoped_report_link(self):
        with web_app.app.test_client() as client:
            job_id = self._job(client)
            status = client.get(f'/vibedash/jobs/{job_id}').get_json()
            self.assertEqual(status['status'], 'completed')
            self.assertEqual(
                status['comparison_report_url'],
                f'/vibedash/jobs/{job_id}/comparison-report.html',
            )
            with patch('vibedash.routes.load_session_data', return_value={
                'analysis_kind': 'period_comparison',
                'report': self._report(),
            }):
                result = client.get(status['result_url'])
        self.assertEqual(result.status_code, 200)
        self.assertIn('Download HTML report', result.get_data(as_text=True))

    def test_export_renders_malformed_json_blocks_without_crashing(self):
        malformed_report = {
            'status': [],
            'input': {'baseline': [], 'current': {'readiness': []}},
            'schema_changes': {'type_changes': [None], 'new_columns': {'not': 'a list'}},
            'numeric_metrics': {'metrics': [None, {'baseline': [], 'current': [], 'confidence_interval': []}]},
            'distribution_drift': {'signals': [None, 'odd signal']},
            'limitations': {'not': 'a list'},
        }
        with web_app.app.test_client() as client:
            job_id = self._job(client)
            with patch('vibedash.routes.load_session_data', return_value={
                'analysis_kind': 'period_comparison',
                'report': malformed_report,
            }):
                response = client.get(f'/vibedash/jobs/{job_id}/comparison-report.html')
        self.assertEqual(response.status_code, 200)
        self.assertIn('<html', response.get_data(as_text=True))

    def test_export_ignores_non_sequence_schema_column_blocks(self):
        malformed_report = self._report()
        malformed_report['schema_changes'] = {
            'new_columns': 123,
            'missing_columns': True,
            'type_changes': 'not-a-list',
        }
        with web_app.app.test_client() as client:
            job_id = self._job(client)
            with patch('vibedash.routes.load_session_data', return_value={
                'analysis_kind': 'period_comparison',
                'report': malformed_report,
            }):
                response = client.get(f'/vibedash/jobs/{job_id}/comparison-report.html')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn('No added columns.', html)
        self.assertIn('No removed columns.', html)
        self.assertIn('No type changes.', html)

    def test_export_and_result_reject_non_mapping_session_contract(self):
        with web_app.app.test_client() as client:
            job_id = self._job(client)
            for malformed_session in ([1], 'not-an-object', 1):
                with patch('vibedash.routes.load_session_data', return_value=malformed_session):
                    export_response = client.get(
                        f'/vibedash/jobs/{job_id}/comparison-report.html'
                    )
                    result_response = client.get(f'/vibedash/jobs/{job_id}/result')
                self.assertEqual(export_response.status_code, 409)
                self.assertEqual(result_response.status_code, 409)

    def test_result_rejects_non_mapping_report_contract(self):
        with web_app.app.test_client() as client:
            job_id = self._job(client)
            for malformed_report in ([], 'not-an-object', 1):
                with patch('vibedash.routes.load_session_data', return_value={
                    'analysis_kind': 'period_comparison',
                    'report': malformed_report,
                }):
                    response = client.get(f'/vibedash/jobs/{job_id}/result')
                self.assertEqual(response.status_code, 409)

    def test_export_escapes_css_style_end_tag_boundary(self):
        with web_app.app.test_client() as client:
            job_id = self._job(client)
            with patch('vibedash.routes.load_session_data', return_value={
                'analysis_kind': 'period_comparison',
                'report': self._report(),
            }), patch('vibedash.routes.Path.read_text', return_value='x </STYLE><script>bad</script> y'):
                response = client.get(f'/vibedash/jobs/{job_id}/comparison-report.html')
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(html.lower().count('</style>'), 1)
        self.assertIn('<\\/style><script>bad</script> y</style>', html)

    def test_completed_comparison_with_expired_session_is_gone(self):
        with web_app.app.test_client() as client:
            job_id = self._job(client)
            with patch('vibedash.routes.load_session_data', return_value=None):
                response = client.get(f'/vibedash/jobs/{job_id}/comparison-report.html')
        self.assertEqual(response.status_code, 410)

    def test_foreign_missing_and_not_ready_are_non_leaking(self):
        with web_app.app.test_client() as owner:
            job_id = self._job(owner)
        with web_app.app.test_client() as stranger:
            self.assertEqual(stranger.get(f'/vibedash/jobs/{job_id}/comparison-report.html').status_code, 404)
        with web_app.app.test_client() as owner:
            running = self._job(owner, status='running')
            self.assertEqual(owner.get(f'/vibedash/jobs/{running}/comparison-report.html').status_code, 409)
            missing = self._job(owner, session_id=False)
            self.assertEqual(owner.get(f'/vibedash/jobs/{missing}/comparison-report.html').status_code, 409)
            other = self._job(owner, comparison=False)
            self.assertEqual(owner.get(f'/vibedash/jobs/{other}/comparison-report.html').status_code, 409)


if __name__ == '__main__':
    unittest.main()
