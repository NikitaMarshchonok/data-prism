import json
import csv
import os
import re
import sqlite3
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import web_app
from werkzeug.datastructures import FileStorage

from vibedash.analysis_jobs import AnalysisJobCapacityError, AnalysisJobStore
from vibedash.exporter import load_session_data
from vibedash import routes


class VibeDashComparisonRouteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        self.old = {
            key: web_app.app.config.get(key)
            for key in (
                'UPLOAD_FOLDER',
                'VIBEDASH_JOB_STORE_PATH',
                'VIBEDASH_RETENTION_HOURS',
                'VIBEDASH_JOB_TIMEOUT_SECONDS',
                'VIBEDASH_MAX_ACTIVE_JOBS_PER_SCOPE',
                'VIBEDASH_MAX_ACTIVE_JOBS',
            )
        }
        web_app.app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=str(root / 'uploads'),
            VIBEDASH_JOB_STORE_PATH=str(root / 'jobs.sqlite3'),
            VIBEDASH_RETENTION_HOURS=24,
            VIBEDASH_JOB_TIMEOUT_SECONDS=600,
            VIBEDASH_MAX_ACTIVE_JOBS_PER_SCOPE=2,
            VIBEDASH_MAX_ACTIVE_JOBS=25,
        )
        Path(web_app.app.config['UPLOAD_FOLDER']).mkdir()

    def tearDown(self):
        web_app.app.config.update(self.old)
        self.tmp.cleanup()

    @staticmethod
    def csv_text(rows=30, columns='value,segment'):
        body = '\n'.join(f'{i},{"north" if i % 2 else "south"}' for i in range(rows))
        return (columns + '\n' + body + '\n').encode()

    def post_data(self, client, *, baseline_name='baseline data.csv', current_name='current.csv',
                  baseline_label='Before', current_label='After', baseline=None, current=None):
        with client.session_transaction() as browser_session:
            browser_session['vibedash_decision_csrf_token'] = 'a' * 64
        return client.post(
            '/vibedash/comparisons/jobs',
            data={
                'baseline_file': (BytesIO(baseline or self.csv_text()), baseline_name),
                'current_file': (BytesIO(current or self.csv_text()), current_name),
                'baseline_label': baseline_label,
                'current_label': current_label,
                'csrf_token': 'a' * 64,
            },
        )

    @patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=True)
    def test_queue_payload_is_bounded_and_readiness_is_returned(self, submit):
        sentinel = b'RAW_SECRET_SENTINEL'
        with web_app.app.test_client() as client:
            response = self.post_data(client, baseline= self.csv_text() + sentinel)
            payload = response.get_json()
            job = AnalysisJobStore(web_app.app.config['VIBEDASH_JOB_STORE_PATH']).get(payload['job_id'])

        self.assertEqual(response.status_code, 202)
        submit.assert_called_once()
        self.assertEqual(job['payload']['analysis_kind'], 'period_comparison')
        self.assertEqual(set(job['payload']), {'analysis_kind', 'baseline', 'current'})
        self.assertEqual(set(job['payload']['baseline']), {'stored_filename', 'filename', 'label'})
        self.assertEqual(set(job['payload']['current']), {'stored_filename', 'filename', 'label'})
        self.assertNotIn(sentinel.decode(), json.dumps(job['payload']))
        for item in (job['payload']['baseline'], job['payload']['current']):
            self.assertRegex(item['stored_filename'], r'^vibedash-[0-9a-f]{32}\.csv$')
        self.assertIn('baseline', payload['readiness'])
        self.assertIn('current', payload['readiness'])
        self.assertEqual(len(list(Path(web_app.app.config['UPLOAD_FOLDER']).glob('*.csv'))), 2)

    @patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=False)
    def test_dispatcher_decline_fails_comparison_job_and_cleans_inputs(self, submit):
        with web_app.app.test_client() as client:
            response = self.post_data(client)

        self.assertEqual(response.status_code, 500)
        submit.assert_called_once()
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])
        with sqlite3.connect(web_app.app.config['VIBEDASH_JOB_STORE_PATH']) as connection:
            statuses = [
                row[0]
                for row in connection.execute('SELECT status FROM analysis_jobs')
            ]
        self.assertEqual(statuses, ['failed'])

    def test_missing_wrong_extension_and_invalid_labels_are_rejected(self):
        with web_app.app.test_client() as client:
            cases = [
                ({'baseline_file': (BytesIO(b'a,b\n1,2\n'), 'b.csv')}, 400),
                ({'baseline_file': (BytesIO(b'a,b\n1,2\n'), 'b.txt'), 'current_file': (BytesIO(b'a,b\n1,2\n'), 'c.csv'), 'baseline_label': 'B', 'current_label': 'C'}, 400),
                ({'baseline_file': (BytesIO(self.csv_text()), 'b.csv'), 'current_file': (BytesIO(self.csv_text()), 'c.csv'), 'baseline_label': 'same', 'current_label': 'SAME'}, 400),
                ({'baseline_file': (BytesIO(self.csv_text()), 'b.csv'), 'current_file': (BytesIO(self.csv_text()), 'c.csv'), 'baseline_label': 'bad\nlabel', 'current_label': 'C'}, 400),
            ]
            for data, expected in cases:
                response = client.post('/vibedash/comparisons/jobs', data=data)
                self.assertEqual(response.status_code, expected)
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])

    def test_missing_csrf_token_is_rejected_before_upload(self):
        with web_app.app.test_client() as client:
            response = client.post(
                '/vibedash/comparisons/jobs',
                data={
                    'baseline_file': (BytesIO(self.csv_text()), 'b.csv'),
                    'current_file': (BytesIO(self.csv_text()), 'c.csv'),
                    'baseline_label': 'Before',
                    'current_label': 'After',
                },
            )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])

    def test_filename_over_255_is_rejected_before_queue(self):
        long_name = 'x' * 256 + '.csv'
        with patch('vibedash.routes.analysis_job_dispatcher.submit') as submit:
            with web_app.app.test_client() as client:
                response = self.post_data(client, baseline_name=long_name)
        self.assertEqual(response.status_code, 400)
        submit.assert_not_called()
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])

    @patch('vibedash.routes.analysis_job_dispatcher.submit')
    def test_blocked_readiness_and_bounds_leave_no_uploads(self, submit):
        blocked = b'constant\n1\n1\n1\n1\n1\n1\n1\n1\n'
        with web_app.app.test_client() as client:
            response = self.post_data(client, baseline=blocked)
        self.assertEqual(response.status_code, 422)
        self.assertFalse(response.get_json()['readiness']['baseline']['analysis_allowed'])
        submit.assert_not_called()

        with patch.object(routes, 'COMPARISON_MAX_ROWS', 2):
            with web_app.app.test_client() as client:
                response = self.post_data(client)
        self.assertEqual(response.status_code, 422)

        with patch.object(routes, 'COMPARISON_MAX_COLUMNS', 1):
            with web_app.app.test_client() as client:
                response = self.post_data(client)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])

    def test_second_save_failure_cleans_first_upload(self):
        original_save = FileStorage.save
        first_path = []

        def save_first_then_fail(storage, destination, *args, **kwargs):
            if not first_path:
                result = original_save(storage, destination, *args, **kwargs)
                first_path.extend(Path(web_app.app.config['UPLOAD_FOLDER']).glob('*.csv'))
                self.assertTrue(first_path[0].is_file())
                return result
            raise OSError('disk')

        with patch(
            'werkzeug.datastructures.FileStorage.save',
            autospec=True,
            side_effect=save_first_then_fail,
        ):
            with web_app.app.test_client() as client:
                response = self.post_data(client)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(len(first_path), 1)
        self.assertFalse(first_path[0].exists())
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])

    def test_upload_save_does_not_follow_symlink(self):
        target = Path(self.tmp.name) / 'outside.csv'
        target.write_bytes(b'keep')
        link = Path(web_app.app.config['UPLOAD_FOLDER']) / ('vibedash-' + ('a' * 32) + '.csv')
        link.symlink_to(target)
        with self.assertRaises(OSError):
            routes._save_comparison_upload(
                FileStorage(stream=BytesIO(b'overwrite'), filename='x.csv'), link
            )
        self.assertEqual(target.read_bytes(), b'keep')
        self.assertTrue(link.is_symlink())

    @patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=True)
    def test_capacity_rejection_cleans_both_uploads(self, submit):
        web_app.app.config['VIBEDASH_MAX_ACTIVE_JOBS_PER_SCOPE'] = 1
        with web_app.app.test_client() as client:
            first = self.post_data(client)
            second = self.post_data(client)
        self.assertEqual(first.status_code, 202)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(len(list(Path(web_app.app.config['UPLOAD_FOLDER']).glob('*.csv'))), 2)

    @patch('vibedash.routes.analysis_job_dispatcher.submit', side_effect=RuntimeError('dispatcher details'))
    def test_dispatcher_failure_marks_job_failed_and_cleans_inputs(self, submit):
        with web_app.app.test_client() as client:
            response = self.post_data(client)
            job = AnalysisJobStore(web_app.app.config['VIBEDASH_JOB_STORE_PATH']).list_for_scope(
                routes._analysis_scope_id(), limit=5
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(job[0]['status'], 'failed')
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])

    @patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=True)
    def test_combined_memory_bound_cleans_both_uploads(self, submit):
        with patch.object(routes, 'COMPARISON_MAX_MEMORY_BYTES', 1):
            with web_app.app.test_client() as client:
                response = self.post_data(client)
        self.assertEqual(response.status_code, 422)
        self.assertIn('memory limit', response.get_json()['error'])
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])

    def test_header_wider_than_64kib_is_checked_as_a_complete_csv_record(self):
        # A fixed-size header sample can miss columns after a very large first
        # field and incorrectly admit an over-wide CSV.
        header = ','.join(['x' * (64 * 1024 + 1)] + [f'c{i}' for i in range(100)])
        body = ','.join(['1'] * 101)
        oversized_header_csv = (header + '\n' + body + '\n').encode()
        with web_app.app.test_client() as client:
            response = self.post_data(client, baseline=oversized_header_csv)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])

    def test_header_parser_restores_process_global_field_limit(self):
        previous_limit = csv.field_size_limit(12345)
        try:
            self.assertEqual(
                routes._read_original_csv_header(BytesIO(b'value\n1\n'), 'utf-8'),
                ['value'],
            )
            self.assertEqual(csv.field_size_limit(), 12345)
        finally:
            csv.field_size_limit(previous_limit)

    def test_stale_running_comparison_cleanup_removes_both_inputs(self):
        with web_app.app.test_client() as client:
            client.get('/vibedash/')
            with client.session_transaction() as browser_session:
                scope_id = browser_session['vibedash_analysis_scope_id']
            store = AnalysisJobStore(web_app.app.config['VIBEDASH_JOB_STORE_PATH'])
            payload = {
                'analysis_kind': 'period_comparison',
                'baseline': {'stored_filename': 'vibedash-' + '1' * 32 + '.csv', 'filename': 'b.csv', 'label': 'Before'},
                'current': {'stored_filename': 'vibedash-' + '2' * 32 + '.csv', 'filename': 'c.csv', 'label': 'After'},
            }
            for item in (payload['baseline'], payload['current']):
                Path(web_app.app.config['UPLOAD_FOLDER'], item['stored_filename']).write_bytes(self.csv_text())
            job = store.create(scope_id, payload)
            store.claim(job['id'])
            with sqlite3.connect(web_app.app.config['VIBEDASH_JOB_STORE_PATH']) as connection:
                connection.execute(
                    "UPDATE analysis_jobs SET started_at = ?, updated_at = ? WHERE id = ?",
                    ('2000-01-01T00:00:00+00:00', '2000-01-01T00:00:00+00:00', job['id']),
                )
            client.get('/vibedash/history')
            self.assertEqual(store.get(job['id'])['status'], 'failed')
        self.assertEqual(list(Path(web_app.app.config['UPLOAD_FOLDER']).iterdir()), [])

    def _worker_payload(self, root):
        upload = root / 'uploads'
        paths = []
        for token, name in (('1' * 32, 'baseline.csv'), ('2' * 32, 'current.csv')):
            path = upload / f'vibedash-{token}.csv'
            path.write_bytes(self.csv_text())
            paths.append(path)
        return {
            'analysis_kind': 'period_comparison',
            'baseline': {'stored_filename': paths[0].name, 'filename': 'baseline.csv', 'label': 'Before'},
            'current': {'stored_filename': paths[1].name, 'filename': 'current.csv', 'label': 'After'},
        }

    def test_worker_success_is_aggregate_only_and_cleans_inputs(self):
        root = Path(self.tmp.name)
        state = root / 'state'
        with patch.dict(os.environ, {'DATA_PRISM_STATE_DIR': str(state)}):
            with web_app.app.app_context():
                payload = self._worker_payload(root)
                result = routes._build_period_comparison_session(
                    payload, run_id='a' * 32, scope_id='a' * 32
                )
            session_data = load_session_data(
                result['session_id'], owner_id='a' * 32
            )
        self.assertFalse(list((root / 'uploads').glob('*.csv')))
        self.assertEqual(session_data['analysis_kind'], 'period_comparison')
        self.assertIn('numeric_metrics', session_data['report'])
        self.assertNotIn('RAW_SECRET_SENTINEL', json.dumps(session_data))
        self.assertEqual(result['manifest']['analysis_contract'], 'vibedash-period-comparison-v1')
        self.assertNotIn('stored_filename', json.dumps(result['manifest']))
        self.assertNotIn('display_filename', json.dumps(result['manifest']))

    def test_worker_failure_still_cleans_inputs(self):
        root = Path(self.tmp.name)
        payload = self._worker_payload(root)
        with patch.dict(os.environ, {'DATA_PRISM_STATE_DIR': str(root / 'state')}):
            with patch('vibedash.routes.build_period_comparison', side_effect=ValueError('engine details')):
                with web_app.app.app_context():
                    with self.assertRaises(ValueError):
                        routes._build_period_comparison_session(
                            payload, run_id='b' * 32, scope_id='b' * 32
                        )
        self.assertFalse(list((root / 'uploads').glob('*.csv')))

    def test_worker_missing_first_input_cleans_second_input(self):
        root = Path(self.tmp.name)
        payload = self._worker_payload(root)
        (root / 'uploads' / payload['baseline']['stored_filename']).unlink()
        with patch.dict(os.environ, {'DATA_PRISM_STATE_DIR': str(root / 'state')}):
            with web_app.app.app_context():
                with self.assertRaises(FileNotFoundError):
                    routes._build_period_comparison_session(
                        payload, run_id='c' * 32, scope_id='c' * 32
                    )
        self.assertFalse(list((root / 'uploads').glob('*.csv')))

    def _complete_comparison_job(self, client, manifest=None):
        with patch('vibedash.routes.analysis_job_dispatcher.submit', return_value=True):
            queued = self.post_data(client).get_json()
        store = AnalysisJobStore(web_app.app.config['VIBEDASH_JOB_STORE_PATH'])
        store.claim(queued['job_id'])
        store.complete(queued['job_id'], '12345678-1234-4234-8234-123456789012', manifest or {
            'manifest_version': 1,
            'analysis_contract': 'vibedash-period-comparison-v1',
            'inputs': {
                'baseline': {'filename': 'baseline.csv', 'label': 'Before', 'content_sha256': 'a' * 64, 'analyzed_rows': 30, 'column_count': 2},
                'current': {'filename': 'current.csv', 'label': 'After', 'content_sha256': 'b' * 64, 'analyzed_rows': 30, 'column_count': 2},
            },
            'comparison': {'status': 'stable', 'counts': {'tested_metrics': 1}},
        })
        return queued

    def test_result_manifest_history_are_scoped_and_comparison_aware(self):
        session_data = {
            'analysis_kind': 'period_comparison',
            'report': {'contract': 'period-comparison-v1', 'status': 'stable'},
            'baseline_filename': 'baseline.csv', 'current_filename': 'current.csv',
            'baseline_label': 'Before', 'current_label': 'After',
            'audit_manifest': {'analysis_contract': 'vibedash-period-comparison-v1'},
        }
        manifest = {
            'manifest_version': 1, 'analysis_contract': 'vibedash-period-comparison-v1',
            'inputs': {
                'baseline': {'filename': 'baseline.csv', 'label': 'Before', 'content_sha256': 'a' * 64, 'analyzed_rows': 30, 'column_count': 2},
                'current': {'filename': 'current.csv', 'label': 'After', 'content_sha256': 'b' * 64, 'analyzed_rows': 30, 'column_count': 2},
            }, 'comparison': {'status': 'stable', 'counts': {}},
        }
        with patch('vibedash.routes.load_session_data', return_value=session_data), patch('vibedash.routes.render_template', return_value='comparison') as render:
            with web_app.app.test_client() as owner:
                queued = self._complete_comparison_job(owner, manifest)
                result = owner.get(queued['status_url'] + '/result')
                owner_manifest = owner.get(queued['status_url'] + '/manifest')
                history = owner.get('/vibedash/history')
                self.assertEqual(result.status_code, 200)
                self.assertEqual(owner_manifest.status_code, 200)
                self.assertEqual(history.status_code, 200)
                self.assertEqual(render.call_args_list[0].args[0], 'vibedash_comparison.html')
                self.assertEqual(render.call_args_list[-1].args[0], 'vibedash_history.html')
                history_job = render.call_args.kwargs['jobs'][0]
                self.assertEqual(history_job['title'], 'Before vs After')
                self.assertEqual(history_job['prompt'], 'Period comparison')
            with web_app.app.test_client() as stranger:
                self.assertEqual(stranger.get(queued['status_url']).status_code, 404)
                self.assertEqual(stranger.get(queued['status_url'] + '/result').status_code, 404)
                self.assertEqual(stranger.get(queued['status_url'] + '/manifest').status_code, 404)


if __name__ == '__main__':
    unittest.main()
