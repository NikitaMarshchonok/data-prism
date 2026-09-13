import json
import sqlite3
import subprocess
import sys
import unittest
import uuid
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vibedash.analysis_jobs import AnalysisJobStore
from vibedash.decision_cases import DecisionCaseStore
from vibedash.pilot_metrics import (
    build_pilot_report, feedback_available, forget_scope, mark_stage,
    record_feedback, scope_token,
)


class PilotMetricsTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'jobs.sqlite3'
        self.store = AnalysisJobStore(self.path)
        self.scope = uuid.uuid4().hex
        self.token = scope_token(self.scope, 'test-secret')

    def job(self, *, consent=True, demo=False, scope=None):
        job = self.store.create(
            scope or self.scope,
            {'prompt': 'PRIVATE-PROMPT', 'filename': 'PRIVATE-FILE.csv',
             'demo_dataset': 'saas_growth' if demo else ''},
            pilot_scope_token=self.token if consent else None,
        )
        self.store.claim(job['id'])
        self.store.complete(job['id'], str(uuid.uuid4()))
        return job['id']

    def report(self, source='upload'):
        return build_pilot_report(self.path)['cohorts'][source]

    def row(self, job_id):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute('SELECT * FROM pilot_analyses WHERE job_id = ?', (job_id,)).fetchone()
            return dict(row) if row else None

    def case(self, job_id, priority=1):
        return DecisionCaseStore(self.path).create(
            self.scope, job_id, priority=priority, owner='PRIVATE-OWNER',
            decision='PRIVATE-DECISION', success_metric='Activation rate',
            target_outcome='Five percentage point increase', review_date='2026-12-01',
            evidence_snapshot={'contract': 'test', 'priority': {'number': priority}},
        )

    def test_no_measurement_without_explicit_consent(self):
        job_id = self.job(consent=False)
        self.case(job_id)
        self.assertIsNone(self.row(job_id))
        self.assertFalse(feedback_available(self.path, job_id))
        self.assertEqual(self.report()['accepted_analyses'], 0)
        self.assertIsNone(self.report()['completion_rate'])

    def test_stages_are_idempotent_and_demo_is_separate(self):
        job_id = self.job()
        self.job(demo=True)
        before = self.row(job_id)
        self.assertFalse(self.store.claim(job_id))
        self.assertFalse(self.store.complete(job_id, str(uuid.uuid4())))
        self.assertFalse(self.store.fail(job_id))
        self.store.get(job_id)
        self.assertEqual(self.row(job_id), before)
        self.assertEqual(self.report()['completed_analyses'], 1)
        self.assertEqual(self.report('demo')['completed_analyses'], 1)
        self.assertEqual(self.report()['failed_analyses'], 0)

    def test_decisions_and_observed_outcomes_count_once_per_analysis(self):
        job_id = self.job()
        first = self.case(job_id)
        self.case(job_id, priority=2)
        cases = DecisionCaseStore(self.path)
        cases.update_outcome(first['id'], self.scope, status='cancelled', actual_outcome='Not attempted')
        self.assertEqual(self.report()['analyses_with_recorded_outcome'], 0)
        cases.update_outcome(first['id'], self.scope, status='validated', actual_outcome='Self-reported test result')
        first_timestamp = self.row(job_id)['outcome_at']
        cases.update_outcome(first['id'], self.scope, status='invalidated', actual_outcome='Corrected result')
        self.assertEqual(self.row(job_id)['outcome_at'], first_timestamp)
        report = self.report()
        self.assertEqual(report['analyses_with_decision'], 1)
        self.assertEqual(report['analyses_with_recorded_outcome'], 1)
        self.assertIsNotNone(report['median_seconds_to_first_decision'])

    def test_feedback_updates_instead_of_accumulating(self):
        job_id = self.job()
        self.assertTrue(record_feedback(self.path, job_id, 'useful', 'none'))
        self.assertTrue(record_feedback(self.path, job_id, 'partly_useful', 'missing_context'))
        report = self.report()
        self.assertEqual(report['feedback_responses'], 1)
        self.assertEqual(report['usefulness']['useful'], 0)
        self.assertEqual(report['usefulness']['partly_useful'], 1)
        with self.assertRaises(ValueError):
            record_feedback(self.path, job_id, 'PRIVATE-FREE-TEXT', 'none')
        self.assertFalse(record_feedback(self.path, uuid.uuid4().hex, 'useful', 'none'))

    def test_withdrawal_is_scoped_and_does_not_recreate_records(self):
        job_id = self.job()
        other_scope = uuid.uuid4().hex
        other = self.store.create(other_scope, {}, pilot_scope_token=scope_token(other_scope, 'test-secret'))
        self.assertEqual(forget_scope(self.path, self.token), 1)
        self.case(job_id)
        self.assertFalse(record_feedback(self.path, job_id, 'useful', 'none'))
        self.assertIsNone(self.row(job_id))
        self.assertIsNotNone(self.row(other['id']))
        self.assertIsNotNone(self.store.get(job_id))
        self.assertEqual(len(DecisionCaseStore(self.path).list_for_scope(self.scope)), 1)

    def test_failed_and_stale_jobs_are_recorded_once(self):
        failed = self.store.create(self.scope, {}, pilot_scope_token=self.token)
        self.store.fail(failed['id'])
        stale = self.store.create(self.scope, {}, pilot_scope_token=self.token)
        self.store.claim(stale['id'])
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute('UPDATE analysis_jobs SET started_at = ? WHERE id = ?', ('2000-01-01T00:00:00.000+00:00', stale['id']))
        self.assertEqual(self.store.fail_stale_running(600), 1)
        self.assertEqual(self.store.fail_stale_running(600), 0)
        report = self.report()
        self.assertEqual(report['failed_analyses'], 2)
        self.assertEqual(report['started_analyses'], 1)
        self.assertEqual(report['completed_analyses'], 0)

    def test_measurement_outlives_jobs_but_expires_after_thirty_days(self):
        job_id = self.job()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute('UPDATE analysis_jobs SET updated_at = ?', ('2000-01-01T00:00:00.000+00:00',))
        self.store.purge_terminal(24)
        self.assertIsNone(self.store.get(job_id))
        self.assertEqual(self.report()['completed_analyses'], 1)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute('UPDATE pilot_analyses SET created_at = ?', ('2000-01-01T00:00:00.000+00:00',))
        self.assertEqual(self.report()['completed_analyses'], 0)
        self.store.purge_terminal(24)
        self.assertIsNone(self.row(job_id))

    def test_measurement_limit_does_not_block_analysis(self):
        with patch('vibedash.pilot_metrics.MAX_RECORDS', 1):
            first = self.job()
            second = self.job()
            self.assertTrue(build_pilot_report(self.path)['record_limit_reached'])
        self.assertIsNotNone(self.row(first))
        self.assertIsNone(self.row(second))
        self.assertEqual(self.store.get(second)['status'], 'completed')

    def test_browser_repeats_are_not_people_and_cohort_window_is_respected(self):
        first = self.job()
        self.job()
        yesterday = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat(timespec='milliseconds')
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute('UPDATE pilot_analyses SET created_at = ? WHERE job_id = ?', (yesterday, first))
        report = self.report()
        self.assertEqual(report['browser_scopes_with_completed_analysis'], 1)
        self.assertEqual(report['browser_scopes_with_repeat_completed_analysis'], 1)
        self.assertEqual(report['browser_scopes_active_on_multiple_utc_dates'], 1)
        self.assertEqual(build_pilot_report(self.path, days=1)['cohorts']['upload']['accepted_analyses'], 1)

    def test_private_text_and_identifiers_never_appear_in_report(self):
        job_id = self.job()
        self.case(job_id)
        contents = json.dumps(self.row(job_id))
        self.assertNotIn('PRIVATE-', contents)
        report = json.dumps(build_pilot_report(self.path))
        for value in (self.scope, self.token, job_id, 'PRIVATE-'):
            self.assertNotIn(value, report)

    def test_report_does_not_create_or_migrate_a_database(self):
        missing = Path(self.directory.name) / 'missing.sqlite3'
        with self.assertRaises(sqlite3.OperationalError):
            build_pilot_report(missing)
        self.assertFalse(missing.exists())
        old = Path(self.directory.name) / 'old.sqlite3'
        with closing(sqlite3.connect(old)) as connection:
            connection.execute('CREATE TABLE legacy (id TEXT)')
        before = old.read_bytes()
        self.assertFalse(build_pilot_report(old)['collection_installed'])
        self.assertEqual(old.read_bytes(), before)
        for days in (0, 31, True, '7'):
            with self.assertRaises(ValueError):
                build_pilot_report(self.path, days=days)

    def test_identifiers_and_stage_allowlist_are_validated(self):
        self.assertEqual(self.token, scope_token(self.scope, 'test-secret'))
        self.assertNotEqual(self.token, scope_token(self.scope, 'other-secret'))
        with self.assertRaises(ValueError):
            scope_token('not-a-scope', 'secret')
        with self.assertRaises(ValueError):
            self.store.create(self.scope, {}, pilot_scope_token='invalid')
        self.assertEqual(self.store.list_for_scope(self.scope), [])
        with closing(sqlite3.connect(self.path)) as connection:
            with self.assertRaises(ValueError):
                mark_stage(connection, uuid.uuid4().hex, 'arbitrary_column', 'now')

    def test_cli_emits_aggregate_json(self):
        self.job()
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(root / 'pilot_report.py'), '--database', str(self.path), '--days', '7'],
            cwd=root, text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)['cohorts']['upload']['completed_analyses'], 1)


if __name__ == '__main__':
    unittest.main()
