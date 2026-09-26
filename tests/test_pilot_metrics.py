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
    build_pilot_report, build_scope_value_feedback, feedback_available,
    forget_scope, mark_stage, record_feedback, scope_token,
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

    def feedback(
        self,
        job_id,
        usefulness='useful',
        blocker='none',
        perceived_time_saved='15_to_30_minutes',
        next_cycle_intent='yes',
    ):
        return record_feedback(
            self.path,
            job_id,
            usefulness,
            blocker,
            perceived_time_saved,
            next_cycle_intent,
        )

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
        self.assertTrue(self.feedback(job_id))
        self.assertTrue(self.feedback(
            job_id,
            'partly_useful',
            'missing_context',
            'under_15_minutes',
            'maybe',
        ))
        report = self.report()
        self.assertEqual(report['feedback_responses'], 1)
        self.assertEqual(report['feedback_eligible_analyses'], 1)
        self.assertEqual(report['feedback_response_rate_among_completed'], 1.0)
        self.assertEqual(report['value_feedback_responses'], 1)
        self.assertEqual(report['value_feedback_response_rate_among_completed'], 1.0)
        self.assertEqual(report['legacy_feedback_responses_without_value_signals'], 0)
        self.assertEqual(report['usefulness']['useful'], 0)
        self.assertEqual(report['usefulness']['partly_useful'], 1)
        self.assertEqual(report['perceived_time_saved']['under_15_minutes'], 1)
        self.assertEqual(report['perceived_time_saved']['15_to_30_minutes'], 0)
        self.assertEqual(report['next_cycle_intent']['maybe'], 1)
        self.assertEqual(report['next_cycle_intent']['yes'], 0)
        with self.assertRaises(ValueError):
            self.feedback(job_id, usefulness='PRIVATE-FREE-TEXT')
        with self.assertRaises(ValueError):
            self.feedback(job_id, perceived_time_saved='about an hour')
        with self.assertRaises(ValueError):
            self.feedback(job_id, next_cycle_intent='contact me')
        self.assertFalse(self.feedback(uuid.uuid4().hex))

    def test_value_feedback_is_aggregated_by_source_without_implying_people(self):
        upload = self.job()
        demo = self.job(demo=True)
        self.assertTrue(self.feedback(upload, perceived_time_saved='over_60_minutes'))
        self.assertTrue(self.feedback(
            demo,
            usefulness='partly_useful',
            blocker='missing_feature',
            perceived_time_saved='none',
            next_cycle_intent='no',
        ))
        upload_report = self.report('upload')
        demo_report = self.report('demo')
        self.assertEqual(upload_report['perceived_time_saved']['over_60_minutes'], 1)
        self.assertEqual(upload_report['next_cycle_intent']['yes'], 1)
        self.assertEqual(upload_report['next_cycle_intent']['no'], 0)
        self.assertEqual(demo_report['perceived_time_saved']['none'], 1)
        self.assertEqual(demo_report['next_cycle_intent']['no'], 1)
        self.assertEqual(demo_report['next_cycle_intent']['yes'], 0)

    def test_scope_value_feedback_excludes_other_scopes_and_identifiers(self):
        own_job = self.job()
        self.assertTrue(self.feedback(own_job))
        other_scope = uuid.uuid4().hex
        other_token = scope_token(other_scope, 'test-secret')
        other = self.store.create(
            other_scope,
            {'prompt': 'OTHER-PRIVATE-PROMPT'},
            pilot_scope_token=other_token,
        )
        self.store.claim(other['id'])
        self.store.complete(other['id'], str(uuid.uuid4()))
        self.assertTrue(record_feedback(
            self.path,
            other['id'],
            'not_useful',
            'missing_feature',
            'over_60_minutes',
            'no',
        ))

        summary = build_scope_value_feedback(self.path, self.token)

        self.assertEqual(summary['completed_opted_in_analyses'], 1)
        self.assertEqual(summary['feedback_responses'], 1)
        self.assertEqual(summary['value_feedback_responses'], 1)
        self.assertEqual(summary['value_feedback_response_rate_among_completed'], 1.0)
        self.assertEqual(summary['perceived_time_saved']['15_to_30_minutes'], 1)
        self.assertEqual(summary['perceived_time_saved']['over_60_minutes'], 0)
        self.assertEqual(summary['next_cycle_intent']['yes'], 1)
        self.assertEqual(summary['next_cycle_intent']['no'], 0)
        serialized = json.dumps(summary)
        for private in (
            self.scope,
            self.token,
            own_job,
            other_scope,
            other_token,
            other['id'],
            'OTHER-PRIVATE-PROMPT',
        ):
            self.assertNotIn(private, serialized)

    def test_scope_value_feedback_preserves_missing_value_signals(self):
        job_id = self.job()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "UPDATE pilot_analyses SET usefulness = 'useful', blocker = 'none' "
                "WHERE job_id = ?",
                (job_id,),
            )

        summary = build_scope_value_feedback(self.path, self.token)

        self.assertEqual(summary['completed_opted_in_analyses'], 1)
        self.assertEqual(summary['feedback_responses'], 1)
        self.assertEqual(summary['value_feedback_responses'], 0)
        self.assertEqual(
            summary['legacy_feedback_responses_without_value_signals'],
            1,
        )
        self.assertEqual(
            summary['value_feedback_response_rate_among_completed'],
            0.0,
        )
        self.assertEqual(sum(summary['perceived_time_saved'].values()), 0)
        self.assertEqual(sum(summary['next_cycle_intent'].values()), 0)

    def test_withdrawal_is_scoped_and_does_not_recreate_records(self):
        job_id = self.job()
        other_scope = uuid.uuid4().hex
        other = self.store.create(other_scope, {}, pilot_scope_token=scope_token(other_scope, 'test-secret'))
        self.assertEqual(forget_scope(self.path, self.token), 1)
        self.case(job_id)
        self.assertFalse(self.feedback(job_id))
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
        with self.assertRaises(sqlite3.OperationalError):
            build_scope_value_feedback(missing, self.token)
        self.assertFalse(missing.exists())
        old = Path(self.directory.name) / 'old.sqlite3'
        with closing(sqlite3.connect(old)) as connection:
            connection.execute('CREATE TABLE legacy (id TEXT)')
        before = old.read_bytes()
        self.assertFalse(build_pilot_report(old)['collection_installed'])
        self.assertFalse(
            build_scope_value_feedback(old, self.token)['collection_installed']
        )
        self.assertEqual(old.read_bytes(), before)
        for days in (0, 31, True, '7'):
            with self.assertRaises(ValueError):
                build_pilot_report(self.path, days=days)
            with self.assertRaises(ValueError):
                build_scope_value_feedback(self.path, self.token, days=days)
        with self.assertRaises(ValueError):
            build_scope_value_feedback(self.path, 'invalid')

    def test_schema_migration_retains_legacy_feedback_and_reports_null_signals(self):
        legacy = Path(self.directory.name) / 'legacy-metrics.sqlite3'
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        legacy_job = uuid.uuid4().hex
        with closing(sqlite3.connect(legacy)) as connection, connection:
            connection.execute("""
                CREATE TABLE pilot_analyses (
                    job_id TEXT PRIMARY KEY,
                    scope_token TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    failed_at TEXT,
                    decision_at TEXT,
                    outcome_at TEXT,
                    usefulness TEXT,
                    blocker TEXT
                )
            """)
            connection.execute(
                """
                INSERT INTO pilot_analyses (
                    job_id, scope_token, source, created_at, completed_at,
                    usefulness, blocker
                ) VALUES (?, ?, 'upload', ?, ?, 'useful', 'none')
                """,
                (legacy_job, self.token, now, now),
            )

        legacy_bytes = legacy.read_bytes()
        # The read-only report remains compatible before an application store
        # has had a chance to migrate the database.
        before = build_pilot_report(legacy)['cohorts']['upload']
        scoped_before = build_scope_value_feedback(legacy, self.token)
        self.assertEqual(legacy.read_bytes(), legacy_bytes)
        self.assertEqual(before['feedback_responses'], 1)
        self.assertEqual(before['value_feedback_responses'], 0)
        self.assertEqual(before['legacy_feedback_responses_without_value_signals'], 1)
        self.assertEqual(sum(before['perceived_time_saved'].values()), 0)
        self.assertEqual(sum(before['next_cycle_intent'].values()), 0)
        self.assertEqual(scoped_before['feedback_responses'], 1)
        self.assertEqual(scoped_before['value_feedback_responses'], 0)
        self.assertEqual(
            scoped_before['legacy_feedback_responses_without_value_signals'],
            1,
        )

        AnalysisJobStore(legacy)
        AnalysisJobStore(legacy)  # The additive migration is idempotent.
        with closing(sqlite3.connect(legacy)) as connection:
            columns = {
                row[1] for row in connection.execute(
                    'PRAGMA table_info(pilot_analyses)'
                )
            }
            retained = connection.execute(
                'SELECT usefulness, blocker FROM pilot_analyses WHERE job_id = ?',
                (legacy_job,),
            ).fetchone()
        self.assertIn('perceived_time_saved', columns)
        self.assertIn('next_cycle_intent', columns)
        self.assertEqual(retained, ('useful', 'none'))
        self.assertTrue(record_feedback(
            legacy,
            legacy_job,
            'useful',
            'none',
            '30_to_60_minutes',
            'yes',
        ))
        with closing(sqlite3.connect(legacy)) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE pilot_analyses SET next_cycle_intent = 'free text' "
                    "WHERE job_id = ?",
                    (legacy_job,),
                )

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
