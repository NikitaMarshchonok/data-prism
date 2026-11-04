"""Frozen issuance, paired arithmetic, release delays, revisions and local privacy."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import copy
from datetime import datetime, timezone
from io import StringIO
import json
import math
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import forecast_ledger as cli
import predict_forecast
from src.forecast_artifact import _digest, create_artifact, save_artifact
from src.forecast_ledger import DATABASE, ForecastLedger
from src.forecasting import ForecastTask, forecast_series
from tests.test_forecasting import frame
from tests.test_forecast_policy import POLICY


TASK = ForecastTask('demand', 'date', 7, 7, 2, '2020-06-10')
NOW = datetime(2020, 7, 1, 12, tzinfo=timezone.utc)


def resign(artifact):
    artifact['sha256'] = _digest({key: value for key, value in artifact.items() if key != 'sha256'})
    return artifact


class ForecastLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = frame()
        report = forecast_series(cls.data, TASK, decision_policy=POLICY)
        cls.artifact = create_artifact(cls.data, TASK, report, decision_policy=POLICY)
        # Controlled valid declarative parameters give independent arithmetic
        # oracles. This does not claim these constants were selected by training.
        cls.artifact['model'] = {'kind': 'lag_ridge', 'mean': [0., 0., 0.],
                                 'scale': [1., 1., 1.], 'coef': [0., 0., 0.], 'intercept': 8.}
        cls.artifact['reference'] = {'kind': 'historical_mean', 'value': 10.}
        resign(cls.artifact)

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'test.forecast-ledger'
        self.clock = patch('src.forecast_ledger._now', return_value=NOW)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.ledger = ForecastLedger.create(self.path)
        self.issued = self.ledger.issue(self.artifact)
        self.pid = self.issued['prediction_id']

    def accept(self, values, **kwargs):
        return self.ledger.accept_actuals(self.pid, values, as_of_date=kwargs.pop('as_of_date', '2020-06-30'), **kwargs)

    def all_actuals(self, value=10.):
        return [{'date': point['date'], 'actual': value} for point in self.issued['predictions']]

    def test_private_explicit_creation_never_replaces_existing_paths(self):
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((self.path / DATABASE).stat().st_mode), 0o600)
        for path in (self.path, self.root / 'missing'):
            with self.assertRaises((ValueError, FileExistsError)):
                if path == self.path:
                    ForecastLedger.create(path)
                else:
                    ForecastLedger(path)
        link = self.root / 'link'
        link.symlink_to(self.path, target_is_directory=True)
        with self.assertRaises(ValueError): ForecastLedger(link)
        (self.path / DATABASE).chmod(0o644)
        with self.assertRaises(ValueError): ForecastLedger(self.path)

    def test_uninitialized_wrong_version_and_missing_database_fail_without_recreation(self):
        with sqlite3.connect(self.path / DATABASE) as connection:
            connection.execute('PRAGMA user_version=2')
        with self.assertRaises(ValueError): ForecastLedger(self.path)
        (self.path / DATABASE).unlink()
        with self.assertRaises(sqlite3.Error): self.ledger.evaluate(self.pid)
        self.assertFalse((self.path / DATABASE).exists())

    def test_unknown_quality_has_no_zero_errors_or_success(self):
        result = self.ledger.evaluate(self.pid)
        self.assertEqual(result['quality_status'], 'unknown')
        self.assertEqual(result['matched_points'], 0)
        self.assertEqual(len(result['pending_dates']), 7)
        for key in ('metrics', 'baseline_metrics', 'mae_relative_improvement', 'business_evaluation'):
            self.assertIsNone(result[key])
        self.assertFalse(result['historical_issuance_verified'])
        self.assertEqual(result['production_readiness'], 'not_established')
        self.assertEqual(result['model_name'], 'lag_ridge')
        self.assertEqual(result['baseline_name'], 'historical_mean')
        self.assertEqual(result['target_unit'], 'number')
        self.assertEqual(result['reporting_delay_days'], 2)

    def test_no_fit_during_journal_issue_and_incomplete_history_does_not_register(self):
        with patch('src.forecast_artifact._fit_lag_model', side_effect=AssertionError('no fit')):
            self.ledger.issue(self.artifact)
            with self.assertRaises(ValueError):
                self.ledger.issue(self.artifact, history=self.data.iloc[:-1], forecast_origin='2020-06-10')
        with sqlite3.connect(self.path / DATABASE) as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM forecasts').fetchone()[0], 2)

    def test_sql_write_failure_rolls_back_forecast_and_all_points(self):
        with sqlite3.connect(self.path / DATABASE) as connection:
            connection.executescript('''CREATE TRIGGER reject_second_point BEFORE INSERT ON points
                WHEN NEW.lead=2 BEGIN SELECT RAISE(ABORT, 'simulated disk/write failure'); END;''')
        with self.assertRaises(sqlite3.Error): self.ledger.issue(self.artifact)
        with sqlite3.connect(self.path / DATABASE) as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM forecasts').fetchone()[0], 1)
            self.assertEqual(connection.execute('SELECT count(*) FROM points').fetchone()[0], 7)

    def test_contract_marker_cannot_adopt_unrelated_database(self):
        with sqlite3.connect(self.path / DATABASE) as connection:
            connection.execute("UPDATE ledger_metadata SET contract='unrelated'")
        with self.assertRaises(ValueError): ForecastLedger(self.path)

    def test_partial_same_date_pairing_and_per_lead_arithmetic(self):
        self.accept([{'date': '2020-06-13', 'actual': 14.}, {'date': '2020-06-11', 'actual': 10.}])
        result = self.ledger.evaluate(self.pid)
        self.assertEqual(result['quality_status'], 'partial')
        self.assertEqual(result['matched_points'], 2)
        self.assertEqual(result['metrics']['mae'], 4.)
        self.assertEqual(result['metrics']['rmse'], math.sqrt(20))
        self.assertEqual(result['metrics']['weighted_absolute_error'], 20.)
        self.assertEqual(result['baseline_metrics']['mae'], 2.)
        self.assertEqual(result['baseline_metrics']['rmse'], math.sqrt(8))
        self.assertEqual(result['baseline_metrics']['weighted_absolute_error'], 10.)
        self.assertEqual(result['mae_relative_improvement'], -1.)
        self.assertEqual([row['lead'] for row in result['matched']], [1, 3])
        self.assertIsNone(result['business_evaluation'])

    def test_complete_facts_retain_baseline_win_and_do_not_train_or_change_predictions(self):
        self.accept(self.all_actuals())
        with patch('src.forecast_artifact._fit_lag_model', side_effect=AssertionError('no fit')):
            result = self.ledger.evaluate(self.pid)
        self.assertEqual(result['quality_status'], 'measured_on_reported_outcomes')
        self.assertEqual(result['baseline_metrics']['mae'], 0)
        self.assertIsNone(result['mae_relative_improvement'])
        self.assertEqual(result['business_evaluation']['status'], 'criteria_not_met_on_reported_outcomes')
        self.assertEqual([row['prediction'] for row in result['matched']], [8.] * 7)
        self.assertEqual([row['baseline'] for row in result['matched']], [10.] * 7)

    def test_policy_gain_requires_full_horizon_and_keeps_failures(self):
        self.accept(self.all_actuals(8.))
        result = self.ledger.evaluate(self.pid)
        self.assertEqual(result['mae_relative_improvement'], 1.)
        self.assertEqual(result['business_evaluation']['status'], 'criteria_met_on_reported_outcomes')
        self.assertEqual(result['production_readiness'], 'not_established')
        self.assertEqual(result['financial_value'], 'not_assessed')

    def test_delay_and_future_asof_reject_without_writing(self):
        for asof in ('2020-06-12', '2020-07-02'):
            with self.assertRaises(ValueError):
                self.accept([{'date': '2020-06-11', 'actual': 10.}], as_of_date=asof)
        self.assertEqual(self.ledger.revisions(self.pid), [])
        self.accept([{'date': '2020-06-11', 'actual': 10.}], as_of_date='2020-06-13')
        self.assertEqual(self.ledger.evaluate(self.pid)['matched_points'], 1)

    def test_invalid_or_unmatched_batch_is_atomic(self):
        first = {'date': '2020-06-11', 'actual': 10.}
        invalid = [{}, {'date': '2020-06-18', 'actual': 10.}, first,
                   {'date': '2020-06-12', 'actual': True},
                   {'date': '2020-06-12', 'actual': '10'},
                   {'date': '2020-06-12', 'actual': float('nan')},
                   {'date': '2020-06-12', 'actual': float('inf')},
                   {'date': '2020-06-12', 'actual': 10 ** 500},
                   {'date': '2020-06-12', 'actual': 2 ** 53},
                   {'date': '2020-06-12', 'actual': 10, 'extra': 1},
                   {'date': '2020-6-12', 'actual': 10}]
        for second in invalid:
            with self.subTest(second=repr(second)):
                with self.assertRaises(ValueError): self.accept([first, second])
                self.assertEqual(self.ledger.revisions(self.pid), [])
        for values in (None, [], {}, [first] * 91):
            with self.assertRaises(ValueError): self.accept(values)

    def test_unrepresentable_scoring_is_not_reported_as_success(self):
        self.accept([{'date': '2020-06-11', 'actual': 1e308}])
        with self.assertRaises((ValueError, OverflowError)):
            self.ledger.evaluate(self.pid)
        output = StringIO()
        with redirect_stdout(output):
            status = cli.main(['--ledger', str(self.path), 'report', '--prediction-id', self.pid])
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(output.getvalue())['status'], 'error')

    def test_duplicate_is_idempotent_and_correction_preserves_old_revision(self):
        values = [{'date': '2020-06-11', 'actual': 10.}]
        self.assertEqual(self.accept(values)['changed'], 1)
        self.assertEqual(self.accept(values)['unchanged'], 1)
        with self.assertRaises(ValueError):
            self.accept([{'date': '2020-06-11', 'actual': 12.}])
        self.assertEqual(len(self.ledger.revisions(self.pid)), 1)
        self.accept([{'date': '2020-06-11', 'actual': 12.}], correction_reason='Source corrected duplicate orders')
        revisions = self.ledger.revisions(self.pid)
        self.assertEqual([row['actual'] for row in revisions], [10., 12.])
        self.assertEqual([row['revision'] for row in revisions], [1, 2])
        self.assertEqual(revisions[1]['reason'], 'Source corrected duplicate orders')
        self.assertEqual(self.ledger.evaluate(self.pid)['metrics']['mae'], 4.)

    def test_invalid_correction_and_backdated_updates_cannot_write(self):
        self.accept([{'date': '2020-06-11', 'actual': 10.}])
        for reason in ('', ' ', 'x' * 501, True, '\ud800'):
            with self.assertRaises((ValueError, UnicodeError)):
                self.accept([{'date': '2020-06-11', 'actual': 12.}], correction_reason=reason)
        with self.assertRaises(ValueError):
            self.accept([{'date': '2020-06-11', 'actual': 12.}],
                        as_of_date='2020-06-29', correction_reason='Backdated')
        self.assertEqual(len(self.ledger.revisions(self.pid)), 1)

    def test_multirow_conflict_rolls_back_new_facts(self):
        self.accept([{'date': '2020-06-12', 'actual': 10.}])
        with self.assertRaises(ValueError):
            self.accept([{'date': '2020-06-11', 'actual': 11.}, {'date': '2020-06-12', 'actual': 12.}])
        self.assertEqual([row['date'] for row in self.ledger.revisions(self.pid)], ['2020-06-12'])

    def test_facts_and_corrections_are_shared_across_overlapping_origins_of_same_model(self):
        more = pd.concat([self.data, frame([212., 213.]).assign(date=pd.date_range('2020-06-09', periods=2))], ignore_index=True)
        other = self.ledger.issue(self.artifact, history=more, forecast_origin='2020-06-12')
        self.accept([{'date': '2020-06-13', 'actual': 14.}])
        result = self.ledger.evaluate(other['prediction_id'])
        self.assertEqual(result['matched'][0]['lead'], 1)
        self.assertEqual(result['matched'][0]['actual'], 14.)
        self.ledger.accept_actuals(other['prediction_id'], [{'date': '2020-06-13', 'actual': 15.}],
                                   as_of_date='2020-06-30', correction_reason='Corrected source')
        self.assertEqual(self.ledger.evaluate(self.pid)['matched'][0]['actual'], 15.)

    def test_same_column_names_do_not_match_other_model_or_other_ledger(self):
        other_model = copy.deepcopy(self.artifact)
        other_model['model_id'] = 'a' * 32
        resign(other_model)
        other = self.ledger.issue(other_model)
        self.accept(self.all_actuals())
        self.assertEqual(self.ledger.evaluate(other['prediction_id'])['quality_status'], 'unknown')
        independent = ForecastLedger.create(self.root / 'independent')
        for pid in (self.pid, '0' * 32, "' OR 1=1 --"):
            with self.assertRaises(ValueError): independent.evaluate(pid)

    def test_unknown_prediction_cannot_write_or_delete_existing_facts(self):
        self.accept(self.all_actuals())
        for pid in ('0' * 32, '', None, "' OR 1=1 --"):
            with self.assertRaises(ValueError):
                self.ledger.accept_actuals(pid, self.all_actuals(), as_of_date='2020-06-30')
            with self.assertRaises(ValueError): self.ledger.delete(pid)
        self.assertEqual(self.ledger.evaluate(self.pid)['matched_points'], 7)

    def test_init_and_confirmed_delete_cli_and_readonly_report_do_not_add_rows(self):
        path = self.root / 'cli.forecast-ledger'
        with redirect_stdout(StringIO()):
            self.assertEqual(cli.main(['--ledger', str(path), 'init']), 0)
            self.assertEqual(cli.main(['--ledger', str(path), 'init']), 1)
            self.assertEqual(cli.main(['--ledger', str(self.path), 'report', '--prediction-id', self.pid]), 0)
        with sqlite3.connect(self.path / DATABASE) as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM forecasts').fetchone()[0], 1)
        with redirect_stdout(StringIO()):
            self.assertEqual(cli.main(['--ledger', str(self.path), 'delete', '--prediction-id', self.pid, '--confirm-delete']), 0)
        with self.assertRaises(ValueError): self.ledger.evaluate(self.pid)

    def test_capacity_limits_fail_without_silent_purge_or_partial_writes(self):
        with patch('src.forecast_ledger.MAX_FORECASTS', 1):
            with self.assertRaises(ValueError): self.ledger.issue(self.artifact)
        with patch('src.forecast_ledger.MAX_OUTCOME_REVISIONS', 1):
            with self.assertRaises(ValueError): self.accept(self.all_actuals())
        self.assertEqual(self.ledger.revisions(self.pid), [])
        self.assertEqual(self.ledger.evaluate(self.pid)['quality_status'], 'unknown')

    def test_deletion_preserves_shared_facts_until_last_dependent_forecast(self):
        other = self.ledger.issue(self.artifact)
        self.accept(self.all_actuals())
        self.assertEqual(self.ledger.delete(self.pid)['outcome_revisions_deleted'], 0)
        with self.assertRaises(ValueError): self.ledger.evaluate(self.pid)
        self.assertEqual(self.ledger.evaluate(other['prediction_id'])['matched_points'], 7)
        self.assertEqual(self.ledger.delete(other['prediction_id'])['outcome_revisions_deleted'], 7)
        with sqlite3.connect(self.path / DATABASE) as connection:
            for name in ('forecasts', 'points', 'outcomes'):
                self.assertEqual(connection.execute(f'SELECT count(*) FROM {name}').fetchone()[0], 0)

    def test_concurrent_duplicate_imports_are_one_revision_not_double_counted(self):
        def import_fact(_):
            return self.accept([{'date': '2020-06-11', 'actual': 12.}])
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(import_fact, range(2)))
        self.assertEqual(sum(row['changed'] for row in results), 1)
        self.assertEqual(sum(row['unchanged'] for row in results), 1)
        self.assertEqual(len(self.ledger.revisions(self.pid)), 1)

    def test_count_domain_failure_is_retained_and_blocks_criteria_success(self):
        artifact = copy.deepcopy(self.artifact)
        artifact['target_unit'] = 'count'
        artifact['model']['intercept'] = -1.
        resign(artifact)
        issued = self.ledger.issue(artifact)
        self.assertEqual(issued['status'], 'domain_constraint_failed')
        self.ledger.accept_actuals(issued['prediction_id'],
                                   [{'date': row['date'], 'actual': 0.} for row in issued['predictions']],
                                   as_of_date='2020-06-30')
        result = self.ledger.evaluate(issued['prediction_id'])
        self.assertEqual(result['metrics']['mae'], 1.)
        self.assertEqual(result['baseline_metrics']['mae'], 10.)
        self.assertEqual(result['business_evaluation']['status'], 'criteria_not_met_on_reported_outcomes')
        with self.assertRaises(ValueError):
            self.ledger.accept_actuals(issued['prediction_id'], [{'date': '2020-06-11', 'actual': -1.}],
                                       as_of_date='2020-06-30', correction_reason='Still invalid')

    def test_zero_error_tie_never_invents_relative_gain(self):
        artifact = copy.deepcopy(self.artifact)
        artifact['model'] = artifact['reference']
        resign(artifact)
        issued = self.ledger.issue(artifact)
        self.ledger.accept_actuals(issued['prediction_id'], self.all_actuals(), as_of_date='2020-06-30')
        report = self.ledger.evaluate(issued['prediction_id'])
        self.assertEqual(report['metrics']['mae'], 0)
        self.assertIsNone(report['mae_relative_improvement'])
        self.assertFalse(report['business_evaluation']['checks']['strictly_lower_weighted_error'])
        json.dumps(report, allow_nan=False)

    def test_no_policy_does_not_invent_business_criteria(self):
        artifact = copy.deepcopy(self.artifact)
        artifact['decision_policy'] = None
        for key in ('metrics', 'baseline_metrics'):
            artifact['evaluation']['holdout'][key].pop('weighted_absolute_error')
        resign(artifact)
        issued = self.ledger.issue(artifact)
        self.ledger.accept_actuals(issued['prediction_id'], self.all_actuals(), as_of_date='2020-06-30')
        report = self.ledger.evaluate(issued['prediction_id'])
        self.assertIsNone(report['business_evaluation'])
        self.assertEqual(set(report['metrics']), {'mae', 'rmse'})

    def test_future_issue_date_is_not_registered(self):
        with patch('src.forecast_ledger._now', return_value=datetime(2020, 6, 9, tzinfo=timezone.utc)):
            with self.assertRaises(ValueError): self.ledger.issue(self.artifact)
        with sqlite3.connect(self.path / DATABASE) as connection:
            self.assertEqual(connection.execute('SELECT count(*) FROM forecasts').fetchone()[0], 1)

    def test_tampered_forecast_or_point_is_not_scored(self):
        with sqlite3.connect(self.path / DATABASE) as connection:
            connection.execute('UPDATE points SET prediction=99 WHERE prediction_id=?', (self.pid,))
        with self.assertRaises(ValueError): self.ledger.evaluate(self.pid)
        with sqlite3.connect(self.path / DATABASE) as connection:
            connection.execute('UPDATE forecasts SET payload=? WHERE id=?', ('{}', self.pid))
        with self.assertRaises(ValueError): self.ledger.revisions(self.pid)

    def test_cli_frozen_issue_actual_import_and_report_survive_process_restart(self):
        model = self.root / 'private.forecast-model.json'
        save_artifact(self.artifact, model)
        command = [sys.executable, 'predict_forecast.py', '--model', str(model), '--ledger', str(self.path)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        issued = json.loads(completed.stdout)
        path = self.root / 'actuals.json'
        path.write_text(json.dumps({'as_of_date': '2020-06-30', 'actuals': self.all_actuals(8.)}))
        base = [sys.executable, 'forecast_ledger.py', '--ledger', str(self.path)]
        subprocess.run(base + ['actuals', '--prediction-id', issued['prediction_id'], '--input', str(path)],
                       capture_output=True, text=True, timeout=30, check=True)
        report = subprocess.run(base + ['report', '--prediction-id', issued['prediction_id']],
                                capture_output=True, text=True, timeout=30, check=True)
        self.assertEqual(json.loads(report.stdout)['metrics']['mae'], 0.)
        self.assertEqual(ForecastLedger(self.path).evaluate(issued['prediction_id'])['matched_points'], 7)

    def test_cli_invalid_json_and_unconfirmed_deletion_fail_without_leaking_values(self):
        path = self.root / 'actuals.json'
        for raw in ('{"as_of_date":"2020-06-30","as_of_date":"2020-06-30","actuals":[]}',
                    '{"as_of_date":"2020-06-30","actuals":[{"date":"2020-06-11","actual":NaN}]}',
                    ' ' * 65537, '[]', '{"secret":"confidential-customer-value"}', '\ud800'):
            path.write_bytes(raw.encode('utf-8', errors='surrogatepass'))
            output = StringIO()
            with redirect_stdout(output):
                code = cli.main(['--ledger', str(self.path), 'actuals', '--prediction-id', self.pid, '--input', str(path)])
            self.assertEqual(code, 1)
            self.assertNotIn('confidential-customer-value', output.getvalue())
        with redirect_stdout(StringIO()):
            self.assertEqual(cli.main(['--ledger', str(self.path), 'delete', '--prediction-id', self.pid]), 1)
        self.assertEqual(self.ledger.evaluate(self.pid)['quality_status'], 'unknown')

    def test_ordinary_prediction_remains_stateless_and_missing_ledger_does_not_get_created(self):
        model = self.root / 'private.forecast-model.json'
        save_artifact(self.artifact, model)
        with redirect_stdout(StringIO()), patch('predict_forecast.ForecastLedger') as ledger:
            self.assertEqual(predict_forecast.main(['--model', str(model)]), 0)
            ledger.assert_not_called()
        missing = self.root / 'missing'
        with redirect_stdout(StringIO()):
            self.assertEqual(predict_forecast.main(['--model', str(model), '--ledger', str(missing)]), 1)
        self.assertFalse(missing.exists())


if __name__ == '__main__':
    unittest.main()
