"""Corruption tests for historical evidence at the delivery boundary."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from evaluate_forecast import main
from src.forecasting import ForecastTask, LEGACY_CANDIDATE_SUITE, forecast_series
from src.forecast_backtest import historical_forecast_backtest, backtest_audit_view
from src.forecast_backtest_result import CONTRACT, validate_historical_backtest
from tests.test_forecast_tasks import daily_frame, confirmed, FORM
from tests.test_forecast_policy import POLICY
from vibedash.forecast_tasks import ForecastTaskError, forecast_task_from_form, run_confirmed_forecast


TASK = ForecastTask('cnt', 'dteday', 7)


class HistoricalEvidenceIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = daily_frame()
        cls.report = historical_forecast_backtest(cls.frame, TASK)
        cls.weighted = historical_forecast_backtest(cls.frame, TASK, decision_policy=POLICY)

    def check(self, report, **kwargs):
        return validate_historical_backtest(report, self.frame, TASK, **kwargs)

    def test_full_evidence_is_checked_without_fitting_mutation_or_acceptance_claim(self):
        before = copy.deepcopy(self.report)
        with patch('src.forecasting._forecast') as fit, patch('src.forecast_backtest.forecast_series') as run:
            result = self.check(self.report)
        fit.assert_not_called()
        run.assert_not_called()
        self.assertEqual(self.report, before)
        self.assertEqual(result['contract'], CONTRACT)
        self.assertEqual(result['checked_windows'], 3)
        self.assertEqual(result['checked_scored_rows'], 21)
        self.assertEqual(result['production_readiness'], 'not_established')
        json.dumps(result, allow_nan=False)

    def test_false_scores_facts_dates_references_and_summary_are_rejected(self):
        mutations = [
            lambda r: r['windows'][0]['metrics'].update(mae=0.),
            lambda r: r['windows'][0]['baseline_metrics'].update(rmse=0.),
            lambda r: r['windows'][0]['evidence'][0].update(actual=12345.),
            lambda r: r['windows'][0]['evidence'][0].update(prediction=12345.),
            lambda r: r['windows'][0]['evidence'][0].update(baseline=12345.),
            lambda r: r['windows'][0]['evidence'].pop(),
            lambda r: r['windows'][0]['evidence'].reverse(),
            lambda r: r['windows'][0]['evidence'][0].update(date='2000-01-01'),
            lambda r: r['windows'][0].update(available_through=r['windows'][0]['end']),
            lambda r: r['windows'][0].update(forecast_origin=r['windows'][0]['start']),
            lambda r: r['windows'][0].update(history_rows=True),
            lambda r: r['windows'][0].update(window_index=True),
            lambda r: r['windows'][0].update(baseline_name='lag_ridge'),
            lambda r: r['windows'][0].update(model_name='unreviewed_model'),
            lambda r: r['windows'][0]['references'].pop('last_value'),
            lambda r: r['windows'][0]['references']['last_value']['forecast'].__setitem__(0, 12345.),
            lambda r: r['windows'][0]['references']['historical_mean']['metrics'].update(mae=0.),
            lambda r: r['windows'][0]['comparison'].update(relative_improvement=1.),
            lambda r: r['windows'][0]['comparison'].update(loss_difference=12345.),
            lambda r: r['windows'][0]['comparison'].update(outcome='unknown'),
            lambda r: r['summary'].update(scored_windows=True),
            lambda r: r['summary'].update(better=12345),
            lambda r: r['windows'].pop(),
            lambda r: r['windows'].reverse(),
            lambda r: r['protocol'].update(method_selection='outer_horizon'),
            lambda r: r['protocol'].update(reporting_delay_days=True),
            lambda r: r.update(required_source_rows=1),
            lambda r: r.update(provided_source_rows=1),
            lambda r: r.update(evaluation_available_through='2000-01-01'),
            lambda r: r.update(predictive_acceptance='approved'),
            lambda r: r.update(production_readiness='approved'),
            lambda r: r.update(status='completed_with_issues'),
        ]
        for index, mutate in enumerate(mutations):
            report = copy.deepcopy(self.report)
            report['result_integrity'] = {'contract': CONTRACT, 'status': 'consistent'}
            mutate(report)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.check(report)

    def test_nonfinite_strings_booleans_and_malformed_containers_fail_closed(self):
        for value in (True, None, '0', float('nan'), float('inf'), 1j, 10 ** 500):
            report = copy.deepcopy(self.report)
            report['windows'][0]['evidence'][0]['prediction'] = value
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                self.check(report)
        for value in (None, [], {}, {'contract': 'daily-rolling-backtest-v1', 'windows': None}):
            with self.assertRaisesRegex(ValueError, '^The historical forecast evidence is incomplete or inconsistent.$'):
                self.check(value)

    def test_business_flags_use_confirmed_weights_and_verified_reported_thresholds(self):
        self.check(self.weighted, decision_policy=POLICY)
        for mutate in (
            lambda r: r['windows'][0]['metrics'].update(weighted_absolute_error=0.),
            lambda r: r['windows'][0]['business_evaluation']['checks'].update(maximum_mae=1),
            lambda r: r['windows'][0]['business_evaluation'].update(relative_improvement=1.),
            lambda r: r['windows'][0]['business_evaluation'].update(financial_value='proven'),
            lambda r: r['summary'].update(criteria_met_windows=12345),
        ):
            report = copy.deepcopy(self.weighted)
            mutate(report)
            with self.assertRaises(ValueError): self.check(report, decision_policy=POLICY)
        with self.assertRaises(ValueError):
            self.check(self.weighted, decision_policy=POLICY | {'underprediction_weight': 2.})

    def test_delayed_nonweekly_and_explicit_origin_windows_bind_to_sorted_source(self):
        for task in (ForecastTask('cnt', 'dteday', 3, 3, 8),
                     ForecastTask('cnt', 'dteday', 3, 3, 8, '2020-06-16')):
            report = historical_forecast_backtest(self.frame, task, candidate_suite=LEGACY_CANDIDATE_SUITE)
            settings = dict(candidate_suite=LEGACY_CANDIDATE_SUITE)
            validate_historical_backtest(report, self.frame.iloc[::-1], task, **settings)
            changed = copy.deepcopy(report)
            changed['windows'][-1]['available_through'] = changed['windows'][-1]['forecast_origin']
            with self.assertRaises(ValueError): validate_historical_backtest(changed, self.frame, task, **settings)
            with self.assertRaises(ValueError): validate_historical_backtest(report, self.frame, TASK, **settings)

    def test_failures_and_insufficient_history_remain_explicit_and_cannot_hide_scored_rows(self):
        with patch('src.forecast_backtest.forecast_series', side_effect=OSError('PRIVATE_SOURCE')):
            failed = historical_forecast_backtest(self.frame, TASK)
        result = self.check(failed)
        self.assertEqual(result['checked_scored_rows'], 0)
        for key, value in (('status', 'scored'), ('reason', 'PRIVATE_SOURCE'),
                           ('metrics', {'mae': 0.}), ('evidence', [])):
            changed = copy.deepcopy(failed)
            changed['windows'][0][key] = value
            with self.assertRaises(ValueError): self.check(changed)
        short = self.frame.iloc[:70]
        report = historical_forecast_backtest(short, TASK)
        validate_historical_backtest(report, short, TASK)
        report['status'] = 'completed'
        with self.assertRaises(ValueError): validate_historical_backtest(report, short, TASK)

    def test_zero_reference_has_no_fabricated_gain(self):
        flat = self.frame.copy()
        flat['cnt'] = 10.
        report = historical_forecast_backtest(flat, TASK, decision_policy=POLICY)
        validate_historical_backtest(report, flat, TASK, decision_policy=POLICY)
        self.assertIsNone(report['windows'][0]['comparison']['relative_improvement'])
        report['windows'][0]['comparison']['relative_improvement'] = 0.
        with self.assertRaises(ValueError): validate_historical_backtest(report, flat, TASK, decision_policy=POLICY)

    def test_negative_count_outputs_and_candidate_failures_are_consistent_failed_evidence(self):
        counts = self.frame.copy()
        counts['cnt'] = np.maximum(0, 300 - 2 * np.arange(len(counts)))
        counts.loc[153:, 'cnt'] = 125
        report = historical_forecast_backtest(counts, TASK, target_unit='count', decision_policy=POLICY)
        self.assertGreater(report['summary']['domain_failed_windows'], 0)
        validate_historical_backtest(report, counts, TASK, target_unit='count', decision_policy=POLICY)
        negative = next(index for index, row in enumerate(report['windows'])
                        if row['domain_constraints']['checks']['future']['negative_values'])
        for mutate in (
            lambda r: r['windows'][negative]['domain_constraints']['checks']['future'].update(negative_values=0),
            lambda r: r['windows'][negative]['domain_constraints'].update(status='nonnegative_point_forecast'),
            lambda r: r['windows'][negative].update(status='scored'),
            lambda r: r['windows'][negative]['business_evaluation'].update(status='criteria_met_on_holdout'),
        ):
            changed = copy.deepcopy(report)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_historical_backtest(changed, counts, TASK, target_unit='count', decision_policy=POLICY)
        from src.forecasting import _forecast
        def fail_forest(history, horizon, period, candidate):
            if candidate == 'lag_random_forest': raise ValueError('Controlled failure')
            return _forecast(history, horizon, period, candidate)
        with patch('src.forecasting._forecast', side_effect=fail_forest):
            failed = historical_forecast_backtest(self.frame, TASK)
        self.assertEqual(failed['summary']['candidate_failures'], 3)
        self.check(failed)
        failed['windows'][0]['candidate_failures'] = True
        with self.assertRaises(ValueError): self.check(failed)

    def test_web_rejects_corrupt_history_before_delivering_or_returning_audit_success(self):
        semantics = confirmed(self.frame)
        task = forecast_task_from_form(self.frame, semantics, FORM)
        primary = forecast_series(self.frame, task['settings'])
        corrupt = copy.deepcopy(self.report)
        corrupt['summary']['better'] = 999
        with patch('vibedash.forecast_tasks.forecast_series', return_value=primary), \
                patch('vibedash.forecast_tasks.historical_forecast_backtest', return_value=corrupt):
            with self.assertRaisesRegex(ForecastTaskError, 'historical forecast evidence'):
                run_confirmed_forecast(self.frame, semantics, task)

    def test_cli_rejects_corrupt_history_before_optional_model_creation(self):
        corrupt = copy.deepcopy(self.report)
        corrupt['windows'][0]['evidence'][0]['actual'] += 1.
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'history.csv'
            self.frame.to_csv(source, index=False)
            model = Path(directory) / 'model.json'
            output = io.StringIO()
            with patch('evaluate_forecast.historical_forecast_backtest', return_value=corrupt), \
                    patch('evaluate_forecast.create_artifact') as save, redirect_stdout(output):
                code = main(['--csv', str(source), '--target', 'cnt', '--time', 'dteday',
                             '--horizon', '7', '--save-model', str(model)])
            save.assert_not_called()
            self.assertFalse(model.exists())
            self.assertEqual(code, 1)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload['status'], 'error')
            self.assertNotIn('historical_backtest', payload)

    def test_audit_includes_only_verified_aggregate_marker_and_old_reports_are_unassessed(self):
        self.assertEqual(backtest_audit_view(self.report)['result_integrity'], {'status': 'not_assessed'})
        report = copy.deepcopy(self.report)
        report['result_integrity'] = self.check(report)
        audit = backtest_audit_view(report)
        self.assertEqual(audit['result_integrity']['status'], 'consistent')
        self.assertNotIn('evidence', json.dumps(audit))
        self.assertNotIn('forecast"', json.dumps(audit))


if __name__ == '__main__': unittest.main()
