"""Real-input rolling evaluation: prefix isolation, baselines and retained losses."""
import copy
import json
import statistics
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.forecasting import ForecastTask, forecast_series, _forecast
from src.forecast_backtest import CONTRACT, historical_forecast_backtest, backtest_audit_view
from src.forecast_scoring import score_predictions
from tests.test_forecast_tasks import daily_frame, confirmed, FORM
from tests.test_forecast_policy import POLICY, POLICY_FORM
from vibedash.forecast_tasks import forecast_task_from_form, run_confirmed_forecast, forecast_audit_view


TASK = ForecastTask('cnt', 'dteday', 7)


class HistoricalForecastBacktestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = daily_frame()
        cls.result = historical_forecast_backtest(cls.frame, TASK)

    def test_real_uploaded_series_gets_three_nonoverlapping_windows_and_independent_errors(self):
        result = self.result
        self.assertEqual(result['contract'], CONTRACT)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['summary']['scheduled_windows'], 3)
        self.assertEqual(result['summary']['scored_windows'], 3)
        self.assertEqual(result['predictive_acceptance'], 'not_established')
        self.assertEqual(result['summary']['better'] + result['summary']['tie'] + result['summary']['worse'], 3)
        previous = None
        for row in result['windows']:
            self.assertLess(row['forecast_origin'], row['start'])
            if previous: self.assertLess(previous, row['start'])
            previous = row['end']
            self.assertEqual(len(row['evidence']), 7)
            for field, metrics in (('prediction', 'metrics'), ('baseline', 'baseline_metrics')):
                actual = [entry['actual'] for entry in row['evidence']]
                expected = sum(abs(value - entry[field]) for value, entry in zip(actual, row['evidence'])) / 7
                self.assertAlmostEqual(row[metrics]['mae'], expected)
        json.dumps(result, allow_nan=False)

    def test_three_calls_receive_only_available_prefix_and_the_declared_issue_date(self):
        task = ForecastTask('cnt', 'dteday', 3, 3, 8, '2020-06-16')
        # Keep the input cutoff at June 8; issue date is eight days later.
        calls = []
        def observe(frame, task, **kwargs):
            calls.append((frame.copy(deep=True), task, kwargs))
            return forecast_series(frame, task, **kwargs)
        with patch('src.forecast_backtest.forecast_series', side_effect=observe):
            result = historical_forecast_backtest(self.frame, task)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(calls), 3)
        for row, (prefix, inner_task, _kwargs) in zip(result['windows'], calls):
            self.assertEqual(list(prefix.columns), ['dteday', 'cnt'])
            self.assertEqual(len(prefix), row['history_rows'])
            np.testing.assert_array_equal(prefix.cnt.to_numpy(), self.frame.cnt.to_numpy()[:len(prefix)])
            self.assertEqual(inner_task.forecast_origin, row['forecast_origin'])
            self.assertEqual((pd.Timestamp(row['forecast_origin']) - prefix.dteday.iloc[-1]).days, 8)

    def test_independent_nonweekly_baselines_advance_through_long_delay(self):
        task = ForecastTask('cnt', 'dteday', 3, 3, 8)
        result = historical_forecast_backtest(self.frame, task)
        for row in result['windows']:
            prefix = self.frame.cnt.iloc[:row['history_rows']].astype(float).tolist()
            expected = {'last_value': [prefix[-1]] * 3,
                        'seasonal_naive': [prefix[-3:][step % 3] for step in range(8, 11)],
                        'historical_mean': [sum(prefix) / len(prefix)] * 3,
                        'historical_median': [statistics.median(prefix)] * 3}
            self.assertEqual(set(row['references']), set(expected))
            for name, values in expected.items():
                np.testing.assert_allclose(row['references'][name]['forecast'], values, rtol=1e-12)
            self.assertEqual(row['baseline_metrics'], row['references'][row['baseline_name']]['metrics'])

    def test_mutating_outer_facts_cannot_change_forecasts_or_selection(self):
        changed = self.frame.copy()
        changed.loc[153:, 'cnt'] += 10000
        after = historical_forecast_backtest(changed, TASK)
        for before, later in zip(self.result['windows'], after['windows']):
            self.assertEqual(before['model_name'], later['model_name'])
            self.assertEqual(before['baseline_name'], later['baseline_name'])
            for name in before['references']:
                self.assertEqual(before['references'][name]['forecast'], later['references'][name]['forecast'])
            self.assertEqual([row['prediction'] for row in before['evidence']],
                             [row['prediction'] for row in later['evidence']])
        self.assertNotEqual(self.result['windows'][-1]['metrics'], after['windows'][-1]['metrics'])

    def test_unreleased_tail_and_pre_origin_gap_never_enter_historical_fits(self):
        task = ForecastTask('cnt', 'dteday', 3, 7, 10)
        before = historical_forecast_backtest(self.frame, task)
        changed = self.frame.copy()
        changed.loc[150:, 'cnt'] += 10000  # Entire final unreleased tail.
        self.assertEqual(before, historical_forecast_backtest(changed, task))
        # Last origin is immediately before target index147; released prefix ends137.
        changed = self.frame.copy()
        changed.loc[137:146, 'cnt'] += 10000
        after = historical_forecast_backtest(changed, task)
        first, second = before['windows'][-1], after['windows'][-1]
        self.assertEqual(first['model_name'], second['model_name'])
        self.assertEqual(first['baseline_name'], second['baseline_name'])
        self.assertEqual(first['evidence'], second['evidence'])

    def test_history_shortfall_does_not_silently_shorten_protocol(self):
        short = self.frame.iloc[:70]
        with patch('src.forecast_backtest.forecast_series') as fit:
            result = historical_forecast_backtest(short, TASK)
        fit.assert_not_called()
        self.assertEqual(result['status'], 'insufficient_history')
        self.assertEqual(result['required_source_rows'], 79)
        self.assertEqual(result['provided_source_rows'], 70)
        self.assertEqual(result['summary']['scored_windows'], 0)
        self.assertEqual(len(result['windows']), 3)
        self.assertTrue(all(row['status'] == 'not_evaluated' for row in result['windows']))
        self.assertIsNone(result['windows'][0]['metrics'])

    def test_failure_keeps_its_slot_without_data_bearing_diagnostics_or_retry(self):
        count = 0
        def controlled(frame, task, **kwargs):
            nonlocal count
            count += 1
            if count == 2: raise OSError('private row diagnostic must not be published')
            return forecast_series(frame, task, **kwargs)
        with patch('src.forecast_backtest.forecast_series', side_effect=controlled):
            result = historical_forecast_backtest(self.frame, TASK)
        self.assertEqual(count, 3)
        self.assertEqual(result['status'], 'completed_with_issues')
        self.assertEqual(result['summary']['scored_windows'], 2)
        self.assertEqual(result['summary']['failed_windows'], 1)
        self.assertEqual(result['windows'][1]['status'], 'error')
        self.assertNotIn('private row diagnostic', json.dumps(result))

    def test_inconsistent_or_wrong_suite_output_fails_window_not_the_whole_report(self):
        for kind in ('metric', 'suite'):
            def corrupt(frame, task, **kwargs):
                result = forecast_series(frame, task, **kwargs)
                if kind == 'metric': result['holdout']['metrics']['mae'] = -1.
                else: result['candidate_suite'] = 'unknown'
                return result
            with patch('src.forecast_backtest.forecast_series', side_effect=corrupt):
                result = historical_forecast_backtest(self.frame, TASK)
            self.assertEqual(result['summary']['failed_windows'], 3)
            self.assertEqual(result['summary']['scored_windows'], 0)

    def test_failed_candidate_and_negative_count_predictions_remain_visible(self):
        def fail_forest(history, horizon, period, candidate):
            if candidate == 'lag_random_forest': raise ValueError('controlled failure')
            return _forecast(history, horizon, period, candidate)
        with patch('src.forecasting._forecast', side_effect=fail_forest):
            result = historical_forecast_backtest(self.frame, TASK)
        self.assertEqual(result['summary']['candidate_failures'], 3)
        self.assertEqual(result['status'], 'completed_with_issues')
        counts = self.frame.copy()
        counts['cnt'] = np.maximum(0, 300 - 2 * np.arange(160))
        counts.loc[153:, 'cnt'] = 125
        result = historical_forecast_backtest(counts, TASK, target_unit='count')
        self.assertGreater(result['summary']['domain_failed_windows'], 0)
        self.assertTrue(any(row['prediction'] < 0 for window in result['windows']
                            for row in window.get('evidence', [])))

    def test_declared_loss_is_used_and_unmet_absolute_criterion_is_preserved(self):
        policy = POLICY | {'maximum_mae': 0.}
        result = historical_forecast_backtest(self.frame, TASK, decision_policy=policy)
        self.assertEqual(result['protocol']['objective'], 'weighted_absolute_error')
        self.assertEqual(result['summary']['criteria_met_windows'], 0)
        for window in result['windows']:
            self.assertFalse(window['business_evaluation']['checks']['maximum_mae'])
            errors = [row['actual'] - row['prediction'] for row in window['evidence']]
            expected = sum(5 * max(error, 0) + max(-error, 0) for error in errors) / 7
            self.assertAlmostEqual(window['metrics']['weighted_absolute_error'], expected)
        self.assertEqual(score_predictions([10., 10.], [8., 13.], decision_policy=policy)['weighted_absolute_error'], 6.5)

    def test_bad_input_unknown_suite_or_negative_count_is_rejected_before_any_fit(self):
        invalid = self.frame.copy()
        invalid.loc[20, 'cnt'] = -1
        for data, kwargs in ((self.frame.drop(index=20), {}), (invalid, {'target_unit': 'count'}),
                             (self.frame, {'candidate_suite': 'unknown'}),
                             (self.frame, {'decision_policy': {}})):
            with patch('src.forecast_backtest.forecast_series') as fit, self.assertRaises(ValueError):
                historical_forecast_backtest(data, TASK, **kwargs)
            fit.assert_not_called()

    def test_web_evaluation_preserves_final_forecast_and_audit_excludes_outer_values(self):
        semantics = confirmed(self.frame)
        task = forecast_task_from_form(self.frame, semantics, FORM | POLICY_FORM)
        original = forecast_series(self.frame, task['settings'], decision_policy=task['decision_policy'])
        report = run_confirmed_forecast(self.frame, semantics, task)
        for key in ('model_name', 'baseline_name', 'holdout', 'future_forecast', 'model_comparison'):
            self.assertEqual(report[key], original[key], key)
        audit = forecast_audit_view(report)['historical_backtest']
        self.assertEqual(audit, backtest_audit_view(report['historical_backtest']))
        encoded = json.dumps(audit, allow_nan=False)
        for forbidden in ('actual', 'evidence', 'predictions', 'future_forecast', 'casual', POLICY['decision']):
            self.assertNotIn(forbidden, encoded)


if __name__ == '__main__': unittest.main()
