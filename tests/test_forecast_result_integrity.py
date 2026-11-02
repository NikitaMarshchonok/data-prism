"""Independent evidence oracles and delivery-boundary corruption tests."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluate_forecast import main as forecast_cli
from src.forecasting import ForecastTask, forecast_series, forecast_task_settings
from src.forecast_result import CONTRACT, validate_forecast_result
from tests.test_forecast_tasks import FORM, confirmed, daily_frame
from tests.test_forecast_policy import POLICY_FORM
from vibedash.forecast_tasks import ForecastTaskError, forecast_task_from_form, run_confirmed_forecast, forecast_audit_view


class ForecastResultIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = daily_frame()
        cls.semantics = confirmed(cls.frame)
        cls.task = forecast_task_from_form(cls.frame, cls.semantics, FORM)
        cls.report = forecast_series(cls.frame, cls.task['settings'])
        cls.policy_task = forecast_task_from_form(cls.frame, cls.semantics, FORM | POLICY_FORM)
        cls.weighted = forecast_series(cls.frame, cls.policy_task['settings'],
                                       decision_policy=cls.policy_task['decision_policy'])

    def check_report(self, report):
        return validate_forecast_result(report, self.frame, self.task['settings'])

    def test_real_report_is_verified_without_modifying_predictions_or_claiming_acceptance(self):
        before = copy.deepcopy(self.report)
        result = self.check_report(self.report)
        self.assertEqual(result['contract'], CONTRACT)
        self.assertEqual(result['status'], 'consistent')
        self.assertEqual(result['checked_holdout_rows'], 7)
        self.assertEqual(result['source_facts'], 'matched')
        self.assertEqual(result['production_readiness'], 'not_established')
        self.assertEqual(self.report, before)
        json.dumps(result, allow_nan=False)

    def test_tampered_errors_facts_and_gain_fail_even_if_report_says_ok(self):
        changes = [
            lambda r: r['holdout']['metrics'].update(mae=0.),
            lambda r: r['holdout']['metrics'].update(rmse=1234.),
            lambda r: r['holdout']['baseline_metrics'].update(mae=0.),
            lambda r: r['holdout'].update(mae_improvement=1.),
            lambda r: r['holdout']['predictions'][0].update(actual=0.),
            lambda r: r['holdout']['predictions'][0].update(prediction=10000.),
            lambda r: r['holdout']['predictions'][0].update(baseline=10000.),
            lambda r: r['predictive_support'].update(status='no_holdout_improvement'),
            lambda r: r['predictive_support'].update(production_readiness='approved'),
            lambda r: r.update(model_name=None),
            lambda r: r.update(business_evaluation={'status': 'criteria_met_on_holdout'}),
        ]
        for mutate in changes:
            report = copy.deepcopy(self.report)
            mutate(report)
            with self.subTest(mutation=changes.index(mutate)), self.assertRaises(ValueError):
                self.check_report(report)

    def test_wrong_dates_missing_duplicate_or_reordered_rows_and_cutoffs_fail(self):
        changes = [
            lambda r: r['holdout']['predictions'].pop(),
            lambda r: r['future_forecast'].clear(),
            lambda r: r['holdout']['predictions'].reverse(),
            lambda r: r['future_forecast'][0].update(date=r['future_forecast'][1]['date']),
            lambda r: r['holdout'].update(start='2000-01-01'),
            lambda r: r['holdout'].update(train_end=r['holdout']['start']),
            lambda r: r.update(future_fit_through='2000-01-01'),
            lambda r: r.update(future_forecast_origin='2000-01-01'),
            lambda r: r.update(test_rows=True),
            lambda r: r['availability'].update(reporting_delay_days=1),
            lambda r: r['availability'].update(historical_vintages_verified=True),
            lambda r: r.update(selection_metric='training accuracy'),
        ]
        for index, mutate in enumerate(changes):
            report = copy.deepcopy(self.report)
            mutate(report)
            with self.subTest(index=index), self.assertRaises(ValueError): self.check_report(report)

    def test_nonfinite_boolean_and_string_evidence_cannot_pass(self):
        for value in (True, '0', None, float('nan'), float('inf'), 1 + 2j, 10 ** 500):
            for part in ('metric', 'prediction', 'future'):
                report = copy.deepcopy(self.report)
                if part == 'metric': report['holdout']['metrics']['mae'] = value
                elif part == 'prediction': report['holdout']['predictions'][0]['prediction'] = value
                else: report['future_forecast'][0]['prediction'] = value
                with self.subTest(part=part, value=repr(value)), self.assertRaises(ValueError): self.check_report(report)

    def test_weighted_metrics_and_boolean_criteria_are_independently_checked(self):
        policy = self.policy_task['decision_policy']
        validate_forecast_result(self.weighted, self.frame, self.policy_task['settings'], decision_policy=policy)
        for mutate in (
            lambda r: r['holdout']['metrics'].update(weighted_absolute_error=0.),
            lambda r: r['business_evaluation'].update(weighted_absolute_error=0.),
            lambda r: r['business_evaluation'].update(relative_improvement=1.),
            lambda r: r['business_evaluation']['checks'].update(maximum_mae=1),
            lambda r: r['business_evaluation']['checks'].update(strictly_lower_weighted_error=False),
            lambda r: r['business_evaluation'].update(status='criteria_not_met_on_holdout'),
            lambda r: r['business_evaluation'].update(financial_value='proven'),
            lambda r: r['business_evaluation'].update(unreviewed_roi=1000.),
            lambda r: r['decision_policy'].update(underprediction_weight=999.),
        ):
            report = copy.deepcopy(self.weighted)
            mutate(report)
            with self.assertRaises(ValueError):
                validate_forecast_result(report, self.frame, self.policy_task['settings'], decision_policy=policy)

    def test_absolute_policy_boundary_is_checked_on_verified_reported_metrics(self):
        # A threshold exactly equal to the computed engine metric must not move
        # because independent fsum differs from numpy summation at roundoff.
        limit = self.weighted['holdout']['metrics']['mae']
        policy = self.policy_task['decision_policy'] | {'maximum_mae': limit}
        report = forecast_series(self.frame, self.policy_task['settings'], decision_policy=policy)
        self.assertTrue(report['business_evaluation']['checks']['maximum_mae'])
        validate_forecast_result(report, self.frame, self.policy_task['settings'], decision_policy=policy)

    def test_hand_calculated_two_day_metric_and_asymmetric_policy_oracle(self):
        # This is a scoring fixture, not a claim that the fitted model produced
        # these particular predictions: actual [10,10], method [8,13], ref [5,15].
        data = self.frame.assign(cnt=10.)
        task = ForecastTask('cnt', 'dteday', 2)
        settings, policy = forecast_task_settings(task), self.policy_task['decision_policy']
        report = forecast_series(data, task, decision_policy=policy)
        for row, prediction, baseline in zip(report['holdout']['predictions'], (8., 13.), (5., 15.)):
            row.update(prediction=prediction, baseline=baseline)
        report.update(model_name='lag_ridge', baseline_name='last_value')
        report['holdout']['metrics'] = {'mae': 2.5, 'rmse': 6.5 ** .5, 'weighted_absolute_error': 6.5}
        report['holdout']['baseline_metrics'] = {'mae': 5., 'rmse': 5., 'weighted_absolute_error': 15.}
        report['holdout']['mae_improvement'] = .5
        report['predictive_support']['status'] = 'observed_holdout_improvement'
        report['business_evaluation'].update(
            status='criteria_met_on_holdout', weighted_absolute_error=6.5,
            baseline_weighted_absolute_error=15., relative_improvement=1 - 6.5 / 15.,
            checks={'strictly_lower_weighted_error': True, 'minimum_improvement': True, 'maximum_mae': True})
        self.assertEqual(validate_forecast_result(report, data, settings, decision_policy=policy)['status'], 'consistent')

    def test_source_order_and_all_availability_modes_are_verified(self):
        for delay, origin in ((0, None), (10, None), (2, '2020-06-10')):
            task = ForecastTask('cnt', 'dteday', 3, 7, delay, origin)
            report = forecast_series(self.frame, task)
            shuffled = self.frame.sample(frac=1, random_state=3)
            result = validate_forecast_result(report, shuffled, forecast_task_settings(task))
            self.assertEqual(result['checked_holdout_rows'], 3)
            source = self.frame.copy()
            source.loc[source.index[-1 - delay if origin is None else -1], 'cnt'] += 1
            with self.assertRaises(ValueError): validate_forecast_result(report, source, forecast_task_settings(task))

    def test_zero_reference_and_tiny_nonzero_errors_keep_distinct_semantics(self):
        data = self.frame.assign(cnt=0.)
        task = ForecastTask('cnt', 'dteday', 7)
        report = forecast_series(data, task)
        self.assertIsNone(report['holdout']['mae_improvement'])
        validate_forecast_result(report, data, forecast_task_settings(task))
        changed = copy.deepcopy(report)
        changed['holdout']['mae_improvement'] = 0.
        with self.assertRaises(ValueError): validate_forecast_result(changed, data, forecast_task_settings(task))
        changed = copy.deepcopy(report)
        changed['holdout']['baseline_metrics']['mae'] = 1e-20
        with self.assertRaises(ValueError): validate_forecast_result(changed, data, forecast_task_settings(task))

        tiny = self.frame.assign(cnt=self.frame.cnt.astype(float) * 1e-15)
        report = forecast_series(tiny, task)
        report['holdout']['metrics']['mae'] = -1e-20
        reference = report['holdout']['baseline_metrics']['mae']
        report['holdout']['mae_improvement'] = 1 - (-1e-20 / reference) if reference > 0 else None
        with self.assertRaises(ValueError): validate_forecast_result(report, tiny, forecast_task_settings(task))

    def test_worker_rechecks_evidence_not_a_cached_integrity_badge(self):
        report = copy.deepcopy(self.report)
        report['result_integrity'] = {'contract': CONTRACT, 'status': 'consistent'}
        report['holdout']['metrics']['mae'] = 0.
        with patch('vibedash.forecast_tasks.forecast_series', return_value=report) as fit, self.assertRaises(ForecastTaskError):
            run_confirmed_forecast(self.frame, self.semantics, self.task)
        fit.assert_called_once()
        for invalid in (None, {}, {'status': 'error'}):
            with patch('vibedash.forecast_tasks.forecast_series', return_value=invalid), self.assertRaises(ForecastTaskError):
                run_confirmed_forecast(self.frame, self.semantics, self.task)

    def test_verified_audit_has_no_per_day_values_or_claim_of_readiness(self):
        report = run_confirmed_forecast(self.frame, self.semantics, self.task)
        audit = forecast_audit_view(report)
        self.assertEqual(audit['result_integrity']['status'], 'consistent')
        self.assertEqual(audit['result_integrity']['production_readiness'], 'not_established')
        encoded = json.dumps(audit, allow_nan=False)
        for forbidden in ('predictions', 'future_forecast', 'actual', 'casual'):
            self.assertNotIn(forbidden, encoded)

    def test_cli_rejects_inconsistent_result_and_accepts_verified_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'series.csv'
            self.frame.to_csv(path, index=False)
            args = ['--csv', str(path), '--target', 'cnt', '--time', 'dteday', '--horizon', '7']
            stream = io.StringIO()
            with redirect_stdout(stream): code = forecast_cli(args)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stream.getvalue())['result_integrity']['status'], 'consistent')
            bad = copy.deepcopy(self.report)
            bad['holdout']['metrics']['mae'] = 0.
            stream = io.StringIO()
            with patch('evaluate_forecast.forecast_series', return_value=bad), redirect_stdout(stream):
                code = forecast_cli(args)
            self.assertEqual(code, 1)
            result = json.loads(stream.getvalue())
            self.assertEqual(result['status'], 'error')
            self.assertIn('evidence-consistency', result['message'])
            self.assertNotIn('holdout', result)


if __name__ == '__main__': unittest.main()
