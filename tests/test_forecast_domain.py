"""A positive future does not rescue an invalid held-out count forecast."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from src.forecast_domain import CONTRACT, evaluate_count_forecast
from src.forecast_policy import evaluate_forecast_policy, weighted_absolute_error
from src.semantic_contract import validate_semantic_contract
from tests.test_forecast_tasks import FORM, confirmed, daily_frame
from tests.test_forecast_policy import POLICY_FORM
from tests import test_vibedash_standalone_export as export_fixture
from vibedash.forecast_tasks import (
    ForecastTaskError, forecast_task_from_form, run_confirmed_forecast, forecast_audit_view,
)


def complete_report():
    return {'status': 'ok', 'holdout': {'predictions': [
        {'prediction': 2.5, 'baseline': 3.5} for _ in range(7)]},
        'future_forecast': [{'prediction': 4.5} for _ in range(7)]}


class CountForecastDomainTests(unittest.TestCase):
    def test_complete_nonnegative_fractional_points_are_valid_not_rounded(self):
        report = complete_report()
        before = copy.deepcopy(report)
        result = evaluate_count_forecast(report, 7)
        self.assertEqual(result['contract'], CONTRACT)
        self.assertEqual(result['status'], 'nonnegative_point_forecast')
        self.assertEqual(result['failed_scopes'], [])
        self.assertEqual(result['checks'], {
            scope: {'checked_values': 7, 'negative_values': 0}
            for scope in ('holdout_method', 'holdout_reference', 'future')})
        self.assertEqual(report, before)
        json.dumps(result, allow_nan=False)

    def test_each_negative_scope_is_recorded_and_never_clipped_or_mutated(self):
        for scope, container, field in (
            ('holdout_method', 'holdout', 'prediction'),
            ('holdout_reference', 'holdout', 'baseline'), ('future', 'future', 'prediction'),
        ):
            report = complete_report()
            rows = report['holdout']['predictions'] if container == 'holdout' else report['future_forecast']
            rows[0][field] = -1e-12  # No hidden rounding tolerance to manufacture validity.
            rows[4][field] = -3.
            before = copy.deepcopy(report)
            result = evaluate_count_forecast(report, 7)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['failed_scopes'], [scope])
            self.assertEqual(result['checks'][scope]['negative_values'], 2)
            self.assertEqual(report, before)

    def test_missing_empty_short_and_long_output_never_passes_vacuously(self):
        for value in (None, {}, {'status': 'error'}, {'status': 'ok'},
                      complete_report() | {'holdout': []}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate_count_forecast(value, 7)
        for part in ('holdout', 'future'):
            for rows in (None, [], [{}], [{}] * 6, [{}] * 8, {}, 'not rows'):
                report = complete_report()
                if part == 'holdout': report['holdout']['predictions'] = rows
                else: report['future_forecast'] = rows
                with self.subTest(part=part, rows=rows), self.assertRaises(ValueError):
                    evaluate_count_forecast(report, 7)

    def test_nonfinite_boolean_string_complex_and_missing_values_are_rejected(self):
        for part, field in (('holdout', 'prediction'), ('holdout', 'baseline'), ('future', 'prediction')):
            for value in (None, True, np.bool_(False), '1', float('nan'), float('inf'),
                          float('-inf'), 1 + 2j, 10 ** 500):
                report = complete_report()
                rows = report['holdout']['predictions'] if part == 'holdout' else report['future_forecast']
                rows[0][field] = value
                with self.subTest(part=part, field=field, value=repr(value)), self.assertRaises(ValueError):
                    evaluate_count_forecast(report, 7)
            report = complete_report()
            rows = report['holdout']['predictions'] if part == 'holdout' else report['future_forecast']
            rows[0] = {}
            with self.assertRaises(ValueError): evaluate_count_forecast(report, 7)

    def test_horizon_must_be_explicit_and_bounded(self):
        for value in (None, True, 0, 91, 7., '7', []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate_count_forecast(complete_report(), value)


class ConfirmedCountForecastDomainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Real Ridge selection, no patched fitting: held-out forecasts go below
        # zero after a declining prefix, then refit on the observed positive jump.
        cls.frame = daily_frame()
        cls.frame['cnt'] = np.maximum(0, 300 - 2 * np.arange(160))
        cls.frame.loc[cls.frame.index[-7:], 'cnt'] = 125
        cls.semantics = confirmed(cls.frame)
        cls.task = forecast_task_from_form(cls.frame, cls.semantics, FORM)
        cls.result = run_confirmed_forecast(cls.frame, cls.semantics, cls.task)

    def test_real_negative_holdout_is_not_rescued_by_positive_future(self):
        report = self.result
        self.assertEqual(report['model_name'], 'lag_ridge')
        self.assertTrue(all(row['prediction'] < 0 for row in report['holdout']['predictions']))
        self.assertTrue(all(row['prediction'] >= 0 for row in report['future_forecast']))
        self.assertEqual(report['domain_constraints']['status'], 'failed')
        self.assertEqual(report['domain_constraints']['failed_scopes'], ['holdout_method'])
        self.assertEqual(report['domain_constraints']['checks']['holdout_method']['negative_values'], 7)
        self.assertEqual(report['business_evaluation']['status'], 'not_specified')

    def test_domain_failure_in_any_scope_blocks_success_but_preserves_numerical_evidence(self):
        task = forecast_task_from_form(self.frame, self.semantics, FORM | POLICY_FORM)
        for scope in ('holdout_method', 'holdout_reference', 'future'):
            report = copy.deepcopy(self.result)
            # Isolate each failure on a structurally complete numerical report.
            for row in report['holdout']['predictions']: row.update(prediction=row['actual'], baseline=1000.)
            for row in report['future_forecast']: row['prediction'] = 3.
            if scope == 'future': report['future_forecast'][0]['prediction'] = -4.
            else: report['holdout']['predictions'][0]['prediction' if scope == 'holdout_method' else 'baseline'] = -4.
            policy = task['decision_policy']
            report['decision_policy'] = policy
            report['selection_metric'] = 'mean validation weighted absolute error'
            # Construct coherent numerical evidence for this domain-only test;
            # independent integrity arithmetic is tested separately.
            actual = np.array([row['actual'] for row in report['holdout']['predictions']])
            for field, key in (('prediction', 'metrics'), ('baseline', 'baseline_metrics')):
                prediction = np.array([row[field] for row in report['holdout']['predictions']])
                report['holdout'][key] = {'mae': float(np.mean(np.abs(actual - prediction))),
                                          'rmse': float(np.sqrt(np.mean((actual - prediction) ** 2))),
                                          'weighted_absolute_error': weighted_absolute_error(actual, prediction, policy)}
            metrics, reference = report['holdout']['metrics'], report['holdout']['baseline_metrics']
            report['holdout']['mae_improvement'] = 1 - metrics['mae'] / reference['mae']
            report['predictive_support']['status'] = 'observed_holdout_improvement'
            report['business_evaluation'] = evaluate_forecast_policy(policy, metrics, reference)
            self.assertEqual(report['business_evaluation']['status'], 'criteria_met_on_holdout')
            before_holdout, before_future = copy.deepcopy(report['holdout']), copy.deepcopy(report['future_forecast'])
            with patch('vibedash.forecast_tasks.forecast_series', return_value=report) as fit:
                result = run_confirmed_forecast(self.frame, self.semantics, task)
            fit.assert_called_once()  # No post-holdout method reselection or retry.
            self.assertEqual(result['business_evaluation']['status'], 'domain_constraint_failed')
            self.assertEqual(result['domain_constraints']['failed_scopes'], [scope])
            self.assertEqual(result['holdout'], before_holdout)
            self.assertEqual(result['future_forecast'], before_future)

    def test_corrupted_completed_count_output_cannot_be_returned_as_valid(self):
        for mutate in (
            lambda report: report.update(future_forecast=[]),
            lambda report: report['holdout']['predictions'].pop(),
            lambda report: report['holdout']['predictions'][0].update(baseline=float('nan')),
        ):
            report = copy.deepcopy(self.result)
            mutate(report)
            with patch('vibedash.forecast_tasks.forecast_series', return_value=report), self.assertRaises(ForecastTaskError):
                run_confirmed_forecast(self.frame, self.semantics, self.task)

    def test_general_numeric_target_does_not_inherit_nonnegativity_assumption(self):
        semantics = copy.deepcopy(self.semantics)
        for column in semantics['columns']:
            if column['is_target']: column['unit'] = 'number'
        semantics = validate_semantic_contract(self.frame, semantics)
        task = forecast_task_from_form(self.frame, semantics, FORM)
        with patch('vibedash.forecast_tasks.forecast_series', return_value=copy.deepcopy(self.result)):
            report = run_confirmed_forecast(self.frame, semantics, task)
        self.assertEqual(report['domain_constraints'], {'status': 'not_assessed'})

    def test_audit_contains_only_aggregate_scope_counts_not_raw_values(self):
        audit = forecast_audit_view(self.result)
        self.assertEqual(audit['domain_constraints'], self.result['domain_constraints'])
        encoded = json.dumps(audit, allow_nan=False)
        for forbidden in ('predictions', 'future_forecast', 'actual', 'casual'):
            self.assertNotIn(forbidden, encoded)


class CountDomainPresentationRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        export_fixture.StandaloneExportTests.setUpClass.__func__(cls)

    def test_existing_presentation_hides_future_table_for_historical_domain_failure(self):
        frame = daily_frame()
        frame['cnt'] = np.maximum(0, 300 - 2 * np.arange(160))
        frame.loc[frame.index[-7:], 'cnt'] = 125
        semantics = confirmed(frame)
        report = run_confirmed_forecast(frame, semantics, forecast_task_from_form(frame, semantics, FORM))
        with patch.dict(self.dashboard, forecast=report):
            html = export_fixture.StandaloneExportTests.render(self)
        self.assertIn('Forecast violates the confirmed count domain', html)
        self.assertIn('Final holdout predictions', html)
        self.assertNotIn('Exploratory future point forecast', html)


if __name__ == '__main__': unittest.main()
