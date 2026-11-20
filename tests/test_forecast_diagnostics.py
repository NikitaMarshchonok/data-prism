"""Hand-calculated error direction, coverage and lead-time oracles."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluate_forecast import main
from src.forecast_diagnostics import CONTRACT, error_profile, build_forecast_diagnostics
from tests.test_forecast_policy import POLICY, POLICY_FORM
from tests.test_forecast_tasks import daily_frame, confirmed, FORM
from vibedash.forecast_tasks import forecast_task_from_form, run_confirmed_forecast, forecast_audit_view, ForecastTaskError


def evidence(predictions):
    return [{'actual': 10., 'prediction': value, 'baseline': 9.} for value in predictions]


def report_fixture():
    return {'status': 'ok', 'task': {'horizon': 2}, 'decision_policy': None,
            'holdout': {'predictions': evidence([8., 13.])},
            'historical_backtest': {'status': 'completed', 'windows': [
                {'status': 'scored', 'evidence': evidence(values)}
                for values in ([8., 13.], [9., 11.], [12., 10.])
            ]}}


class ForecastErrorDiagnosticsTests(unittest.TestCase):
    def test_zero_mean_error_does_not_hide_opposing_errors_or_asymmetric_penalties(self):
        result = error_profile([10., 10., 10., 10.], [8., 13., 10., 9.], decision_policy=POLICY)
        self.assertEqual(result['observations'], 4)
        self.assertEqual(result['underpredicted_days'], 2)
        self.assertEqual(result['overpredicted_days'], 1)
        self.assertEqual(result['exact_days'], 1)
        self.assertEqual(result['mean_signed_error'], 0.)
        self.assertEqual(result['underprediction_mae_contribution'], .75)
        self.assertEqual(result['overprediction_mae_contribution'], .75)
        self.assertEqual(result['mae'], 1.5)
        self.assertEqual(result['maximum_underprediction'], 2.)
        self.assertEqual(result['maximum_overprediction'], 3.)
        self.assertEqual(result['underprediction_weighted_contribution'], 3.75)
        self.assertEqual(result['overprediction_weighted_contribution'], .75)
        self.assertEqual(result['weighted_absolute_error'], 4.5)

    def test_equal_weights_reduce_to_mae_and_missing_policy_does_not_invent_costs(self):
        plain = error_profile([10., 10.], [8., 13.])
        self.assertEqual(plain['mean_signed_error'], .5)
        self.assertIsNone(plain['weighted_absolute_error'])
        weighted = error_profile([10., 10.], [8., 13.], decision_policy=POLICY | {'underprediction_weight': 1.})
        self.assertEqual(weighted['weighted_absolute_error'], 2.5)
        self.assertEqual(weighted['weighted_absolute_error'], weighted['mae'])
        # Magnitudes of negative targets do not change the sign convention.
        negative = error_profile([-10., -10.], [-12., -7.])
        self.assertEqual(negative, plain)

    def test_exact_days_are_counted_without_rounding_small_nonzero_errors(self):
        exact = error_profile([0., 0.], [0., 0.])
        self.assertEqual(exact['exact_days'], 2)
        tiny = error_profile([0., 0.], [-1e-20, 1e-20])
        self.assertEqual(tiny['exact_days'], 0)
        self.assertEqual(tiny['underpredicted_days'], 1)
        self.assertEqual(tiny['overpredicted_days'], 1)

    def test_leads_are_grouped_by_issue_date_distance_without_readding_main_holdout(self):
        report = report_fixture()
        before = copy.deepcopy(report)
        result = build_forecast_diagnostics(report)
        self.assertEqual(result['contract'], CONTRACT)
        self.assertEqual(result['sign_convention'], 'prediction_minus_actual')
        first, second = result['historical_by_lead']
        self.assertEqual((first['lead_days'], second['lead_days']), (1, 2))
        self.assertEqual(first['observations'], 3)
        self.assertAlmostEqual(first['model']['mae'], 5 / 3)
        self.assertAlmostEqual(first['model']['mean_signed_error'], -1 / 3)
        self.assertAlmostEqual(second['model']['mae'], 4 / 3)
        self.assertAlmostEqual(second['model']['mean_signed_error'], 4 / 3)
        self.assertEqual(first['baseline']['mean_signed_error'], -1.)
        self.assertEqual(result['holdout']['model']['observations'], 2)
        self.assertEqual(result['production_readiness'], 'not_established')
        self.assertEqual(report, before)
        json.dumps(result, allow_nan=False)

    def test_failed_windows_are_missing_not_zero_and_numeric_domain_failures_remain_included(self):
        report = report_fixture()
        report['historical_backtest']['windows'][1] = {'status': 'error'}
        report['historical_backtest']['windows'][2]['status'] = 'scored_domain_failure'
        result = build_forecast_diagnostics(report)
        self.assertEqual(result['scored_origins'], 2)
        self.assertEqual(result['historical_by_lead'][0]['observations'], 2)
        self.assertEqual(result['historical_by_lead'][0]['model']['mae'], 2.)
        self.assertEqual(result['historical_by_lead'][0]['model']['mean_signed_error'], 0.)
        report['historical_backtest']['status'] = 'insufficient_history'
        report['historical_backtest']['windows'] = [{'status': 'not_evaluated'}] * 3
        result = build_forecast_diagnostics(report)
        self.assertEqual(result['scored_origins'], 0)
        for lead in result['historical_by_lead']:
            self.assertEqual(lead['observations'], 0)
            self.assertIsNone(lead['model'])
            self.assertIsNone(lead['baseline'])

    def test_incomplete_or_nonfinite_evidence_is_rejected_instead_of_silently_dropped(self):
        for actual, prediction in (([], []), ([1.], [1., 2.]), ([True], [1.]),
                                   ([1.], [None]), ([1.], ['1']), ([1.], [float('inf')]),
                                   ([1.], [float('nan')]), ([10 ** 500], [1.])):
            with self.subTest(actual=repr(actual)), self.assertRaises(ValueError):
                error_profile(actual, prediction)
        for mutate in (
            lambda r: r['holdout']['predictions'].pop(),
            lambda r: r['historical_backtest']['windows'][0]['evidence'].pop(),
            lambda r: r['historical_backtest']['windows'][0]['evidence'][0].update(actual=float('nan')),
            lambda r: r['historical_backtest']['windows'][0].update(status='unknown'),
            lambda r: r['historical_backtest']['windows'].pop(),
            lambda r: r['task'].update(horizon=True),
        ):
            report = report_fixture()
            mutate(report)
            with self.assertRaises(ValueError): build_forecast_diagnostics(report)

    def test_web_and_cli_include_profiles_that_reconcile_with_verified_errors(self):
        data = daily_frame()
        semantics = confirmed(data)
        task = forecast_task_from_form(data, semantics, FORM | POLICY_FORM)
        report = run_confirmed_forecast(data, semantics, task)
        diagnostics = report['error_diagnostics']
        self.assertAlmostEqual(diagnostics['holdout']['model']['mae'], report['holdout']['metrics']['mae'])
        self.assertAlmostEqual(diagnostics['holdout']['model']['weighted_absolute_error'],
                               report['holdout']['metrics']['weighted_absolute_error'])
        self.assertNotIn('error_diagnostics', forecast_audit_view(report))
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'daily.csv'
            data.to_csv(source, index=False)
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = main(['--csv', str(source), '--target', 'cnt', '--time', 'dteday', '--horizon', '7'])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stream.getvalue())['error_diagnostics']['contract'], CONTRACT)

    def test_diagnostics_failure_cannot_return_a_partial_report(self):
        data = daily_frame()
        semantics = confirmed(data)
        task = forecast_task_from_form(data, semantics, FORM)
        with patch('vibedash.forecast_tasks.build_forecast_diagnostics', side_effect=ValueError('Controlled failure')):
            with self.assertRaises(ForecastTaskError): run_confirmed_forecast(data, semantics, task)


if __name__ == '__main__': unittest.main()
