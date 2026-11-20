"""Validation-only model ranking: independent daily errors and source cutoffs."""
import copy
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from evaluate_forecast import main
from src.forecasting import ForecastTask, _forecast, forecast_series, forecast_task_settings, LEGACY_CANDIDATE_SUITE
from src.forecast_result import validate_forecast_result
from src.forecast_selection_result import CONTRACT, validate_forecast_selection
from tests.test_forecast_tasks import daily_frame, confirmed, FORM
from tests.test_forecast_policy import POLICY
from vibedash.forecast_tasks import forecast_task_from_form, run_confirmed_forecast, ForecastTaskError, forecast_audit_view


TASK = ForecastTask('cnt', 'dteday', 7)


class ForecastSelectionIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = daily_frame()
        cls.report = forecast_series(cls.frame, TASK)
        cls.settings = forecast_task_settings(TASK)

    def check(self, report):
        return validate_forecast_selection(report, self.frame, self.settings)

    def test_real_selection_is_recomputed_without_fitting_or_changing_the_report(self):
        before = copy.deepcopy(self.report)
        with patch('src.forecasting._forecast') as fit:
            result = self.check(self.report)
        fit.assert_not_called()
        self.assertEqual(before, self.report)
        self.assertEqual(result['contract'], CONTRACT)
        self.assertEqual(result['checked_candidates'], 6)
        self.assertEqual(result['checked_folds'], 3)
        self.assertEqual(result['checked_prediction_rows'], 126)
        self.assertEqual(result['production_readiness'], 'not_established')
        json.dumps(result, allow_nan=False)

    def test_corrupted_folds_facts_scores_and_candidate_catalogue_cannot_pass(self):
        mutations = (
            lambda r: r['validation_folds'][0].update(train_end=r['holdout']['start']),
            lambda r: r['validation_folds'][0].update(train_rows=True),
            lambda r: r['validation_folds'][0].update(validation_start='2000-01-01'),
            lambda r: r['validation_folds'].reverse(),
            lambda r: r.update(cv_folds=True),
            lambda r: r['model_comparison'][0]['fold_predictions'][0][0].update(actual=12345.),
            lambda r: r['model_comparison'][0]['fold_predictions'][0][0].update(prediction=12345.),
            lambda r: r['model_comparison'][0]['fold_predictions'][0].pop(),
            lambda r: r['model_comparison'][0]['fold_predictions'][0].reverse(),
            lambda r: r['model_comparison'][0]['fold_metrics'][0].update(mae=0.),
            lambda r: r['model_comparison'][0].update(mean_mae=0.),
            lambda r: r['model_comparison'][0].update(mean_objective_loss=-1e-20),
            lambda r: r['model_comparison'].pop(),
            lambda r: r['model_comparison'].reverse(),
            lambda r: r['model_comparison'][0].update(selected=True),
            lambda r: r['model_comparison'][0].update(selected=1),
            lambda r: r.update(baseline_candidates=['last_value']),
            lambda r: r.update(candidate_suite='unknown'),
            lambda r: r.pop('validation_evidence_contract'),
        )
        for index, mutate in enumerate(mutations):
            report = copy.deepcopy(self.report)
            mutate(report)
            with self.subTest(index=index), self.assertRaises(ValueError): self.check(report)

    def test_coherent_counterfeit_baseline_errors_still_fail_prefix_reconstruction(self):
        report = copy.deepcopy(self.report)
        row = report['model_comparison'][0]
        for predictions, metrics in zip(row['fold_predictions'], row['fold_metrics']):
            for value in predictions: value['prediction'] = value['actual']
            metrics.update(mae=0., rmse=0.)
        row.update(mean_mae=0., mean_objective_loss=0.)
        for item in report['model_comparison']: item['selected'] = item is row
        report.update(model_name='last_value', baseline_name='last_value')
        with self.assertRaises(ValueError): self.check(report)

    def test_holdout_outcomes_and_reported_gains_cannot_choose_a_different_method(self):
        report = copy.deepcopy(self.report)
        report['holdout']['metrics'].update(mae=0., rmse=0.)
        report['holdout']['mae_improvement'] = 1.
        self.check(report)  # Selection verifier deliberately ignores final outcomes.
        report['model_name'] = 'lag_random_forest'
        for row in report['model_comparison']:
            row['selected'] = row['model_name'] == 'lag_random_forest'
        with self.assertRaises(ValueError): self.check(report)

    def test_declared_asymmetric_loss_changes_ranking_and_mae_ranking_is_rejected(self):
        data = self.frame.assign(cnt=np.tile([8., 12.], 80))
        def controlled(history, horizon, period, candidate):
            if candidate not in ('lag_ridge', 'lag_random_forest'):
                return _forecast(history, horizon, period, candidate)
            upcoming = np.array([8. if (len(history) + step) % 2 == 0 else 12. for step in range(horizon)])
            return upcoming - 1.5 if candidate == 'lag_ridge' else upcoming + 2.
        with patch('src.forecasting._forecast', side_effect=controlled):
            plain = forecast_series(data, TASK)
            weighted = forecast_series(data, TASK, decision_policy=POLICY)
        self.assertEqual(plain['model_name'], 'lag_ridge')
        self.assertEqual(weighted['model_name'], 'lag_random_forest')
        validate_forecast_result(plain, data, self.settings)
        result = validate_forecast_result(weighted, data, self.settings, decision_policy=POLICY)
        self.assertEqual(result['validation_selection']['objective'], 'weighted_absolute_error')
        weighted['model_name'] = 'lag_ridge'
        for row in weighted['model_comparison']: row['selected'] = row['model_name'] == 'lag_ridge'
        with self.assertRaises(ValueError):
            validate_forecast_result(weighted, data, self.settings, decision_policy=POLICY)

    def test_delay_longer_than_horizon_and_legacy_suite_preserve_cutoffs_and_order(self):
        for origin in (None, '2020-06-16'):
            task = ForecastTask('cnt', 'dteday', 3, 3, 8, origin)
            report = forecast_series(self.frame, task, candidate_suite=LEGACY_CANDIDATE_SUITE)
            result = validate_forecast_result(report, self.frame.iloc[::-1], forecast_task_settings(task))
            self.assertEqual(result['validation_selection']['checked_candidates'], 4)
            report['validation_folds'][-1]['train_end'] = report['validation_folds'][-1]['forecast_origin']
            with self.assertRaises(ValueError):
                validate_forecast_result(report, self.frame, forecast_task_settings(task))

    def test_exact_ties_keep_first_simple_baseline_and_failed_candidates_cannot_be_selected(self):
        flat = self.frame.assign(cnt=10.)
        report = forecast_series(flat, TASK)
        self.assertEqual(report['model_name'], 'last_value')
        validate_forecast_result(report, flat, self.settings)
        report['model_name'] = report['baseline_name'] = 'seasonal_naive'
        for row in report['model_comparison']: row['selected'] = row['model_name'] == 'seasonal_naive'
        with self.assertRaises(ValueError): validate_forecast_result(report, flat, self.settings)
        def fail_ridge(history, horizon, period, candidate):
            if candidate == 'lag_ridge': raise ValueError('controlled failure')
            return _forecast(history, horizon, period, candidate)
        with patch('src.forecasting._forecast', side_effect=fail_ridge):
            report = forecast_series(self.frame, TASK)
        self.check(report)
        failed = next(row for row in report['model_comparison'] if row['status'] == 'error')
        self.assertEqual(failed['fold_predictions'], [])
        failed['selected'] = True
        with self.assertRaises(ValueError): self.check(report)

    def test_nonfinite_rows_incomplete_evidence_and_cached_badges_fail_closed(self):
        for value in (True, None, '0', float('nan'), float('inf'), 10 ** 500):
            report = copy.deepcopy(self.report)
            report['model_comparison'][0]['fold_predictions'][0][0]['prediction'] = value
            report['result_integrity'] = {'status': 'consistent', 'validation_selection': {'status': 'consistent'}}
            with self.subTest(value=repr(value)), self.assertRaises(ValueError): self.check(report)
        for report in (None, {}, [], {'validation_evidence_contract': 'forecast-validation-evidence-v1'}):
            with self.assertRaisesRegex(ValueError, '^The forecast validation or selection evidence is incomplete or inconsistent.$'):
                self.check(report)

    def test_finite_extreme_constant_does_not_overflow_the_independent_median(self):
        for value in (1e308, sys.float_info.max):
            data = self.frame.assign(cnt=value)
            with patch('src.forecasting._fit_lag_model', side_effect=ValueError('Controlled unsupported scale')):
                report = forecast_series(data, TASK)
            self.assertEqual(report['status'], 'ok')
            validate_forecast_result(report, data, self.settings)

    def test_web_rechecks_selection_and_audit_keeps_only_aggregate_counts(self):
        semantics = confirmed(self.frame)
        task = forecast_task_from_form(self.frame, semantics, FORM)
        corrupt = copy.deepcopy(self.report)
        corrupt['model_comparison'][0]['mean_objective_loss'] = 0.
        with patch('vibedash.forecast_tasks.forecast_series', return_value=corrupt), \
                patch('vibedash.forecast_tasks.historical_forecast_backtest') as backtest:
            with self.assertRaises(ForecastTaskError): run_confirmed_forecast(self.frame, semantics, task)
            backtest.assert_not_called()
        report = run_confirmed_forecast(self.frame, semantics, task)
        audit = forecast_audit_view(report)
        self.assertEqual(audit['result_integrity']['validation_selection']['status'], 'consistent')
        self.assertNotIn('fold_predictions', json.dumps(audit))
        self.assertNotIn('actual', json.dumps(audit))

    def test_cli_refuses_forged_validation_before_model_creation(self):
        report = copy.deepcopy(self.report)
        report['model_comparison'][0]['fold_predictions'][0][0]['actual'] += 1.
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'series.csv'
            self.frame.to_csv(source, index=False)
            output = io.StringIO()
            with patch('evaluate_forecast.forecast_series', return_value=report), \
                    patch('evaluate_forecast.create_artifact') as create, redirect_stdout(output):
                code = main(['--csv', str(source), '--target', 'cnt', '--time', 'dteday',
                             '--horizon', '7', '--save-model', str(Path(directory) / 'model.json')])
            create.assert_not_called()
            self.assertEqual(code, 1)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload['status'], 'error')
            self.assertNotIn('fold_predictions', payload)


if __name__ == '__main__': unittest.main()
