"""Outer evaluation plumbing, independent arithmetic and preserved failures."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluate_forecast_benchmark import DEFAULT_CONFIG, main
from src.forecast_benchmark import (
    CANDIDATES, SCENARIOS, baseline_predictions, compare_mae, evaluate_window,
    generate_series, run_forecast_benchmark, score_predictions, validate_benchmark_config,
)
from src.forecasting import LEGACY_CANDIDATE_SUITE


def protocol():
    return json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))


def stub_engine(data, task, **kwargs):
    """Plumbing fixture, never used as evidence of actual algorithm performance."""
    return {
        'contract': 'daily-forecast-v3', 'status': 'ok',
        'candidate_suite': LEGACY_CANDIDATE_SUITE,
        'future_forecast_origin': task.forecast_origin,
        'future_fit_through': data.date.iloc[-1],
        'availability': {'history_mode': 'available_history_only',
                         'reporting_delay_days': task.reporting_delay_days},
        'model_name': 'last_value', 'baseline_name': 'last_value',
        'future_forecast': [{'date': date, 'prediction': float(data.target.iloc[-1])}
                            for date in pd.date_range(pd.Timestamp(task.forecast_origin)
                                + pd.Timedelta(days=1), periods=task.horizon).strftime('%Y-%m-%d')],
        'model_comparison': [
            {'model_name': name, 'status': 'ok', 'mean_mae': float(index + 1),
             'mean_objective_loss': float(index + 1),
             'fold_metrics': [{'mae': float(index + 1)}] * 3, 'selected': index == 0}
            for index, name in enumerate(CANDIDATES)]}


class ForecastBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise all generators with the real candidate selector, with a small
        # valid protocol. The full checked-in protocol is run separately by CI.
        cls.small = protocol() | {'seeds': [17], 'initial_history_days': 80,
                                  'horizon_days': 2, 'outer_windows': 1, 'reporting_delays': [0]}
        cls.real_report = run_forecast_benchmark(cls.small)

    def test_real_engine_completes_all_scenarios_without_claiming_acceptance(self):
        report = self.real_report
        self.assertEqual(report['protocol_integrity'], 'passed', report['totals'])
        self.assertEqual(report['predictive_acceptance'], 'not_established')
        self.assertEqual(report['series_count'], 7)
        self.assertEqual([row['scenario'] for row in report['series']], list(SCENARIOS))
        self.assertEqual(report['totals']['scheduled_windows'], 7)
        constant = report['series'][0]['windows'][0]
        self.assertEqual(constant['model_metrics'], {'mae': 0., 'rmse': 0.})
        self.assertIsNone(constant['selected_baseline_comparison']['relative_mae_improvement'])
        self.assertEqual(report['series'][1]['windows'][0]['engine']['model_name'], 'seasonal_naive')
        self.assertEqual(set(report['implementation']['source_sha256']),
                         {'forecast_benchmark.py', 'forecast_scoring.py', 'forecasting.py', 'forecast_policy.py'})
        json.dumps(report, allow_nan=False)

    def test_independent_error_oracle_and_zero_reference_semantics(self):
        self.assertEqual(score_predictions([1., 5.], [3., 1.]), {'mae': 3., 'rmse': 10 ** .5})
        self.assertEqual(compare_mae(2., 1.)['outcome'], 'worse')
        self.assertEqual(compare_mae(0., 1.)['relative_mae_improvement'], 1.)
        zero = compare_mae(1., 0.)
        self.assertEqual(zero['outcome'], 'worse')
        self.assertIsNone(zero['relative_mae_improvement'])
        self.assertFalse(zero['relative_improvement_defined'])
        for actual, prediction in (([], []), ([1], []), ([True], [0.]),
                                   ([1.], [np.inf]), ([np.nan], [1.]), ([1.], ['1'])):
            with self.subTest(actual=actual, prediction=prediction), self.assertRaises(ValueError):
                score_predictions(actual, prediction)

    def test_baseline_oracle_bridges_delay_longer_than_season(self):
        history = [10., 20., 30., 40., 50., 60., 70.]
        predicted = baseline_predictions(history, 4, 10)
        self.assertEqual(predicted['last_value'], [70.] * 4)
        self.assertEqual(predicted['seasonal_naive'], [40., 50., 60., 70.])

    def test_configuration_rejects_dropped_cases_ambiguous_values_and_unbounded_work(self):
        changes = [{'extra': 1}, {'contract': 'v0'}, {'generator_version': True},
                   {'scenarios': list(SCENARIOS[:-1])}, {'seeds': []}, {'seeds': [17, 17]},
                   {'seeds': [True]}, {'seeds': [2**32]}, {'reporting_delays': [0, 0]},
                   {'reporting_delays': [31]}, {'reporting_delays': [False]},
                   {'outer_windows': True}, {'horizon_days': 0}, {'initial_history_days': 10000},
                   {'initial_history_days': 60, 'horizon_days': 14, 'reporting_delays': [30]},
                   {'seeds': [1, 2, 3, 4, 5], 'outer_windows': 5, 'reporting_delays': [0, 2, 4]}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_benchmark_config(protocol() | change)
        for value in (None, [], {}):
            with self.assertRaises(ValueError): validate_benchmark_config(value)
        original = protocol()
        validated = validate_benchmark_config(original)
        original['seeds'].append(999)
        self.assertNotIn(999, validated['seeds'])

    def test_frozen_schedule_inputs_and_scores_have_no_outer_actuals(self):
        config = protocol()
        data = generate_series('trend', 17, config)
        calls = []
        def observe(frame, task, **kwargs):
            calls.append((frame.copy(), task))
            return stub_engine(frame, task)
        with patch('src.forecast_benchmark.forecast_series', side_effect=observe):
            result = evaluate_window(data, config, 0, 2)
            changed = data.copy()
            # Change unavailable gap AND all outer outcomes. Neither is an input.
            changed.loc[140:, 'target'] += 10000.
            after = evaluate_window(changed, config, 0, 2)
        self.assertEqual(result['history_rows'], 140)
        self.assertEqual(calls[0][0].to_dict(), calls[1][0].to_dict())
        self.assertEqual(calls[0][1].forecast_origin, data.date.iloc[141])
        self.assertEqual(calls[0][0].date.iloc[-1], data.date.iloc[139])
        self.assertEqual(result['engine'], after['engine'])
        self.assertEqual(result['forecast'], after['forecast'])
        self.assertEqual(result['input_sha256'], after['input_sha256'])
        self.assertNotEqual(result['model_metrics'], after['model_metrics'])
        for row in (result, after):
            self.assertEqual(row['model_metrics'], score_predictions(row['actual'], row['forecast']))

    def test_real_forecaster_does_not_see_mutated_outer_outcomes(self):
        data = generate_series('trend', 17, self.small)
        before = evaluate_window(data, self.small, 0, 0)
        data.loc[80:, 'target'] += 10000.
        after = evaluate_window(data, self.small, 0, 0)
        self.assertEqual(before['status'], 'scored')
        self.assertEqual(before['engine'], after['engine'])
        self.assertEqual(before['forecast'], after['forecast'])
        self.assertNotEqual(before['model_metrics'], after['model_metrics'])

    def test_all_72_scheduled_runs_retained_and_same_origins_across_delays(self):
        with patch('src.forecast_benchmark.forecast_series', side_effect=stub_engine):
            report = run_forecast_benchmark(protocol())
            repeat = run_forecast_benchmark(protocol())
        self.assertEqual(report, repeat)
        self.assertEqual(report['series_count'], 12)  # exact oracles not duplicated by seed
        self.assertEqual(report['totals']['scheduled_windows'], 72)
        self.assertEqual(report['totals']['scored_windows'], 72)
        for series in report['series']:
            windows = series['windows']
            self.assertEqual(len(windows), 6)
            self.assertEqual([row['forecast_origin'] for row in windows[:3]],
                             [row['forecast_origin'] for row in windows[3:]])
            for index in range(2):
                self.assertLess(windows[index]['dates'][-1], windows[index + 1]['dates'][0])
                self.assertEqual(windows[index + 1]['history_rows'] - windows[index]['history_rows'], 7)

    def test_baseline_reference_uses_validation_not_best_outer_error_and_losses_survive(self):
        config = protocol()
        data = generate_series('weekly_exact', None, config)
        def worse_model(frame, task, **kwargs):
            report = stub_engine(frame, task)
            report['model_name'] = 'lag_ridge'
            report['model_comparison'][0]['selected'] = False
            report['model_comparison'][2].update(selected=True, mean_mae=.5, mean_objective_loss=.5,
                                                   fold_metrics=[{'mae': .5}] * 3)
            for row in report['future_forecast']: row['prediction'] = 1000.
            return report
        with patch('src.forecast_benchmark.forecast_series', side_effect=worse_model):
            window = evaluate_window(data, config, 0, 0)
            report = run_forecast_benchmark(config)
        self.assertEqual(window['status'], 'scored')
        self.assertEqual(window['engine']['baseline_name'], 'last_value')
        self.assertEqual(window['baselines']['seasonal_naive']['metrics']['mae'], 0.)
        self.assertGreater(window['baselines']['last_value']['metrics']['mae'], 0.)
        self.assertEqual(window['selected_baseline_comparison']['outcome'], 'worse')
        self.assertEqual(report['protocol_integrity'], 'passed')
        self.assertEqual(report['totals']['vs_validation_selected_baseline']['worse'], 72)
        self.assertEqual(report['predictive_acceptance'], 'not_established')

    def test_engine_failures_are_recorded_without_dropping_other_windows(self):
        count = 0
        def failing(frame, task, **kwargs):
            nonlocal count
            count += 1
            if count == 1: raise RuntimeError('controlled failure')
            if count == 2: return {'status': 'error', 'message': 'No forecast.'}
            return stub_engine(frame, task)
        with patch('src.forecast_benchmark.forecast_series', side_effect=failing):
            report = run_forecast_benchmark(protocol())
        self.assertEqual(count, 72)
        self.assertEqual(report['status'], 'completed_with_failures')
        self.assertEqual(report['totals']['failed_windows'], 2)
        self.assertEqual(report['totals']['scored_windows'], 70)
        failed = report['series'][0]['windows'][0]
        self.assertEqual(failed['error']['message'], 'controlled failure')
        self.assertEqual(set(failed['baselines']), {'last_value', 'seasonal_naive'})
        json.dumps(report, allow_nan=False)

    def test_candidate_failure_stays_visible_even_if_selected_method_completes(self):
        def failed_candidate(frame, task, **kwargs):
            report = stub_engine(frame, task)
            report['model_comparison'][-1] = {'model_name': 'lag_random_forest', 'status': 'error',
                                             'selected': False, 'reason': 'controlled candidate failure'}
            return report
        with patch('src.forecast_benchmark.forecast_series', side_effect=failed_candidate):
            report = run_forecast_benchmark(self.small)
        self.assertEqual(report['protocol_integrity'], 'failed')
        self.assertEqual(report['totals']['scored_windows'], 7)
        self.assertEqual(report['totals']['candidate_failures'], 7)
        row = report['series'][0]['windows'][0]
        self.assertEqual(row['status'], 'scored_with_candidate_failures')
        self.assertEqual(row['engine']['candidates'][-1]['reason'], 'controlled candidate failure')

    def test_malformed_predictions_or_selection_cannot_pass(self):
        config = protocol()
        data = generate_series('trend', 17, config)
        def mutate(kind):
            def altered(frame, task, **kwargs):
                report = stub_engine(frame, task)
                if kind == 'nan': report['future_forecast'][0]['prediction'] = float('nan')
                if kind == 'bool': report['future_forecast'][0]['prediction'] = True
                if kind == 'missing_date': report['future_forecast'].pop()
                if kind == 'wrong_date': report['future_forecast'][0]['date'] = '2000-01-01'
                if kind == 'wrong_cutoff': report['future_fit_through'] = task.forecast_origin
                if kind == 'wrong_selection': report['model_name'] = 'lag_ridge'
                if kind == 'wrong_baseline': report['baseline_name'] = 'seasonal_naive'
                if kind == 'missing_candidate': report['model_comparison'].pop()
                if kind == 'bad_mean': report['model_comparison'][0]['mean_mae'] = .1
                if kind == 'negative_score': report['model_comparison'][0]['fold_metrics'][0]['mae'] = -1.
                return report
            return altered
        for kind in ('nan', 'bool', 'missing_date', 'wrong_date', 'wrong_cutoff', 'wrong_selection',
                     'wrong_baseline', 'missing_candidate', 'bad_mean', 'negative_score'):
            with self.subTest(kind=kind), patch('src.forecast_benchmark.forecast_series', side_effect=mutate(kind)):
                row = evaluate_window(data, config, 0, 2)
                self.assertEqual(row['status'], 'error')
                self.assertEqual(row['forecast'], [])
                self.assertIsNone(row['model_metrics'])
                json.dumps(row, allow_nan=False)

    def test_cli_writes_same_report_and_separates_execution_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            for integrity, code in (('passed', 0), ('failed', 2)):
                report = copy.deepcopy(self.real_report) | {'protocol_integrity': integrity}
                stream = io.StringIO()
                with patch('evaluate_forecast_benchmark.run_forecast_benchmark', return_value=report), redirect_stdout(stream):
                    self.assertEqual(main(['--output', str(path)]), code)
                self.assertEqual(json.loads(stream.getvalue()), json.loads(path.read_text()))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config', str(Path(directory) / 'missing.json')]), 1)


if __name__ == '__main__': unittest.main()
