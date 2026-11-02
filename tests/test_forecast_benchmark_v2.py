"""The extended references must not rewrite the original evaluation protocol."""
import json
import unittest
from unittest.mock import patch

from src.forecasting import CANDIDATES, CANDIDATE_SUITE, BASELINES, LEGACY_CANDIDATE_SUITE
from src.forecast_benchmark import baseline_predictions, evaluate_window, generate_series, run_forecast_benchmark
from tests.test_forecast_benchmark import protocol, stub_engine


def extended_stub(frame, task, *, candidate_suite):
    report = stub_engine(frame, task)
    report['candidate_suite'] = candidate_suite
    if candidate_suite == CANDIDATE_SUITE:
        report['model_comparison'] = [
            {'model_name': name, 'status': 'ok', 'selected': index == 0,
             'mean_mae': float(index + 1), 'mean_objective_loss': float(index + 1),
             'fold_metrics': [{'mae': float(index + 1)}] * 3}
            for index, name in enumerate(CANDIDATES)]
    return report


class ExtendedForecastBenchmarkTests(unittest.TestCase):
    def test_independent_mean_and_median_reference_oracles(self):
        prediction = baseline_predictions([1., 2., 3., 100.] * 2, 3, 10, candidate_suite=CANDIDATE_SUITE)
        self.assertEqual(prediction['historical_mean'], [26.5] * 3)
        self.assertEqual(prediction['historical_median'], [2.5] * 3)
        with self.assertRaises(ValueError):
            baseline_predictions([1.] * 7, 3, 0, candidate_suite='unknown')
        with self.assertRaises(ValueError):
            baseline_predictions([1.] * 4, 3, 0, candidate_suite=CANDIDATE_SUITE)

    def test_both_protocols_keep_identical_sources_dates_cutoffs_and_all_scheduled_cases(self):
        old_config = protocol()
        new_config = old_config | {'contract': 'forecast-benchmark-v2'}
        self.assertEqual(old_config['contract'], 'forecast-benchmark-v1')
        with patch('src.forecast_benchmark.forecast_series', side_effect=extended_stub):
            old = run_forecast_benchmark(old_config)
            new = run_forecast_benchmark(new_config)
        self.assertEqual(old['candidate_suite'], LEGACY_CANDIDATE_SUITE)
        self.assertEqual(new['candidate_suite'], CANDIDATE_SUITE)
        self.assertEqual(len(old['candidates']), 4)
        self.assertEqual(len(new['candidates']), 6)
        self.assertNotEqual(old['protocol_sha256'], new['protocol_sha256'])
        self.assertEqual(new['protocol_integrity'], 'passed')
        self.assertEqual(old['totals']['scheduled_windows'], new['totals']['scheduled_windows'])
        for first, second in zip(old['series'], new['series']):
            self.assertEqual(first['data_sha256'], second['data_sha256'])
            for before, after in zip(first['windows'], second['windows']):
                for key in ('input_sha256', 'forecast_origin', 'available_through', 'dates', 'actual'):
                    self.assertEqual(before[key], after[key], key)
                for name in ('last_value', 'seasonal_naive'):
                    self.assertEqual(before['baselines'][name], after['baselines'][name])
                self.assertEqual(set(after['baselines']), set(BASELINES))

    def test_v2_rejects_legacy_output_instead_of_silently_downgrading(self):
        config = protocol() | {'contract': 'forecast-benchmark-v2'}
        data = generate_series('trend', 17, config)
        with patch('src.forecast_benchmark.forecast_series', side_effect=stub_engine):
            row = evaluate_window(data, config, 0, 2)
        self.assertEqual(row['status'], 'error')
        self.assertEqual(row['forecast'], [])
        self.assertEqual(set(row['baselines']), set(BASELINES))

    def test_new_constant_reference_can_be_selected_without_changing_outer_scoring(self):
        config = protocol() | {'contract': 'forecast-benchmark-v2', 'initial_history_days': 80,
                               'horizon_days': 2, 'outer_windows': 1, 'reporting_delays': [0], 'seeds': [17]}
        def choose_median(frame, task, *, candidate_suite):
            report = extended_stub(frame, task, candidate_suite=candidate_suite)
            report['model_name'] = report['baseline_name'] = 'historical_median'
            report['model_comparison'][0]['selected'] = False
            row = next(row for row in report['model_comparison'] if row['model_name'] == 'historical_median')
            row.update(selected=True, mean_mae=.1, mean_objective_loss=.1, fold_metrics=[{'mae': .1}] * 3)
            for prediction in report['future_forecast']: prediction['prediction'] = float(frame.target.median())
            return report
        with patch('src.forecast_benchmark.forecast_series', side_effect=choose_median):
            report = run_forecast_benchmark(config)
        self.assertEqual(report['protocol_integrity'], 'passed')
        self.assertEqual(report['totals']['vs_validation_selected_baseline']['tie'], 7)
        self.assertEqual(report['predictive_acceptance'], 'not_established')
        json.dumps(report, allow_nan=False)


if __name__ == '__main__': unittest.main()
