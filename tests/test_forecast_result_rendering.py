"""Forecast presentation must not promote training or unknown status to success."""
import copy
import unittest
from unittest.mock import patch

from tests import test_vibedash_standalone_export as export_fixture
from tests.test_forecast_tasks import FORM, confirmed, daily_frame
from tests.test_forecast_policy import POLICY_FORM
from vibedash.forecast_tasks import forecast_task_from_form, run_confirmed_forecast
from src.forecasting import ForecastTask
from src.forecast_backtest import historical_forecast_backtest


class ForecastResultRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        export_fixture.StandaloneExportTests.setUpClass.__func__(cls)
        frame = daily_frame()
        semantics = confirmed(frame)
        cls.forecast = run_confirmed_forecast(
            frame, semantics, forecast_task_from_form(frame, semantics, FORM),
        )

    def render_forecast(self, report):
        with patch.dict(self.dashboard, forecast=report):
            return export_fixture.StandaloneExportTests.render(self)

    def test_each_support_status_has_an_explicit_non_production_explanation(self):
        headings = {
            'baseline_selected': 'Simple baseline retained',
            'no_holdout_improvement': 'No improvement over the baseline',
            'observed_holdout_improvement': 'Lower MAE on this holdout — exploratory result',
            'unknown': 'Forecast support not established',
        }
        for status, heading in headings.items():
            with self.subTest(status=status):
                report = copy.deepcopy(self.forecast)
                report['predictive_support']['status'] = status
                html = self.render_forecast(report)
                self.assertIn(heading, html)
                self.assertIn('Not approved for operational use', html)
                if status != 'observed_holdout_improvement':
                    self.assertNotIn('Lower MAE on this holdout', html)

    def test_negative_count_forecast_is_warned_and_future_table_is_not_shown(self):
        report = copy.deepcopy(self.forecast)
        report['domain_constraints']['status'] = 'failed'
        html = self.render_forecast(report)
        self.assertIn('Forecast violates the confirmed count domain', html)
        self.assertIn('Final holdout predictions', html)
        self.assertNotIn('Exploratory future point forecast', html)

    def test_undefined_and_negative_improvement_are_not_positive_accuracy_claims(self):
        report = copy.deepcopy(self.forecast)
        report['holdout']['mae_improvement'] = None
        self.assertIn('Not defined', self.render_forecast(report))
        report['holdout']['mae_improvement'] = -.063
        report['task']['target_column'] = '<script>untrusted</script>'
        html = self.render_forecast(report)
        self.assertIn('-6.3%', html)
        self.assertIn('&lt;script&gt;untrusted&lt;/script&gt;', html)
        self.assertNotIn('<script>untrusted</script>', html)
        self.assertIn('not accuracy percent', html)

    def test_decision_failure_is_visible_even_when_mae_improved_and_text_is_escaped(self):
        frame = daily_frame()
        semantics = confirmed(frame)
        fields = POLICY_FORM | {'forecast_decision': '<script>untrusted decision</script>',
                                'forecast_maximum_mae': '0'}
        report = run_confirmed_forecast(frame, semantics, forecast_task_from_form(frame, semantics, FORM | fields))
        self.assertEqual(report['predictive_support']['status'], 'observed_holdout_improvement')
        self.assertEqual(report['business_evaluation']['status'], 'criteria_not_met_on_holdout')
        html = self.render_forecast(report)
        self.assertIn('Decision criteria not met or not established', html)
        self.assertIn('Absolute MAE limit: Not met', html)
        self.assertIn('&lt;script&gt;untrusted decision&lt;/script&gt;', html)
        self.assertNotIn('<script>untrusted decision</script>', html)
        self.assertIn('not financial savings', html)

    def test_rolling_evaluation_has_accessible_window_table_and_no_accuracy_claim(self):
        html = self.render_forecast(self.forecast)
        self.assertIn('Historical rolling-origin evaluation', html)
        self.assertIn('Per-window errors and retained unsuccessful results', html)
        self.assertIn('aria-label="Historical forecast comparison windows" tabindex="0"', html)
        self.assertIn('scope="col"', html)
        self.assertIn('scope="row"', html)
        self.assertIn('3 of 3 windows scored', html)
        self.assertIn('not an accuracy percentage', html)
        self.assertIn('This does not reselect the final future model', html)

    def test_short_history_and_failed_windows_are_not_rendered_as_success(self):
        report = copy.deepcopy(self.forecast)
        report['historical_backtest'] = historical_forecast_backtest(daily_frame().iloc[:70], ForecastTask('cnt', 'dteday', 7))
        html = self.render_forecast(report)
        self.assertIn('Not evaluated — insufficient history', html)
        self.assertIn('Required: 79 daily rows; supplied: 70', html)
        self.assertIn('Not scored', html)
        report = copy.deepcopy(self.forecast)
        backtest = report['historical_backtest']
        backtest['status'] = 'completed_with_issues'
        backtest['summary'].update(failed_windows=1, scored_windows=2)
        backtest['windows'][0].update(status='error', metrics=None, baseline_metrics=None, comparison=None)
        html = self.render_forecast(report)
        self.assertIn('Evaluation has issues: 1 failed windows', html)
        self.assertIn('Evaluation failed', html)
        self.assertIn('Not scored', html)

    def test_window_labels_are_escaped_and_weighted_errors_are_visible(self):
        frame = daily_frame()
        semantics = confirmed(frame)
        report = run_confirmed_forecast(frame, semantics, forecast_task_from_form(frame, semantics, FORM | POLICY_FORM))
        report['historical_backtest']['windows'][0]['model_name'] = '<script>untrusted method</script>'
        html = self.render_forecast(report)
        self.assertIn('&lt;script&gt;untrusted method&lt;/script&gt;', html)
        self.assertNotIn('<script>untrusted method</script>', html)
        self.assertIn('Weighted loss:', html)
        self.assertIn('weighted absolute error under the declared preferences', html)


if __name__ == '__main__':
    unittest.main()
