"""Forecast presentation must not promote training or unknown status to success."""
import copy
import unittest
from unittest.mock import patch

from tests import test_vibedash_standalone_export as export_fixture
from tests.test_forecast_tasks import FORM, confirmed, daily_frame
from vibedash.forecast_tasks import forecast_task_from_form, run_confirmed_forecast


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


if __name__ == '__main__':
    unittest.main()
