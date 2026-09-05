import json
import unittest

import pandas as pd

from vibedash.generator_bridge import generate_dashboard_data
from vibedash.insight_engine import EvidenceBasedInsightEngine
from vibedash.spec import Chart, VizSpec


class EvidenceBasedInsightEngineTests(unittest.TestCase):
    def test_quality_metrics_are_exact_and_json_serializable(self):
        data = pd.DataFrame(
            {
                "value": [1.0, 1.0, None, 4.0],
                "group": ["A", "A", "B", "C"],
                "constant": ["same"] * 4,
            }
        )

        insights = EvidenceBasedInsightEngine(data).generate()
        quality = next(item for item in insights if item["insight_id"] == "data-quality")

        self.assertEqual(quality["metrics"]["missing_cells"], 1)
        self.assertEqual(quality["metrics"]["constant_column_count"], 1)
        self.assertEqual(quality["sample_size"], 4)
        json.dumps(insights)

    def test_strong_correlation_reports_pair_value_and_support(self):
        data = pd.DataFrame(
            {
                "feature_a": list(range(120)),
                "feature_b": [3 * value + 5 for value in range(120)],
            }
        )

        insights = EvidenceBasedInsightEngine(data).generate()
        correlation = next(
            item for item in insights if item["insight_id"] == "strongest-correlation"
        )

        self.assertAlmostEqual(correlation["metrics"]["pearson_r"], 1.0)
        self.assertEqual(correlation["sample_size"], 120)
        self.assertEqual(correlation["confidence"], "high")
        self.assertIn("not causation", correlation["recommendation"])

    def test_outliers_are_flagged_without_recommending_automatic_deletion(self):
        data = pd.DataFrame({"amount": list(range(1, 31)) + [1000]})

        insights = EvidenceBasedInsightEngine(data).generate()
        outliers = next(item for item in insights if item["insight_id"] == "potential-outliers")

        self.assertEqual(outliers["metrics"]["outlier_count"], 1)
        self.assertIn("Do not remove them automatically", outliers["recommendation"])

    def test_category_concentration_is_data_backed(self):
        data = pd.DataFrame({"segment": ["enterprise"] * 80 + ["smb"] * 20})

        insights = EvidenceBasedInsightEngine(data).generate()
        concentration = next(
            item for item in insights if item["insight_id"] == "category-concentration"
        )

        self.assertEqual(concentration["metrics"]["top_value"], "enterprise")
        self.assertEqual(concentration["metrics"]["top_share"], 0.8)

    def test_time_change_uses_early_and_late_windows(self):
        data = pd.DataFrame(
            {
                "event_date": pd.date_range("2026-01-01", periods=20, freq="D"),
                "revenue": list(range(100, 120)),
            }
        )

        insights = EvidenceBasedInsightEngine(data).generate()
        trend = next(item for item in insights if item["insight_id"] == "strongest-trend")

        self.assertEqual(trend["metrics"]["time_column"], "event_date")
        self.assertEqual(trend["metrics"]["period_count"], 20)
        self.assertGreater(trend["metrics"]["relative_change"], 0)

    def test_dashboard_contract_contains_insights_and_shape(self):
        data = pd.DataFrame({"amount": [10, 20, 30], "region": ["A", "A", "B"]})

        result = generate_dashboard_data(data, VizSpec(title="Test dashboard"))

        self.assertEqual(result["df_shape"], (3, 2))
        self.assertTrue(result["insights"])
        self.assertEqual(result["insights"][0]["type"], "evidence")
        self.assertIn("statistical_validation", result)
        self.assertEqual(result["statistical_validation"]["status"], "insufficient_data")
        self.assertIn("pattern_analysis", result)
        self.assertEqual(result["pattern_analysis"]["status"], "insufficient_data")

    def test_chart_errors_escape_untrusted_column_names(self):
        malicious_column = '<img src=x onerror="alert(1)">'
        data = pd.DataFrame({"safe": [1, 2, 3]})
        spec = VizSpec(
            title="Security test",
            charts=[Chart(type="hist", x=malicious_column, title="Invalid chart")],
        )

        result = generate_dashboard_data(data, spec)

        self.assertNotIn("<img", result["charts"][0]["html"])
        self.assertIn("&lt;img", result["charts"][0]["html"])


if __name__ == "__main__":
    unittest.main()
