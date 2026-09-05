import json
import unittest

import numpy as np
import pandas as pd

from vibedash.ai_analyzer import DataScienceAI
from vibedash.statistical_engine import (
    StatisticalValidationEngine,
    _benjamini_hochberg,
)


class StatisticalValidationEngineTests(unittest.TestCase):
    def test_group_difference_reports_effect_interval_and_adjusted_p_value(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(
            {
                "group": ["control"] * 50 + ["treatment"] * 50,
                "revenue": np.r_[
                    rng.normal(100, 8, 50),
                    rng.normal(125, 8, 50),
                ],
            }
        )

        report = StatisticalValidationEngine(data).analyze()
        result = next(item for item in report["tests"] if item["family"] == "group-comparison")

        self.assertEqual(report["correction"], "Benjamini-Hochberg false discovery rate (FDR)")
        self.assertTrue(result["significant"])
        self.assertEqual(result["effect_size"]["name"], "Hedges' g")
        self.assertEqual(result["effect_size"]["magnitude"], "large")
        self.assertNotEqual(result["confidence_interval"]["lower"], 0)
        self.assertGreaterEqual(result["adjusted_p_value"], result["raw_p_value"])
        json.dumps(report)

    def test_inconclusive_group_result_does_not_claim_equivalence(self):
        values = list(range(20))
        data = pd.DataFrame(
            {
                "group": ["A"] * 20 + ["B"] * 20,
                "value": values + values,
            }
        )

        report = StatisticalValidationEngine(data).analyze()
        result = report["tests"][0]

        self.assertFalse(result["significant"])
        self.assertIn("does not prove that the groups are equivalent", result["interpretation"])

    def test_correlation_reports_fisher_confidence_interval(self):
        x = np.linspace(0, 20, 80)
        y = 4 * x + np.sin(x)
        data = pd.DataFrame({"feature_a": x, "feature_b": y})

        report = StatisticalValidationEngine(data).analyze()
        result = report["tests"][0]

        self.assertEqual(result["family"], "correlation")
        self.assertTrue(result["significant"])
        self.assertEqual(result["confidence_interval"]["parameter"], "Pearson r")
        self.assertLessEqual(
            result["confidence_interval"]["lower"],
            result["effect_size"]["value"],
        )
        self.assertGreaterEqual(
            result["confidence_interval"]["upper"],
            result["effect_size"]["value"],
        )
        self.assertIn("does not imply causation", result["interpretation"])

    def test_benjamini_hochberg_preserves_order_and_never_reduces_p_values(self):
        raw = [0.04, 0.01, 0.03, 0.80]

        adjusted = _benjamini_hochberg(raw)

        self.assertEqual(len(adjusted), len(raw))
        self.assertEqual(adjusted[1], min(adjusted))
        for raw_value, adjusted_value in zip(raw, adjusted):
            self.assertGreaterEqual(adjusted_value, raw_value)

    def test_insufficient_data_returns_explanatory_json_report(self):
        data = pd.DataFrame({"value": [1, 2, 3]})

        report = StatisticalValidationEngine(data).analyze()

        self.assertEqual(report["status"], "insufficient_data")
        self.assertEqual(report["tests"], [])
        self.assertIn("No eligible hypotheses", report["summary"])
        json.dumps(report)

    def test_chat_statistical_analysis_uses_the_validated_report(self):
        data = pd.DataFrame(
            {
                "group": ["A"] * 20 + ["B"] * 20,
                "value": list(range(20)) + list(range(10, 30)),
            }
        )

        result = DataScienceAI(data).analyze_question("run statistical tests")

        self.assertIn("statistical_validation", result)
        self.assertIn("Benjamini-Hochberg", result["answer"])
        self.assertNotIn("p-value < 0.05 означает", result["answer"])
        self.assertTrue(any(item.get("type") == "evidence" for item in result["insights"]))


if __name__ == "__main__":
    unittest.main()
