import unittest

import numpy as np
import pandas as pd

from src.model_reliability import analyze_model_reliability


class ModelReliabilityTests(unittest.TestCase):
    def test_numeric_split_shift_is_flagged_without_calling_it_production_drift(self):
        train = pd.DataFrame({"value": np.linspace(0, 1, 80)})
        test = pd.DataFrame({"value": np.linspace(5, 6, 20)})

        report = analyze_model_reliability(
            train, test, pd.Series(np.linspace(0, 1, 80)), pd.Series(np.linspace(5, 6, 20)),
            np.linspace(5, 6, 20), "regression"
        )

        self.assertEqual(report["feature_stability"][0]["severity"], "high")
        self.assertIn("not production drift monitoring", report["methodology_notes"][0])

    def test_categorical_shift_uses_total_variation_distance(self):
        train = pd.DataFrame({"region": ["A"] * 72 + ["B"] * 8})
        test = pd.DataFrame({"region": ["A"] * 2 + ["B"] * 18})

        report = analyze_model_reliability(
            train, test, pd.Series([0] * 40 + [1] * 40), pd.Series([0] * 10 + [1] * 10),
            np.array([0] * 10 + [1] * 10), "classification"
        )

        stability = report["feature_stability"][0]
        self.assertEqual(stability["method"], "total variation distance")
        self.assertEqual(stability["severity"], "high")

    def test_classification_subgroup_gap_identifies_worst_group(self):
        test = pd.DataFrame({"segment": ["A"] * 10 + ["B"] * 10})
        target = pd.Series([0, 1] * 10)
        predictions = np.r_[target.iloc[:10], 1 - target.iloc[10:]]

        report = analyze_model_reliability(
            pd.DataFrame({"segment": ["A", "B"] * 40}), test,
            pd.Series([0, 1] * 40), target, predictions, "classification"
        )

        gap = report["subgroup_performance"][0]
        self.assertTrue(gap["material_gap"])
        self.assertEqual(gap["worst_group"], "B")
        self.assertEqual(gap["absolute_gap"], 1.0)

    def test_small_subgroups_are_not_reported(self):
        test = pd.DataFrame({"segment": ["A"] * 9 + ["rare"]})
        target = pd.Series(range(10), dtype=float)

        report = analyze_model_reliability(
            pd.DataFrame({"segment": ["A", "rare"] * 20}), test,
            pd.Series(range(40), dtype=float), target, target.to_numpy(), "regression"
        )

        self.assertEqual(report["subgroup_performance"], [])


if __name__ == "__main__":
    unittest.main()
