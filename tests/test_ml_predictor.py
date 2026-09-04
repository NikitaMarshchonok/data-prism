import unittest

import numpy as np
import pandas as pd

from src.ml_predictor import predict_target


class ModelEvaluationTests(unittest.TestCase):
    def test_classification_handles_categories_missing_values_and_leakage(self):
        row_count = 120
        target = pd.Series(["yes" if index % 3 == 0 else "no" for index in range(row_count)])
        data = pd.DataFrame(
            {
                "customer_id": list(range(1000, 1000 + row_count)),
                "age": [20 + index % 40 for index in range(row_count)],
                "region": ["north", "south", None, "west"] * 30,
                "copied_target": target,
                "converted": target,
            }
        )

        result = predict_target(data, "converted")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["task_type"], "classification")
        self.assertIn("baseline_accuracy", result["metrics"])
        self.assertIn("customer_id", result["dropped_features"])
        self.assertIn("copied_target", result["dropped_features"])
        self.assertGreater(result["train_rows"], result["test_rows"])
        self.assertTrue(result["feature_importance_plot"].startswith("data:image/png;base64,"))

    def test_regression_reports_baseline_and_multiple_metrics(self):
        rng = np.random.default_rng(42)
        feature = np.linspace(0, 20, 160)
        data = pd.DataFrame(
            {
                "feature": feature,
                "segment": ["A", "B"] * 80,
                "target": feature * 4.0 + rng.normal(0, 0.5, len(feature)),
            }
        )

        result = predict_target(data, "target")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["task_type"], "regression")
        self.assertIn("mae", result["metrics"])
        self.assertIn("rmse", result["metrics"])
        self.assertIn("r2", result["metrics"])
        self.assertIn("baseline_mae", result["metrics"])
        self.assertGreater(result["metrics"]["mae_improvement"], 0)

    def test_too_small_dataset_returns_explanatory_error(self):
        data = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 0]})

        result = predict_target(data, "target")

        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["feature_importance_plot"])
        self.assertIn("минимум", result["metric"])


if __name__ == "__main__":
    unittest.main()
