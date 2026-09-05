import unittest
from unittest.mock import patch

import pandas as pd

from src.dashboard_generator import generate_dashboard_data


class DashboardGeneratorTests(unittest.TestCase):
    @patch("src.dashboard_generator.generate_ai_summary_openai", return_value="summary")
    @patch(
        "src.dashboard_generator.predict_target",
        return_value={
            "target_col": "target",
            "task_type": "classification",
            "model_name": "LogisticRegression",
            "metric": "Accuracy: 100%",
            "model_comparison": [
                {
                    "model_name": "LogisticRegression",
                    "mean_score": 0.95,
                    "std_score": 0.02,
                    "status": "ok",
                    "selected": True,
                }
            ],
            "cv_folds": 5,
            "cv_metric": "balanced accuracy; higher is better",
            "diagnostics": {"type": "classification", "per_class_metrics": []},
            "reliability": {"status": "ok", "scope": "random holdout reliability audit"},
            "feature_importance_method": "holdout permutation importance",
            "feature_importance_plot": None,
        },
    )
    def test_selected_target_is_passed_to_ml(self, predict_target, _ai_summary):
        data = pd.DataFrame(
            {
                "feature": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = generate_dashboard_data(data, target_column="target")

        predict_target.assert_called_once_with(data, "target")
        ml_card = result[-1]
        self.assertEqual(ml_card["model_name"], "LogisticRegression")
        self.assertEqual(ml_card["cv_folds"], 5)
        self.assertEqual(len(ml_card["model_comparison"]), 1)
        self.assertEqual(ml_card["diagnostics"]["type"], "classification")
        self.assertEqual(ml_card["reliability"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
