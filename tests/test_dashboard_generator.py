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
            "metric": "Accuracy: 100%",
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

        generate_dashboard_data(data, target_column="target")

        predict_target.assert_called_once_with(data, "target")


if __name__ == "__main__":
    unittest.main()
