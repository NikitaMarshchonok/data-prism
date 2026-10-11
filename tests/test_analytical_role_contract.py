import unittest

import numpy as np
import pandas as pd

from src.data_drift import create_baseline_profile
from src.ml_predictor import _prepare_features
from vibedash.anomaly_segmentation_engine import AnomalySegmentationEngine
from src.column_semantics import profile_dataframe_columns
from vibedash.period_comparison import build_period_comparison
from vibedash.statistical_engine import StatisticalValidationEngine


def _bike_like_frame(rows: int = 80) -> pd.DataFrame:
    index = np.arange(1, rows + 1)
    return pd.DataFrame(
        {
            "instant": index,
            "dteday": pd.date_range("2025-01-01", periods=rows, freq="D").astype(str),
            "season": (index % 4) + 1,
            "yr": index % 2,
            "mnth": (index % 12) + 1,
            "holiday": (index % 11 == 0).astype(int),
            "weekday": index % 7,
            "workingday": (index % 7 < 5).astype(int),
            "weathersit": (index % 3) + 1,
            "temp": 0.3 + index * 0.002,
            "atemp": 0.32 + index * 0.0021,
            "hum": 0.5 + np.sin(index / 8) * 0.1,
            "windspeed": 0.2 + np.cos(index / 9) * 0.03,
            "casual": 100 + index * 3,
            "registered": 500 + index * 9,
            "cnt": 600 + index * 12,
        }
    )


class AnalyticalRoleContractTests(unittest.TestCase):
    def test_nullable_numeric_dtypes_keep_roles_and_missingness(self):
        data = pd.DataFrame(
            {
                "row_id": pd.Series([1, 2, 3, 4], dtype="Int64"),
                "region_code": pd.Series([1, 2, 1, pd.NA], dtype="Int64"),
                "revenue": pd.Series([10.5, pd.NA, 12.5, 13.5], dtype="Float64"),
            }
        )

        profiles = profile_dataframe_columns(data)

        self.assertEqual(profiles["row_id"]["role"], "identifier")
        self.assertEqual(profiles["region_code"]["role"], "category")
        self.assertEqual(profiles["revenue"]["role"], "measure")
        self.assertEqual(profiles["revenue"]["finite_count"], 3)

    def test_explicit_category_name_wins_over_short_sequential_values(self):
        profiles = profile_dataframe_columns(
            pd.DataFrame({"season": [1, 2, 3, 4], "value": [1, 2, 3, 4]})
        )

        self.assertEqual(profiles["season"]["role"], "category")
        self.assertEqual(profiles["value"]["role"], "measure")

    def test_bike_schema_has_one_shared_role_assignment(self):
        profiles = profile_dataframe_columns(_bike_like_frame())

        self.assertEqual(profiles["instant"]["role"], "identifier")
        self.assertEqual(profiles["dteday"]["role"], "temporal")
        for column in ("season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"):
            self.assertEqual(profiles[column]["role"], "category")
        for column in ("temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"):
            self.assertEqual(profiles[column]["role"], "measure")

    def test_statistics_and_pattern_engines_never_treat_codes_as_measures(self):
        data = _bike_like_frame()
        forbidden = {
            "instant", "dteday", "season", "yr", "mnth", "holiday",
            "weekday", "workingday", "weathersit",
        }

        statistics = StatisticalValidationEngine(data).analyze(max_results=100)
        correlations = [
            test for test in statistics["tests"] if test["family"] == "correlation"
        ]
        self.assertTrue(correlations)
        self.assertTrue(
            all(not (set(test["columns"]) & forbidden) for test in correlations)
        )

        patterns = AnomalySegmentationEngine(data).analyze()
        self.assertEqual(patterns["status"], "ok")
        self.assertFalse(set(patterns["features"]) & forbidden)
        self.assertIn("cnt", patterns["features"])

    def test_drift_and_period_comparison_use_roles_not_storage_dtype(self):
        data = _bike_like_frame()
        baseline = data.iloc[:40].copy()
        baseline["yr"] = 0
        profile = create_baseline_profile(baseline)

        self.assertEqual(profile["columns"]["instant"]["feature_type"], "identifier")
        self.assertEqual(profile["columns"]["dteday"]["feature_type"], "temporal")
        self.assertEqual(profile["columns"]["season"]["feature_type"], "categorical")
        self.assertEqual(profile["columns"]["yr"]["feature_type"], "categorical")
        self.assertEqual(profile["columns"]["cnt"]["feature_type"], "numeric")

        comparison = build_period_comparison(data.iloc[:40], data.iloc[40:])
        compared = {
            metric["column"]
            for metric in comparison["numeric_metrics"]["metrics"]
        }
        self.assertFalse(
            compared
            & {
                "instant", "dteday", "season", "yr", "mnth", "holiday",
                "weekday", "workingday", "weathersit",
            }
        )

    def test_ml_drops_identifiers_and_dates_and_encodes_numeric_categories(self):
        data = _bike_like_frame()
        prepared, dropped = _prepare_features(
            data.drop(columns=["cnt"]),
            data["cnt"],
        )

        self.assertIn("instant", dropped)
        self.assertIn("dteday", dropped)
        self.assertEqual(prepared["season"].dtype, object)
        self.assertEqual(prepared["workingday"].dtype, object)
        self.assertTrue(pd.api.types.is_numeric_dtype(prepared["temp"]))


if __name__ == "__main__":
    unittest.main()
