import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data_drift import (
    compare_to_baseline,
    create_baseline_profile,
    load_baseline_profile,
    save_baseline_profile,
)


class DataDriftTests(unittest.TestCase):
    def test_profile_contains_aggregates_without_raw_rows(self):
        data = pd.DataFrame(
            {
                "amount": list(range(20)),
                "region": ["north", "south"] * 10,
            }
        )

        profile = create_baseline_profile(data, dataset_name="reference.csv")

        self.assertEqual(profile["row_count"], 20)
        self.assertEqual(profile["dataset_name"], "reference.csv")
        self.assertIn("bin_proportions", profile["columns"]["amount"])
        self.assertIn("category_proportions", profile["columns"]["region"])
        self.assertNotIn("data", profile)
        self.assertNotIn("values", profile["columns"]["amount"])

    def test_detects_numeric_and_categorical_distribution_drift(self):
        baseline = pd.DataFrame(
            {
                "amount": list(range(100)),
                "region": ["north", "south"] * 50,
            }
        )
        current = pd.DataFrame(
            {
                "amount": list(range(1000, 1100)),
                "region": ["north"] * 100,
            }
        )

        report = compare_to_baseline(current, create_baseline_profile(baseline))
        by_feature = {item["feature"]: item for item in report["feature_drift"]}

        self.assertEqual(report["status"], "critical")
        self.assertEqual(by_feature["amount"]["metric"], "PSI")
        self.assertEqual(by_feature["amount"]["severity"], "high")
        self.assertEqual(by_feature["region"]["metric"], "TV distance")
        self.assertEqual(by_feature["region"]["severity"], "high")

    def test_identical_batch_is_reported_as_stable(self):
        data = pd.DataFrame(
            {
                "amount": list(range(100)),
                "region": ["north", "south"] * 50,
            }
        )

        report = compare_to_baseline(data.copy(), create_baseline_profile(data))

        self.assertEqual(report["status"], "stable")
        self.assertTrue(all(item["score"] == 0 for item in report["feature_drift"]))

    def test_reports_schema_and_type_changes(self):
        baseline = pd.DataFrame(
            {
                "amount": list(range(20)),
                "region": ["north", "south"] * 10,
                "removed": [1] * 20,
            }
        )
        current = pd.DataFrame(
            {
                "amount": [f"item-{index}" for index in range(20)],
                "region": ["north", "south"] * 10,
                "new_feature": list(range(20)),
            }
        )

        report = compare_to_baseline(current, create_baseline_profile(baseline))

        self.assertEqual(report["schema_changes"]["missing_columns"], ["removed"])
        self.assertEqual(report["schema_changes"]["new_columns"], ["new_feature"])
        self.assertEqual(
            report["schema_changes"]["type_changes"],
            [
                {
                    "feature": "amount",
                    "baseline_type": "numeric",
                    "current_type": "categorical",
                }
            ],
        )
        self.assertEqual(report["status"], "critical")

    def test_missing_rate_change_contributes_to_severity(self):
        baseline = pd.DataFrame({"value": list(range(20))})
        current = pd.DataFrame({"value": [None] * 10 + list(range(10))})

        report = compare_to_baseline(current, create_baseline_profile(baseline))
        drift = report["feature_drift"][0]

        self.assertEqual(drift["missing_rate_delta"], 0.5)
        self.assertEqual(drift["severity"], "high")

    def test_profile_round_trip_is_validated(self):
        profile = create_baseline_profile(
            pd.DataFrame({"value": list(range(20))}),
            dataset_name="reference.csv",
        )

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "baseline.json"
            save_baseline_profile(profile, path)
            loaded = load_baseline_profile(path)

        self.assertEqual(loaded, profile)

    def test_unsupported_profile_version_is_rejected(self):
        profile = create_baseline_profile(pd.DataFrame({"value": list(range(20))}))
        profile["profile_version"] = 999

        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "baseline.json"
            path.write_text(json.dumps(profile), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_baseline_profile(path)

    def test_duplicate_string_column_names_are_rejected(self):
        data = pd.DataFrame([[1, 2]], columns=[1, "1"])

        with self.assertRaisesRegex(ValueError, "must be unique"):
            create_baseline_profile(data)


if __name__ == "__main__":
    unittest.main()
