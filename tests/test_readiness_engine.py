import json
import unittest

import numpy as np
import pandas as pd

from vibedash.readiness_engine import DatasetReadinessEngine


class DatasetReadinessEngineTests(unittest.TestCase):
    def test_clean_dataset_is_ready_with_full_score(self):
        data = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=40),
                "revenue": np.arange(40, dtype=float),
                "region": ["north", "south"] * 20,
            }
        )

        report = DatasetReadinessEngine(data).assess()

        self.assertEqual(report["contract"], "dataset-readiness-v1")
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["score"], 100)
        self.assertTrue(report["analysis_allowed"])
        self.assertEqual(report["counts"]["blocked"], 0)
        json.dumps(report)

    def test_small_constant_dataset_is_blocked_with_remediation(self):
        data = pd.DataFrame({"constant": [1] * 5})

        report = DatasetReadinessEngine(data).assess()

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["analysis_allowed"])
        self.assertGreaterEqual(report["counts"]["blocked"], 2)
        issues = [
            check for check in report["checks"] if check["status"] == "blocked"
        ]
        self.assertTrue(all(check["recommendation"] for check in issues))

    def test_quality_and_privacy_risks_produce_actionable_warnings(self):
        data = pd.DataFrame(
            {
                "customer_email": [f"user-{index}@example.com" for index in range(80)],
                "revenue": [float(index) if index % 2 else np.nan for index in range(80)],
                "region": ["north", "south"] * 40,
            }
        )

        report = DatasetReadinessEngine(data).assess()
        warnings = {
            check["check_id"]: check
            for check in report["checks"]
            if check["status"] == "warning"
        }

        self.assertEqual(report["status"], "ready_with_warnings")
        self.assertIn("missingness", warnings)
        self.assertIn("sensitive-schema", warnings)
        self.assertIn("high-cardinality-fields", warnings)
        self.assertLess(report["score"], 100)

    def test_duplicate_rows_can_block_materially_distorted_data(self):
        unique = pd.DataFrame({"value": range(10), "group": ["a", "b"] * 5})
        data = pd.concat([unique, unique, unique, unique], ignore_index=True)

        report = DatasetReadinessEngine(data).assess()
        duplicate_check = next(
            check for check in report["checks"] if check["check_id"] == "duplicate-rows"
        )

        self.assertEqual(duplicate_check["status"], "blocked")
        self.assertEqual(duplicate_check["metrics"]["duplicate_rate"], 0.75)
        self.assertEqual(report["status"], "blocked")

    def test_duplicate_column_names_are_blocked_without_crashing(self):
        data = pd.DataFrame(np.arange(80).reshape(40, 2), columns=["value", "value"])

        report = DatasetReadinessEngine(data).assess()
        identity = next(
            check for check in report["checks"] if check["check_id"] == "column-identity"
        )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(identity["status"], "blocked")

    def test_unique_date_dimension_is_not_treated_as_an_identifier(self):
        data = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=80).astype(str),
                "revenue": range(80),
            }
        )

        report = DatasetReadinessEngine(data).assess()
        high_cardinality = next(
            check
            for check in report["checks"]
            if check["check_id"] == "high-cardinality-fields"
        )

        self.assertEqual(high_cardinality["status"], "passed")


if __name__ == "__main__":
    unittest.main()
