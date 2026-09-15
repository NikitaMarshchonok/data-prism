import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from vibedash.period_comparison import PeriodComparisonError, build_period_comparison


class PeriodComparisonTests(unittest.TestCase):
    def periods(self, n=30):
        return (
            pd.DataFrame({"value": np.arange(n, dtype=float), "group": ["a", "b"] * (n // 2)}),
            pd.DataFrame({"value": np.arange(n, dtype=float) + 2, "group": ["a", "b"] * (n // 2)}),
        )

    def test_contract_and_strict_json(self):
        baseline, current = self.periods()
        report = build_period_comparison(baseline, current)
        self.assertEqual(report["contract"], "period-comparison-v1")
        json.dumps(report, allow_nan=False)
        self.assertLessEqual(report["numeric_metrics"]["display_count"], 8)

    def test_identical_is_stable_and_deterministic(self):
        baseline, _ = self.periods()
        left = build_period_comparison(baseline, baseline.copy())
        right = build_period_comparison(baseline, baseline.copy())
        self.assertEqual(left, right)
        self.assertEqual(left["status"], "stable")

    def test_schema_changes(self):
        baseline, current = self.periods()
        current["new"] = 1
        report = build_period_comparison(baseline, current)
        self.assertIn("new", report["schema_changes"]["new_columns"])
        self.assertEqual(report["status"], "review_required")

    def test_insufficient_and_zero_baseline(self):
        baseline = pd.DataFrame({"x": [0] * 10, "y": range(10)})
        current = pd.DataFrame({"x": range(10), "y": range(10, 20)})
        report = build_period_comparison(baseline, current)
        metrics = {m["column"]: m for m in report["numeric_metrics"]["metrics"]}
        self.assertIsNone(metrics["x"]["percent_change"])
        self.assertEqual(metrics["x"]["status"], "ok")

    def test_validation_and_readiness(self):
        baseline, current = self.periods()
        with self.assertRaises(ValueError):
            build_period_comparison(baseline, current, max_metrics=True)
        with self.assertRaises(ValueError):
            build_period_comparison(baseline, current, baseline_label="bad\nlabel")
        with self.assertRaises(PeriodComparisonError) as error:
            build_period_comparison(baseline.iloc[:2], current)
        self.assertIn("readiness", error.exception.reason)

    def test_strong_numeric_and_categorical_change_requires_review(self):
        baseline = pd.DataFrame({"value": np.arange(40), "segment": ["old"] * 40})
        current = pd.DataFrame({"value": np.arange(40) + 100, "segment": ["new"] * 40})
        report = build_period_comparison(baseline, current)
        self.assertEqual(report["status"], "review_required")
        self.assertTrue(report["distribution_drift"]["signals"])

    def test_delta_and_ci_direction(self):
        baseline = pd.DataFrame({"value": np.arange(20, dtype=float)})
        current = pd.DataFrame({"value": np.arange(20, dtype=float) + 5})
        metric = build_period_comparison(baseline, current)["numeric_metrics"]["metrics"][0]
        self.assertEqual(metric["estimate"]["direction"], "current_minus_baseline")
        self.assertEqual(metric["absolute_change"], 5.0)
        self.assertLess(metric["confidence_interval"]["lower"], 5)
        self.assertGreater(metric["confidence_interval"]["upper"], 5)

    def test_bh_is_applied_before_display(self):
        baseline = pd.DataFrame({f"x{i}": np.arange(40) for i in range(4)})
        current = pd.DataFrame({f"x{i}": np.arange(40) + (20 if i == 0 else 0) for i in range(4)})
        report = build_period_comparison(baseline, current, max_metrics=1)
        block = report["numeric_metrics"]
        self.assertEqual(block["tested_count"], 4)
        self.assertEqual(block["display_count"], 1)
        self.assertTrue(all(m["adjusted_p_value"] is not None for m in block["metrics"]))
        self.assertTrue(all(m["raw_p_value"] <= m["adjusted_p_value"] for m in block["metrics"]))

    def test_fdr_significance_uses_full_precision_before_report_rounding(self):
        baseline = pd.DataFrame({"value": np.arange(20, dtype=float)})
        current = pd.DataFrame({"value": np.arange(20, dtype=float) + 1})
        # The displayed p-value rounds to 0.05, but its actual value is just
        # above the alpha threshold and must not be marked significant.
        with patch(
            "vibedash.period_comparison.stats.ttest_ind",
            return_value=SimpleNamespace(pvalue=0.050000004, df=20.0),
        ):
            metric = build_period_comparison(baseline, current)["numeric_metrics"]["metrics"][0]
        self.assertEqual(metric["raw_p_value"], 0.05)
        self.assertFalse(metric["significant"])

    def test_removed_and_type_change_are_review_required(self):
        baseline = pd.DataFrame({"value": np.arange(20), "removed": np.arange(20)})
        current = pd.DataFrame({"value": np.arange(20) + 1, "removed": ["x"] * 20})
        report = build_period_comparison(baseline, current)
        self.assertEqual(report["status"], "review_required")
        self.assertNotEqual(report["status"], "blocked")

    def test_one_zero_variance_group_is_valid_welch(self):
        baseline = pd.DataFrame({"value": [1] * 10, "varying": np.arange(10)})
        current = pd.DataFrame({"value": np.arange(10), "varying": np.arange(10)})
        metric = {m["column"]: m for m in build_period_comparison(baseline, current)["numeric_metrics"]["metrics"]}["value"]
        self.assertEqual(metric["status"], "ok")
        self.assertIsInstance(metric["raw_p_value"], float)
        self.assertIsNotNone(metric["confidence_interval"])

    def test_both_constant_is_insufficient(self):
        baseline = pd.DataFrame({"value": [1] * 10, "varying": np.arange(10)})
        current = pd.DataFrame({"value": [2] * 10, "varying": np.arange(10)})
        metric = {m["column"]: m for m in build_period_comparison(baseline, current)["numeric_metrics"]["metrics"]}["value"]
        self.assertEqual(metric["status"], "insufficient_evidence")

    def test_negative_baseline_percent_is_null_with_reason(self):
        baseline = pd.DataFrame({"value": [-5] * 10, "varying": np.arange(10)})
        current = pd.DataFrame({"value": [-3] * 10, "varying": np.arange(10)})
        metric = {m["column"]: m for m in build_period_comparison(baseline, current)["numeric_metrics"]["metrics"]}["value"]
        self.assertIsNone(metric["percent_change"])
        self.assertIn("strictly positive", metric["percent_change_reason"])

    def test_labels_are_distinct_after_normalization(self):
        baseline, current = self.periods()
        with self.assertRaises(ValueError):
            build_period_comparison(baseline, current, baseline_label=" Baseline ", current_label="baseline")

    def test_non_dataframes_and_bounds(self):
        baseline, current = self.periods()
        with self.assertRaises(TypeError):
            build_period_comparison([], current)
        with patch("vibedash.period_comparison.MAX_ROWS", 2):
            with self.assertRaises(ValueError):
                build_period_comparison(baseline, current)
        with patch("vibedash.period_comparison.MAX_COLUMNS", 1):
            with self.assertRaises(ValueError):
                build_period_comparison(baseline, current)

    def test_memory_bound_and_nonfinite_values_are_bounded(self):
        baseline, current = self.periods()
        with patch("vibedash.period_comparison.MAX_MEMORY_BYTES", 1):
            with self.assertRaises(ValueError):
                build_period_comparison(baseline, current)

        baseline = pd.DataFrame({"value": [1, 2, np.inf, 4, 5, 6, 7, 8]})
        current = pd.DataFrame({"value": [2, 3, 4, 5, 6, 7, 8, 9]})
        metric = build_period_comparison(baseline, current)["numeric_metrics"]["metrics"][0]
        self.assertEqual(metric["baseline"]["n"], 7)
        self.assertEqual(metric["baseline"]["missing_rate"], 0.125)

    def test_combined_resource_bounds_are_enforced(self):
        baseline = pd.DataFrame({"value": np.arange(20)})
        current = pd.DataFrame({"value": np.arange(20) + 1})
        with patch("vibedash.period_comparison.MAX_TOTAL_ROWS", 30):
            with self.assertRaises(ValueError):
                build_period_comparison(baseline, current)
        with patch("vibedash.period_comparison.MAX_TOTAL_COLUMNS", 1):
            with self.assertRaises(ValueError):
                build_period_comparison(baseline, current)
        each = int(
            baseline.memory_usage(index=True, deep=True).sum()
        )
        with patch("vibedash.period_comparison.MAX_MEMORY_BYTES", each + 1):
            with self.assertRaises(ValueError):
                build_period_comparison(baseline, current)

    def test_complex_numeric_columns_are_not_coerced_to_real(self):
        baseline = pd.DataFrame({'value': [complex(i, i) for i in range(20)]})
        current = pd.DataFrame({'value': [complex(i + 1, i + 1) for i in range(20)]})
        report = build_period_comparison(baseline, current)
        self.assertEqual(report['numeric_metrics']['eligible_count'], 0)

    def test_non_string_column_keys_are_normalized_positionally(self):
        # Pandas treats True and 1 as equal during label lookup even though
        # their string names are distinct and the readiness contract permits
        # both columns.
        baseline = pd.DataFrame(
            np.column_stack([np.arange(20, dtype=float), np.arange(20, dtype=float) + 100]),
            columns=[True, 1],
        )
        current = baseline.copy()
        current.iloc[:, 0] += 2
        current.iloc[:, 1] += 3

        report = build_period_comparison(baseline, current)

        metrics = {metric['column']: metric for metric in report['numeric_metrics']['metrics']}
        self.assertEqual(set(metrics), {'True', '1'})
        self.assertEqual(metrics['True']['absolute_change'], 2.0)
        self.assertEqual(metrics['1']['absolute_change'], 3.0)

    def test_categorical_values_are_not_returned(self):
        sentinel = "RAW_SECRET_SENTINEL"
        baseline = pd.DataFrame({"value": np.arange(20), "category": [sentinel] * 20})
        current = pd.DataFrame({"value": np.arange(20), "category": ["other"] * 20})
        report = build_period_comparison(baseline, current)
        self.assertNotIn(sentinel, json.dumps(report))

    def test_inferential_candidate_cap_and_counts(self):
        baseline = pd.DataFrame({f"x{i}": np.arange(40) + i for i in range(34)})
        current = pd.DataFrame({f"x{i}": np.arange(40) + i + 1 for i in range(34)})
        block = build_period_comparison(baseline, current)["numeric_metrics"]
        self.assertEqual(block["eligible_count"], 34)
        self.assertEqual(block["candidate_count"], 32)
        self.assertEqual(block["truncated_count"], 2)

    def test_inferential_cap_bounds_actual_welch_calls(self):
        baseline = pd.DataFrame({f"x{i}": np.arange(40, dtype=float) for i in range(40)})
        current = pd.DataFrame({f"x{i}": np.arange(40, dtype=float) + 1 for i in range(40)})
        with patch(
            "vibedash.period_comparison.stats.ttest_ind",
            wraps=__import__("scipy").stats.ttest_ind,
        ) as ttest:
            block = build_period_comparison(baseline, current)["numeric_metrics"]
        self.assertEqual(block["candidate_count"], 32)
        self.assertEqual(ttest.call_count, 32)

    def test_candidate_cap_does_not_hide_later_testable_metrics(self):
        baseline = pd.DataFrame({f"constant{i}": [1] * 40 for i in range(32)})
        current = pd.DataFrame({f"constant{i}": [2] * 40 for i in range(32)})
        baseline["important"] = np.arange(40, dtype=float)
        current["important"] = baseline["important"] + 100

        block = build_period_comparison(baseline, current)["numeric_metrics"]

        self.assertEqual(block["candidate_count"], 32)
        self.assertEqual(block["tested_count"], 1)
        self.assertIn("important", {metric["column"] for metric in block["metrics"]})

    def test_duplicate_normalized_headers_are_readiness_error(self):
        baseline = pd.DataFrame([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]], columns=["value", " value "])
        current = baseline.copy()
        with self.assertRaises(PeriodComparisonError):
            build_period_comparison(baseline, current)


if __name__ == "__main__":
    unittest.main()
