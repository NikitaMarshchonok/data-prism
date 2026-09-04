import unittest
from unittest.mock import patch

import pandas as pd

from vibedash.generator_bridge import (
    UnsafeExpressionError,
    _apply_filters,
    _evaluate_filter_condition,
    _evaluate_metric,
)
from vibedash.spec import Filter


class SafeFilterDslTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "Revenue": [1000, 6000, 8000, 9000],
                "Region": ["APAC", "EMEA", "NA", "EMEA"],
                "Sales Amount": [10, 20, 30, 40],
            }
        )

    def test_combined_filter_expression(self):
        filters = [
            Filter(
                field="Region",
                where="Revenue > 5000 and Region == 'EMEA'",
            )
        ]

        result = _apply_filters(self.data, filters)

        self.assertEqual(result["Revenue"].tolist(), [6000, 9000])

    def test_membership_and_backtick_column_filter(self):
        mask = _evaluate_filter_condition(
            self.data,
            "`Sales Amount` >= 20 and Region in ['EMEA', 'NA']",
        )

        self.assertEqual(self.data.loc[mask, "Revenue"].tolist(), [6000, 8000, 9000])

    @patch("os.system")
    def test_filter_rejects_python_code_without_executing_it(self, system):
        filters = [
            Filter(
                field="Revenue",
                where="__import__('os').system('echo compromised') or Revenue > 0",
            )
        ]

        result = _apply_filters(self.data, filters)

        system.assert_not_called()
        pd.testing.assert_frame_equal(result, self.data)


class SafeMetricDslTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            {
                "Revenue": [100.0, 200.0, 300.0],
                "Sales Amount": [10.0, 20.0, 30.0],
                "Region": ["APAC", "EMEA", "EMEA"],
            }
        )

    def test_allowed_aggregates_and_arithmetic(self):
        result = _evaluate_metric(self.data, "sum(Revenue) / count()")

        self.assertEqual(result, 200.0)
        self.assertEqual(_evaluate_metric(self.data, "nunique(Region)"), 2)
        self.assertEqual(_evaluate_metric(self.data, "mean(`Sales Amount`)"), 20.0)

    @patch("os.system")
    def test_metric_rejects_python_code_without_executing_it(self, system):
        expression = "__import__('os').system('echo compromised')"

        with self.assertRaises(UnsafeExpressionError):
            _evaluate_metric(self.data, expression)

        system.assert_not_called()

    def test_metric_rejects_attribute_access(self):
        with self.assertRaises(UnsafeExpressionError):
            _evaluate_metric(self.data, "(1).__class__")


if __name__ == "__main__":
    unittest.main()
