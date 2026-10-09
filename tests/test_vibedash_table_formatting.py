import math
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from vibedash.generator_bridge import _format_audit_value, _generate_tables
from vibedash.spec import VizSpec


class VibeDashTableFormattingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "Plan": ["Growth", "Growth", "Starter"],
                "CustomerCount": [142, 68, 107],
                "MonthlyRevenue": [33_326.856, 7_210.1, 19_002.0],
                "ConversionRate": [0.0708, 0.052, 0.09125],
                "ChurnRate": [0.052, 0.063, float("nan")],
                "TicketCount": [13, 7, 11],
                "Renewed": [True, False, True],
            }
        )

    @staticmethod
    def _table_by_title(tables, title):
        return next(table for table in tables if table["title"] == title)

    def test_overview_formats_business_values_without_mutating_source(self):
        original = self.frame.copy(deep=True)

        tables = _generate_tables(self.frame, VizSpec(title="Audit table"))
        overview = self._table_by_title(tables, "Data Overview")
        first_row = dict(zip(overview["headers"], overview["rows"][0]))
        third_row = dict(zip(overview["headers"], overview["rows"][2]))

        self.assertEqual(first_row["CustomerCount"], "142")
        self.assertEqual(first_row["MonthlyRevenue"], "$33,326.86")
        self.assertEqual(first_row["ConversionRate"], "7.08%")
        self.assertEqual(first_row["ChurnRate"], "5.20%")
        self.assertEqual(first_row["TicketCount"], "13")
        self.assertEqual(first_row["Renewed"], "Yes")
        self.assertEqual(third_row["ChurnRate"], "—")
        assert_frame_equal(self.frame, original)

    def test_statistics_use_readable_precision_and_semantic_units(self):
        tables = _generate_tables(self.frame, VizSpec(title="Audit table"))
        statistics = self._table_by_title(
            tables,
            "Numeric Columns Statistics",
        )
        rows = {
            row[0]: dict(zip(statistics["headers"][1:], row[1:]))
            for row in statistics["rows"]
        }

        self.assertEqual(rows["count"]["CustomerCount"], "3")
        self.assertEqual(rows["count"]["ChurnRate"], "2")
        self.assertEqual(rows["mean"]["CustomerCount"], "105.67")
        self.assertEqual(rows["mean"]["MonthlyRevenue"], "$19,846.32")
        self.assertEqual(rows["mean"]["ConversionRate"], "7.13%")
        self.assertEqual(rows["mean"]["ChurnRate"], "5.75%")
        self.assertNotIn("333333", rows["mean"]["CustomerCount"])

    def test_category_counts_and_non_finite_values_are_review_safe(self):
        tables = _generate_tables(self.frame, VizSpec(title="Audit table"))
        plans = self._table_by_title(tables, "Top Values: Plan")

        self.assertEqual(plans["rows"][0], ["Growth", "2"])
        self.assertEqual(_format_audit_value(self.frame, None, math.inf), "—")
        self.assertEqual(_format_audit_value(self.frame, None, -math.inf), "—")


if __name__ == "__main__":
    unittest.main()
