import unittest
from unittest.mock import patch

import pandas as pd

from vibedash.generator_bridge import (
    _generate_ai_summary,
    _generate_kpis,
    _generate_tables,
)
from vibedash.spec import Chart, Filter, Metric, VizSpec, parse_prompt_to_viz_spec


PILOT_COLUMNS = [
    "Date",
    "Region",
    "Plan",
    "AcquisitionChannel",
    "CustomerCount",
    "MonthlyRevenue",
    "ConversionRate",
    "ChurnRate",
    "TicketCount",
    "NPSScore",
    "Renewed",
]


class VibeDashAutomaticSemanticsTests(unittest.TestCase):
    @patch("vibedash.spec._improve_with_ollama", return_value=None)
    def test_automatic_spec_uses_bounded_semantic_metrics(self, _ollama):
        spec = parse_prompt_to_viz_spec(
            "Сравни динамику выручки, клиентов, конверсии и оттока",
            PILOT_COLUMNS,
        )

        self.assertLessEqual(len(spec.metrics), 8)
        self.assertEqual(
            len({metric.expr for metric in spec.metrics}),
            len(spec.metrics),
        )
        metrics = {metric.expr: metric for metric in spec.metrics}
        self.assertEqual(metrics["sum(MonthlyRevenue)"].fmt, "currency")
        self.assertEqual(metrics["mean(MonthlyRevenue)"].fmt, "currency")
        self.assertEqual(metrics["mean(ConversionRate)"].fmt, "percent")
        self.assertEqual(metrics["mean(ChurnRate)"].fmt, "percent")
        self.assertEqual(metrics["mean(NPSScore)"].title, "Average NPS Score")
        self.assertNotIn("Total Amount", [metric.title for metric in spec.metrics])
        self.assertNotIn("Average Value", [metric.title for metric in spec.metrics])

        frame = pd.DataFrame(
            {
                "CustomerCount": [100, 140],
                "MonthlyRevenue": [10_000.0, 14_000.0],
                "ConversionRate": [0.03, 0.05],
                "ChurnRate": [0.07, 0.05],
                "TicketCount": [9, 11],
                "NPSScore": [60, 70],
            }
        )
        rendered = {
            kpi["title"]: kpi["value"]
            for kpi in _generate_kpis(frame, spec.metrics)
        }
        self.assertEqual(rendered["Average Monthly Revenue"], "$12,000.00")
        self.assertEqual(rendered["Average Conversion Rate"], "4.0%")

    @patch("vibedash.spec._improve_with_ollama", return_value=None)
    def test_automatic_spec_never_invents_gauge_targets_or_date_rankings(self, _ollama):
        spec = parse_prompt_to_viz_spec(
            "Покажи динамику и различия между регионами и тарифами",
            PILOT_COLUMNS,
        )

        self.assertFalse(any(chart.type == "gauge" for chart in spec.charts))
        self.assertFalse(
            any(
                chart.type in {"bar", "pie"}
                and chart.x == "Date"
                and not chart.y
                for chart in spec.charts
            )
        )
        trend = next(chart for chart in spec.charts if chart.type == "line")
        self.assertEqual(trend.x, "Date")
        self.assertEqual(trend.y, "MonthlyRevenue")
        grouped = {chart.group: chart for chart in spec.charts if chart.group}
        self.assertEqual(grouped["Region"].agg, "sum")
        self.assertEqual(grouped["Region"].y, "MonthlyRevenue")
        self.assertEqual(grouped["Plan"].agg, "sum")
        self.assertEqual(spec.filters[0].field, "Region")

    @patch("vibedash.spec._improve_with_ollama")
    def test_llm_output_is_sanitized_before_rendering(self, improve):
        improve.return_value = VizSpec(
            title="Unsafe automatic dashboard",
            metrics=[
                Metric(title="Rate", expr="mean(ConversionRate)", fmt="number"),
                Metric(title="Rate duplicate", expr="mean(ConversionRate)", fmt="number"),
            ],
            charts=[
                Chart(type="gauge", y="CustomerCount", agg="mean"),
                Chart(
                    type="bar",
                    x="Date",
                    y="CustomerCount",
                    agg="count",
                    top=10,
                    title="Top 10 Date",
                ),
                Chart(type="hist", x="CustomerCount"),
            ],
        )

        spec = parse_prompt_to_viz_spec("Analyze", PILOT_COLUMNS)

        self.assertEqual(len(spec.metrics), 1)
        self.assertEqual(spec.metrics[0].fmt, "percent")
        self.assertEqual([chart.type for chart in spec.charts], ["hist"])

    @patch("vibedash.spec._improve_with_ollama", return_value=None)
    def test_category_only_dataset_does_not_fail_generation(self, _ollama):
        spec = parse_prompt_to_viz_spec(
            "Покажи распределение по регионам",
            ["Date", "Region", "Plan"],
        )

        self.assertEqual(spec.metrics, [])
        self.assertTrue(spec.charts)
        self.assertFalse(any(chart.x == "Date" for chart in spec.charts))
        self.assertEqual(spec.filters[0].field, "Region")

    def test_audit_tables_skip_dates_and_near_unique_labels(self):
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2026-01-01", periods=30).astype(str),
                "ExternalReference": [f"row-{index}" for index in range(30)],
                "Region": ["Europe", "North America"] * 15,
                "Plan": ["Starter", "Growth", "Enterprise"] * 10,
                "MonthlyRevenue": range(30),
            }
        )

        tables = _generate_tables(frame, VizSpec(title="Audit"))
        titles = {table["title"] for table in tables}

        self.assertIn("Top Values: Region", titles)
        self.assertIn("Top Values: Plan", titles)
        self.assertNotIn("Top Values: Date", titles)
        self.assertNotIn("Top Values: ExternalReference", titles)

    def test_summary_does_not_claim_an_empty_filter_was_applied(self):
        frame = pd.DataFrame({"Region": ["Europe", "North America"]})
        configured = VizSpec(
            title="Regions",
            filters=[Filter(field="Region")],
        )
        applied = VizSpec(
            title="Regions",
            filters=[Filter(field="Region", values=["Europe"])],
        )

        configured_summary = _generate_ai_summary(frame, configured)
        applied_summary = _generate_ai_summary(frame, applied)

        self.assertIn("Configured 1 filter field", configured_summary)
        self.assertNotIn("Applied 1 filters", configured_summary)
        self.assertIn("Applied 1 filters", applied_summary)

    @patch("vibedash.spec._improve_with_ollama", return_value=None)
    def test_dataframe_profile_excludes_ids_and_near_unique_categories(
        self, _ollama
    ):
        frame = pd.DataFrame(
            {
                "Rk": range(1, 31),
                "Player": [f"Player {index}" for index in range(30)],
                "Age": [20 + index % 12 for index in range(30)],
                "Pos": ["C", "LW", "RW"] * 10,
                "Team": ["A", "B", "C", "D", "E"] * 6,
                "Shot_percent": [10.0 + index / 10 for index in range(30)],
            }
        )

        spec = parse_prompt_to_viz_spec(
            "Compare player performance",
            list(frame.columns),
            dataframe=frame,
        )

        chart_categories = {
            category
            for chart in spec.charts
            for category in [
                chart.group,
                chart.x if chart.type in {"bar", "pie"} and not chart.y else None,
            ]
            if category
        }
        self.assertNotIn("Rk", chart_categories)
        self.assertNotIn("Player", chart_categories)
        self.assertEqual(chart_categories, {"Pos", "Team"})
        self.assertEqual([filter_obj.field for filter_obj in spec.filters], ["Pos"])
        metrics = {metric.expr: metric for metric in spec.metrics}
        self.assertNotIn("mean(Rk)", metrics)
        self.assertEqual(metrics["mean(Shot_percent)"].fmt, "percent")

    @patch("vibedash.spec._improve_with_ollama")
    def test_dataframe_profile_sanitizes_llm_categories(self, improve):
        improve.return_value = VizSpec(
            title="Unsafe categories",
            charts=[
                Chart(type="bar", x="RowID", top=10),
                Chart(type="bar", x="ExternalReference", top=10),
                Chart(type="bar", x="Region", top=10),
            ],
            filters=[
                Filter(field="RowID"),
                Filter(field="ExternalReference"),
                Filter(field="Region"),
            ],
        )
        frame = pd.DataFrame(
            {
                "RowID": range(30),
                "ExternalReference": [f"ref-{index}" for index in range(30)],
                "Region": ["Europe", "North America"] * 15,
            }
        )

        spec = parse_prompt_to_viz_spec(
            "Analyze regions",
            list(frame.columns),
            dataframe=frame,
        )

        self.assertEqual([chart.x for chart in spec.charts], ["Region"])
        self.assertEqual([filter_obj.field for filter_obj in spec.filters], ["Region"])


if __name__ == "__main__":
    unittest.main()
