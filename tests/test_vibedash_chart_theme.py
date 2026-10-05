import unittest
from pathlib import Path

import pandas as pd
import plotly.express as px

from vibedash.chart_theme import (
    CHART_CONFIG,
    ACCENT,
    BLUE,
    apply_evidence_chart_theme,
    chart_kind_label,
    humanize_chart_label,
)
from vibedash.generator_bridge import _create_chart_html, _generate_charts
from vibedash.spec import Chart


class VibeDashChartThemeTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "Plan": ["Starter", "Growth", "Enterprise"],
                "Date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                "ChurnRate": [0.074, 0.063, 0.055],
                "CustomerCount": [80, 140, 220],
                "MonthlyRevenue": [4_000, 18_000, 62_000],
            }
        )

    def test_column_names_are_presented_as_business_labels(self):
        self.assertEqual(humanize_chart_label("MonthlyRevenue"), "Monthly Revenue")
        self.assertEqual(humanize_chart_label("nps_score"), "NPS Score")
        self.assertEqual(
            humanize_chart_label('<img src=x onerror="alert(1)">'),
            "&lt;img Src=x Onerror=&quot;alert(1)&quot;&gt;",
        )

    def test_bar_theme_uses_percent_axis_and_removes_inner_title(self):
        chart = Chart(
            type="bar",
            y="ChurnRate",
            agg="mean",
            group="Plan",
            title="Average Churn by Plan",
        )
        grouped = self.frame.groupby("Plan")["ChurnRate"].mean().reset_index()
        figure = px.bar(grouped, x="Plan", y="ChurnRate", title=chart.title)

        themed = apply_evidence_chart_theme(figure, chart, self.frame)

        self.assertIsNone(themed.layout.title.text)
        self.assertEqual(themed.layout.paper_bgcolor, "rgba(0,0,0,0)")
        self.assertEqual(themed.layout.xaxis.title.text, "Plan")
        self.assertEqual(themed.layout.yaxis.title.text, "Churn Rate")
        self.assertEqual(themed.layout.yaxis.tickformat, ".1%")
        self.assertEqual(themed.data[0].marker.color, ACCENT)
        self.assertEqual(list(themed.data[0].text), ["5.5%", "6.3%", "7.4%"])

    def test_scatter_theme_uses_currency_and_restrained_markers(self):
        chart = Chart(
            type="scatter",
            x="CustomerCount",
            y="MonthlyRevenue",
            title="Customer Count vs Revenue",
        )
        figure = px.scatter(self.frame, x=chart.x, y=chart.y, title=chart.title)

        themed = apply_evidence_chart_theme(figure, chart, self.frame)

        self.assertEqual(themed.layout.xaxis.title.text, "Customer Count")
        self.assertEqual(themed.layout.yaxis.title.text, "Monthly Revenue")
        self.assertEqual(themed.layout.yaxis.tickprefix, "$")
        self.assertEqual(themed.data[0].marker.color, BLUE)
        self.assertAlmostEqual(themed.data[0].marker.opacity, 0.72)

    def test_date_axis_keeps_plotly_date_formatting(self):
        chart = Chart(
            type="line",
            x="Date",
            y="MonthlyRevenue",
            title="Daily Revenue Trend",
        )
        figure = px.line(self.frame, x=chart.x, y=chart.y, title=chart.title)

        themed = apply_evidence_chart_theme(figure, chart, self.frame)

        self.assertIsNone(themed.layout.xaxis.tickformat)
        self.assertEqual(themed.layout.xaxis.title.text, "Date")

    def test_chart_html_has_responsive_product_configuration(self):
        html = _create_chart_html(
            self.frame,
            Chart(type="hist", x="CustomerCount", title="Customer distribution"),
        )

        self.assertIn('"responsive": true', html)
        self.assertIn('"displaylogo": false', html)
        self.assertNotIn('"text":"Customer distribution"', html)
        self.assertEqual(CHART_CONFIG["displaylogo"], False)

    def test_generated_chart_metadata_supports_card_styling(self):
        charts = _generate_charts(
            self.frame,
            [Chart(type="hist", x="CustomerCount", title="Customer distribution")],
        )

        self.assertEqual(charts[0]["type"], "hist")
        self.assertEqual(charts[0]["kind"], "Distribution")
        self.assertEqual(chart_kind_label("scatter"), "Relationship")

    def test_evidence_template_loads_responsive_chart_controller(self):
        repository = Path(__file__).resolve().parents[1]
        template = (repository / "templates" / "vibedash_evidence.html").read_text()
        controller = (repository / "static" / "vibedash_charts.js").read_text()

        self.assertIn("vibedash_charts.js", template)
        self.assertIn("ResizeObserver", controller)
        self.assertIn("Plotly.Plots.resize", controller)


if __name__ == "__main__":
    unittest.main()
