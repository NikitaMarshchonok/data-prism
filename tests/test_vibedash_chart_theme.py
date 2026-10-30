import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.column_semantics import profile_dataframe_columns

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

    def test_numeric_category_codes_have_discrete_axis_without_numeric_format(self):
        frame = pd.DataFrame({"season": [1, 2, 3, 4], "cnt": [15, 30, 45, 60]})
        chart = Chart(type="bar", group="season", y="cnt", agg="sum")
        figure = px.bar(frame, x="season", y="cnt")
        figure.update_xaxes(tickformat="~s", tickprefix="$", ticksuffix="%")
        apply_evidence_chart_theme(figure, chart, frame)
        self.assertEqual(figure.layout.xaxis.type, "category")
        self.assertIsNone(figure.layout.xaxis.tickformat)
        self.assertIsNone(figure.layout.xaxis.tickprefix)
        self.assertIsNone(figure.layout.xaxis.ticksuffix)
        self.assertEqual(list(figure.data[0].x), [1, 2, 3, 4])
        self.assertEqual(list(figure.data[0].y), [15, 30, 45, 60])

    def test_confirmed_role_wins_over_measure_like_name_and_numeric_dtype(self):
        frame = pd.DataFrame({"RevenueCode": [1000, 3000, 7000], "value": [10, 20, 30]})
        chart = Chart(type="scatter", x="RevenueCode", y="value")
        profiles = profile_dataframe_columns(frame)
        profiles["RevenueCode"].update(role="category", unit="none")
        figure = px.scatter(frame, x=chart.x, y=chart.y)
        apply_evidence_chart_theme(figure, chart, frame, column_profiles=profiles)
        self.assertEqual(figure.layout.xaxis.type, "category")
        self.assertIn("%{x}", figure.data[0].hovertemplate)
        self.assertNotIn("$%{x", figure.data[0].hovertemplate)
        self.assertNotIn("%{x:,.2f}", figure.data[0].hovertemplate)

    def test_confirmed_numeric_measure_is_not_forced_to_category(self):
        frame = pd.DataFrame({"season": [1, 2, 3, 4], "value": [10, 20, 30, 40]})
        profiles = profile_dataframe_columns(frame)
        profiles["season"].update(role="measure", unit="number")
        chart = Chart(type="scatter", x="season", y="value")
        figure = px.scatter(frame, x=chart.x, y=chart.y)
        apply_evidence_chart_theme(figure, chart, frame, column_profiles=profiles)
        self.assertNotEqual(figure.layout.xaxis.type, "category")
        self.assertEqual(figure.layout.xaxis.tickformat, "~s")

    def test_string_numeric_labels_keep_leading_zeros_and_trace_order(self):
        frame = pd.DataFrame({"Code": ["010", "002", "001"], "value": [10, 20, 30]})
        chart = Chart(type="bar", x="Code", y="value")
        figure = px.bar(frame, x=chart.x, y=chart.y)
        apply_evidence_chart_theme(figure, chart, frame)
        self.assertEqual(figure.layout.xaxis.type, "category")
        self.assertEqual(list(figure.data[0].x), ["010", "002", "001"])
        self.assertIsNone(figure.layout.xaxis.categoryorder)

    def test_date_like_labels_confirmed_as_categories_stay_discrete(self):
        frame = pd.DataFrame({"PeriodLabel": ["2026-01-01", "2026-04-01"], "value": [1, 2]})
        profiles = profile_dataframe_columns(frame)
        profiles["PeriodLabel"].update(role="category", unit="none")
        chart = Chart(type="bar", x="PeriodLabel", y="value")
        figure = px.bar(frame, x=chart.x, y=chart.y)
        apply_evidence_chart_theme(figure, chart, frame, column_profiles=profiles)
        self.assertEqual(figure.layout.xaxis.type, "category")

    def test_scatter_category_on_y_has_raw_hover_labels(self):
        frame = pd.DataFrame({"value": [50, 75, 100], "segment": [1, 2, 3]})
        chart = Chart(type="scatter", x="value", y="segment")
        figure = px.scatter(frame, x=chart.x, y=chart.y)
        apply_evidence_chart_theme(figure, chart, frame)
        self.assertEqual(figure.layout.yaxis.type, "category")
        self.assertIn("%{y}<extra>", figure.data[0].hovertemplate)

    def test_category_count_axis_and_labels_remain_numeric_counts(self):
        frame = pd.DataFrame({"season": [1, 1, 2], "Status": [1, 2, 1]})
        chart = Chart(type="bar", group="season", y="Status", agg="count")
        figure = px.bar(frame.groupby("season")["Status"].count().reset_index(), x="season", y="Status")
        apply_evidence_chart_theme(figure, chart, frame)
        self.assertEqual(figure.layout.xaxis.type, "category")
        self.assertEqual(figure.layout.yaxis.type, "linear")
        self.assertEqual(figure.layout.yaxis.tickformat, ",.0f")
        self.assertEqual(figure.layout.yaxis.title.text, "Count")
        self.assertEqual(list(figure.data[0].text), ["2", "1"])

    def test_count_of_currency_values_is_not_formatted_as_money(self):
        chart = Chart(type="bar", group="Plan", y="MonthlyRevenue", agg="count")
        grouped = self.frame.groupby("Plan")["MonthlyRevenue"].count().reset_index()
        figure = px.bar(grouped, x="Plan", y="MonthlyRevenue")
        apply_evidence_chart_theme(figure, chart, self.frame)
        self.assertIsNone(figure.layout.yaxis.tickprefix)
        self.assertEqual(list(figure.data[0].text), ["1", "1", "1"])
        self.assertNotIn("$%{y", figure.data[0].hovertemplate)

    def test_count_chart_numeric_codes_reach_serialized_html(self):
        frame = pd.DataFrame({"season": [3, 3, 1, 4, 2]})
        seen = []
        def capture(figure, chart, data, **kwargs):
            seen.append(apply_evidence_chart_theme(figure, chart, data, **kwargs))
        with patch("vibedash.generator_bridge.apply_evidence_chart_theme", side_effect=capture):
            html = _create_chart_html(frame, Chart(type="bar", x="season", top=3))
        self.assertEqual(seen[0].layout.xaxis.type, "category")
        self.assertEqual(list(seen[0].data[0].x), [3, 1, 4])
        self.assertEqual(list(seen[0].data[0].y), [2, 1, 1])
        self.assertIn('"type":"category"', html)

    def test_multi_trace_category_axis_preserves_both_series(self):
        frame = pd.DataFrame({"season": [1, 2, 3], "value": [10, 20, 30]})
        figure = go.Figure([go.Bar(x=[1, 2], y=[10, 20]), go.Bar(x=[2, 3], y=[30, 40])])
        before = [trace.to_plotly_json()["x"] for trace in figure.data]
        apply_evidence_chart_theme(figure, Chart(type="bar", x="season", y="value"), frame)
        self.assertEqual(figure.layout.xaxis.type, "category")
        self.assertEqual([trace.to_plotly_json()["x"] for trace in figure.data], before)


if __name__ == "__main__":
    unittest.main()
