import math
import re
import unittest
from pathlib import Path

from flask import Flask, render_template

from src.demo_data import create_saas_growth_demo
from vibedash.audit_report import build_audit_report_context
from vibedash.generator_bridge import generate_dashboard_data
from vibedash.plain_language import (
    KNOWN_CHECKS,
    KNOWN_INSIGHT_IDS,
    KNOWN_PRIORITY_CATEGORIES,
    KNOWN_TEST_FAMILIES,
    explain_check,
    explain_insight,
    explain_priority,
    explain_statistics,
    explain_test,
)
from vibedash.spec import create_saas_demo_viz_spec


ROOT = Path(__file__).resolve().parents[1]


class EngineVocabularyCoverageTests(unittest.TestCase):
    """A new engine result type must not silently ship without a translation."""

    def source(self, relative):
        return (ROOT / relative).read_text(encoding="utf-8")

    def test_every_non_passing_readiness_check_has_an_explanation(self):
        emitted = set(
            re.findall(
                r'ReadinessCheck\(\s*"([a-z-]+)",\s*"[^"]*",\s*"(warning|blocked)"',
                self.source("vibedash/readiness_engine.py"),
            )
        )
        self.assertGreaterEqual(len(emitted), 10)
        self.assertEqual(emitted - KNOWN_CHECKS, set())

    def test_every_insight_id_has_an_explanation(self):
        emitted = set(re.findall(r'insight_id="([a-z-]+)"', self.source("vibedash/insight_engine.py")))
        self.assertGreaterEqual(len(emitted), 8)
        self.assertEqual(emitted - KNOWN_INSIGHT_IDS, set())

    def test_every_fixed_brief_category_has_an_explanation(self):
        emitted = set(re.findall(r'"category": "([a-z-]+)"', self.source("vibedash/decision_brief.py")))
        self.assertEqual(emitted - KNOWN_PRIORITY_CATEGORIES, set())

    def test_every_statistical_family_has_an_explanation(self):
        emitted = set(re.findall(r'family="([a-z-]+)"', self.source("vibedash/statistical_engine.py")))
        self.assertGreaterEqual(len(emitted), 4)
        self.assertEqual(emitted - KNOWN_TEST_FAMILIES, set())


class CheckExplanationTests(unittest.TestCase):
    METRICS = {
        "missingness": {"missing_rate": 0.25, "empty_column_count": 1, "missing_cells": 10},
        "duplicate-rows": {"duplicate_rows": 30, "duplicate_rate": 0.3},
    }

    def test_every_known_check_produces_complete_text(self):
        for check_id, status in sorted(KNOWN_CHECKS):
            with self.subTest(check_id=check_id, status=status):
                plain = explain_check(
                    {"check_id": check_id, "status": status, "metrics": self.METRICS.get(check_id, {})}
                )
                self.assertIsNotNone(plain)
                self.assertTrue(all(plain[key] for key in ("headline", "meaning", "action")))

    def test_passed_and_unknown_checks_fall_back(self):
        self.assertIsNone(explain_check({"check_id": "missingness", "status": "passed", "metrics": {}}))
        self.assertIsNone(explain_check({"check_id": "brand-new", "status": "warning"}))
        self.assertIsNone(explain_check(None))

    def test_missing_metric_falls_back_instead_of_guessing(self):
        self.assertIsNone(explain_check({"check_id": "missingness", "status": "warning", "metrics": {}}))
        self.assertIsNone(
            explain_check({"check_id": "duplicate-rows", "status": "warning", "metrics": {"duplicate_rate": math.nan}})
        )

    def test_missingness_wording_scales_with_severity(self):
        small = explain_check({"check_id": "missingness", "status": "warning", "metrics": {"missing_rate": 0.002}})
        large = explain_check({"check_id": "missingness", "status": "blocked", "metrics": {"missing_rate": 0.6}})
        self.assertIn("rarely changes", small["meaning"])
        self.assertIn("too much", large["headline"])
        self.assertEqual(
            explain_check({"check_id": "missingness", "status": "warning", "metrics": {"missing_rate": 0.0004}})["headline"],
            "less than 0.1% of the values are missing.",
        )


class InsightAndTestExplanationTests(unittest.TestCase):
    def test_negative_relationship_reads_as_opposite_direction(self):
        plain = explain_insight(
            {"insight_id": "strongest-correlation", "metrics": {"column_x": "Price", "column_y": "Units", "pearson_r": -0.81}}
        )
        self.assertEqual(plain["headline"], "When “Price” is higher, “Units” is usually lower too.")
        self.assertIn("strong", plain["meaning"])
        self.assertIn("does not prove", plain["meaning"])

    def test_trend_direction_and_magnitude(self):
        plain = explain_insight(
            {"insight_id": "strongest-trend", "metrics": {"value_column": "Sales", "time_column": "Date", "relative_change": -0.234}}
        )
        self.assertEqual(plain["headline"], "“Sales” fell by 23% over the period.")
        self.assertIn("seasonality", plain["meaning"])

    def test_structural_dependency_names_components(self):
        plain = explain_insight(
            {
                "insight_id": "observed-additive-relationship",
                "metrics": {"column": "Total", "component_columns": ["Online", "Retail"]},
            }
        )
        self.assertEqual(plain["headline"], "“Total” is simply the sum of “Online”, “Retail”.")

    def test_unknown_or_incomplete_insights_fall_back(self):
        self.assertIsNone(explain_insight({"insight_id": "dataset-overview", "metrics": {}}))
        self.assertIsNone(explain_insight({"insight_id": "potential-outliers", "metrics": {"column": "X"}}))
        self.assertIsNone(explain_insight({"insight_id": "strongest-correlation", "metrics": {"column_x": "A"}}))

    def test_group_comparison_names_the_higher_group(self):
        sentence = explain_test(
            {
                "family": "group-comparison",
                "significant": True,
                "estimates": {
                    "value_column": "Revenue",
                    "group_column": "Plan",
                    "group_a": "Basic",
                    "group_b": "Pro",
                    "group_a_mean": 10.0,
                    "group_b_mean": 25.0,
                },
            }
        )
        self.assertEqual(sentence, "“Revenue” is reliably higher for “Pro” than for “Basic” (by “Plan”).")

    def test_non_significant_tests_do_not_claim_a_pattern(self):
        sentence = explain_test(
            {"family": "correlation", "significant": False, "columns": ["A", "B"], "effect_size": {"value": 0.1}}
        )
        self.assertEqual(sentence, "No reliable link between “A” and “B”.")
        self.assertIsNone(explain_test({"family": "unknown"}))

    def test_statistics_explanations(self):
        withheld = explain_statistics({"status": "inference_withheld", "reason": "temporal_design_unverified"})
        self.assertIn("runs over time", withheld["headline"])
        ok = explain_statistics({"status": "ok", "candidate_test_count": 21, "significant_test_count": 12})
        self.assertEqual(ok["headline"], "We checked 21 possible patterns; 12 held up under strict testing.")
        self.assertIsNone(explain_statistics({"status": "inference_withheld", "reason": "something-new"}))

    def test_long_column_names_are_bounded(self):
        plain = explain_insight(
            {"insight_id": "category-concentration", "metrics": {"column": "c" * 500, "top_value": "v", "top_share": 0.9}}
        )
        self.assertLess(len(plain["headline"]), 120)


class ReportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frame = create_saas_growth_demo()
        spec = create_saas_demo_viz_spec()
        cls.temporal = generate_dashboard_data(frame, spec, prompt="")
        cls.cross_sectional = generate_dashboard_data(frame.drop(columns=["Date"]), spec, prompt="")

    def test_demo_report_explains_every_visible_item(self):
        for dashboard in (self.temporal, self.cross_sectional):
            report = build_audit_report_context({"dashboard_data": dashboard})
            items = report["priorities"] + report["findings"] + report["readiness"]["issues"]
            self.assertTrue(items)
            for item in items:
                with self.subTest(title=item["title"]):
                    self.assertIsNotNone(item["plain"])
            self.assertIsNotNone(report["statistics"]["plain"])

    def test_validated_signal_priority_uses_the_matching_test(self):
        priority = next(
            item
            for item in self.cross_sectional["decision_brief"]["priorities"]
            if item["category"] == "validated-signal"
        )
        plain = explain_priority(priority, self.cross_sectional)
        self.assertIn("strict statistical checks", plain["meaning"])
        self.assertIsNone(explain_priority({"category": "validated-signal", "title": "missing"}, self.cross_sectional))
        self.assertIsNone(explain_priority({"category": "unknown"}, self.cross_sectional))

    def test_quality_insight_is_not_repeated_under_findings(self):
        report = build_audit_report_context({"dashboard_data": self.temporal})
        self.assertNotIn("Data quality assessment", [item["title"] for item in report["findings"]])

    def test_rendered_report_keeps_technical_detail_and_escapes_names(self):
        app = Flask(__name__, template_folder=str(ROOT / "templates"))
        dashboard = dict(self.temporal)
        dashboard["insights"] = [
            {
                "insight_id": "category-concentration",
                "category": "segment",
                "title": "Category concentration",
                "statement": "Engine statement.",
                "confidence": "high",
                "severity": "warning",
                "metrics": {"column": "<b>Plan</b>", "top_value": "Pro", "top_share": 0.8},
            },
            {
                "insight_id": "future-type",
                "category": "novel",
                "title": "Novel finding",
                "statement": "Original engine wording.",
                "confidence": "high",
                "severity": "info",
            },
        ]
        report = build_audit_report_context({"dashboard_data": dashboard})
        with app.test_request_context("/"):
            html = render_template("vibedash_audit_report.html", report=report, report_css="")
        self.assertIn("Technical detail", html)
        self.assertIn("Engine statement.", html)
        self.assertIn("&lt;b&gt;Plan&lt;/b&gt;", html)
        self.assertNotIn("<b>Plan</b>", html)
        self.assertIn("Original engine wording.", html)


if __name__ == "__main__":
    unittest.main()
