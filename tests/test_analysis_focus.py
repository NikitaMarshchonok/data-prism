"""Task relevance must not invent intent or alter inferential results."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.analysis_context import build_analysis_context, is_relevant
from src.column_semantics import profile_dataframe_columns
from src.semantic_contract import infer_semantic_contract
from vibedash.decision_brief import build_decision_brief
from vibedash.generator_bridge import generate_dashboard_data
from vibedash.insight_engine import EvidenceBasedInsightEngine
from vibedash.spec import Metric, VizSpec
from vibedash.statistical_engine import StatisticalValidationEngine


def _frame(dated=False):
    rng = np.random.default_rng(827)
    n = 160
    temp = rng.normal(size=n)
    cnt = 100 + np.linspace(0, 100, n) + rng.normal(0, 5, n)
    frame = pd.DataFrame({
        "temp": temp,
        "atemp": 2 * temp + rng.normal(0, .001, n),
        "registered": .6 * cnt + rng.normal(0, 9, n),
        "cnt": cnt,
        "season": [1, 2] * (n // 2),
        "holiday": [0] * (n - 10) + [1] * 10,
    })
    if dated:
        frame["dteday"] = pd.date_range("2011-01-01", periods=n)
    return frame


def _context(data, prompt="Analyze cnt"):
    return build_analysis_context(prompt, data.columns, profile_dataframe_columns(data))


READY = {"status": "ready", "status_label": "Ready", "checks": []}


class AnalysisFocusTests(unittest.TestCase):
    def test_exact_names_aliases_and_boundaries_not_fuzzy_task_guesses(self):
        data = pd.DataFrame({"cnt": [1., 2.], "discount": [3., 4.], "MonthlyRevenue": [5., 6.]})
        self.assertEqual(_context(data, "Analyze discount")["focus_columns"], ["discount"])
        self.assertEqual(_context(data, "Покажи MONTHLY REVENUE")["focus_columns"], ["MonthlyRevenue"])
        self.assertEqual(_context(data, "Please predict business success")["source"], "unfocused")

    def test_context_contains_schema_only_and_preserves_explicit_groups(self):
        data = _frame(dated=True)
        context = _context(data, "Analyze cnt over dteday by season. Private business instruction.")
        self.assertEqual(context["focus_columns"], ["cnt"])
        self.assertEqual(context["group_columns"], ["season"])
        self.assertEqual(context["time_columns"], ["dteday"])
        self.assertNotIn("Private business instruction", json.dumps(context))

    def test_explicit_measure_precedes_confirmed_target_and_exclusion_wins(self):
        data = _frame()
        profiles = profile_dataframe_columns(data)
        profiles["registered"]["is_target"] = True
        fallback = build_analysis_context("Analyze this dataset", data.columns, profiles)
        self.assertEqual(fallback["source"], "confirmed_target")
        self.assertEqual(fallback["focus_columns"], ["registered"])
        explicit = build_analysis_context("Analyze cnt. Ignore registered.", data.columns, profiles)
        self.assertEqual(explicit["focus_columns"], ["cnt"])
        self.assertEqual(explicit["excluded_columns"], ["registered"])

    def test_relevance_uses_structured_columns_not_narrative_mentions(self):
        context = _context(_frame())
        self.assertFalse(is_relevant({"title": "cnt", "statement": "cnt"}, context))
        self.assertTrue(is_relevant({"metrics": {"value_column": "cnt"}}, context))
        self.assertTrue(is_relevant({"columns": ["season", "cnt"]}, context))

    def test_strongest_relationship_must_include_requested_measure(self):
        data = _frame()
        general = EvidenceBasedInsightEngine(data).generate()
        focused = EvidenceBasedInsightEngine(data, analysis_context=_context(data)).generate()
        general_pair = next(item for item in general if item["category"] == "relationship")["metrics"]
        focused_pair = next(item for item in focused if item["category"] == "relationship")["metrics"]
        self.assertEqual({general_pair["column_x"], general_pair["column_y"]}, {"temp", "atemp"})
        self.assertIn("cnt", {focused_pair["column_x"], focused_pair["column_y"]})
        self.assertAlmostEqual(focused_pair["pearson_r"], data["cnt"].corr(data["registered"]), places=6)

    def test_focused_trend_and_outliers_do_not_fall_back_to_other_measures(self):
        data = _frame(dated=True)
        data.loc[0, "temp"] = 99999
        insights = EvidenceBasedInsightEngine(data, analysis_context=_context(data)).generate(max_insights=10)
        trend = next(item for item in insights if item["category"] == "trend")
        self.assertEqual(trend["metrics"]["value_column"], "cnt")
        self.assertEqual(trend["confidence"], "exploratory")
        for item in insights:
            if item["insight_id"] == "potential-outliers":
                self.assertEqual(item["metrics"]["column"], "cnt")
        self.assertFalse(any(item["category"] == "segment" for item in insights))

    def test_category_concentration_uses_requested_group_not_other_category(self):
        data = _frame()
        insights = EvidenceBasedInsightEngine(data, analysis_context=_context(data, "cnt by season")).generate(max_insights=10)
        concentration = next(item for item in insights if item["category"] == "segment")
        self.assertEqual(concentration["metrics"]["column"], "season")
        self.assertEqual(concentration["metrics"]["top_share"], .5)

    def test_weak_requested_relationship_does_not_invent_or_substitute_signal(self):
        data = _frame()
        data["cnt"] = np.random.default_rng(67).normal(size=len(data))
        data["registered"] = np.random.default_rng(68).normal(size=len(data))
        insights = EvidenceBasedInsightEngine(data, analysis_context=_context(data)).generate()
        self.assertFalse(any(item["category"] == "relationship" for item in insights))

    def test_display_ranking_preserves_all_p_values_and_fdr_family(self):
        data = _frame()
        general = StatisticalValidationEngine(data).analyze(max_results=100)
        focused = StatisticalValidationEngine(data, analysis_context=_context(data)).analyze(max_results=100)
        self.assertIn("cnt", focused["tests"][0]["columns"])
        self.assertEqual(general["candidate_test_count"], focused["candidate_test_count"])
        self.assertEqual(general["significant_test_count"], focused["significant_test_count"])
        general_tests = {(item["family"], frozenset(item["columns"])): item for item in general["tests"]}
        for result in focused["tests"]:
            original = general_tests[result["family"], frozenset(result["columns"])]
            for key in ("raw_p_value", "adjusted_p_value"):
                self.assertAlmostEqual(result[key], original[key], places=13)
            self.assertEqual(result["significant"], original["significant"])

    def test_requested_measure_enters_bounded_scan_even_if_last_in_schema(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({f"metric_{i}": rng.normal(size=100) for i in range(12)})
        engine = StatisticalValidationEngine(data, analysis_context=_context(data, "Analyze metric_11"))
        self.assertEqual(engine._numeric_columns()[0], "metric_11")
        self.assertTrue(any("metric_11" in item["columns"] for item in engine.analyze()["tests"]))

    def test_explicit_exclusions_reach_descriptive_and_statistical_engines(self):
        data = _frame()
        context = _context(data, "Analyze cnt. Не используй registered.")
        tests = StatisticalValidationEngine(data, analysis_context=context).analyze(max_results=100)["tests"]
        self.assertTrue(all("registered" not in item["columns"] for item in tests))
        insights = EvidenceBasedInsightEngine(data, analysis_context=context).generate()
        self.assertFalse(any(item["category"] == "relationship" for item in insights))

    def test_brief_deduplicates_statistical_and_descriptive_pair(self):
        data = _frame()
        context = _context(data)
        dashboard = {
            "analysis_context": context,
            "statistical_validation": StatisticalValidationEngine(data, analysis_context=context).analyze(max_results=100),
            "insights": EvidenceBasedInsightEngine(data, analysis_context=context).generate(),
        }
        brief = build_decision_brief(dashboard, READY, max_priorities=10)
        signal = next(item for item in brief["priorities"] if item["category"] == "validated-signal")
        self.assertIn("cnt", signal["title"])
        self.assertFalse(any(item["category"] == "evidence-relationship" for item in brief["priorities"]))

    def test_brief_never_substitutes_unrelated_significant_result(self):
        data = _frame()
        report = StatisticalValidationEngine(data).analyze(max_results=1)
        brief = build_decision_brief({"analysis_context": _context(data), "statistical_validation": report}, READY)
        self.assertEqual(brief["priorities"], [])
        self.assertEqual(brief["status"], "insufficient_evidence")

    def test_temporal_guardrail_stays_first_and_kpis_are_unchanged(self):
        data = _frame(dated=True)
        contract = infer_semantic_contract(data, confirmed=True)
        spec = VizSpec(title="Task", metrics=[Metric(title="cnt", expr="sum(cnt)", fmt="number")])
        with patch("vibedash.generator_bridge.AnomalySegmentationEngine") as patterns:
            patterns.return_value.analyze.return_value = {"status": "insufficient_data"}
            result = generate_dashboard_data(data, spec, semantic_contract=contract, prompt="Analyze cnt over dteday")
            general = generate_dashboard_data(data, spec, semantic_contract=contract)
        self.assertEqual(result["kpis"], general["kpis"])
        self.assertEqual(result["analysis_context"]["focus_columns"], ["cnt"])
        self.assertEqual(result["statistical_validation"]["inference_status"], "withheld")
        self.assertEqual(result["decision_brief"]["priorities"][0]["category"], "assumption-review")
        self.assertTrue(any(item["category"] == "evidence-trend" for item in result["decision_brief"]["priorities"]))


if __name__ == "__main__":
    unittest.main()
