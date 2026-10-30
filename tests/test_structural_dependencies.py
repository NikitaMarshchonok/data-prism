"""Arithmetic provenance must not become an independent explanatory signal."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from flask import Flask, render_template

from src.analysis_context import build_analysis_context, finding_columns
from src.column_semantics import profile_dataframe_columns
from src.structural_dependencies import (
    MAX_COLUMNS, MAX_ROWS, additive_pair_dependency, detect_additive_dependencies,
)
from vibedash.decision_brief import build_decision_brief
from vibedash.ai_analyzer import DataScienceAI
from vibedash.generator_bridge import generate_dashboard_data
from vibedash.insight_engine import EvidenceBasedInsightEngine
from vibedash.spec import Metric, VizSpec
from vibedash.statistical_engine import StatisticalValidationEngine


def frame(dated=False):
    rng = np.random.default_rng(915)
    casual = rng.integers(10, 300, 120)
    registered = rng.integers(200, 3000, 120)
    result = pd.DataFrame({"casual": casual, "registered": registered, "cnt": casual + registered})
    if dated:
        result["dteday"] = pd.date_range("2011-01-01", periods=len(result))
    return result


def detect(data, profiles=None, context=None):
    return detect_additive_dependencies(data, profiles or profile_dataframe_columns(data), context)


class StructuralDependencyTests(unittest.TestCase):
    def test_exact_identity_verified_on_every_row_and_serializable(self):
        data = frame()
        dependencies = detect(data)
        self.assertEqual(len(dependencies), 1)
        identity = dependencies[0]
        self.assertEqual(identity["equation"], "cnt = casual + registered")
        self.assertEqual(identity["evaluated_rows"], len(data))
        self.assertEqual(identity["coverage"], 1.)
        json.dumps(dependencies, allow_nan=False)
        self.assertIsNotNone(additive_pair_dependency(["cnt", "registered"], dependencies))
        self.assertIsNone(additive_pair_dependency(["casual", "registered"], dependencies))

    def test_single_mismatch_outside_preview_is_not_an_identity(self):
        data = frame()
        data.loc[37, "cnt"] += 1
        self.assertEqual(detect(data), [])

    def test_tiny_mismatch_is_not_rounded_into_identity(self):
        data = frame().astype(float)
        data.loc[37, "cnt"] += 1e-8
        self.assertEqual(detect(data), [])

    def test_missing_infinite_and_constant_operands_are_not_assessed(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                data = frame().astype(float)
                data.loc[37, "casual"] = value
                self.assertEqual(detect(data), [])
        data = frame()
        data["casual"] = 5
        data["cnt"] = data["registered"] + 5
        self.assertEqual(detect(data), [])

    def test_small_samples_fractional_and_unsafe_integer_ranges_are_not_assessed(self):
        self.assertEqual(detect(frame().head(29)), [])
        fractional = frame().astype(float) / 8
        self.assertEqual(detect(fractional), [])
        unsafe = frame().astype("int64")
        unsafe["registered"] += 2 ** 53
        unsafe["cnt"] = unsafe["registered"] + unsafe["casual"]
        self.assertEqual(detect(unsafe), [])

    def test_compatible_units_required_and_roles_respected(self):
        data = frame()
        profiles = profile_dataframe_columns(data)
        for profile in profiles.values():
            profile["unit"] = "count"
        self.assertEqual(len(detect(data, profiles)), 1)
        profiles["cnt"]["unit"] = "currency"
        self.assertEqual(detect(data, profiles), [])
        profiles["cnt"]["unit"] = "count"
        for role in ("identifier", "temporal", "category"):
            profiles["cnt"]["role"] = role
            self.assertEqual(detect(data, profiles), [])

    def test_signed_integer_identity_does_not_invent_positive_parts(self):
        data = frame() - 150
        data["cnt"] = data["casual"] + data["registered"]
        self.assertEqual(len(detect(data)), 1)

    def test_independent_noise_is_not_promoted(self):
        rng = np.random.default_rng(93)
        data = pd.DataFrame({name: rng.integers(0, 5000, 500) for name in ("x", "y", "value")})
        self.assertEqual(detect(data), [])

    def test_safe_integer_boundary_and_nullable_integers(self):
        data = frame()
        data["registered"] += 2 ** 51
        data["cnt"] = data["registered"] + data["casual"]
        self.assertEqual(len(detect(data)), 1)
        nullable = frame().astype("Int64")
        self.assertEqual(len(detect(nullable)), 1)
        nullable.loc[37, "cnt"] = pd.NA
        self.assertEqual(detect(nullable), [])

    def test_extended_precision_is_not_silently_rounded(self):
        if np.dtype(np.longdouble).itemsize <= 8:
            self.skipTest("Extended precision is unavailable on this platform")
        data = frame().astype(np.longdouble)
        data.loc[37, "cnt"] += np.longdouble("0.0000000000000001")
        self.assertEqual(detect(data, {name: {"role": "measure"} for name in data.columns}), [])

    def test_scan_is_deterministic_and_role_codes_cannot_create_identity(self):
        data = frame()
        self.assertEqual(detect(data), detect(data.sample(frac=1, random_state=7)))
        data["casual"] = np.tile([0, 1, 2], 40)
        data["registered"] = np.tile([1, 2, 0], 40)
        data["cnt"] = data["casual"] + data["registered"]
        profiles = {name: {"role": "category"} for name in data.columns}
        self.assertEqual(detect(data, profiles), [])

    def test_caps_and_explicit_exclusions(self):
        data = frame()
        self.assertEqual(detect(data, context={"excluded_columns": ["casual"]}), [])
        too_large = pd.concat([data] * (MAX_ROWS // len(data) + 1), ignore_index=True)
        self.assertEqual(detect(too_large), [])
        rng = np.random.default_rng(14)
        wide = pd.DataFrame({f"feature_{i}": rng.integers(50000, 60000, len(data))
                             for i in range(MAX_COLUMNS - 2)})
        wide = pd.concat([data[["casual", "registered"]], wide, data[["cnt"]]], axis=1)
        self.assertEqual(detect(wide), [])  # cnt is outside the schema-order cap
        self.assertEqual(len(detect(wide, context={"focus_columns": ["cnt"]})), 1)

    def test_focused_evidence_explains_identity_instead_of_total_component_correlation(self):
        data = frame()
        profiles = profile_dataframe_columns(data)
        context = build_analysis_context("Analyze cnt", data.columns, profiles)
        insights = EvidenceBasedInsightEngine(data, analysis_context=context).generate()
        identity = next(item for item in insights if item["category"] == "structural-dependency")
        self.assertEqual(identity["confidence"], "exploratory")
        self.assertIn("not independent discoveries", identity["statement"])
        self.assertIn("available at prediction time", identity["recommendation"])
        self.assertEqual(finding_columns(identity), {"cnt", "casual", "registered"})
        self.assertFalse(any(item["category"] == "relationship" for item in insights))

    def test_focus_on_component_still_receives_structural_explanation(self):
        data = frame()
        insights = EvidenceBasedInsightEngine(data, analysis_context={"focus_columns": ["registered"]}).generate()
        self.assertTrue(any(item["category"] == "structural-dependency" for item in insights))

    def test_other_relationships_are_not_suppressed(self):
        data = frame()
        data["feature_a"] = np.linspace(.1, 1.5, len(data))
        data["feature_b"] = data["feature_a"] * 2 + .1
        insights = EvidenceBasedInsightEngine(data).generate(max_insights=10)
        correlation = next(item for item in insights if item["category"] == "relationship")
        self.assertEqual({correlation["metrics"]["column_x"], correlation["metrics"]["column_y"]},
                         {"feature_a", "feature_b"})

    def test_statistics_and_fdr_family_are_retained_but_labelled(self):
        data = frame()
        report = StatisticalValidationEngine(data).analyze(max_results=20)
        baseline = StatisticalValidationEngine(data, structural_dependencies=[]).analyze(max_results=20)
        self.assertEqual(report["candidate_test_count"], baseline["candidate_test_count"])
        self.assertEqual(report["significant_test_count"], baseline["significant_test_count"])
        original = {test["test_id"]: test for test in baseline["tests"]}
        for test in report["tests"]:
            for key in ("raw_p_value", "adjusted_p_value", "effect_size", "confidence_interval"):
                self.assertEqual(test[key], original[test["test_id"]][key])
            if "cnt" in test["columns"]:
                self.assertIsNotNone(test["structural_dependency"])
                self.assertIn("not an independent discovery", test["interpretation"])
                self.assertIn("Observed arithmetic", test["limitations"][0])
        evidence = StatisticalValidationEngine(data).evidence_insights(report, max_results=20)
        self.assertFalse(any("cnt" in item["title"] for item in evidence))

    def test_brief_does_not_promote_structural_test_even_without_test_annotation(self):
        data = frame()
        profiles = profile_dataframe_columns(data)
        context = build_analysis_context("Analyze cnt", data.columns, profiles)
        stats = StatisticalValidationEngine(data, structural_dependencies=[]).analyze(max_results=20)
        insights = EvidenceBasedInsightEngine(data, analysis_context=context).generate()
        brief = build_decision_brief({"statistical_validation": stats, "insights": insights,
                                     "structural_dependencies": detect(data), "analysis_context": context},
                                    {"status": "ready", "status_label": "Ready"})
        self.assertFalse(any(item["category"] == "validated-signal" for item in brief["priorities"]))
        self.assertEqual(brief["priorities"][0]["category"], "evidence-structural-dependency")

    def test_temporal_guard_and_kpis_unchanged_bridge_scans_once(self):
        data = frame(dated=True)
        spec = VizSpec(title="Counts", metrics=[Metric(title="Total", expr="sum(cnt)", fmt="number")])
        with patch("vibedash.generator_bridge.detect_additive_dependencies", wraps=detect_additive_dependencies) as scan:
            result = generate_dashboard_data(data, spec, prompt="Analyze cnt over dteday")
        self.assertEqual(scan.call_count, 1)
        self.assertEqual(result["statistical_validation"]["inference_status"], "withheld")
        self.assertEqual(result["statistical_validation"]["tests"], [])
        self.assertEqual(len(result["structural_dependencies"]), 1)
        self.assertEqual(result["kpis"][0]["value"], f"{data['cnt'].sum():,}")
        categories = [item["category"] for item in result["decision_brief"]["priorities"]]
        self.assertIn("evidence-structural-dependency", categories)
        self.assertNotIn("validated-signal", categories)

    def test_chat_explains_arithmetic_without_promoting_total_component_pair(self):
        data = frame()
        correlation = DataScienceAI(data).analyze_question("correlation")
        self.assertIn("cnt = casual + registered", correlation["answer"])
        self.assertIn("не являются независимыми открытиями", correlation["answer"])
        self.assertNotIn("registered ↔ cnt", correlation["answer"])
        self.assertNotIn("Самые коррелирующие колонки", " ".join(
            item for item in correlation["insights"] if isinstance(item, str)))
        statistics = DataScienceAI(data).analyze_question("statistical tests")
        self.assertIn("observed arithmetic relationship", statistics["answer"])
        self.assertNotIn("**Linear association: registered vs cnt** — FDR significant", statistics["answer"])

    def test_equation_metadata_is_escaped_in_both_html_reports_and_exports(self):
        data = frame().rename(columns={"cnt": "cnt<script>untrusted</script>"})
        insights = EvidenceBasedInsightEngine(data).generate()
        app = Flask(__name__, template_folder=str(Path(__file__).resolve().parents[1] / "templates"))
        for endpoint in ("analysis_history", "decision_cases", "export", "index"):
            app.add_url_rule("/" + endpoint, endpoint="vibedash." + endpoint, view_func=lambda: "")
        for template in ("vibedash_evidence.html", "vibedash_preview.html"):
            for export in (False, True):
                with self.subTest(template=template, export=export), app.test_request_context("/"):
                    html = render_template(template, dashboard_data={"insights": insights, "df_shape": data.shape},
                                           viz_spec=VizSpec(title="Counts"), filename="test.csv", prompt="Analyze",
                                           session_id="test", export_mode=export)
                    self.assertIn("cnt&lt;script&gt;untrusted&lt;/script&gt;", html)
                    self.assertNotIn("cnt<script>untrusted</script>", html)


if __name__ == "__main__":
    unittest.main()
