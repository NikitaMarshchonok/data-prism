import json
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from src.group_statistics import (
    MAX_BOOTSTRAP_ROWS,
    compare_group_means,
    holm,
    newcombe_difference,
    rate_difference_test,
    welch_anova,
    wilson_interval,
)
from vibedash.decision_brief import build_decision_brief
from vibedash.group_quality import evaluate_group_comparisons
from vibedash.plain_language import explain_test
from vibedash.statistical_engine import MAX_GROUP_LEVELS, StatisticalValidationEngine


def _family(report, family):
    return [test for test in report["tests"] if test["family"] == family]


def _plans(rng, sizes=(60, 60, 60), shifts=(0.0, 4.0, 12.0)):
    labels = np.repeat(["Starter", "Growth", "Enterprise"][: len(sizes)], sizes)
    revenue = rng.normal(50.0, 6.0, len(labels)) + np.repeat(shifts, sizes)
    return pd.DataFrame({"Plan": labels, "Revenue": revenue})


class GroupStatisticsPrimitiveTests(unittest.TestCase):
    def test_welch_anova_matches_scipy(self):
        rng = np.random.default_rng(1)
        groups = [rng.normal(0, 1, 20), rng.normal(0.5, 3, 35), rng.normal(1, 2, 12)]
        ours = welch_anova(groups)
        reference = stats.f_oneway(*groups, equal_var=False)
        self.assertAlmostEqual(ours.statistic, float(reference.statistic), places=9)
        self.assertAlmostEqual(ours.p_value, float(reference.pvalue), places=9)

    def test_welch_anova_needs_spread_in_every_group(self):
        self.assertIsNone(welch_anova([np.ones(10), np.arange(10.0), np.arange(10.0) * 2]))
        self.assertIsNone(welch_anova([np.arange(10.0)]))

    def test_rate_test_matches_pearson_chi_square(self):
        result = rate_difference_test([30, 45, 12], [100, 120, 80])
        table = np.array([[30, 45, 12], [70, 75, 68]])
        chi2, p_value = stats.chi2_contingency(table, correction=False)[:2]
        self.assertEqual(result.method, "Pearson chi-square test")
        self.assertAlmostEqual(result.statistic, chi2, places=9)
        self.assertAlmostEqual(result.p_value, p_value, places=12)

    def test_sparse_tables_use_exact_methods(self):
        two = rate_difference_test([1, 8], [20, 22])
        self.assertEqual(two.method, "Fisher's exact test")
        self.assertAlmostEqual(two.p_value, stats.fisher_exact([[1, 19], [8, 14]]).pvalue, places=12)
        many = rate_difference_test([2, 5, 1], [20, 22, 18])
        self.assertIn("Monte Carlo", many.method)
        self.assertEqual(many.p_value, rate_difference_test([2, 5, 1], [20, 22, 18]).p_value)

    def test_rate_test_rejects_tables_without_variation(self):
        self.assertIsNone(rate_difference_test([0, 0], [10, 12]))
        self.assertIsNone(rate_difference_test([10, 12], [10, 12]))

    def test_published_interval_values(self):
        low, high = wilson_interval(3, 10)
        self.assertAlmostEqual(low, 0.1078, places=4)
        self.assertAlmostEqual(high, 0.6032, places=4)
        # Newcombe (1998), example (a): 56/70 versus 48/80.
        low, high = newcombe_difference(56, 70, 48, 80)
        self.assertAlmostEqual(low, 0.0524, places=4)
        self.assertAlmostEqual(high, 0.3339, places=4)

    def test_holm_is_monotone_and_keeps_order(self):
        self.assertEqual(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_bootstrap_comparison_is_deterministic_and_bounded(self):
        rng = np.random.default_rng(3)
        groups = [rng.lognormal(3, 0.8, n) for n in (30, 50, 70)]
        first, second = compare_group_means(groups), compare_group_means(groups)
        self.assertEqual(first.p_value, second.p_value)
        self.assertIn("bootstrap", first.method)
        self.assertEqual(len(first.pairs), 3)
        for pair in first.pairs:
            low, high = pair.confidence_interval
            self.assertLess(low, pair.difference)
            self.assertGreater(high, pair.difference)

    def test_strong_differences_get_a_fine_p_value_floor(self):
        # 0.05 / 90 hypotheses is below the 0.001 floor of 999 resamples; the
        # extension to 9,999 resamples keeps strong effects able to pass FDR.
        rng = np.random.default_rng(31)
        result = compare_group_means([rng.normal(mean, 1.0, 40) for mean in (0.0, 1.0, 2.0)])
        self.assertLess(result.p_value, 0.05 / 90)
        self.assertIn("9999 resamples", result.method)

    def test_large_files_use_the_asymptotic_test(self):
        rng = np.random.default_rng(4)
        size = MAX_BOOTSTRAP_ROWS // 3 + 10
        result = compare_group_means([rng.normal(0, 1, size) for _ in range(3)])
        self.assertNotIn("bootstrap", result.method)
        self.assertAlmostEqual(result.p_value, result.anova.p_value)


class MultiGroupEngineTests(unittest.TestCase):
    def test_three_plans_are_compared_and_the_highest_is_named(self):
        report = StatisticalValidationEngine(_plans(np.random.default_rng(5))).analyze()
        result = _family(report, "multi-group-comparison")[0]
        estimates = result["estimates"]
        self.assertTrue(result["significant"])
        self.assertEqual(estimates["highest_group"], "Enterprise")
        self.assertEqual(estimates["lowest_group"], "Starter")
        self.assertEqual(result["effect_size"]["magnitude"], "large")
        self.assertGreater(result["confidence_interval"]["lower"], 0)
        self.assertTrue(estimates["pairwise"])
        self.assertTrue(all(pair["ci_lower"] <= pair["difference"] <= pair["ci_upper"] for pair in estimates["pairwise"]))
        self.assertIn("Enterprise", result["interpretation"])
        json.dumps(report)

    def test_inconclusive_result_reports_no_pairs_and_no_equivalence(self):
        report = StatisticalValidationEngine(_plans(np.random.default_rng(6), shifts=(0.0, 0.0, 0.0))).analyze()
        result = _family(report, "multi-group-comparison")[0]
        self.assertFalse(result["significant"])
        self.assertEqual(result["estimates"]["pairwise"], [])
        self.assertIn("does not prove that the groups are equivalent", result["interpretation"])

    def test_rare_levels_are_left_out_and_disclosed(self):
        frame = _plans(np.random.default_rng(7))
        rare = pd.DataFrame({"Plan": ["Legacy"] * 3, "Revenue": [500.0, 520.0, 510.0]})
        report = StatisticalValidationEngine(pd.concat([frame, rare], ignore_index=True)).analyze()
        result = _family(report, "multi-group-comparison")[0]
        self.assertEqual(result["estimates"]["group_count"], 3)
        self.assertEqual(result["estimates"]["excluded_rows"], 3)
        self.assertNotIn("Legacy", [group["group"] for group in result["estimates"]["groups"]])
        self.assertTrue(any("3 rows" in note for note in result["limitations"]))

    def test_columns_with_too_many_levels_are_not_scanned(self):
        rng = np.random.default_rng(8)
        levels = [f"Store {index:02d}" for index in range(MAX_GROUP_LEVELS + 1)]
        frame = pd.DataFrame({"Store": np.repeat(levels, 10), "Revenue": rng.normal(100, 10, 10 * len(levels))})
        report = StatisticalValidationEngine(frame).analyze()
        self.assertEqual(_family(report, "multi-group-comparison"), [])

    def test_two_level_columns_keep_the_welch_t_test(self):
        report = StatisticalValidationEngine(_plans(np.random.default_rng(9), sizes=(40, 40), shifts=(0.0, 10.0))).analyze()
        self.assertEqual(_family(report, "multi-group-comparison"), [])
        self.assertTrue(_family(report, "group-comparison")[0]["test_id"].startswith("welch-"))

    def test_dated_rows_still_withhold_inference(self):
        frame = _plans(np.random.default_rng(10))
        frame["Date"] = pd.date_range("2026-01-01", periods=len(frame), freq="D").strftime("%Y-%m-%d")
        report = StatisticalValidationEngine(frame).analyze()
        self.assertEqual(report["inference_status"], "withheld")
        self.assertEqual(report["tests"], [])

    def test_large_practical_effects_lead_over_negligible_significant_ones(self):
        rng = np.random.default_rng(11)
        n = 4000
        region = rng.choice(["North", "South", "East", "West"], n)
        plan = rng.choice(["Starter", "Growth", "Enterprise"], n)
        frame = pd.DataFrame({
            "Region": region,
            "Plan": plan,
            "Revenue": rng.normal(100, 10, n) + (region == "North") * 0.8 + (plan == "Enterprise") * 15,
        })
        report = StatisticalValidationEngine(frame).analyze()
        leading = report["tests"][0]
        self.assertEqual(leading["columns"], ["Revenue", "Plan"])
        north = next(test for test in report["tests"] if test["columns"] == ["Revenue", "Region"])
        self.assertTrue(north["significant"])
        self.assertIn(north["effect_size"]["magnitude"], {"negligible", "small"})
        brief = build_decision_brief({"statistical_validation": report}, {})
        signal = next(item for item in brief["priorities"] if item["category"] == "validated-signal")
        self.assertEqual(signal["title"], leading["title"])


class RateComparisonEngineTests(unittest.TestCase):
    def frame(self, rng, rates=(0.9, 0.75, 0.55), size=150):
        labels = np.repeat(["Enterprise", "Growth", "Starter"], size)
        renewed = np.where(rng.random(len(labels)) < np.repeat(rates, size), "Yes", "No")
        return pd.DataFrame({"Plan": labels, "Renewed": renewed})

    def test_renewal_rate_by_plan_names_segments_with_intervals(self):
        report = StatisticalValidationEngine(self.frame(np.random.default_rng(12))).analyze()
        result = _family(report, "rate-comparison")[0]
        estimates = result["estimates"]
        self.assertTrue(result["significant"])
        self.assertEqual(estimates["positive_level"], "Yes")
        self.assertEqual(estimates["highest_segment"], "Enterprise")
        self.assertEqual(estimates["lowest_segment"], "Starter")
        self.assertGreater(result["confidence_interval"]["lower"], 0)
        for segment in estimates["segments"]:
            self.assertLessEqual(segment["ci_lower"], segment["rate"])
            self.assertGreaterEqual(segment["ci_upper"], segment["rate"])
        json.dumps(report)

    def test_without_an_explicit_yes_the_rarer_level_is_the_event(self):
        frame = self.frame(np.random.default_rng(13), rates=(0.2, 0.1, 0.05))
        frame["Renewed"] = frame["Renewed"].map({"Yes": "Churned", "No": "Active"})
        result = _family(StatisticalValidationEngine(frame).analyze(), "rate-comparison")[0]
        self.assertEqual(result["estimates"]["positive_level"], "Churned")

    def test_two_binary_columns_are_tested_once_with_the_focus_as_outcome(self):
        rng = np.random.default_rng(14)
        frame = pd.DataFrame({
            "Trial": rng.choice(["Yes", "No"], 300),
            "Renewed": rng.choice(["Yes", "No"], 300),
        })
        context = {"focus_columns": ["Trial"]}
        results = _family(StatisticalValidationEngine(frame, analysis_context=context).analyze(), "rate-comparison")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["estimates"]["outcome_column"], "Trial")

    def test_equal_rates_do_not_raise_a_claim(self):
        report = StatisticalValidationEngine(self.frame(np.random.default_rng(17), rates=(0.3, 0.3, 0.3))).analyze()
        result = _family(report, "rate-comparison")[0]
        self.assertFalse(result["significant"])
        self.assertIn("does not prove that the rates are equal", result["interpretation"])


class PlainLanguageTests(unittest.TestCase):
    def test_multi_group_sentence_uses_units_and_flags_small_effects(self):
        test = {
            "family": "multi-group-comparison",
            "significant": True,
            "effect_size": {"magnitude": "small"},
            "estimates": {
                "value_column": "Conversion", "group_column": "Plan", "unit": "ratio",
                "highest_group": "Enterprise", "lowest_group": "Starter",
                "highest_mean": 0.063, "lowest_mean": 0.027,
            },
        }
        sentence = explain_test(test)
        self.assertIn("highest for “Enterprise” (average 6.3%)", sentence)
        self.assertIn("real but small", sentence)
        test["significant"] = False
        self.assertEqual(explain_test(test), "No reliable difference in “Conversion” between the “Plan” groups.")

    def test_rate_sentence_names_both_rates(self):
        test = {
            "family": "rate-comparison",
            "significant": True,
            "effect_size": {"magnitude": "medium"},
            "estimates": {
                "outcome_column": "Renewed", "positive_level": "Yes", "segment_column": "Plan",
                "highest_segment": "Enterprise", "lowest_segment": "Starter",
                "highest_rate": 0.9, "lowest_rate": 0.55,
            },
        }
        self.assertEqual(
            explain_test(test),
            "The share of “Renewed” = “Yes” depends on “Plan”: 90% for “Enterprise” versus 55% for “Starter”.",
        )


class GroupQualityGateTests(unittest.TestCase):
    def test_false_alarms_detection_and_pairwise_errors_meet_thresholds(self):
        report = evaluate_group_comparisons(seeds=300)
        failed = [item for item in report["scenarios"] if not item["passed"]]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
