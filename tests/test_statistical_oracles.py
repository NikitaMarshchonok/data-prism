import unittest

import numpy as np
import pandas as pd
from scipy import stats

from src.data_drift import compare_to_baseline, create_baseline_profile
from src.statistical_primitives import (
    benjamini_hochberg,
    pearson_confidence_interval,
    pearson_summary,
    population_stability_index,
    total_variation_distance,
    welch_independent_summary,
)
from vibedash.period_comparison import build_period_comparison
from vibedash.statistical_engine import StatisticalValidationEngine


class StatisticalPrimitiveOracleTests(unittest.TestCase):
    """Frozen, independently reviewable reference examples for core math."""

    def test_pearson_and_fisher_interval_match_reference_values(self):
        x = np.arange(1, 9, dtype=float)
        y = np.array([2, 1, 4, 3, 6, 5, 8, 7], dtype=float)

        result = pearson_summary(x, y)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.correlation, 0.9047619047619048, places=14)
        self.assertAlmostEqual(result.p_value, 0.00200827550542948, places=14)
        self.assertAlmostEqual(result.confidence_interval[0], 0.552062825228524, places=14)
        self.assertAlmostEqual(result.confidence_interval[1], 0.982824222137223, places=14)

    def test_welch_interval_and_hedges_g_match_reference_values(self):
        left = np.arange(1, 9, dtype=float)
        right = np.arange(2, 18, 2, dtype=float)

        result = welch_independent_summary(left, right)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.statistic, -2.32379000772445, places=14)
        self.assertAlmostEqual(result.p_value, 0.0418067730194469, places=14)
        self.assertAlmostEqual(result.degrees_of_freedom, 10.2941176470588, places=13)
        self.assertAlmostEqual(result.mean_difference, -4.5, places=14)
        self.assertAlmostEqual(result.standard_error, 1.93649167310371, places=14)
        np.testing.assert_allclose(
            result.confidence_interval,
            [-8.79812025287847, -0.201879747121529],
            rtol=0.0,
            atol=1e-10,
        )
        self.assertAlmostEqual(result.hedges_g, -1.09851891274247, places=14)

    def test_benjamini_hochberg_matches_canonical_four_value_example(self):
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])

        np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.008], atol=1e-15)

    def test_psi_and_total_variation_match_binary_reference_values(self):
        expected = [0.5, 0.5]
        actual = [0.75, 0.25]

        self.assertAlmostEqual(
            population_stability_index(expected, actual),
            0.2746530721670274,
            places=14,
        )
        self.assertAlmostEqual(total_variation_distance(expected, actual), 0.25, places=14)

    def test_invalid_probability_and_p_value_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            benjamini_hochberg([0.01, float("nan")])
        with self.assertRaises(ValueError):
            benjamini_hochberg([0.01, 1.01])
        with self.assertRaises(ValueError):
            population_stability_index([0.5, 0.5], [1.0])
        with self.assertRaises(ValueError):
            total_variation_distance([0.5, -0.5], [0.5, 0.5])


class AnalyticalContractIntegrationTests(unittest.TestCase):
    def test_statistical_report_preserves_pearson_oracle_values(self):
        data = pd.DataFrame(
            {
                "metric_x": np.arange(1, 9, dtype=float),
                "metric_y": np.array([2, 1, 4, 3, 6, 5, 8, 7], dtype=float),
            }
        )

        report = StatisticalValidationEngine(data).analyze()
        result = report["tests"][0]

        self.assertEqual(result["method"], "Pearson correlation")
        self.assertEqual(result["statistic"], 0.904762)
        self.assertAlmostEqual(result["raw_p_value"], 0.00200827550542948, places=14)
        self.assertEqual(result["confidence_interval"]["lower"], 0.552063)
        self.assertEqual(result["confidence_interval"]["upper"], 0.982824)

    def test_statistical_report_preserves_oracle_values(self):
        data = pd.DataFrame(
            {
                "group": ["left"] * 8 + ["right"] * 8,
                "value": list(np.arange(1, 9, dtype=float))
                + list(np.arange(2, 18, 2, dtype=float)),
            }
        )

        report = StatisticalValidationEngine(data).analyze()
        result = report["tests"][0]

        self.assertEqual(result["method"], "Welch's independent-samples t-test")
        self.assertEqual(result["statistic"], -2.32379)
        self.assertAlmostEqual(result["raw_p_value"], 0.0418067730194469, places=14)
        self.assertEqual(result["effect_size"]["value"], -1.098519)
        self.assertEqual(result["confidence_interval"]["lower"], -8.79812)
        self.assertEqual(result["confidence_interval"]["upper"], -0.20188)

    def test_period_comparison_uses_the_same_welch_contract_and_direction(self):
        baseline = pd.DataFrame({"value": np.arange(1, 9, dtype=float)})
        current = pd.DataFrame({"value": np.arange(2, 18, 2, dtype=float)})

        report = build_period_comparison(baseline, current)
        metric = report["numeric_metrics"]["metrics"][0]

        self.assertEqual(metric["estimate"]["direction"], "current_minus_baseline")
        self.assertEqual(metric["estimate"]["value"], 4.5)
        self.assertAlmostEqual(metric["raw_p_value"], 0.04180677, places=8)
        self.assertEqual(metric["hedges_g"], 1.09851891)
        self.assertEqual(metric["confidence_interval"]["lower"], 0.20187975)
        self.assertEqual(metric["confidence_interval"]["upper"], 8.79812025)

    def test_public_categorical_drift_matches_total_variation_oracle(self):
        baseline = pd.DataFrame({"segment": ["a"] * 50 + ["b"] * 50})
        current = pd.DataFrame({"segment": ["a"] * 75 + ["b"] * 25})

        report = compare_to_baseline(current, create_baseline_profile(baseline))
        drift = report["feature_drift"][0]

        self.assertEqual(drift["metric"], "TV distance")
        self.assertEqual(drift["score"], 0.25)
        self.assertEqual(drift["severity"], "high")

    def test_public_numeric_drift_matches_psi_oracle(self):
        baseline = pd.DataFrame({"value": [0.0] * 50 + [1.0] * 50})
        current = pd.DataFrame({"value": [0.0] * 75 + [1.0] * 25})

        report = compare_to_baseline(current, create_baseline_profile(baseline))
        drift = report["feature_drift"][0]

        self.assertEqual(drift["metric"], "PSI")
        self.assertEqual(drift["score"], 0.274653)
        self.assertEqual(drift["severity"], "high")


class DeterministicCalibrationTests(unittest.TestCase):
    """Seeded simulations catch direction or calibration regressions."""

    def test_welch_null_false_positive_rate_is_not_inflated(self):
        rng = np.random.default_rng(20261011)
        rejections = 0
        simulations = 300
        for _ in range(simulations):
            left = rng.normal(0.0, 1.0, 30)
            right = rng.normal(0.0, 1.8, 30)
            result = welch_independent_summary(left, right)
            assert result is not None
            rejections += result.p_value <= 0.05

        false_positive_rate = rejections / simulations
        self.assertGreaterEqual(false_positive_rate, 0.01)
        self.assertLessEqual(false_positive_rate, 0.09)

    def test_fisher_interval_has_expected_seeded_coverage(self):
        rng = np.random.default_rng(20261011)
        population_correlation = 0.5
        covariance = np.array([[1.0, population_correlation], [population_correlation, 1.0]])
        covered = 0
        simulations = 300
        for _ in range(simulations):
            sample = rng.multivariate_normal([0.0, 0.0], covariance, size=80)
            correlation = float(stats.pearsonr(sample[:, 0], sample[:, 1]).statistic)
            lower, upper = pearson_confidence_interval(correlation, len(sample))
            covered += lower <= population_correlation <= upper

        coverage = covered / simulations
        self.assertGreaterEqual(coverage, 0.90)
        self.assertLessEqual(coverage, 0.99)

    def test_bh_global_null_rate_is_bounded(self):
        rng = np.random.default_rng(20261011)
        experiments = 500
        experiments_with_rejection = 0
        for _ in range(experiments):
            adjusted = benjamini_hochberg(rng.uniform(size=20).tolist())
            experiments_with_rejection += any(value <= 0.05 for value in adjusted)

        rejection_rate = experiments_with_rejection / experiments
        self.assertGreaterEqual(rejection_rate, 0.02)
        self.assertLessEqual(rejection_rate, 0.08)

    def test_distribution_distances_are_symmetric_and_grow_with_shift(self):
        baseline = [0.5, 0.5]
        shifted = [[0.5 + shift, 0.5 - shift] for shift in (0.0, 0.1, 0.2, 0.3)]
        psi = [population_stability_index(baseline, value) for value in shifted]
        tv = [total_variation_distance(baseline, value) for value in shifted]

        self.assertEqual(psi, sorted(psi))
        self.assertEqual(tv, sorted(tv))
        self.assertAlmostEqual(
            population_stability_index(baseline, shifted[-1]),
            population_stability_index(shifted[-1], baseline),
            places=14,
        )
        self.assertAlmostEqual(
            total_variation_distance(baseline, shifted[-1]),
            total_variation_distance(shifted[-1], baseline),
            places=14,
        )


if __name__ == "__main__":
    unittest.main()
