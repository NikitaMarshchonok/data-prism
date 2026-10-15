import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.ml_predictor import (
    _expected_calibration_error,
    _is_exact_target_proxy,
    predict_target,
)


class _FitAuditTransformer(BaseEstimator, TransformerMixin):
    fitted_maxima = []

    def fit(self, values, target=None):
        numeric = np.asarray(values, dtype=float)
        type(self).fitted_maxima.append(float(np.nanmax(numeric)))
        return self

    def transform(self, values):
        return np.asarray(values, dtype=float)


class PredictiveMLOracleTests(unittest.TestCase):
    def test_exact_numeric_copy_is_a_proxy_even_when_dtype_and_name_differ(self):
        target = pd.Series(np.arange(40), name="outcome")
        feature = pd.Series(np.arange(40, dtype=float), name="measurement")

        self.assertTrue(
            _is_exact_target_proxy(
                feature,
                target,
                feature_name="measurement",
                target_name="outcome",
            )
        )

    def test_exact_numeric_target_proxy_is_excluded_and_signal_remains_supported(self):
        rng = np.random.default_rng(4101)
        feature = rng.normal(size=360)
        target = 4.0 * feature + rng.normal(scale=0.35, size=len(feature))
        data = pd.DataFrame(
            {
                "signal": feature,
                "noise": rng.normal(size=len(feature)),
                "target_proxy": target * -7.0 + 13.0,
                "target": target,
            }
        )

        original = predict_target(data, "target")
        transformed = data.copy()
        transformed["signal"] = transformed["signal"] * -11.0 + 8.0
        transformed = transformed.sample(frac=1.0, random_state=991)
        changed = predict_target(transformed, "target")

        for report in (original, changed):
            self.assertEqual(report["status"], "ok")
            self.assertIn("target_proxy", report["dropped_features"])
            self.assertEqual(
                report["predictive_support"]["status"],
                "supported_on_holdout",
            )
            self.assertLessEqual(
                report["predictive_support"]["permutation_p_value"],
                0.05,
            )
            self.assertGreater(report["metrics"]["mae_improvement"], 0.75)

    def test_seeded_classification_noise_is_not_promoted_as_predictive(self):
        rng = np.random.default_rng(2026)
        row_count = 320
        data = pd.DataFrame(
            {f"noise_{index}": rng.normal(size=row_count) for index in range(5)}
        )
        data["target"] = rng.choice(["a", "b", "c"], size=row_count)

        report = predict_target(data, "target")

        self.assertEqual(report["status"], "ok")
        self.assertEqual(
            report["predictive_support"]["status"],
            "not_supported_on_holdout",
        )
        self.assertGreater(
            report["predictive_support"]["permutation_p_value"],
            0.05,
        )

    def test_known_classification_signal_beats_balanced_baseline(self):
        rng = np.random.default_rng(4102)
        feature = rng.normal(size=360)
        target = np.where(
            feature + rng.normal(scale=0.35, size=len(feature)) > 0,
            "yes",
            "no",
        )
        data = pd.DataFrame(
            {
                "signal": feature,
                "noise": rng.normal(size=len(feature)),
                "target": target,
            }
        )

        report = predict_target(data, "target")

        self.assertEqual(
            report["predictive_support"]["status"],
            "supported_on_holdout",
        )
        self.assertGreater(report["metrics"]["balanced_accuracy_lift"], 0.25)
        self.assertEqual(report["metrics"]["baseline_balanced_accuracy"], 0.5)
        self.assertLessEqual(
            report["predictive_support"]["permutation_p_value"],
            0.05,
        )

    def test_preprocessors_never_fit_on_reserved_holdout_rows(self):
        rng = np.random.default_rng(4103)
        train_feature = rng.normal(size=80)
        test_feature = np.full(20, 1_000_000.0)
        X_train = pd.DataFrame({"feature": train_feature})
        X_test = pd.DataFrame({"feature": test_feature})
        y_train = pd.Series(["yes", "no"] * 40)
        y_test = pd.Series(["yes", "no"] * 10)
        data = pd.DataFrame(
            {
                "feature": np.r_[train_feature, test_feature],
                "target": np.r_[y_train, y_test],
            }
        )
        _FitAuditTransformer.fitted_maxima = []

        with patch(
            "src.ml_predictor._split_data",
            return_value=(
                X_train,
                X_test,
                y_train,
                y_test,
                "oracle holdout",
            ),
        ), patch(
            "src.ml_predictor._build_preprocessor",
            side_effect=lambda *_: _FitAuditTransformer(),
        ):
            report = predict_target(data, "target")

        self.assertEqual(report["status"], "ok")
        self.assertTrue(_FitAuditTransformer.fitted_maxima)
        self.assertLess(max(_FitAuditTransformer.fitted_maxima), 100.0)

    def test_top_label_calibration_error_matches_known_oracles(self):
        perfect = _expected_calibration_error(
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([True, True, True, True]),
        )
        overconfident = _expected_calibration_error(
            np.array([0.9, 0.9, 0.9, 0.9]),
            np.array([True, False, True, False]),
        )

        self.assertAlmostEqual(perfect, 0.0)
        self.assertAlmostEqual(overconfident, 0.4)


if __name__ == "__main__":
    unittest.main()
