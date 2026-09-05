"""Holdout split-stability and subgroup model diagnostics."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, mean_absolute_error


MIN_DISTRIBUTION_ROWS = 5
MIN_SUBGROUP_ROWS = 5
MAX_FEATURES = 12
MAX_GROUPS = 10


def analyze_model_reliability(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    predictions,
    task_type: str,
) -> Dict[str, Any]:
    """Audit a random holdout without claiming production drift or fairness."""
    feature_stability = _feature_stability(X_train, X_test)
    subgroup_performance = _subgroup_performance(
        X_test,
        y_test,
        predictions,
        task_type,
    )
    target_stability = _target_stability(y_train, y_test, task_type)
    high_shift = sum(item["severity"] == "high" for item in feature_stability)
    material_gaps = sum(item["material_gap"] for item in subgroup_performance)
    status = "warning" if high_shift or material_gaps else "ok"
    return {
        "status": status,
        "scope": "random holdout reliability audit",
        "feature_stability": feature_stability,
        "target_stability": target_stability,
        "subgroup_performance": subgroup_performance,
        "summary": (
            f"Found {high_shift} high split-shift feature(s) and "
            f"{material_gaps} material subgroup performance gap(s)."
        ),
        "methodology_notes": [
            "Split stability compares the random train and holdout partitions; it is not production drift monitoring.",
            "Numeric features use the two-sample KS distance; categorical features use total variation distance.",
            "Subgroup results are descriptive and require enough holdout rows in every reported group.",
            "A performance gap is not by itself evidence of unlawful bias or unfairness.",
        ],
    }


def _feature_stability(X_train: pd.DataFrame, X_test: pd.DataFrame) -> List[Dict[str, Any]]:
    results = []
    for column in list(X_train.columns)[:MAX_FEATURES]:
        train = X_train[column]
        test = X_test[column]
        if pd.api.types.is_numeric_dtype(train):
            train_values = _finite_numeric(train)
            test_values = _finite_numeric(test)
            if len(train_values) < MIN_DISTRIBUTION_ROWS or len(test_values) < MIN_DISTRIBUTION_ROWS:
                continue
            statistic, p_value = stats.ks_2samp(train_values, test_values)
            score = float(statistic)
            result = {
                "feature": str(column),
                "feature_type": "numeric",
                "method": "two-sample KS distance",
                "distance": round(score, 6),
                "p_value_descriptive": round(float(p_value), 6),
            }
        else:
            train_distribution = _category_distribution(train)
            test_distribution = _category_distribution(test)
            levels = train_distribution.index.union(test_distribution.index)
            score = float(
                0.5
                * np.abs(
                    train_distribution.reindex(levels, fill_value=0).to_numpy()
                    - test_distribution.reindex(levels, fill_value=0).to_numpy()
                ).sum()
            )
            result = {
                "feature": str(column),
                "feature_type": "categorical",
                "method": "total variation distance",
                "distance": round(score, 6),
                "p_value_descriptive": None,
            }
        result["severity"] = _shift_severity(score)
        results.append(result)
    return sorted(results, key=lambda item: (-item["distance"], item["feature"]))


def _target_stability(y_train: pd.Series, y_test: pd.Series, task_type: str) -> Dict[str, Any]:
    if task_type == "regression":
        train = _finite_numeric(y_train)
        test = _finite_numeric(y_test)
        statistic, p_value = stats.ks_2samp(train, test)
        return {
            "method": "two-sample KS distance",
            "distance": round(float(statistic), 6),
            "p_value_descriptive": round(float(p_value), 6),
            "severity": _shift_severity(float(statistic)),
        }
    train_distribution = _category_distribution(y_train)
    test_distribution = _category_distribution(y_test)
    levels = train_distribution.index.union(test_distribution.index)
    distance = float(
        0.5
        * np.abs(
            train_distribution.reindex(levels, fill_value=0).to_numpy()
            - test_distribution.reindex(levels, fill_value=0).to_numpy()
        ).sum()
    )
    return {
        "method": "total variation distance",
        "distance": round(distance, 6),
        "p_value_descriptive": None,
        "severity": _shift_severity(distance),
    }


def _subgroup_performance(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions,
    task_type: str,
) -> List[Dict[str, Any]]:
    results = []
    prediction_series = pd.Series(predictions, index=X_test.index)
    target_series = pd.Series(np.asarray(y_test), index=X_test.index)
    for column in X_test.select_dtypes(exclude=[np.number]).columns:
        groups = X_test[column].fillna("__MISSING__").astype(str)
        group_counts = groups.value_counts()
        if not 2 <= len(group_counts) <= MAX_GROUPS or int(group_counts.min()) < MIN_SUBGROUP_ROWS:
            continue
        rows = []
        for group in group_counts.index:
            mask = groups == group
            if task_type == "classification":
                score = float(accuracy_score(target_series[mask], prediction_series[mask]))
            else:
                score = float(mean_absolute_error(target_series[mask], prediction_series[mask]))
            rows.append({"group": str(group), "rows": int(mask.sum()), "metric_value": round(score, 6)})
        values = [row["metric_value"] for row in rows]
        gap = float(max(values) - min(values))
        if task_type == "classification":
            worst = min(rows, key=lambda item: item["metric_value"])
            relative_gap = gap
            metric = "accuracy"
            material = gap >= 0.15
        else:
            worst = max(rows, key=lambda item: item["metric_value"])
            overall = float(mean_absolute_error(target_series, prediction_series))
            relative_gap = gap / overall if overall > 0 else 0.0
            metric = "MAE"
            material = relative_gap >= 0.50
        results.append(
            {
                "feature": str(column),
                "metric": metric,
                "groups": rows,
                "absolute_gap": round(gap, 6),
                "relative_gap": round(float(relative_gap), 6),
                "worst_group": worst["group"],
                "material_gap": bool(material),
            }
        )
    return sorted(results, key=lambda item: (-item["relative_gap"], item["feature"]))


def _finite_numeric(values: pd.Series) -> np.ndarray:
    return (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )


def _category_distribution(values: pd.Series) -> pd.Series:
    return values.fillna("__MISSING__").astype(str).value_counts(normalize=True)


def _shift_severity(distance: float) -> str:
    if distance >= 0.25:
        return "high"
    if distance >= 0.10:
        return "moderate"
    return "low"
