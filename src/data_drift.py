"""Persistent baseline profiles and baseline-to-current data drift analysis."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


PROFILE_VERSION = 1
MAX_PROFILE_FEATURES = 100
MAX_CATEGORIES = 20
NUMERIC_BINS = 10
EPSILON = 1e-6
MISSING_CATEGORY = "__DATA_PRISM_MISSING__"
OTHER_CATEGORY = "__DATA_PRISM_OTHER__"
CATEGORY_VALUE_PREFIX = "__DATA_PRISM_VALUE__:"
SEVERITY_ORDER = {"low": 0, "moderate": 1, "high": 2}


def create_baseline_profile(
    data: pd.DataFrame,
    *,
    dataset_name: str | None = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable aggregate profile without storing raw rows."""
    if not isinstance(data, pd.DataFrame) or data.empty or data.shape[1] == 0:
        raise ValueError("Baseline dataset must contain at least one row and one column.")

    columns: Dict[str, Dict[str, Any]] = {}
    column_map = _column_map(data)
    for feature, column in list(column_map.items())[:MAX_PROFILE_FEATURES]:
        series = data[column]
        feature_type = _feature_type(series)
        profile: Dict[str, Any] = {
            "feature_type": feature_type,
            "dtype": str(series.dtype),
            "missing_rate": round(_missing_rate(series, feature_type), 8),
            "non_null_count": int(series.notna().sum()),
        }
        if feature_type == "numeric":
            profile.update(_numeric_profile(series))
        else:
            profile.update(_categorical_profile(series))
        columns[feature] = profile

    return {
        "profile_version": PROFILE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "row_count": int(len(data)),
        "column_count": int(data.shape[1]),
        "profiled_column_count": len(columns),
        "columns_truncated": data.shape[1] > MAX_PROFILE_FEATURES,
        "columns": columns,
        "privacy_note": "The profile contains aggregate statistics and category labels, not raw rows.",
    }


def save_baseline_profile(profile: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    """Atomically persist a validated aggregate baseline profile."""
    _validate_profile(profile)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(profile, output, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_baseline_profile(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load and validate a persisted baseline profile."""
    with Path(path).open("r", encoding="utf-8") as source:
        profile = json.load(source)
    _validate_profile(profile)
    return profile


def compare_to_baseline(
    current_data: pd.DataFrame,
    baseline_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare a current batch with a previously persisted baseline profile."""
    if not isinstance(current_data, pd.DataFrame) or current_data.empty or current_data.shape[1] == 0:
        raise ValueError("Current dataset must contain at least one row and one column.")
    _validate_profile(baseline_profile)

    baseline_columns = baseline_profile["columns"]
    current_columns = _column_map(current_data)
    missing_columns = sorted(set(baseline_columns) - set(current_columns))
    new_columns = sorted(set(current_columns) - set(baseline_columns))
    type_changes: List[Dict[str, str]] = []
    feature_drift: List[Dict[str, Any]] = []

    for feature in sorted(set(baseline_columns) & set(current_columns)):
        baseline = baseline_columns[feature]
        current_series = current_data[current_columns[feature]]
        current_type = _feature_type(current_series)
        if current_type != baseline["feature_type"]:
            type_changes.append(
                {
                    "feature": feature,
                    "baseline_type": baseline["feature_type"],
                    "current_type": current_type,
                }
            )
            continue

        if current_type == "numeric":
            result = _compare_numeric(feature, current_series, baseline)
        else:
            result = _compare_categorical(feature, current_series, baseline)
        feature_drift.append(result)

    feature_drift.sort(
        key=lambda item: (-SEVERITY_ORDER[item["severity"]], -item["score"], item["feature"])
    )
    high_count = sum(item["severity"] == "high" for item in feature_drift)
    moderate_count = sum(item["severity"] == "moderate" for item in feature_drift)

    if missing_columns or type_changes or high_count:
        status = "critical"
    elif new_columns or moderate_count:
        status = "warning"
    else:
        status = "stable"

    return {
        "status": status,
        "baseline_created_at": baseline_profile["created_at"],
        "baseline_dataset_name": baseline_profile.get("dataset_name"),
        "baseline_rows": baseline_profile["row_count"],
        "current_rows": int(len(current_data)),
        "feature_drift": feature_drift,
        "schema_changes": {
            "missing_columns": missing_columns,
            "new_columns": new_columns,
            "type_changes": type_changes,
        },
        "summary": (
            f"Detected {high_count} high and {moderate_count} moderate feature drift signal(s); "
            f"{len(missing_columns)} missing, {len(new_columns)} new, and "
            f"{len(type_changes)} type-changed column(s)."
        ),
        "methodology_notes": [
            "Numeric drift uses Population Stability Index over bins fixed by the baseline.",
            "Categorical drift uses total variation distance over baseline categories and an other bucket.",
            "Missing-rate changes contribute to feature severity.",
            "Thresholds are operational screening rules and should be calibrated for each business context.",
            (
                "This compares an explicit baseline with the current batch; "
                "it does not schedule checks or retain drift history."
            ),
        ],
    }


def _numeric_profile(series: pd.Series) -> Dict[str, Any]:
    values = _finite_numeric(series)
    if not values:
        return {"bin_edges": [], "bin_proportions": [], "mean": None, "std": None}

    quantiles = np.quantile(values, np.linspace(0, 1, NUMERIC_BINS + 1))
    internal_edges = sorted({float(value) for value in quantiles[1:-1] if np.isfinite(value)})
    proportions = _bin_proportions(values, internal_edges)
    return {
        "bin_edges": [round(value, 12) for value in internal_edges],
        "bin_proportions": [round(value, 12) for value in proportions],
        "mean": round(float(np.mean(values)), 12),
        "std": round(float(np.std(values)), 12),
    }


def _categorical_profile(series: pd.Series) -> Dict[str, Any]:
    values = _category_values(series)
    counts = values.value_counts()
    categories = [
        str(value)
        for value in counts.index
        if value != MISSING_CATEGORY
    ][: MAX_CATEGORIES - 2]
    tracked_categories = categories + [MISSING_CATEGORY, OTHER_CATEGORY]
    proportions = _categorical_proportions(values, categories)
    return {
        "categories": tracked_categories,
        "category_proportions": [round(value, 12) for value in proportions],
    }


def _compare_numeric(
    feature: str,
    current: pd.Series,
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    values = _finite_numeric(current)
    baseline_proportions = baseline.get("bin_proportions", [])
    if values and baseline_proportions:
        current_proportions = _bin_proportions(values, baseline.get("bin_edges", []))
        score = _population_stability_index(baseline_proportions, current_proportions)
    else:
        score = 0.0
    return _feature_result(
        feature=feature,
        feature_type="numeric",
        metric="PSI",
        score=score,
        baseline_missing=float(baseline["missing_rate"]),
        current_missing=_missing_rate(current, "numeric"),
    )


def _compare_categorical(
    feature: str,
    current: pd.Series,
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    tracked = list(baseline.get("categories", []))
    categories = [value for value in tracked if value not in {MISSING_CATEGORY, OTHER_CATEGORY}]
    current_proportions = _categorical_proportions(_category_values(current), categories)
    baseline_proportions = baseline.get("category_proportions", [])
    score = (
        0.5
        * sum(
            abs(float(left) - float(right))
            for left, right in zip(baseline_proportions, current_proportions)
        )
        if len(baseline_proportions) == len(current_proportions)
        else 0.0
    )
    return _feature_result(
        feature=feature,
        feature_type="categorical",
        metric="TV distance",
        score=score,
        baseline_missing=float(baseline["missing_rate"]),
        current_missing=float(current.isna().mean()) if len(current) else 0.0,
    )


def _feature_result(
    *,
    feature: str,
    feature_type: str,
    metric: str,
    score: float,
    baseline_missing: float,
    current_missing: float,
) -> Dict[str, Any]:
    distribution_severity = _distribution_severity(score, metric)
    missing_delta = abs(current_missing - baseline_missing)
    missing_severity = _missing_severity(missing_delta)
    severity = max(
        (distribution_severity, missing_severity),
        key=lambda value: SEVERITY_ORDER[value],
    )
    return {
        "feature": feature,
        "feature_type": feature_type,
        "metric": metric,
        "score": round(float(score), 6),
        "baseline_missing_rate": round(baseline_missing, 6),
        "current_missing_rate": round(current_missing, 6),
        "missing_rate_delta": round(current_missing - baseline_missing, 6),
        "severity": severity,
    }


def _feature_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return "numeric"
    return "categorical"


def _finite_numeric(series: pd.Series) -> List[float]:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
        .tolist()
    )


def _category_values(series: pd.Series) -> pd.Series:
    values = series.astype(str).map(lambda value: f"{CATEGORY_VALUE_PREFIX}{value}")
    return values.where(series.notna(), MISSING_CATEGORY)


def _missing_rate(series: pd.Series, feature_type: str) -> float:
    if feature_type == "numeric":
        missing = (
            pd.to_numeric(series, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .isna()
        )
        return float(missing.mean()) if len(missing) else 0.0
    return float(series.isna().mean()) if len(series) else 0.0


def _bin_proportions(values: Iterable[float], edges: Iterable[float]) -> List[float]:
    numeric_values = np.asarray(list(values), dtype=float)
    edge_values = np.asarray(list(edges), dtype=float)
    bin_ids = np.searchsorted(edge_values, numeric_values, side="right")
    counts = np.bincount(bin_ids, minlength=len(edge_values) + 1).astype(float)
    return (counts / counts.sum()).tolist()


def _categorical_proportions(values: pd.Series, categories: List[str]) -> List[float]:
    known = set(categories)
    bucketed = values.map(
        lambda value: value if value == MISSING_CATEGORY or value in known else OTHER_CATEGORY
    )
    ordered = categories + [MISSING_CATEGORY, OTHER_CATEGORY]
    counts = bucketed.value_counts(normalize=True)
    return [float(counts.get(category, 0.0)) for category in ordered]


def _population_stability_index(expected: Iterable[float], actual: Iterable[float]) -> float:
    expected_values = np.asarray(list(expected), dtype=float)
    actual_values = np.asarray(list(actual), dtype=float)
    expected_values = np.clip(expected_values, EPSILON, None)
    actual_values = np.clip(actual_values, EPSILON, None)
    expected_values /= expected_values.sum()
    actual_values /= actual_values.sum()
    return float(np.sum((actual_values - expected_values) * np.log(actual_values / expected_values)))


def _distribution_severity(score: float, metric: str) -> str:
    if metric == "PSI":
        if score >= 0.25:
            return "high"
        if score >= 0.10:
            return "moderate"
        return "low"
    if score >= 0.25:
        return "high"
    if score >= 0.10:
        return "moderate"
    return "low"


def _missing_severity(delta: float) -> str:
    if delta >= 0.20:
        return "high"
    if delta >= 0.05:
        return "moderate"
    return "low"


def _validate_profile(profile: Dict[str, Any]) -> None:
    if not isinstance(profile, dict):
        raise ValueError("Baseline profile must be a dictionary.")
    if profile.get("profile_version") != PROFILE_VERSION:
        raise ValueError("Unsupported baseline profile version.")
    required = {"created_at", "row_count", "column_count", "columns"}
    if not required.issubset(profile) or not isinstance(profile["columns"], dict):
        raise ValueError("Baseline profile is incomplete.")
    for feature, column in profile["columns"].items():
        if not isinstance(feature, str) or not isinstance(column, dict):
            raise ValueError("Baseline profile contains an invalid column entry.")
        feature_type = column.get("feature_type")
        if feature_type not in {"numeric", "categorical"}:
            raise ValueError("Baseline profile contains an unsupported feature type.")
        if "missing_rate" not in column:
            raise ValueError("Baseline profile is missing column statistics.")
        if feature_type == "numeric" and not {
            "bin_edges",
            "bin_proportions",
        }.issubset(column):
            raise ValueError("Baseline numeric profile is incomplete.")
        if feature_type == "numeric" and len(column["bin_proportions"]) not in {
            0,
            len(column["bin_edges"]) + 1,
        }:
            raise ValueError("Baseline numeric bins are inconsistent.")
        if feature_type == "categorical" and not {
            "categories",
            "category_proportions",
        }.issubset(column):
            raise ValueError("Baseline categorical profile is incomplete.")
        if feature_type == "categorical" and len(column["categories"]) != len(
            column["category_proportions"]
        ):
            raise ValueError("Baseline categorical buckets are inconsistent.")


def _column_map(data: pd.DataFrame) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for column in data.columns:
        feature = str(column)
        if feature in result:
            raise ValueError("Dataset column names must be unique after string conversion.")
        result[feature] = column
    return result
