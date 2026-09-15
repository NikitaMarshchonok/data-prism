"""Bounded, deterministic comparison of two in-memory data periods."""

from __future__ import annotations

import json
import math
import warnings
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats

from src.data_drift import compare_to_baseline, create_baseline_profile
from .readiness_engine import DatasetReadinessEngine

MAX_ROWS = 100_000
MAX_COLUMNS = 100
# Resource limits apply to the comparison as a whole, not independently to
# each uploaded frame.  Keeping these explicit prevents two individually
# valid periods from doubling the in-process work budget.
MAX_TOTAL_ROWS = 100_000
MAX_TOTAL_COLUMNS = 100
MAX_REQUEST_BYTES = 100 * 1024 * 1024
# Object-backed pandas columns can use substantially more memory than their
# CSV representation.  Keep the in-memory comparison bounded independently of
# the request body's byte limit.
MAX_MEMORY_BYTES = 256 * 1024 * 1024
MAX_INFERENTIAL_CANDIDATES = 32
MIN_SAMPLE = 8


class PeriodComparisonError(ValueError):
    """Safe, typed error for invalid or non-analyzable period inputs."""

    def __init__(self, reason: str, *, readiness: Dict[str, Any] | None = None):
        self.reason = str(reason)
        self.readiness = readiness
        super().__init__(self.reason)


def build_period_comparison(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    baseline_label: str = "Baseline",
    current_label: str = "Current",
    max_metrics: int = 8,
) -> Dict[str, Any]:
    """Build a JSON-safe aggregate comparison without retaining rows."""
    _validate_frame(baseline_df, "baseline")
    _validate_frame(current_df, "current")
    # Pandas label lookup treats some distinct non-string labels as equal
    # (notably ``True`` and ``1``).  Downstream drift/readiness helpers also
    # use string feature names, so use positional, normalized labels for the
    # complete comparison contract.  Duplicate normalized names are rejected
    # by the readiness engine instead of being silently merged.
    baseline_df = _normalized_frame(baseline_df)
    current_df = _normalized_frame(current_df)
    if len(baseline_df) + len(current_df) > MAX_TOTAL_ROWS:
        raise ValueError(f"comparison exceeds the {MAX_TOTAL_ROWS} combined row limit")
    if baseline_df.shape[1] + current_df.shape[1] > MAX_TOTAL_COLUMNS:
        raise ValueError(f"comparison exceeds the {MAX_TOTAL_COLUMNS} combined column limit")
    combined_memory = int(
        baseline_df.memory_usage(index=True, deep=True).sum()
        + current_df.memory_usage(index=True, deep=True).sum()
    )
    if combined_memory > MAX_MEMORY_BYTES:
        raise ValueError(f"comparison exceeds the {MAX_MEMORY_BYTES} combined byte memory limit")
    baseline_label = _label(baseline_label, "baseline_label")
    current_label = _label(current_label, "current_label")
    if baseline_label.casefold() == current_label.casefold():
        raise ValueError("baseline_label and current_label must be distinct")
    if isinstance(max_metrics, bool) or not isinstance(max_metrics, int) or not 1 <= max_metrics <= 8:
        raise ValueError("max_metrics must be an integer between 1 and 8")

    baseline_readiness = DatasetReadinessEngine(baseline_df).assess()
    current_readiness = DatasetReadinessEngine(current_df).assess()
    if not baseline_readiness["analysis_allowed"] or not current_readiness["analysis_allowed"]:
        blocked = "baseline" if not baseline_readiness["analysis_allowed"] else "current"
        readiness = {"baseline": baseline_readiness, "current": current_readiness}
        raise PeriodComparisonError(
            f"{blocked} period is not eligible for comparison (dataset readiness is blocked)",
            readiness=readiness,
        )

    # The existing drift contract is used in memory only; timestamps are omitted
    # because they would make equivalent inputs produce different JSON.
    profile = create_baseline_profile(baseline_df, dataset_name=baseline_label)
    drift = compare_to_baseline(current_df, profile)
    schema_changes = drift.get("schema_changes", {"missing_columns": [], "new_columns": [], "type_changes": []})
    eligible_metrics = _metric_candidates(baseline_df, current_df)
    # The cap bounds the number of Welch tests, but an alphabetically early
    # block of constant/undersized columns must not crowd every analyzable
    # metric out of the candidate set.  Keep testable metrics first, then use
    # the remaining slots for explicit insufficient-evidence diagnostics.
    inferential_metrics = [
        metric for metric in eligible_metrics if metric["_inferential_eligible"]
    ]
    non_inferential_metrics = [
        metric for metric in eligible_metrics if not metric["_inferential_eligible"]
    ]
    inferential_count = min(len(inferential_metrics), MAX_INFERENTIAL_CANDIDATES)
    metrics = inferential_metrics[:inferential_count] + non_inferential_metrics[
        : MAX_INFERENTIAL_CANDIDATES - inferential_count
    ]
    truncated_count = max(0, len(eligible_metrics) - len(metrics))
    for metric in metrics:
        inferential_eligible = metric.pop("_inferential_eligible", False)
        column_key = metric.pop("_column_key", metric["column"])
        if inferential_eligible:
            _populate_inferential_metric(metric, baseline_df, current_df, column_key)

    # Apply BH only after all selected Welch tests have run, so the correction
    # covers every selected tested metric while excluding non-finite tests.
    adjusted = _benjamini_hochberg(
        [m["raw_p_value"] for m in metrics if m["raw_p_value"] is not None]
    )
    adjusted_iter = iter(adjusted)
    for metric in metrics:
        if metric["raw_p_value"] is not None:
            adjusted_p_value = next(adjusted_iter)
            metric["significant"] = bool(adjusted_p_value <= 0.05)
            metric["adjusted_p_value"] = _round(adjusted_p_value)
            # Round only at the public-report boundary. BH must consume the
            # full-precision raw p-value so borderline results are not moved
            # across the significance threshold by display rounding.
            metric["raw_p_value"] = _round(metric["raw_p_value"])
        else:
            metric["adjusted_p_value"] = None
            metric["significant"] = False
    metrics.sort(key=lambda m: (m["adjusted_p_value"] if m["adjusted_p_value"] is not None else 1.0, -abs(m["absolute_change"] or 0.0), m["column"]))
    displayed = metrics[:max_metrics]
    significant = sum(1 for m in metrics if m["significant"])
    drift_signals = _safe_drift_signals(drift.get("feature_drift", []))
    drift_status = drift.get("status", "stable")
    if drift_status in {"critical", "warning"} or schema_changes.get("missing_columns") or schema_changes.get("type_changes") or schema_changes.get("new_columns") or significant:
        status = "review_required"
    else:
        status = "stable"
    summary = (
        f"Compared {len(eligible_metrics)} eligible numeric metric(s); tested {sum(m['raw_p_value'] is not None for m in metrics)} "
        f"and displayed {len(displayed)}. {significant} statistically significant after FDR; "
        f"{len(drift_signals)} distribution drift signal(s)."
    )
    if truncated_count:
        summary += f" {truncated_count} candidate(s) were truncated by the inferential limit."
    result = {
        "contract": "period-comparison-v1",
        "status": status,
        "summary": summary,
        "input": {
            "baseline": {"label": baseline_label, "rows": int(len(baseline_df)), "columns": int(baseline_df.shape[1]), "readiness": baseline_readiness},
            "current": {"label": current_label, "rows": int(len(current_df)), "columns": int(current_df.shape[1]), "readiness": current_readiness},
        },
        "schema_changes": schema_changes,
        "distribution_drift": {"status": drift_status, "signals": drift_signals},
        "numeric_metrics": {"eligible_count": len(eligible_metrics), "candidate_count": len(metrics), "tested_count": sum(m["raw_p_value"] is not None for m in metrics), "truncated_count": truncated_count, "display_count": len(displayed), "significant_count": significant, "metrics": displayed},
        "limitations": [
            "Observational comparison; it is noncausal and does not establish that period changes caused outcomes.",
            "Rows are treated as independent observations from the represented populations.",
            "Seasonality, changing population mix, and confounding may explain differences.",
            "FDR and drift thresholds are screening rules, not proof of material impact.",
        ],
    }
    # Enforce the public JSON contract at the boundary.
    json.dumps(result, allow_nan=False)
    return result


def _validate_frame(frame: Any, name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name}_df must be a pandas DataFrame")
    rows, columns = frame.shape
    if rows > MAX_ROWS:
        raise ValueError(f"{name}_df exceeds the {MAX_ROWS} row limit")
    if columns > MAX_COLUMNS:
        raise ValueError(f"{name}_df exceeds the {MAX_COLUMNS} column limit")
    memory_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    if memory_bytes > MAX_MEMORY_BYTES:
        raise ValueError(f"{name}_df exceeds the {MAX_MEMORY_BYTES} byte memory limit")


def _normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a shallow frame copy with stable string column labels.

    Assigning labels on a shallow copy avoids copying the potentially large
    data blocks while ensuring helpers never perform ambiguous pandas lookups
    for non-string labels.
    """
    normalized = frame.copy(deep=False)
    normalized.columns = [str(column).strip() for column in frame.columns]
    return normalized


def _label(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if not value or len(value) > 80 or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{name} must be non-empty, at most 80 characters, and contain no control characters")
    return value


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _metric_candidates(baseline: pd.DataFrame, current: pd.DataFrame) -> List[Dict[str, Any]]:
    common = sorted(set(baseline.columns) & set(current.columns), key=str)
    result: List[Dict[str, Any]] = []
    for column in common:
        if pd.api.types.is_bool_dtype(baseline[column]) or pd.api.types.is_bool_dtype(current[column]):
            continue
        # Complex values do not have a meaningful scalar mean difference for
        # this report; coercing them to float would silently discard the
        # imaginary component.
        if pd.api.types.is_complex_dtype(baseline[column]) or pd.api.types.is_complex_dtype(current[column]):
            continue
        if not (pd.api.types.is_numeric_dtype(baseline[column]) and pd.api.types.is_numeric_dtype(current[column])):
            continue
        b, c = _finite(baseline[column]), _finite(current[column])
        bm = float(np.mean(b)) if len(b) else None
        cm = float(np.mean(c)) if len(c) else None
        diff = (cm - bm) if bm is not None and cm is not None else None
        entry: Dict[str, Any] = {
            "column": str(column), "status": "ok", "baseline": _descriptive(b, baseline[column]),
            "current": _descriptive(c, current[column]), "absolute_change": _round(diff),
            "percent_change": None, "raw_p_value": None, "adjusted_p_value": None,
            "confidence_interval": None, "hedges_g": None, "magnitude": None, "significant": False,
        }
        entry["_column_key"] = column
        if bm is not None and bm > 0 and diff is not None:
            entry["percent_change"] = _round(100.0 * diff / bm)
            entry["percent_change_reason"] = "Reported relative to a strictly positive baseline mean."
        else:
            entry["percent_change_reason"] = "Not reported because baseline mean is not strictly positive."
        inferential_eligible = not (
            len(b) < MIN_SAMPLE
            or len(c) < MIN_SAMPLE
            or (np.var(b) == 0 and np.var(c) == 0)
        )
        entry["_inferential_eligible"] = inferential_eligible
        if not inferential_eligible:
            entry["status"] = "insufficient_evidence"
            result.append(entry)
            continue
        result.append(entry)
    return result


def _populate_inferential_metric(
    entry: Dict[str, Any],
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    column_key: Any,
) -> None:
    """Run the bounded inferential calculation for one selected metric."""
    b, c = _finite(baseline[column_key]), _finite(current[column_key])
    # Welch's implementation emits a RuntimeWarning for a constant group even
    # though the one-constant/one-varying case is still analyzable. Non-finite
    # outputs become an explicit insufficient-evidence result.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        test = stats.ttest_ind(c, b, equal_var=False)
        baseline_variance = float(np.var(b, ddof=1))
        current_variance = float(np.var(c, ddof=1))
    se = math.sqrt(current_variance / len(c) + baseline_variance / len(b))
    dof = float(test.df)
    if (
        not np.isfinite(se)
        or se <= 0
        or not np.isfinite(dof)
        or dof <= 0
        or not np.isfinite(test.pvalue)
    ):
        entry["status"] = "insufficient_evidence"
        return
    crit = float(stats.t.ppf(0.975, dof))
    delta = float(np.mean(c) - np.mean(b))
    pooled = math.sqrt(
        ((len(c) - 1) * current_variance + (len(b) - 1) * baseline_variance)
        / (len(c) + len(b) - 2)
    )
    d = delta / pooled if pooled else None
    correction = 1.0 - 3.0 / (4.0 * (len(b) + len(c)) - 9.0)
    g = d * correction if d is not None else None
    entry["method"] = "Welch independent-samples t-test"
    entry["null_hypothesis"] = "The baseline and current population means are equal."
    entry["estimate"] = {
        "parameter": "mean_difference",
        "direction": "current_minus_baseline",
        "value": _round(delta),
    }
    entry["alpha"] = 0.05
    entry["fdr_method"] = "Benjamini-Hochberg"
    entry["raw_p_value"] = float(test.pvalue)
    entry["confidence_interval"] = {
        "level": 0.95,
        "parameter": "current_minus_baseline_mean",
        "lower": _round(delta - crit * se),
        "upper": _round(delta + crit * se),
    }
    entry["hedges_g"] = _round(g)
    entry["magnitude"] = _magnitude(g)


def _descriptive(values: np.ndarray, original: pd.Series) -> Dict[str, Any]:
    # Non-finite numeric values are excluded from the estimates and should be
    # reported as unavailable observations rather than silently counted as
    # observed in the missingness rate.
    finite_mask = np.isfinite(pd.to_numeric(original, errors="coerce").to_numpy(dtype=float))
    unavailable = int((~finite_mask).sum())
    return {"n": int(len(values)), "mean": _round(np.mean(values) if len(values) else None), "median": _round(np.median(values) if len(values) else None), "missing_rate": _round(float(unavailable / len(original)) if len(original) else 1.0)}


def _benjamini_hochberg(p_values: Iterable[float]) -> List[float]:
    values = list(p_values)
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    adjusted = [1.0] * len(values)
    running = 1.0
    for rank in range(len(values), 0, -1):
        index = order[rank - 1]
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def _safe_drift_signals(signals: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in signal.items() if k in {"feature", "feature_type", "metric", "score", "severity", "baseline_missing_rate", "current_missing_rate", "missing_rate_delta"}} for signal in signals]


def _round(value: Any) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 8)


def _magnitude(value: float | None) -> str | None:
    if value is None:
        return None
    absolute = abs(value)
    return "negligible" if absolute < 0.2 else "small" if absolute < 0.5 else "medium" if absolute < 0.8 else "large"
