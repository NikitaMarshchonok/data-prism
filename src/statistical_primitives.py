"""Validated statistical primitives shared by analytical modules.

The functions in this module deliberately operate on plain numeric vectors.
Product-facing modules remain responsible for eligibility rules, semantic
column selection, reporting, and interpretation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import stats


DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_PROBABILITY_EPSILON = 1e-6


@dataclass(frozen=True)
class WelchResult:
    """Complete two-sided Welch comparison for ``left - right``."""

    statistic: float
    p_value: float
    degrees_of_freedom: float
    mean_difference: float
    standard_error: float
    confidence_interval: tuple[float, float]
    hedges_g: float


@dataclass(frozen=True)
class PearsonResult:
    """Two-sided Pearson result with a Fisher-z confidence interval."""

    correlation: float
    p_value: float
    sample_size: int
    confidence_interval: tuple[float, float]


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Adjust finite p-values while preserving their original order."""
    if len(p_values) == 0:
        return []
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("p-values must be a finite one-dimensional sequence")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be between 0 and 1")

    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = np.empty(len(values), dtype=float)
    running_minimum = 1.0
    for index in range(len(values) - 1, -1, -1):
        rank = index + 1
        candidate = float(ranked[index] * len(values) / rank)
        running_minimum = min(running_minimum, candidate)
        adjusted_ranked[index] = min(running_minimum, 1.0)

    adjusted = np.empty(len(values), dtype=float)
    adjusted[order] = adjusted_ranked
    return [float(value) for value in adjusted]


def pearson_confidence_interval(
    correlation: float,
    sample_size: int,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Return a Fisher-z interval for a Pearson correlation coefficient."""
    correlation = float(correlation)
    if not np.isfinite(correlation) or not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must be finite and between -1 and 1")
    if isinstance(sample_size, bool) or not isinstance(sample_size, (int, np.integer)):
        raise TypeError("sample_size must be an integer")
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    _validate_confidence_level(confidence_level)
    if sample_size <= 3:
        return -1.0, 1.0
    if abs(correlation) == 1.0:
        return correlation, correlation

    fisher_z = float(np.arctanh(correlation))
    critical = float(stats.norm.ppf(0.5 + confidence_level / 2.0))
    margin = critical / float(np.sqrt(sample_size - 3))
    return float(np.tanh(fisher_z - margin)), float(np.tanh(fisher_z + margin))


def pearson_summary(
    left: Iterable[float],
    right: Iterable[float],
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> PearsonResult | None:
    """Return a two-sided Pearson test and Fisher-z interval."""
    _validate_confidence_level(confidence_level)
    left_values = _finite_vector(left, "left")
    right_values = _finite_vector(right, "right")
    if len(left_values) != len(right_values):
        raise ValueError("Pearson vectors must have equal length")
    if (
        len(left_values) < 2
        or np.unique(left_values).size < 2
        or np.unique(right_values).size < 2
    ):
        return None

    result = stats.pearsonr(left_values, right_values)
    correlation = float(result.statistic)
    p_value = float(result.pvalue)
    if not np.isfinite(correlation) or not np.isfinite(p_value):
        return None
    interval = pearson_confidence_interval(
        correlation,
        len(left_values),
        confidence_level=confidence_level,
    )
    return PearsonResult(
        correlation=correlation,
        p_value=float(np.clip(p_value, 0.0, 1.0)),
        sample_size=len(left_values),
        confidence_interval=interval,
    )


def welch_independent_summary(
    left: Iterable[float],
    right: Iterable[float],
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> WelchResult | None:
    """Return Welch inference and Hedges' g for ``left - right``.

    ``None`` is returned when a finite inferential result cannot be defined,
    including fewer than two observations per group or zero variance in both
    groups. Callers can then expose an explicit insufficient-evidence state.
    """
    _validate_confidence_level(confidence_level)
    left_values = _finite_vector(left, "left")
    right_values = _finite_vector(right, "right")
    if len(left_values) < 2 or len(right_values) < 2:
        return None

    left_variance = float(np.var(left_values, ddof=1))
    right_variance = float(np.var(right_values, ddof=1))
    standard_error_squared = (
        left_variance / len(left_values) + right_variance / len(right_values)
    )
    if not np.isfinite(standard_error_squared) or standard_error_squared <= 0.0:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        test = stats.ttest_ind(
            left_values,
            right_values,
            equal_var=False,
            nan_policy="omit",
        )
    statistic = float(test.statistic)
    p_value = float(test.pvalue)
    degrees_of_freedom = float(test.df)
    if not all(np.isfinite(value) for value in (statistic, p_value, degrees_of_freedom)):
        return None
    if degrees_of_freedom <= 0.0:
        return None

    mean_difference = float(np.mean(left_values) - np.mean(right_values))
    standard_error = float(np.sqrt(standard_error_squared))
    critical = float(
        stats.t.ppf(0.5 + confidence_level / 2.0, degrees_of_freedom)
    )
    margin = critical * standard_error

    pooled_denominator = len(left_values) + len(right_values) - 2
    pooled_variance = (
        (len(left_values) - 1) * left_variance
        + (len(right_values) - 1) * right_variance
    ) / pooled_denominator
    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        return None
    cohen_d = mean_difference / float(np.sqrt(pooled_variance))
    correction = 1.0 - (
        3.0 / (4.0 * (len(left_values) + len(right_values)) - 9.0)
    )
    hedges_g = float(cohen_d * correction)

    return WelchResult(
        statistic=statistic,
        p_value=float(np.clip(p_value, 0.0, 1.0)),
        degrees_of_freedom=degrees_of_freedom,
        mean_difference=mean_difference,
        standard_error=standard_error,
        confidence_interval=(mean_difference - margin, mean_difference + margin),
        hedges_g=hedges_g,
    )


def population_stability_index(
    expected: Iterable[float],
    actual: Iterable[float],
    *,
    epsilon: float = DEFAULT_PROBABILITY_EPSILON,
) -> float:
    """Return PSI for two equally sized probability vectors."""
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    expected_values, actual_values = _paired_probability_vectors(expected, actual)
    expected_values = np.clip(expected_values, epsilon, None)
    actual_values = np.clip(actual_values, epsilon, None)
    expected_values /= expected_values.sum()
    actual_values /= actual_values.sum()
    return float(
        np.sum(
            (actual_values - expected_values)
            * np.log(actual_values / expected_values)
        )
    )


def total_variation_distance(
    expected: Iterable[float],
    actual: Iterable[float],
) -> float:
    """Return total variation distance for two probability vectors."""
    expected_values, actual_values = _paired_probability_vectors(expected, actual)
    expected_values /= expected_values.sum()
    actual_values /= actual_values.sum()
    return float(0.5 * np.sum(np.abs(expected_values - actual_values)))


def _finite_vector(values: Iterable[float], name: str) -> np.ndarray:
    result = np.asarray(list(values), dtype=float)
    if result.ndim != 1 or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional sequence")
    return result


def _paired_probability_vectors(
    expected: Iterable[float],
    actual: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    expected_values = _finite_vector(expected, "expected")
    actual_values = _finite_vector(actual, "actual")
    if len(expected_values) == 0 or len(expected_values) != len(actual_values):
        raise ValueError("probability vectors must be non-empty and equally sized")
    if np.any(expected_values < 0.0) or np.any(actual_values < 0.0):
        raise ValueError("probability vectors cannot contain negative values")
    if expected_values.sum() <= 0.0 or actual_values.sum() <= 0.0:
        raise ValueError("probability vectors must have positive mass")
    return expected_values, actual_values


def _validate_confidence_level(confidence_level: float) -> None:
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and between 0 and 1")
