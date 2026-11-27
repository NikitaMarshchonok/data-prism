"""Comparisons across several independent groups.

Plain numeric functions: eligibility, column selection and wording stay in
the product modules.  Means are compared with Welch's heteroscedastic ANOVA.
Up to ``MAX_BOOTSTRAP_ROWS`` rows its p-value and the pairwise comparisons come
from a seeded bootstrap of the studentized statistics on group-centred data
(bootstrap-t; pairwise by the single-step max-|t| method), which keeps the
false alarm rate close to nominal on skewed data with unequal groups, where
the F approximation is anti-conservative.  Larger files use the asymptotic F
test and pairwise Welch t-tests (Holm-adjusted, Bonferroni intervals).  Rates are compared with Pearson's chi-square test when every
expected count is at least five, Fisher's exact test for sparse 2x2 tables,
and a seeded Monte Carlo version of the exact conditional test otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from scipy import stats


MIN_EXPECTED_COUNT = 5.0
MONTE_CARLO_DRAWS = 4999
MONTE_CARLO_SEED = 20261127
BOOTSTRAP_DRAWS = 999
# When the first draws leave the observed statistic in the extreme tail, more
# draws refine the p-value floor from 0.001 to 0.0001.  Without this a strong
# difference could never pass the FDR step in a scan of ~90 hypotheses
# (0.05 / 90 < 0.001).
BOOTSTRAP_EXTENDED_DRAWS = 9999
BOOTSTRAP_EXTEND_BELOW = 5
BOOTSTRAP_BATCH = 200
MAX_BOOTSTRAP_ROWS = 5000


@dataclass(frozen=True)
class WelchAnovaResult:
    statistic: float
    p_value: float
    df_between: float
    df_within: float
    omega_squared: float
    sizes: tuple[int, ...]
    means: tuple[float, ...]


@dataclass(frozen=True)
class PairwiseDifference:
    left: int
    right: int
    difference: float
    p_value: float
    adjusted_p_value: float
    confidence_interval: tuple[float, float]


@dataclass(frozen=True)
class GroupMeansComparison:
    anova: WelchAnovaResult
    p_value: float
    method: str
    pairs: list[PairwiseDifference]


@dataclass(frozen=True)
class RateTestResult:
    statistic: float
    p_value: float
    method: str
    cramers_v: float
    min_expected_count: float


def welch_anova(groups: Sequence[np.ndarray]) -> WelchAnovaResult | None:
    """Welch's F test for equal means without assuming equal variances.

    Returns ``None`` when a group has fewer than two values or no variation,
    because the precision weights ``n / s^2`` are then undefined.
    """
    if len(groups) < 2:
        return None
    sizes = np.array([len(group) for group in groups], dtype=float)
    if np.any(sizes < 2):
        return None
    means = np.array([float(np.mean(group)) for group in groups])
    variances = np.array([float(np.var(group, ddof=1)) for group in groups])
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        return None
    k = len(groups)
    weights = sizes / variances
    total_weight = float(weights.sum())
    weighted_mean = float(np.sum(weights * means) / total_weight)
    between = float(np.sum(weights * (means - weighted_mean) ** 2) / (k - 1))
    lam = float(np.sum((1.0 - weights / total_weight) ** 2 / (sizes - 1.0)))
    denominator = 1.0 + 2.0 * (k - 2) / (k * k - 1.0) * lam
    statistic = between / denominator
    df_between = float(k - 1)
    df_within = float((k * k - 1.0) / (3.0 * lam)) if lam > 0 else float("inf")
    p_value = float(stats.f.sf(statistic, df_between, df_within))
    if not all(np.isfinite(value) for value in (statistic, p_value)):
        return None
    total = float(sizes.sum())
    # omega^2 recovered from the F statistic (approximate for Welch's F).
    numerator = df_between * (statistic - 1.0)
    omega_squared = max(0.0, numerator / (numerator + total))
    return WelchAnovaResult(
        statistic=float(statistic),
        p_value=float(np.clip(p_value, 0.0, 1.0)),
        df_between=df_between,
        df_within=df_within,
        omega_squared=float(omega_squared),
        sizes=tuple(int(size) for size in sizes),
        means=tuple(float(mean) for mean in means),
    )


def _welch_f(means: np.ndarray, variances: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Vectorised Welch F over the last axis (one row per replicate)."""
    k = sizes.shape[-1]
    weights = sizes / variances
    total = weights.sum(axis=-1, keepdims=True)
    centre = (weights * means).sum(axis=-1, keepdims=True) / total
    between = (weights * (means - centre) ** 2).sum(axis=-1) / (k - 1)
    lam = ((1.0 - weights / total) ** 2 / (sizes - 1.0)).sum(axis=-1)
    return between / (1.0 + 2.0 * (k - 2) / (k * k - 1.0) * lam)


def _bootstrap_null(
    centred: Sequence[np.ndarray],
    sizes: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch F and max pairwise |t| on resamples of group-centred values."""
    k = len(centred)
    f_star = np.empty(draws)
    max_t = np.empty(draws)
    for start in range(0, draws, BOOTSTRAP_BATCH):
        count = min(BOOTSTRAP_BATCH, draws - start)
        boot_means = np.empty((count, k))
        boot_vars = np.empty((count, k))
        for index, values in enumerate(centred):
            sample = values[rng.integers(0, len(values), size=(count, len(values)), dtype=np.int32)]
            boot_means[:, index] = sample.mean(axis=1)
            boot_vars[:, index] = sample.var(axis=1, ddof=1)
        # A resample with no spread has no defined weight; it counts against
        # significance (inf), instead of silently dropping out as NaN.
        degenerate = np.any(boot_vars <= 0.0, axis=1)
        safe_vars = np.where(boot_vars > 0.0, boot_vars, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            f_values = _welch_f(boot_means, safe_vars, sizes)
            boot_se = np.sqrt(safe_vars[:, left] / sizes[left] + safe_vars[:, right] / sizes[right])
            t_values = np.abs(boot_means[:, left] - boot_means[:, right]) / boot_se
            t_max = np.nanmax(np.where(np.isnan(t_values), -np.inf, t_values), axis=1)
        f_star[start:start + count] = np.where(degenerate, np.inf, f_values)
        max_t[start:start + count] = np.where(degenerate, np.inf, t_max)
    return f_star, max_t


def compare_group_means(
    groups: Sequence[np.ndarray],
    confidence_level: float = 0.95,
    *,
    seed: int = MONTE_CARLO_SEED,
) -> GroupMeansComparison | None:
    """Welch ANOVA plus all pairwise differences ``mean[left] - mean[right]``.

    Pairwise p-values are family-wise adjusted and the intervals hold
    simultaneously at ``confidence_level``.
    """
    anova = welch_anova(groups)
    if anova is None:
        return None
    sizes = np.array(anova.sizes, dtype=float)
    if sizes.sum() > MAX_BOOTSTRAP_ROWS:
        return GroupMeansComparison(
            anova=anova,
            p_value=anova.p_value,
            method="Welch's ANOVA with Holm-adjusted pairwise Welch tests",
            pairs=pairwise_welch(groups, confidence_level),
        )
    rng = np.random.default_rng(seed)
    k = len(groups)
    pairs = list(combinations(range(k), 2))
    left = np.array([pair[0] for pair in pairs])
    right = np.array([pair[1] for pair in pairs])
    means = np.array(anova.means)
    variances = np.array([float(np.var(group, ddof=1)) for group in groups])
    se = np.sqrt(variances[left] / sizes[left] + variances[right] / sizes[right])
    observed_t = np.abs(means[left] - means[right]) / se
    centred = [np.asarray(group, dtype=float) - float(np.mean(group)) for group in groups]
    f_star, max_t = _bootstrap_null(centred, sizes, left, right, BOOTSTRAP_DRAWS, rng)
    if np.sum(f_star >= anova.statistic) < BOOTSTRAP_EXTEND_BELOW:
        more_f, more_t = _bootstrap_null(
            centred, sizes, left, right, BOOTSTRAP_EXTENDED_DRAWS - BOOTSTRAP_DRAWS, rng
        )
        f_star, max_t = np.concatenate([f_star, more_f]), np.concatenate([max_t, more_t])
    draws = len(f_star)
    p_value = float((1 + np.sum(f_star >= anova.statistic)) / (draws + 1))
    critical = float(np.quantile(max_t, confidence_level, method="higher"))
    result = []
    for index, (a, b) in enumerate(pairs):
        difference = float(means[a] - means[b])
        adjusted = float((1 + np.sum(max_t >= observed_t[index])) / (draws + 1))
        margin = critical * float(se[index])
        result.append(PairwiseDifference(a, b, difference, adjusted, adjusted, (difference - margin, difference + margin)))
    return GroupMeansComparison(
        anova=anova,
        p_value=p_value,
        method=f"Welch's ANOVA, bootstrap-t ({draws} resamples) with max-|t| pairwise comparisons",
        pairs=result,
    )


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjustment (family-wise error), original order kept."""
    values = np.asarray(p_values, dtype=float)
    m = len(values)
    if m == 0:
        return []
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, float(values[index]) * (m - rank)))
        adjusted[index] = running
    return [float(value) for value in adjusted]


def pairwise_welch(groups: Sequence[np.ndarray], confidence_level: float = 0.95) -> list[PairwiseDifference]:
    """All pairwise Welch comparisons ``mean[left] - mean[right]``.

    P-values are Holm-adjusted across the pairs; intervals use the Bonferroni
    level so that all of them hold simultaneously at ``confidence_level``.
    """
    pairs = list(combinations(range(len(groups)), 2))
    if not pairs:
        return []
    level = 1.0 - (1.0 - confidence_level) / len(pairs)
    raw = []
    for left, right in pairs:
        a, b = groups[left], groups[right]
        va, vb = float(np.var(a, ddof=1)) / len(a), float(np.var(b, ddof=1)) / len(b)
        se = float(np.sqrt(va + vb))
        difference = float(np.mean(a) - np.mean(b))
        df = (va + vb) ** 2 / (va * va / (len(a) - 1) + vb * vb / (len(b) - 1))
        statistic = difference / se
        p_value = float(2.0 * stats.t.sf(abs(statistic), df))
        margin = float(stats.t.ppf(0.5 + level / 2.0, df)) * se
        raw.append((left, right, difference, p_value, (difference - margin, difference + margin)))
    adjusted = holm([item[3] for item in raw])
    return [
        PairwiseDifference(left, right, difference, p_value, adjusted_p, interval)
        for (left, right, difference, p_value, interval), adjusted_p in zip(raw, adjusted)
    ]


def wilson_interval(successes: int, total: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("total must be positive")
    z = float(stats.norm.ppf(0.5 + confidence_level / 2.0))
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def newcombe_difference(
    successes_a: int, total_a: int, successes_b: int, total_b: int, confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Newcombe hybrid score interval for ``p_a - p_b`` (method 10)."""
    p_a, p_b = successes_a / total_a, successes_b / total_b
    low_a, high_a = wilson_interval(successes_a, total_a, confidence_level)
    low_b, high_b = wilson_interval(successes_b, total_b, confidence_level)
    difference = p_a - p_b
    lower = difference - np.sqrt((p_a - low_a) ** 2 + (high_b - p_b) ** 2)
    upper = difference + np.sqrt((high_a - p_a) ** 2 + (p_b - low_b) ** 2)
    return float(max(-1.0, lower)), float(min(1.0, upper))


def _chi_square(successes: np.ndarray, sizes: np.ndarray, rate: float) -> np.ndarray:
    expected_success = sizes * rate
    expected_failure = sizes * (1.0 - rate)
    failures = sizes - successes
    return (
        np.sum((successes - expected_success) ** 2 / expected_success, axis=-1)
        + np.sum((failures - expected_failure) ** 2 / expected_failure, axis=-1)
    )


def rate_difference_test(successes: Sequence[int], sizes: Sequence[int]) -> RateTestResult | None:
    """Test equal event rates across groups in a 2 x k table.

    Returns ``None`` when no group has the event or every group has it.
    """
    x = np.asarray(successes, dtype=float)
    n = np.asarray(sizes, dtype=float)
    if len(x) < 2 or len(x) != len(n) or np.any(n <= 0) or np.any(x < 0) or np.any(x > n):
        return None
    total, events = float(n.sum()), float(x.sum())
    if events <= 0 or events >= total:
        return None
    rate = events / total
    min_expected = float(min((n * rate).min(), (n * (1.0 - rate)).min()))
    statistic = float(_chi_square(x, n, rate))
    k = len(x)
    if min_expected >= MIN_EXPECTED_COUNT:
        p_value = float(stats.chi2.sf(statistic, k - 1))
        method = "Pearson chi-square test"
    elif k == 2:
        table = [[int(x[0]), int(n[0] - x[0])], [int(x[1]), int(n[1] - x[1])]]
        p_value = float(stats.fisher_exact(table).pvalue)
        method = "Fisher's exact test"
    else:
        rng = np.random.default_rng(MONTE_CARLO_SEED)
        draws = rng.multivariate_hypergeometric(n.astype(np.int64), int(events), size=MONTE_CARLO_DRAWS)
        simulated = _chi_square(draws.astype(float), n, rate)
        exceed = int(np.sum(simulated >= statistic - 1e-9 * max(1.0, statistic)))
        p_value = (1.0 + exceed) / (MONTE_CARLO_DRAWS + 1.0)
        method = f"Exact conditional test (Monte Carlo, {MONTE_CARLO_DRAWS} draws)"
    phi2 = statistic / total
    phi2_corrected = max(0.0, phi2 - (k - 1) / (total - 1.0))
    k_corrected = k - (k - 1) ** 2 / (total - 1.0)
    r_corrected = 2.0 - 1.0 / (total - 1.0)
    scale = min(k_corrected - 1.0, r_corrected - 1.0)
    cramers_v = float(np.sqrt(phi2_corrected / scale)) if scale > 0 else 0.0
    return RateTestResult(
        statistic=statistic,
        p_value=float(np.clip(p_value, 0.0, 1.0)),
        method=method,
        cramers_v=cramers_v,
        min_expected_count=min_expected,
    )
