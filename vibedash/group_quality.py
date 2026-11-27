"""Synthetic ground-truth evaluation of segment comparisons.

Each scenario draws many seeded datasets whose truth is known and runs the
full statistical engine on them, so column selection, level filtering and the
FDR step are measured together with the tests.  Thresholds are the contract:
a change that raises false alarms or loses power fails the test suite.

Null scenarios include the hard cases for mean and rate tests in business
data: skewed (log-normal) values, unequal group sizes and variances, and
sparse rates where Pearson's chi-square approximation is not trusted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .statistical_engine import DEFAULT_ALPHA, StatisticalValidationEngine


SEGMENTS = ("North", "South", "East", "West")


def _labels(sizes) -> np.ndarray:
    return np.repeat(np.array(SEGMENTS[: len(sizes)]), sizes)


def _lognormal_equal_means(rng: np.random.Generator, sizes, sigmas, mean: float = 100.0) -> np.ndarray:
    parts = []
    for size, sigma in zip(sizes, sigmas):
        mu = np.log(mean) - sigma * sigma / 2.0
        parts.append(rng.lognormal(mu, sigma, size))
    return np.concatenate(parts)


def _means_null_normal(rng: np.random.Generator) -> pd.DataFrame:
    sizes = (30, 30, 30, 30)
    return pd.DataFrame({"Region": _labels(sizes), "Revenue": rng.normal(100, 15, sum(sizes))})


def _means_null_skewed(rng: np.random.Generator) -> pd.DataFrame:
    sizes = (40, 60, 120, 240)
    values = _lognormal_equal_means(rng, sizes, (0.9, 0.6, 0.75, 0.5))
    return pd.DataFrame({"Region": _labels(sizes), "Revenue": values})


def _means_null_skewed_small(rng: np.random.Generator) -> pd.DataFrame:
    sizes = (10, 20, 40, 80)
    values = _lognormal_equal_means(rng, sizes, (1.2, 0.8, 0.6, 0.5))
    return pd.DataFrame({"Region": _labels(sizes), "Revenue": values})


def _means_one_segment_higher(rng: np.random.Generator) -> pd.DataFrame:
    sizes = (50, 50, 50, 50)
    shift = np.repeat([0.0, 0.0, 0.0, 9.0], sizes)
    return pd.DataFrame({"Region": _labels(sizes), "Revenue": rng.normal(100, 15, sum(sizes)) + shift})


def _rates(rng: np.random.Generator, sizes, rates) -> pd.DataFrame:
    labels = _labels(sizes)
    probability = np.repeat(rates, sizes)
    renewed = np.where(rng.random(len(labels)) < probability, "Yes", "No")
    return pd.DataFrame({"Region": labels, "Renewed": renewed})


def _rates_null_balanced(rng: np.random.Generator) -> pd.DataFrame:
    return _rates(rng, (100, 100, 100), (0.3, 0.3, 0.3))


def _rates_null_sparse(rng: np.random.Generator) -> pd.DataFrame:
    return _rates(rng, (12, 30, 80, 200), (0.08, 0.08, 0.08, 0.08))


def _rates_one_segment_higher(rng: np.random.Generator) -> pd.DataFrame:
    return _rates(rng, (150, 150, 150, 150), (0.10, 0.10, 0.10, 0.22))


@dataclass(frozen=True)
class GroupScenario:
    name: str
    family: str
    metric: str
    threshold: float
    generate: Callable[[np.random.Generator], pd.DataFrame]
    description: str
    expected_top: str | None = None


GROUP_SCENARIOS = (
    GroupScenario("means_null_normal", "multi-group-comparison", "false_alarm", 0.07, _means_null_normal,
                  "Four equal-mean normal groups of 30."),
    GroupScenario("means_null_skewed", "multi-group-comparison", "false_alarm", 0.08, _means_null_skewed,
                  "Four log-normal groups with equal means, unequal sizes (40-240) and unequal spread."),
    GroupScenario("means_null_skewed_small", "multi-group-comparison", "known_limit", 0.20, _means_null_skewed_small,
                  "Known limit: a 10-row group with very uneven values (log-normal sigma 1.2). Means of so few skewed "
                  "values are unstable; the bootstrap reduces but does not remove false alarms."),
    GroupScenario("means_one_higher", "multi-group-comparison", "detection", 0.80, _means_one_segment_higher,
                  "One of four groups of 50 is 0.6 standard deviations higher (theoretical ANOVA power about 0.86); "
                  "detected and named highest.", "West"),
    GroupScenario("rates_null_balanced", "rate-comparison", "false_alarm", 0.07, _rates_null_balanced,
                  "Three segments of 100 with the same 30% rate."),
    GroupScenario("rates_null_sparse", "rate-comparison", "false_alarm", 0.07, _rates_null_sparse,
                  "Four segments (12-200 rows) with the same 8% rate: sparse cells, exact test path."),
    GroupScenario("rates_one_higher", "rate-comparison", "detection", 0.80, _rates_one_segment_higher,
                  "One of four segments has a 22% rate instead of 10%; detected and named highest.", "West"),
)


def _find(report: dict, family: str) -> dict | None:
    return next((test for test in report.get("tests") or [] if test.get("family") == family), None)


def _top(test: dict) -> str | None:
    estimates = test.get("estimates") or {}
    return estimates.get("highest_group") or estimates.get("highest_segment")


def evaluate_group_comparisons(seeds: int = 300, *, alpha: float = DEFAULT_ALPHA, base_seed: int = 20261127) -> dict:
    """Rates of false alarms, detection, and pairwise errors under partial nulls."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    report = {"seeds": seeds, "alpha": alpha, "scenarios": []}
    for offset, scenario in enumerate(GROUP_SCENARIOS):
        rng = np.random.default_rng(base_seed + offset)
        hits = 0
        null_pair_errors = 0
        for _ in range(seeds):
            frame = scenario.generate(rng)
            result = StatisticalValidationEngine(frame, alpha=alpha).analyze(max_results=20)
            test = _find(result, scenario.family)
            if test is None:
                continue
            if scenario.metric in {"false_alarm", "known_limit"}:
                hits += bool(test.get("significant"))
                continue
            hits += bool(test.get("significant") and _top(test) == scenario.expected_top)
            pairs = (test.get("estimates") or {}).get("pairwise") or []
            null_pair_errors += any(
                scenario.expected_top not in (pair.get("higher"), pair.get("lower")) for pair in pairs
            )
        rate = hits / seeds
        # The threshold is a property of the method, not of one batch of seeds:
        # allow two binomial standard errors so a gate run on a few hundred
        # seeds does not fail (or pass) on sampling noise alone.
        allowance = 2.0 * float(np.sqrt(scenario.threshold * (1.0 - scenario.threshold) / seeds))
        if scenario.metric in {"false_alarm", "known_limit"}:
            passed = rate <= scenario.threshold + allowance
        else:
            passed = rate >= scenario.threshold - allowance
        entry = {
            "name": scenario.name,
            "family": scenario.family,
            "description": scenario.description,
            "metric": scenario.metric,
            "threshold": scenario.threshold,
            "rate": rate,
            "allowance": round(allowance, 4),
            "passed": bool(passed),
        }
        if scenario.family == "multi-group-comparison" and scenario.metric == "detection":
            # Three of the four groups are equal: any reported pair among them is
            # a family-wise error, which the max-|t| adjustment keeps near alpha.
            entry["null_pair_error_rate"] = null_pair_errors / seeds
            pair_allowance = 2.0 * float(np.sqrt(alpha * (1.0 - alpha) / seeds))
            entry["passed"] = bool(passed and entry["null_pair_error_rate"] <= alpha + pair_allowance)
        report["scenarios"].append(entry)
    report["passed"] = all(item["passed"] for item in report["scenarios"])
    return report


if __name__ == "__main__":  # pragma: no cover - manual reporting helper
    import json

    print(json.dumps(evaluate_group_comparisons(1000), indent=2))
