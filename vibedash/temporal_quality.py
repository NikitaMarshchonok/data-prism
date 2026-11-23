"""Synthetic ground-truth evaluation of the temporal trend detector.

Each scenario generates many seeded series whose truth is known (trend or no
trend) and measures how often the detector reports a reliable trend.  The
thresholds are the contract: a change to the algorithm that raises the false
alarm rate or loses power fails the test suite.

A naive baseline (ordinary least-squares slope test that assumes independent
observations) is evaluated on the same series to show what the
autocorrelation correction buys.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .temporal_engine import DEFAULT_ALPHA, mann_kendall_trend


SERIES_LENGTH = 60
LEVEL = 100.0
NOISE_SD = 10.0


def _ar1_noise(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    innovations = rng.normal(0.0, sd * np.sqrt(1 - phi**2), n)
    noise = np.empty(n)
    noise[0] = rng.normal(0.0, sd)
    for index in range(1, n):
        noise[index] = phi * noise[index - 1] + innovations[index]
    return noise


@dataclass(frozen=True)
class TrendScenario:
    name: str
    has_trend: bool
    threshold: float
    generate: Callable[[np.random.Generator], np.ndarray]
    description: str


def _white_noise(rng: np.random.Generator) -> np.ndarray:
    return LEVEL + rng.normal(0.0, NOISE_SD, SERIES_LENGTH)


def _autocorrelated_noise(rng: np.random.Generator) -> np.ndarray:
    return LEVEL + _ar1_noise(rng, SERIES_LENGTH, 0.6, NOISE_SD)


def _linear_trend(rng: np.random.Generator) -> np.ndarray:
    # +1% of the level per period, noise SD of 10% of the level.
    return LEVEL + 0.01 * LEVEL * np.arange(SERIES_LENGTH) + rng.normal(0.0, NOISE_SD, SERIES_LENGTH)


def _autocorrelated_trend(rng: np.random.Generator) -> np.ndarray:
    return LEVEL + 0.01 * LEVEL * np.arange(SERIES_LENGTH) + _ar1_noise(rng, SERIES_LENGTH, 0.6, NOISE_SD)


SCENARIOS = (
    TrendScenario("white_noise", False, 0.07, _white_noise, "Independent noise, no trend: false alarm rate."),
    TrendScenario("ar1_noise", False, 0.10, _autocorrelated_noise, "AR(1) phi=0.6 noise, no trend: false alarm rate."),
    TrendScenario("linear_trend", True, 0.80, _linear_trend, "+1%/period trend in independent noise: detection rate."),
    TrendScenario("ar1_trend", True, 0.80, _autocorrelated_trend, "+1%/period trend in AR(1) noise: detection rate."),
)


def _engine_detects(values: np.ndarray, alpha: float) -> bool:
    result = mann_kendall_trend(values, alpha=alpha)
    lower, upper = result["slope_ci"]
    return result["p_value"] <= alpha and (lower > 0 or upper < 0)


def _naive_detects(values: np.ndarray, alpha: float) -> bool:
    return stats.linregress(np.arange(len(values)), values).pvalue <= alpha


def evaluate_trend_detector(seeds: int = 1000, *, alpha: float = DEFAULT_ALPHA, base_seed: int = 20261123) -> dict:
    """Return per-scenario detection rates for the engine and the naive baseline."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    report = {"seeds": seeds, "alpha": alpha, "series_length": SERIES_LENGTH, "scenarios": []}
    for offset, scenario in enumerate(SCENARIOS):
        rng = np.random.default_rng(base_seed + offset)
        engine_hits = naive_hits = 0
        for _ in range(seeds):
            values = scenario.generate(rng)
            engine_hits += _engine_detects(values, alpha)
            naive_hits += _naive_detects(values, alpha)
        engine_rate = engine_hits / seeds
        passed = engine_rate >= scenario.threshold if scenario.has_trend else engine_rate <= scenario.threshold
        report["scenarios"].append(
            {
                "name": scenario.name,
                "description": scenario.description,
                "metric": "detection_rate" if scenario.has_trend else "false_alarm_rate",
                "threshold": scenario.threshold,
                "engine_rate": engine_rate,
                "naive_ols_rate": naive_hits / seeds,
                "passed": bool(passed),
            }
        )
    report["passed"] = all(item["passed"] for item in report["scenarios"])
    return report


if __name__ == "__main__":  # pragma: no cover - manual reporting helper
    import json

    print(json.dumps(evaluate_trend_detector(), indent=2))
