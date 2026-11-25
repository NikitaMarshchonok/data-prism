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

import pandas as pd

from .temporal_engine import DEFAULT_ALPHA, mann_kendall_trend
from .temporal_patterns import (
    MIN_LEVEL_SHIFT,
    MIN_SEASONAL_AMPLITUDE,
    last_period_check,
    level_shift_test,
    seasonality_test,
)


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


# --------------------------------------------------------------------------
# Seasonality, level shifts and last-period alerts
# --------------------------------------------------------------------------

_DAILY_STARTS = list(pd.date_range("2024-01-01", periods=70, freq="D"))  # 10 calendar weeks
_MONTHLY_STARTS = list(pd.date_range("2022-01-01", periods=36, freq="MS"))
_WEEKLY_STARTS = list(pd.date_range("2023-01-02", periods=104, freq="W-MON"))
_LAST_STARTS = list(pd.date_range("2024-01-01", periods=52, freq="W-MON"))
_WEEKDAY_EFFECT = np.array([0.0, 0.03, 0.05, 0.06, 0.08, -0.10, -0.12])  # 20% peak to trough
_MONTH_EFFECT = 0.08 * np.sin(2 * np.pi * np.arange(12) / 12)  # 16% peak to trough


@dataclass(frozen=True)
class PatternScenario:
    name: str
    check: str
    has_effect: bool
    threshold: float
    generate: Callable[[np.random.Generator], np.ndarray]
    starts: list
    description: str


def _season_detects(values: np.ndarray, starts: list, cycle: str, alpha: float) -> bool:
    result = seasonality_test(values, starts, cycle=cycle)
    return bool(result and result["p_value"] <= alpha and result["amplitude"] >= MIN_SEASONAL_AMPLITUDE)


def _shift_detects(values: np.ndarray, starts: list, alpha: float) -> bool:
    result = level_shift_test(values, starts)
    return bool(
        result
        and not result["at_edge"]
        and result["p_value"] <= alpha
        and result["relative_shift"] is not None
        and abs(result["relative_shift"]) >= MIN_LEVEL_SHIFT
    )


def _last_detects(values: np.ndarray, starts: list, alpha: float) -> bool:
    result = last_period_check(values, starts)
    return bool(result and result["p_value"] <= alpha)


def _days(n: int) -> np.ndarray:
    return np.arange(n)


PATTERN_SCENARIOS = (
    PatternScenario("weekday_white_noise", "weekday", False, 0.07,
                    lambda rng: LEVEL + rng.normal(0.0, NOISE_SD, 70), _DAILY_STARTS,
                    "Daily independent noise, no weekly pattern: false alarm rate."),
    PatternScenario("weekday_ar1_trend", "weekday", False, 0.07,
                    lambda rng: LEVEL + 0.5 * _days(70) + _ar1_noise(rng, 70, 0.6, NOISE_SD), _DAILY_STARTS,
                    "Daily AR(1) noise with a trend, no weekly pattern: false alarm rate."),
    PatternScenario("weekday_pattern", "weekday", True, 0.90,
                    lambda rng: LEVEL * (1 + _WEEKDAY_EFFECT[_days(70) % 7]) + _ar1_noise(rng, 70, 0.6, NOISE_SD),
                    _DAILY_STARTS, "20% weekday pattern in AR(1) noise over 10 weeks: detection rate."),
    PatternScenario("month_ar1_trend", "month", False, 0.10,
                    lambda rng: LEVEL + 1.0 * _days(36) + _ar1_noise(rng, 36, 0.6, NOISE_SD), _MONTHLY_STARTS,
                    "Monthly AR(1) noise with a trend, no yearly pattern: false alarm rate."),
    PatternScenario("month_pattern", "month", True, 0.75,
                    lambda rng: LEVEL * (1 + _MONTH_EFFECT[_days(36) % 12]) + _ar1_noise(rng, 36, 0.6, NOISE_SD / 2),
                    _MONTHLY_STARTS, "16% yearly pattern in AR(1) noise over 3 years: detection rate."),
    PatternScenario("shift_ar1_noise", "level_shift", False, 0.07,
                    lambda rng: LEVEL + _ar1_noise(rng, 104, 0.6, NOISE_SD), _WEEKLY_STARTS,
                    "Weekly AR(1) noise, no shift: false alarm rate."),
    PatternScenario("shift_exponential_growth", "level_shift", False, 0.07,
                    lambda rng: LEVEL * 1.02 ** _days(104) * (1 + rng.normal(0.0, 0.1, 104)), _WEEKLY_STARTS,
                    "2%/week compound growth, no shift: curved growth must not look like a jump."),
    PatternScenario("shift_linear_trend", "level_shift", False, 0.07,
                    lambda rng: LEVEL + 1.0 * _days(104) + _ar1_noise(rng, 104, 0.6, NOISE_SD), _WEEKLY_STARTS,
                    "Linear trend in AR(1) noise, no shift: false alarm rate."),
    PatternScenario("shift_step", "level_shift", True, 0.80,
                    lambda rng: LEVEL + 20.0 * (_days(104) >= 52) + rng.normal(0.0, NOISE_SD, 104), _WEEKLY_STARTS,
                    "+20% step in independent noise: detection rate."),
    PatternScenario("shift_drop_in_trend", "level_shift", True, 0.80,
                    lambda rng: LEVEL + 0.5 * _days(104) - 30.0 * (_days(104) >= 73) + _ar1_noise(rng, 104, 0.4, NOISE_SD),
                    _WEEKLY_STARTS, "-30 drop on top of a rising trend in AR(1) noise: detection rate."),
    PatternScenario("last_ar1_noise", "latest_period", False, 0.07,
                    lambda rng: LEVEL + _ar1_noise(rng, 52, 0.6, NOISE_SD), _LAST_STARTS,
                    "AR(1) noise, normal last week: false alert rate."),
    PatternScenario("last_heavy_tails", "latest_period", False, 0.10,
                    lambda rng: LEVEL + 3.0 * rng.standard_t(3, 52), _LAST_STARTS,
                    "Heavy-tailed (t3) noise, normal last week: false alert rate."),
    PatternScenario("last_drop", "latest_period", True, 0.75,
                    lambda rng: np.r_[LEVEL + rng.normal(0.0, NOISE_SD, 51), 0.6 * LEVEL + rng.normal(0.0, NOISE_SD)],
                    _LAST_STARTS, "Last week 40% below normal: detection rate."),
)


def evaluate_pattern_detectors(seeds: int = 300, *, alpha: float = DEFAULT_ALPHA, base_seed: int = 20261124) -> dict:
    """Return per-scenario rates for the seasonality, level-shift and last-period checks."""
    if seeds < 1:
        raise ValueError("seeds must be positive")
    report = {"seeds": seeds, "alpha": alpha, "scenarios": []}
    for offset, scenario in enumerate(PATTERN_SCENARIOS):
        rng = np.random.default_rng(base_seed + offset)
        hits = 0
        for _ in range(seeds):
            values = scenario.generate(rng)
            if scenario.check in ("weekday", "month"):
                hits += _season_detects(values, scenario.starts, scenario.check, alpha)
            elif scenario.check == "level_shift":
                hits += _shift_detects(values, scenario.starts, alpha)
            else:
                hits += _last_detects(values, scenario.starts, alpha)
        rate = hits / seeds
        passed = rate >= scenario.threshold if scenario.has_effect else rate <= scenario.threshold
        report["scenarios"].append(
            {
                "name": scenario.name,
                "check": scenario.check,
                "description": scenario.description,
                "metric": "detection_rate" if scenario.has_effect else "false_alarm_rate",
                "threshold": scenario.threshold,
                "rate": rate,
                "passed": bool(passed),
            }
        )
    report["passed"] = all(item["passed"] for item in report["scenarios"])
    return report


# --------------------------------------------------------------------------
# Why a KPI changed
# --------------------------------------------------------------------------

_DRIVER_DATES = pd.date_range("2024-01-01", "2024-12-29", freq="D")  # 52 complete weeks
_DRIVER_TEMPORAL = {
    "status": "ok",
    "time_column": "Date",
    "series": [
        {"column": "Revenue", "unit": "currency", "grain": "week", "first_period": "2024-01-01", "last_period": "2024-12-23"},
        {"column": "Conversion", "unit": "ratio", "grain": "week", "first_period": "2024-01-01", "last_period": "2024-12-23"},
    ],
}
_REGIONS = {"Europe": 300.0, "US": 500.0, "Asia": 200.0}


def _driver_frame(rng: np.random.Generator, *, europe_drop: float = 0.0, starter_surge: bool = False) -> pd.DataFrame:
    """Daily rows per region and plan, AR(1) demand noise, optional late changes."""
    days = len(_DRIVER_DATES)
    late = _DRIVER_DATES >= pd.Timestamp("2024-12-02")
    frames = []
    for region, base in _REGIONS.items():
        demand = base * (1 + _ar1_noise(rng, days, 0.5, 0.1))
        if region == "Europe":
            demand = demand * np.where(late, 1 - europe_drop, 1.0)
        starter_rows = rng.random(days) < np.where(late & starter_surge, 0.9, 0.3)
        frames.append(pd.DataFrame({
            "Date": _DRIVER_DATES, "Region": region, "Plan": "Pro", "Revenue": demand,
            "Conversion": 0.06 + rng.normal(0, 0.004, days),
        }))
        frames.append(pd.DataFrame({
            "Date": _DRIVER_DATES[starter_rows], "Region": region, "Plan": "Starter",
            "Revenue": 0.2 * demand[starter_rows], "Conversion": 0.02 + rng.normal(0, 0.004, int(starter_rows.sum())),
        }))
    return pd.concat(frames, ignore_index=True)


def evaluate_driver_analysis(seeds: int = 200, *, alpha: float = DEFAULT_ALPHA, base_seed: int = 20261125) -> dict:
    """Ground truth for the driver analysis: false alarms, detection and attribution."""
    from .driver_analysis import DriverAnalysisEngine

    if seeds < 1:
        raise ValueError("seeds must be positive")
    scenarios = (
        ("no_change", {}, "No change: share of measures flagged as changing more than usual.", "false_alarm", 0.07),
        ("europe_drop", {"europe_drop": 0.5}, "Europe (30% of revenue) halves: revenue flagged and Europe named.", "attribution", 0.80),
        ("mix_shift", {"starter_surge": True}, "More low-converting Starter rows: conversion flagged as a mix shift.", "attribution", 0.80),
    )
    report = {"seeds": seeds, "alpha": alpha, "scenarios": []}
    for offset, (name, kwargs, description, metric, threshold) in enumerate(scenarios):
        rng = np.random.default_rng(base_seed + offset)
        hits = trials = 0
        for _ in range(seeds):
            frame = _driver_frame(rng, **kwargs)
            result = DriverAnalysisEngine(frame, _DRIVER_TEMPORAL, alpha=alpha).analyze()
            measures = {item["column"]: item for item in result["measures"]}
            if metric == "false_alarm":
                trials += len(measures)
                hits += sum(item["material"] for item in measures.values())
                continue
            trials += 1
            if name == "europe_drop":
                revenue = measures.get("Revenue") or {}
                top = (revenue.get("dimensions") or [{}])[0]
                hits += bool(
                    revenue.get("material")
                    and top.get("dimension") == "Region"
                    and (top.get("driver") or {}).get("segment") == "Europe"
                )
            else:
                conversion = measures.get("Conversion") or {}
                top = (conversion.get("dimensions") or [{}])[0]
                hits += bool(
                    conversion.get("material")
                    and top.get("dimension") == "Plan"
                    and top.get("mix_story")
                    and (top.get("mix_driver") or {}).get("segment") == "Starter"
                )
        rate = hits / trials if trials else 0.0
        passed = rate <= threshold if metric == "false_alarm" else rate >= threshold
        report["scenarios"].append(
            {"name": name, "description": description, "metric": metric, "threshold": threshold, "rate": rate, "passed": bool(passed)}
        )
    report["passed"] = all(item["passed"] for item in report["scenarios"])
    return report


if __name__ == "__main__":  # pragma: no cover - manual reporting helper
    import json

    print(json.dumps({
        "trend": evaluate_trend_detector(),
        "patterns": evaluate_pattern_detectors(1000),
        "drivers": evaluate_driver_analysis(500),
    }, indent=2))
