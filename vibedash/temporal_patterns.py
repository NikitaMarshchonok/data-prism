"""Seasonality, level shifts and last-period checks for prepared time series.

Each check answers one question a business owner asks about a metric over
time and is calibrated on synthetic series with known truth
(``temporal_quality.py``):

* Seasonality: does the metric follow a weekly (day-of-week) or yearly
  (month-of-year) pattern?  Within-cycle permutation test on the detrended
  series, plus a minimum practical amplitude.
* Level shift: did the level change abruptly at some point?  CUSUM statistic
  with an AR(1) sieve bootstrap null, required under both a linear and a
  log-linear trend so that curved growth is not mistaken for a jump.
* Last period: is the latest complete period unusual compared with how the
  metric normally moves?  Rolling one-step baseline errors with a robust scale
  and a heavy-tailed reference distribution.

All randomness uses fixed seeds, so the same data always gives the same
result.  Functions return ``None`` when the series cannot support the check.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats


RANDOM_SEED = 20261124
PERMUTATIONS = 999
BOOTSTRAP_REPLICATES = 499

# Practical-significance floors: with long histories even tiny effects become
# statistically detectable, and a 2% weekday wobble is not worth a manager's
# attention.  Both are relative to the typical level of the metric.
MIN_SEASONAL_AMPLITUDE = 0.10
MIN_LEVEL_SHIFT = 0.10

SEASONAL_CYCLES = {
    "weekday": {"length": 7, "min_blocks": 8},
    "month": {"length": 12, "min_blocks": 2},
}
WEEKDAY_LABELS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTH_LABELS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

CHANGE_POINT_MIN_PERIODS = 24
CHANGE_POINT_MAX_PERIODS = 400
LAST_PERIOD_MIN_PERIODS = 16
LAST_PERIOD_BASELINE_WINDOW = 4
# Student t with 5 degrees of freedom: close to calibrated for normal noise
# and still near nominal for heavy-tailed business data (measured: 2-6%
# false alerts at a 5% level, versus 6-12% with a normal reference).
LAST_PERIOD_REFERENCE_DF = 5


def _rng(*parts: int) -> np.random.Generator:
    return np.random.default_rng([RANDOM_SEED, *parts])


def _theil_sen_slope(values: np.ndarray) -> float:
    n = len(values)
    i, j = np.triu_indices(n, k=1)
    return float(np.median((values[j] - values[i]) / (j - i)))


# --------------------------------------------------------------------------
# Seasonality
# --------------------------------------------------------------------------

def _season_positions(period_starts: Sequence[pd.Timestamp], cycle: str) -> tuple[np.ndarray, np.ndarray]:
    starts = pd.DatetimeIndex(period_starts)
    if cycle == "weekday":
        positions = starts.day_of_week.to_numpy()
        # Blocks are calendar weeks (Monday to Sunday).
        monday = (starts - pd.to_timedelta(positions, unit="D")).normalize()
        blocks = ((monday - monday[0]).days // 7).to_numpy()
    else:
        positions = (starts.month - 1).to_numpy()
        months = (starts.year - starts[0].year) * 12 + starts.month - starts[0].month
        blocks = (np.asarray(months) // 12)
    return positions.astype(int), blocks.astype(int)


def seasonality_test(
    values: Sequence[float],
    period_starts: Sequence[pd.Timestamp],
    *,
    cycle: str,
) -> Optional[dict[str, Any]]:
    """Test for a repeating day-of-week or month-of-year pattern."""
    if cycle not in SEASONAL_CYCLES:
        raise ValueError(f"Unknown cycle: {cycle}")
    length = SEASONAL_CYCLES[cycle]["length"]
    y = np.asarray(values, dtype=float)
    if len(y) != len(period_starts) or len(y) < length * 2:
        return None
    finite = np.isfinite(y)
    level = float(np.median(y[finite])) if finite.any() else math.nan
    if not math.isfinite(level) or level <= 0:
        return None

    # Remove a monotonic trend first; otherwise later positions in each cycle
    # sit systematically higher in a growing series.
    index = np.arange(len(y), dtype=float)
    slope = _theil_sen_slope(y[finite]) if finite.sum() >= 3 else 0.0
    detrended = y - slope * index

    positions, blocks = _season_positions(period_starts, cycle)
    complete = []
    for block in np.unique(blocks):
        mask = (blocks == block) & finite
        if mask.sum() != length:
            continue
        row = np.full(length, np.nan)
        row[positions[mask]] = detrended[mask]
        if np.isfinite(row).all():
            complete.append(row)
    if len(complete) < SEASONAL_CYCLES[cycle]["min_blocks"]:
        return None

    centred = np.asarray(complete)
    centred = centred - centred.mean(axis=1, keepdims=True)
    profile = centred.mean(axis=0)
    statistic = float(np.sum(profile**2))

    # Under no seasonality, positions inside each cycle are exchangeable, so
    # shuffling them within each block gives the exact null distribution.
    rng = _rng(len(y), length, len(centred))
    order = np.argsort(rng.random((PERMUTATIONS, *centred.shape)), axis=2)
    shuffled = np.take_along_axis(np.broadcast_to(centred, (PERMUTATIONS, *centred.shape)), order, axis=2)
    null = np.sum(shuffled.mean(axis=1) ** 2, axis=1)
    p_value = float((1 + np.sum(null >= statistic * (1 - 1e-12))) / (PERMUTATIONS + 1))

    # The observed profile also contains noise, so its spread overstates the
    # true pattern (most for weak patterns).  Shrink it by the share of the
    # profile energy that exceeds what pure noise produces on average.
    shrinkage = math.sqrt(max(0.0, 1.0 - float(null.mean()) / statistic)) if statistic > 0 else 0.0
    relative = shrinkage * profile / level
    labels = WEEKDAY_LABELS if cycle == "weekday" else MONTH_LABELS
    peak, trough = int(np.argmax(profile)), int(np.argmin(profile))
    return {
        "method": f"Within-cycle permutation test on the detrended series ({PERMUTATIONS} permutations)",
        "cycle": cycle,
        "cycles_used": int(len(centred)),
        "p_value": p_value,
        "amplitude": float(relative[peak] - relative[trough]),
        "peak": {"label": labels[peak], "relative": float(relative[peak])},
        "trough": {"label": labels[trough], "relative": float(relative[trough])},
        "profile": [{"label": labels[i], "relative": float(relative[i])} for i in range(length)],
        "shrinkage": float(shrinkage),
        "min_amplitude": MIN_SEASONAL_AMPLITUDE,
    }


# --------------------------------------------------------------------------
# Level shift
# --------------------------------------------------------------------------

def _cusum_statistic(series: np.ndarray, projection: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray]:
    n = series.shape[-1]
    residuals = series - series @ projection.T
    cusum = np.cumsum(residuals, axis=-1)[..., margin - 1 : n - margin]
    return np.abs(cusum).max(axis=-1) / math.sqrt(n), np.abs(cusum).argmax(axis=-1) + margin


def _shift_on_scale(z: np.ndarray, rng: np.random.Generator) -> tuple[float, int, float]:
    n = len(z)
    index = np.arange(n, dtype=float)
    design = np.column_stack([np.ones(n), index])
    projection = design @ np.linalg.pinv(design)
    margin = max(4, n // 10)
    observed, location = _cusum_statistic(z, projection, margin)
    location = int(location)

    # Null model: the trend plus AR(1) noise estimated without a shift.  A
    # real shift inflates the AR coefficient, which makes the test
    # conservative rather than producing false alarms.
    residuals = z - projection @ z
    centred = residuals - residuals.mean()
    denominator = float(centred @ centred)
    rho = float(centred[:-1] @ centred[1:]) / denominator if denominator > 0 else 0.0
    rho = min(max(rho, 0.0), 0.95)
    innovations = residuals[1:] - rho * residuals[:-1]
    innovations = innovations - innovations.mean()
    burn_in = 50
    draws = rng.choice(innovations, size=(BOOTSTRAP_REPLICATES, n + burn_in))
    simulated = np.empty_like(draws)
    simulated[:, 0] = draws[:, 0]
    for step in range(1, n + burn_in):
        simulated[:, step] = rho * simulated[:, step - 1] + draws[:, step]
    null, _ = _cusum_statistic(simulated[:, burn_in:], projection, margin)
    p_value = float((1 + np.sum(null >= observed)) / (BOOTSTRAP_REPLICATES + 1))

    step_design = np.column_stack([np.ones(n), index, (index >= location).astype(float)])
    coefficients = np.linalg.lstsq(step_design, z, rcond=None)[0]
    return p_value, location, float(coefficients[2])


def level_shift_test(values: Sequence[float], period_starts: Sequence[pd.Timestamp]) -> Optional[dict[str, Any]]:
    """Find the single most prominent abrupt change in level, if testable."""
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(y)
    y = y[finite]
    starts = [start for start, keep in zip(period_starts, finite) if keep]
    n = len(y)
    if not CHANGE_POINT_MIN_PERIODS <= n <= CHANGE_POINT_MAX_PERIODS or np.ptp(y) == 0:
        return None

    level = float(np.median(y))
    p_linear, location, shift_linear = _shift_on_scale(y, _rng(n, 1))
    positive = bool(np.all(y > 0))
    if positive:
        # A shift must also hold against exponential growth; otherwise curved
        # growth would be reported as a jump.  The reported size and date
        # come from the multiplicative (log) model.
        p_log, location, shift_log = _shift_on_scale(np.log(y), _rng(n, 2))
        p_value = max(p_linear, p_log)
        relative = math.expm1(shift_log)
    else:
        p_value = p_linear
        relative = shift_linear / abs(level) if level != 0 else None

    before = y[:location]
    after = y[location:]
    return {
        "method": (
            "CUSUM level-shift test with an AR(1) sieve bootstrap "
            f"({BOOTSTRAP_REPLICATES} replicates), required under linear and log-linear trends"
        ),
        "p_value": float(p_value),
        "change_period": starts[location].date().isoformat(),
        "periods_before": int(len(before)),
        "periods_after": int(len(after)),
        "relative_shift": float(relative) if relative is not None else None,
        "absolute_shift": float(shift_linear),
        "median_before": float(np.median(before)),
        "median_after": float(np.median(after)),
        # Candidate dates exclude the first and last tenth of the series; a
        # maximum on that boundary means the real change may lie beyond it
        # (typically a change in the last few periods), so its date is unknown.
        "at_edge": bool(location <= max(4, n // 10) or location >= n - max(4, n // 10)),
        "min_shift": MIN_LEVEL_SHIFT,
    }


# --------------------------------------------------------------------------
# Last period
# --------------------------------------------------------------------------

def robust_unusualness(history: Sequence[float], latest: float) -> Optional[tuple[float, float, float]]:
    """Robust z-score of ``latest`` against ``history`` and its two-sided p-value.

    Returns ``(z, p_value, centre)`` or ``None`` when the history has no spread.
    The median and MAD resist past outliers; the Student t reference keeps the
    false-alert rate near nominal for heavy-tailed business data.
    """
    values = np.asarray(history, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or not math.isfinite(latest):
        return None
    centre = float(np.median(values))
    scale = 1.4826 * float(np.median(np.abs(values - centre)))
    if not math.isfinite(scale) or scale <= 0:
        return None
    score = (latest - centre) / scale
    return float(score), float(2.0 * stats.t.sf(abs(score), LAST_PERIOD_REFERENCE_DF)), centre


def last_period_check(values: Sequence[float], period_starts: Sequence[pd.Timestamp]) -> Optional[dict[str, Any]]:
    """Compare the latest complete period with the metric's normal movement."""
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(y)
    y = y[finite]
    starts = [start for start, keep in zip(period_starts, finite) if keep]
    window = LAST_PERIOD_BASELINE_WINDOW
    if len(y) < LAST_PERIOD_MIN_PERIODS:
        return None

    # Baseline: the average of the previous few periods, evaluated at every
    # past period (rolling origin) so its typical error is known.
    kernel = np.ones(window) / window
    baseline = np.convolve(y, kernel, mode="valid")[:-1]
    errors = y[window:] - baseline
    history, latest = errors[:-1], float(errors[-1])
    unusual = robust_unusualness(history, latest)
    if unusual is None:
        return None
    score, p_value, centre = unusual
    expected = float(baseline[-1] + centre)
    observed = float(y[-1])
    return {
        "method": (
            f"Previous-{window}-period average baseline; rolling one-step errors; robust z-score "
            f"against Student t({LAST_PERIOD_REFERENCE_DF})"
        ),
        "period": starts[-1].date().isoformat(),
        "observed": observed,
        "expected": expected,
        "relative_deviation": (observed - expected) / abs(expected) if expected != 0 else None,
        "robust_z": float(score),
        "p_value": p_value,
        "history_periods": int(len(history)),
    }
