"""Bounded daily single-series forecasting with fixed-origin temporal evaluation.

No contemporaneous covariates, random splits, IID significance tests, or automatic
imputation. Every validation forecast predicts its entire horizon recursively
without observing the actual target inside that horizon.
"""
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CONTRACT = "daily-forecast-v1"
CANDIDATES = ("last_value", "seasonal_naive", "lag_ridge", "lag_random_forest")
VALIDATION_FOLDS = 3
MAX_ROWS = 5000


@dataclass(frozen=True)
class ForecastTask:
    target_column: str
    time_column: str
    horizon: int
    seasonal_period: int = 7

    def __post_init__(self):
        for value in (self.target_column, self.time_column):
            if not isinstance(value, str) or not value.strip() or len(value) > 255:
                raise ValueError("Target and time column names must be explicit non-empty strings.")
        if self.target_column == self.time_column:
            raise ValueError("Target and time columns must differ.")
        for name in ("horizon", "seasonal_period"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 90:
                raise ValueError(f"{name} must be an integer from 1 to 90 days.")


def validate_forecast_task(value):
    """Validate a typed task or strict mapping without guessing task fields."""
    if isinstance(value, ForecastTask):
        return value
    if isinstance(value, Mapping):
        allowed = {"target_column", "time_column", "horizon", "seasonal_period"}
        if set(value) - allowed:
            raise ValueError("Unknown forecast task fields.")
        try:
            return ForecastTask(**value)
        except TypeError as error:
            raise ValueError("Forecast task requires target_column, time_column and horizon.") from error
    raise ValueError("An explicit forecast task is required.")


def _series(frame, task):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Forecast input must be a dataframe with unique columns.")
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"Forecast input must contain from 1 to {MAX_ROWS} rows.")
    if task.target_column not in frame or task.time_column not in frame:
        raise ValueError("The requested target or time column is missing; no substitute is selected.")
    raw_time = frame[task.time_column]
    # A daily contract must not silently reinterpret epochs, time zones or
    # intraday rows as business dates. The caller must aggregate explicitly.
    if pd.api.types.is_datetime64_any_dtype(raw_time):
        if getattr(raw_time.dt, "tz", None) is not None:
            raise ValueError("Use timezone-naive daily business dates.")
        dates = raw_time
    else:
        if not raw_time.map(lambda value: isinstance(value, str)).all():
            raise ValueError("Time values must be ISO YYYY-MM-DD dates.")
        if not raw_time.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError("Time values must be ISO YYYY-MM-DD dates.")
        dates = pd.to_datetime(raw_time, format="%Y-%m-%d", errors="coerce")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Invalid or intraday dates are not supported by the daily contract.")
    if dates.duplicated().any():
        raise ValueError("Duplicate dates: aggregate to one observation per day explicitly.")
    raw_target = frame[task.target_column]
    if (not pd.api.types.is_numeric_dtype(raw_target)
            or pd.api.types.is_bool_dtype(raw_target)
            or pd.api.types.is_complex_dtype(raw_target)):
        raise ValueError("Forecast target must be a finite real numeric measure.")
    values = raw_target.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all():
        raise ValueError("Target contains missing or infinite values; no automatic imputation is applied.")
    if (pd.api.types.is_integer_dtype(raw_target)
            and raw_target.map(lambda value: abs(int(value)) > 2 ** 53 - 1).any()):
        raise ValueError("Integer targets exceed exact float64 precision; rescale explicitly before modelling.")
    order = np.argsort(dates.to_numpy(), kind="stable")
    dates = pd.DatetimeIndex(dates.iloc[order])
    if not (dates[1:] - dates[:-1] == pd.Timedelta(days=1)).all():
        raise ValueError("Missing or irregular days: define a regular daily series before forecasting.")
    minimum = max(30, 4 * task.seasonal_period) + (VALIDATION_FOLDS + 1) * task.horizon
    if len(frame) < minimum:
        raise ValueError(f"Insufficient history: at least {minimum} daily observations are required.")
    return dates, values[order]


def validate_forecast_input(frame, task):
    """Preflight the full history without training or silently truncating it."""
    task = validate_forecast_task(task)
    _series(frame, task)
    return task


def _lags(period):
    return sorted({1, period, 2 * period})


def _forecast(history, horizon, period, candidate):
    """Only the prefix is accepted: future actuals cannot enter recursion."""
    if candidate == "last_value":
        predicted = np.repeat(history[-1], horizon)
    elif candidate == "seasonal_naive":
        predicted = np.asarray([history[-period + (step % period)] for step in range(horizon)])
    else:
        lags = _lags(period)
        start = max(lags)
        X = np.asarray([[history[index - lag] for lag in lags]
                        for index in range(start, len(history))])
        y = history[start:]
        if candidate == "lag_ridge":
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        elif candidate == "lag_random_forest":
            model = RandomForestRegressor(n_estimators=60, min_samples_leaf=3,
                                          max_depth=8, random_state=42, n_jobs=1)
        else:
            raise ValueError("Unknown forecast candidate.")
        model.fit(X, y)
        buffer = list(history)
        predictions = []
        for _ in range(horizon):
            prediction = float(model.predict([[buffer[-lag] for lag in lags]])[0])
            buffer.append(prediction)
            predictions.append(prediction)
        predicted = np.asarray(predictions)
    predicted = np.asarray(predicted, dtype=float)
    if len(predicted) != horizon or not np.isfinite(predicted).all():
        raise ValueError("Forecast produced non-finite or incomplete predictions.")
    return predicted


def _metrics(actual, prediction):
    with np.errstate(over="raise", invalid="raise"):
        error = np.asarray(actual) - np.asarray(prediction)
        metrics = {"mae": float(np.mean(np.abs(error))),
                   "rmse": float(np.sqrt(np.mean(error ** 2)))}
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Forecast metrics are not finite.")
    return metrics


def _date(value):
    return value.strftime("%Y-%m-%d")


def forecast_series(frame, task):
    """Evaluate, lock the selected method, then refit for future forecasts.

    Validation chooses both the method and the reference baseline, never the
    final holdout. Reported improvement is descriptive, not a significance or
    deployment decision. Extra input columns are deliberately unused.
    """
    try:
        task = validate_forecast_task(task)
        dates, values = _series(frame, task)
        h, period = task.horizon, task.seasonal_period
        holdout_start = len(values) - h
        origins = [holdout_start - h * remaining for remaining in range(VALIDATION_FOLDS, 0, -1)]
        folds = [{"train_start": _date(dates[0]), "train_end": _date(dates[origin - 1]),
                  "train_rows": origin, "validation_start": _date(dates[origin]),
                  "validation_end": _date(dates[origin + h - 1]), "validation_rows": h}
                 for origin in origins]
        comparison = []
        for candidate in CANDIDATES:
            try:
                scores = [_metrics(values[origin:origin + h],
                                   _forecast(values[:origin].copy(), h, period, candidate))
                          for origin in origins]
                comparison.append({"model_name": candidate, "status": "ok",
                                   "mean_mae": float(np.mean([score["mae"] for score in scores])),
                                   "fold_metrics": scores, "selected": False})
            except (ValueError, FloatingPointError, OverflowError):
                comparison.append({"model_name": candidate, "status": "error",
                                   "mean_mae": None, "fold_metrics": [], "selected": False,
                                   "reason": "Candidate could not complete all temporal folds."})
        successful = [row for row in comparison if row["status"] == "ok"]
        baselines = [row for row in successful if row["model_name"] in CANDIDATES[:2]]
        if not baselines:
            raise ValueError("No baseline completed temporal validation.")
        # Stable ordering favours the simple baseline on exact ties.
        selected = min(successful, key=lambda row: row["mean_mae"])
        baseline = min(baselines, key=lambda row: row["mean_mae"])
        selected["selected"] = True
        name, baseline_name = selected["model_name"], baseline["model_name"]
        history = values[:holdout_start].copy()
        prediction = _forecast(history, h, period, name)
        baseline_prediction = (prediction.copy() if name == baseline_name else
                               _forecast(history, h, period, baseline_name))
        metrics = _metrics(values[holdout_start:], prediction)
        baseline_metrics = _metrics(values[holdout_start:], baseline_prediction)
        improvement = (1 - metrics["mae"] / baseline_metrics["mae"]
                       if baseline_metrics["mae"] > 0 else None)
        if improvement is not None and not np.isfinite(improvement):
            improvement = None
        # This refit is explicitly AFTER the locked holdout evaluation. Its
        # in-sample history is not used to overwrite the holdout metrics.
        future = _forecast(values.copy(), h, period, name)
        future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=h, freq="D")
        return {
            "contract": CONTRACT, "status": "ok", "task_type": "forecast",
            "task": asdict(task), "target_col": task.target_column, "frequency": "D",
            "model_name": name, "baseline_name": baseline_name,
            "selection_metric": "mean validation MAE", "model_comparison": comparison,
            "cv_folds": VALIDATION_FOLDS, "validation_folds": folds,
            "train_rows": holdout_start, "test_rows": h,
            "holdout": {"train_end": _date(dates[holdout_start - 1]),
                        "start": _date(dates[holdout_start]), "end": _date(dates[-1]),
                        "metrics": metrics, "baseline_metrics": baseline_metrics,
                        "mae_improvement": improvement,
                        "predictions": [{"date": _date(day), "actual": float(actual),
                                         "prediction": float(pred), "baseline": float(base)}
                                        for day, actual, pred, base in zip(dates[holdout_start:],
                                            values[holdout_start:], prediction, baseline_prediction)]},
            "predictive_support": {
                "status": ("baseline_selected" if name in CANDIDATES[:2] else
                           "observed_holdout_improvement" if metrics["mae"] < baseline_metrics["mae"]
                           else "no_holdout_improvement"),
                "statistical_significance": "not_assessed",
                "production_readiness": "not_established"},
            "future_fit_through": _date(dates[-1]),
            "future_forecast": [{"date": _date(day), "prediction": float(pred)}
                                for day, pred in zip(future_dates, future)],
            "unused_columns": [str(column) for column in frame.columns
                               if column not in {task.target_column, task.time_column}],
            "prediction_intervals": None,
            "evaluation_notes": [
                "One complete target observation per day, assumed available at each forecast origin.",
                "Three expanding-window validation folds select the method and baseline before the final holdout.",
                "Every fold and holdout predicts a full horizon without updates from actual targets inside it.",
                "Only past target lags are used; all other columns are excluded, including future weather and target components.",
                "Future forecasts refit the locked method on all observed history after evaluation.",
                "Holdout gains are descriptive; no IID permutation p-values, calibrated intervals, production monitoring or serving are claimed.",
            ],
        }
    except (ValueError, TypeError, FloatingPointError, OverflowError) as error:
        return {"contract": CONTRACT, "status": "error", "task_type": "forecast",
                "message": str(error), "future_forecast": [], "model_comparison": [],
                "predictive_support": {"status": "not_evaluated"}}
