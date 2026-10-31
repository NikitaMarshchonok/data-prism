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

from src.forecast_policy import (
    validate_forecast_policy, weighted_absolute_error, evaluate_forecast_policy,
)


CONTRACT = "daily-forecast-v1"
DELAY_CONTRACT = "daily-forecast-v2"
AVAILABILITY_CONTRACT = "forecast-origin-availability-v1"
CANDIDATES = ("last_value", "seasonal_naive", "lag_ridge", "lag_random_forest")
VALIDATION_FOLDS = 3
MAX_ROWS = 5000


@dataclass(frozen=True)
class ForecastTask:
    target_column: str
    time_column: str
    horizon: int
    seasonal_period: int = 7
    reporting_delay_days: int = 0

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
        if type(self.reporting_delay_days) is not int or not 0 <= self.reporting_delay_days <= 30:
            raise ValueError("reporting_delay_days must be an integer from 0 to 30 days.")


def validate_forecast_task(value):
    """Validate a typed task or strict mapping without guessing task fields."""
    if isinstance(value, ForecastTask):
        return value
    if isinstance(value, Mapping):
        allowed = {"target_column", "time_column", "horizon", "seasonal_period", "reporting_delay_days"}
        if set(value) - allowed:
            raise ValueError("Unknown forecast task fields.")
        try:
            return ForecastTask(**value)
        except TypeError as error:
            raise ValueError("Forecast task requires target_column, time_column and horizon.") from error
    raise ValueError("An explicit forecast task is required.")


def forecast_task_settings(task):
    """Keep the zero-delay v1 task representation readable for retained jobs."""
    result = asdict(validate_forecast_task(task))
    if result['reporting_delay_days'] == 0:
        result.pop('reporting_delay_days')
    return result


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
    minimum = (max(30, 4 * task.seasonal_period) + (VALIDATION_FOLDS + 1) * task.horizon
               + 3 * task.reporting_delay_days)
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


def _metrics(actual, prediction, policy=None):
    with np.errstate(over="raise", invalid="raise"):
        error = np.asarray(actual) - np.asarray(prediction)
        metrics = {"mae": float(np.mean(np.abs(error))),
                   "rmse": float(np.sqrt(np.mean(error ** 2)))}
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError("Forecast metrics are not finite.")
    if policy is not None:
        metrics['weighted_absolute_error'] = weighted_absolute_error(actual, prediction, policy)
    return metrics


def _date(value):
    return value.strftime("%Y-%m-%d")


def forecast_series(frame, task, *, decision_policy=None):
    """Evaluate, lock the selected method, then refit for future forecasts.

    Validation chooses both the method and the reference baseline, never the
    final holdout. Reported improvement is descriptive, not a significance or
    deployment decision. Extra input columns are deliberately unused.
    """
    try:
        task = validate_forecast_task(task)
        policy = validate_forecast_policy(decision_policy) if decision_policy is not None else None
        objective = 'weighted_absolute_error' if policy is not None else 'mae'
        dates, values = _series(frame, task)
        h, period, delay = task.horizon, task.seasonal_period, task.reporting_delay_days
        # origin is the first date being forecast, not the last available label.
        # Bridge unavailable days recursively; never insert their actual targets.
        def predict_at(origin, candidate):
            return _forecast(values[:origin - delay].copy(), h + delay, period, candidate)[delay:]

        # Validation/holdout outcomes must themselves have been released at the
        # final origin. Otherwise delayed labels can leak through model selection
        # even when every individual fit uses a properly embargoed prefix.
        evaluation_end = len(values) - delay
        holdout_start = evaluation_end - h
        # The chosen method must also have been selectable at the HOLDOUT origin,
        # not only at the later final origin. Leave a release gap between the
        # last validation outcome and the holdout start before locking selection.
        selection_end = holdout_start - delay
        origins = [selection_end - h * remaining for remaining in range(VALIDATION_FOLDS, 0, -1)]
        folds = [{"train_start": _date(dates[0]), "train_end": _date(dates[origin - delay - 1]),
                  "forecast_origin": _date(dates[origin - 1]), "unavailable_days": delay,
                  "train_rows": origin - delay, "validation_start": _date(dates[origin]),
                  "validation_end": _date(dates[origin + h - 1]), "validation_rows": h}
                 for origin in origins]
        comparison = []
        for candidate in CANDIDATES:
            try:
                scores = [_metrics(values[origin:origin + h],
                                   predict_at(origin, candidate), policy)
                          for origin in origins]
                comparison.append({"model_name": candidate, "status": "ok",
                                   "mean_mae": float(np.mean([score["mae"] for score in scores])),
                                   "mean_objective_loss": float(np.mean([score[objective] for score in scores])),
                                   "fold_metrics": scores, "selected": False})
            except (ValueError, FloatingPointError, OverflowError):
                comparison.append({"model_name": candidate, "status": "error",
                                   "mean_mae": None, "mean_objective_loss": None, "fold_metrics": [], "selected": False,
                                   "reason": "Candidate could not complete all temporal folds."})
        successful = [row for row in comparison if row["status"] == "ok"]
        baselines = [row for row in successful if row["model_name"] in CANDIDATES[:2]]
        if not baselines:
            raise ValueError("No baseline completed temporal validation.")
        # Stable ordering favours the simple baseline on exact ties.
        selected = min(successful, key=lambda row: row["mean_objective_loss"])
        baseline = min(baselines, key=lambda row: row["mean_objective_loss"])
        selected["selected"] = True
        name, baseline_name = selected["model_name"], baseline["model_name"]
        prediction = predict_at(holdout_start, name)
        baseline_prediction = (prediction.copy() if name == baseline_name else
                               predict_at(holdout_start, baseline_name))
        metrics = _metrics(values[holdout_start:evaluation_end], prediction, policy)
        baseline_metrics = _metrics(values[holdout_start:evaluation_end], baseline_prediction, policy)
        improvement = (1 - metrics["mae"] / baseline_metrics["mae"]
                       if baseline_metrics["mae"] > 0 else None)
        if improvement is not None and not np.isfinite(improvement):
            improvement = None
        # This refit is explicitly AFTER the locked holdout evaluation. Its
        # in-sample history is not used to overwrite the holdout metrics.
        future = predict_at(len(values), name)
        future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=h, freq="D")
        return {
            "contract": DELAY_CONTRACT if delay else CONTRACT, "status": "ok", "task_type": "forecast",
            "task": forecast_task_settings(task), "target_col": task.target_column, "frequency": "D",
            "availability": {"contract": AVAILABILITY_CONTRACT, "reporting_delay_days": delay,
                             "origin_convention": "end_of_completed_day",
                             "evaluation_outcomes_available_through": _date(dates[evaluation_end - 1]),
                             "selection_outcomes_available_through": _date(dates[selection_end - 1]),
                             "unavailable_tail_rows": delay,
                             "source": "caller_assertion", "historical_vintages_verified": False},
            "model_name": name, "baseline_name": baseline_name,
            "selection_metric": ("mean validation weighted absolute error" if policy else "mean validation MAE"),
            "decision_policy": policy,
            "business_evaluation": (evaluate_forecast_policy(policy, metrics, baseline_metrics) if policy else
                                    {'status': 'not_specified', 'production_readiness': 'not_established'}),
            "model_comparison": comparison,
            "cv_folds": VALIDATION_FOLDS, "validation_folds": folds,
            "train_rows": holdout_start - delay, "test_rows": h,
            "holdout": {"train_end": _date(dates[holdout_start - delay - 1]),
                        "forecast_origin": _date(dates[holdout_start - 1]), "unavailable_days": delay,
                        "start": _date(dates[holdout_start]), "end": _date(dates[evaluation_end - 1]),
                        "metrics": metrics, "baseline_metrics": baseline_metrics,
                        "mae_improvement": improvement,
                        "predictions": [{"date": _date(day), "actual": float(actual),
                                         "prediction": float(pred), "baseline": float(base)}
                                        for day, actual, pred, base in zip(dates[holdout_start:evaluation_end],
                                            values[holdout_start:evaluation_end], prediction, baseline_prediction)]},
            "predictive_support": {
                "status": ("baseline_selected" if name in CANDIDATES[:2] else
                           "observed_holdout_improvement" if metrics["mae"] < baseline_metrics["mae"]
                           else "no_holdout_improvement"),
                "statistical_significance": "not_assessed",
                "production_readiness": "not_established"},
            "future_forecast_origin": _date(dates[-1]),
            "future_fit_through": _date(dates[len(values) - delay - 1]),
            "future_forecast": [{"date": _date(day), "prediction": float(pred)}
                                for day, pred in zip(future_dates, future)],
            "unused_columns": [str(column) for column in frame.columns
                               if column not in {task.target_column, task.time_column}],
            "prediction_intervals": None,
            "evaluation_notes": [
                f"One complete daily series with an asserted fixed reporting delay of {delay} whole calendar days.",
                "Each origin is the end of a completed day; targets after origin minus delay are withheld from fitting and prediction inputs.",
                "Three expanding-window validation folds select the method and baseline before the final holdout.",
                "Validation outcomes are released by the holdout origin; selection cannot see its unreleased pre-holdout gap.",
                "Validation and holdout outcomes end at the final available cutoff; unreleased tail labels cannot influence selection or scoring.",
                "Every fold and holdout recursively bridges the unavailable days and predicts the full horizon without actual target updates.",
                "Only past target lags are used; all other columns are excluded, including future weather and target components.",
                "Future forecasts refit the locked method on the available prefix only, using the last input date as their origin.",
                "Availability is an assertion, not reconstructed release timestamps or historical data vintages. Revised targets and variable delays remain unsupported.",
                "Holdout gains are descriptive; no IID permutation p-values, calibrated intervals, production monitoring or serving are claimed.",
            ],
        }
    except (ValueError, TypeError, FloatingPointError, OverflowError) as error:
        contract = DELAY_CONTRACT if isinstance(task, ForecastTask) and task.reporting_delay_days else CONTRACT
        return {"contract": contract, "status": "error", "task_type": "forecast",
                "message": str(error), "future_forecast": [], "model_comparison": [],
                "predictive_support": {"status": "not_evaluated"}}
