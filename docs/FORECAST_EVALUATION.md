# Daily forecast evaluation

This is the first bounded forecasting engine, not a production model service.
It is available through Python and a local CLI. VibeDash's upload form and
background jobs do not yet submit forecasting tasks. The confirmed VibeDash
semantic contract is not yet connected to this engine; callers explicitly
identify the numeric target and daily time column.

## Task and availability assumptions

`daily-forecast-v1` supports one series with one finite real numeric observation
per calendar day. The target observation for a completed day is assumed known
at that day's end. A forecast origin is the end of the last observed day, and
the horizon is the next 1–90 days. Target reporting delays require a different
availability design and are not supported here.

Dates must be timezone-naive midnight datetimes or ISO `YYYY-MM-DD` strings.
The engine sorts chronologically without modifying its input. Duplicate days,
missing days, missing/non-finite targets, ambiguous columns, unsupported types,
unknown task fields and insufficient history fail explicitly. It does not
silently aggregate multiple entities, fill gaps, interpolate targets, substitute
a missing target or truncate history. At most 5,000 daily observations are
accepted. Initial history must be at least `max(30, 4 * seasonal_period)` days
before three validation horizons and one final holdout horizon.
Integer targets outside exact float64 precision are rejected rather than
silently losing integer information during model conversion.

Only target lags are predictors. All other columns are unused, even if they
perfectly explain the target. For the bike example, neither the contemporaneous
`casual`/`registered` components nor observed future weather enter the model.
This narrow boundary avoids guessing covariate availability. It is not a
general proof that an arbitrary feature is safe for forecasting.

## Evaluation protocol

1. Reserve the last complete horizon as the final holdout.
2. On earlier history, use three non-overlapping validation horizons, with an
   expanding training prefix. Each has the same length as the requested horizon.
3. Compare last-value and seasonal-naive baselines, standardized lag Ridge,
   and lag Random Forest. Lags are 1, the seasonal period, and twice that
   period. Seasonal period defaults to seven days but is an explicit task
   assumption, not a detected seasonal property. Hyperparameters are fixed.
4. Fit each learned transform/model on its prefix only. Predict every horizon
   recursively without inserting any actual target from inside that horizon.
   Seasonal naive repeats the last observed seasonal cycle for longer horizons.
5. Select the method and reference baseline by mean validation MAE, before
   accessing final holdout outcomes. Exact ties favour the earlier/simple
   baseline. A candidate failing a fold is recorded as failed and cannot win.
6. Evaluate the locked method and reference baseline on the final holdout.
   Report MAE, RMSE, per-day errors and descriptive relative MAE improvement.
   When baseline MAE is zero, relative improvement is null, not an invented
   percentage. Do not reselect a method using the holdout results.
7. Separately refit the locked method on all observed history to predict the
   next horizon. This later fit does not overwrite the original holdout metrics.

The protocol follows the separation of past training data and future
evaluation used in [scikit-learn's lagged-feature example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html)
and the rolling-origin approach described in
[Forecasting: Principles and Practice](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation).

## Usage

```bash
python evaluate_forecast.py --csv /path/to/day.csv \
  --target cnt --time dteday --horizon 7 --seasonal-period 7
```

The CLI reads a local UTF-8 comma-delimited CSV (at most 16 MiB and 64 uniquely
named columns), performs no upload, leaves the input unchanged and prints
strict JSON. Exit code 0 means the protocol completed, **not** that a learned
model beat its baseline; inspect `predictive_support` and holdout metrics.
Exit code 1 means an input, task or evaluation failure. The report includes
actual holdout values and forecasts and should be treated as data-bearing.

```python
from src.forecasting import ForecastTask, forecast_series

task = ForecastTask(target_column="cnt", time_column="dteday",
                    horizon=7, seasonal_period=7)
result = forecast_series(dataframe, task)

# The shared predictive entry point also accepts the explicit task:
from src.ml_predictor import predict_target
result = predict_target(dataframe, forecast_task=task)
```

The legacy independent-row predictor now rejects an explicitly missing target
instead of choosing another. Without a forecast task, dated data requires
explicit `sampling_design="independent_rows"`. That parameter is a caller
assertion about sampling, not a test proving independence; it is unsuitable for
evaluating future demand. Undated legacy calls retain inferred-target behaviour
for compatibility and still require task/domain review.

## Limits and acceptance

Three recent validation horizons and one holdout are narrow historical evidence,
not validation across all seasons, entities, regimes or business costs. A learned
method can fail to beat its reference on the final holdout. The status reports
that outcome instead of claiming success or choosing a replacement after seeing
the answers. Noise can yield a chance positive descriptive improvement. No IID
permutation significance, calibrated prediction intervals, fairness assurance,
automatic deployment, model persistence, outcome monitoring or retraining are
claimed. Target constraints such as non-negative demand are not inferred or
silently imposed; the current contract is for general numeric point forecasts.

Tests include exact naive/seasonal oracles, known trend sensitivity, retaining a
perfect seasonal baseline, full-horizon prefix isolation, scaler-fit isolation,
holdout mutation without reselection, row-order invariance, unused covariates,
malformed tasks, missing/duplicate dates, non-finite targets and bounded CLI
failure paths. The CI quality gate adds independent-row prediction and daily
forecasting scenarios without weakening the historical 0.8 MAE improvement or
0.05 IID permutation thresholds.
