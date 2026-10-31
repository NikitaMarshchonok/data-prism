# Daily forecast evaluation

This is the first bounded forecasting engine, not a production model service.
It is available through Python, a local CLI, and an opt-in VibeDash background
job. The browser workflow binds the task to the reviewed semantic contract;
Python/CLI callers explicitly identify the numeric target and daily time column.

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
5. Select the method and reference baseline by mean validation MAE (or the
   explicitly predeclared weighted absolute error described below), before
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

### VibeDash

Upload a CSV, review column meanings, then select **Daily forecast evaluation
(pilot)**. Select one numeric measure target and one daily time column, set the
horizon and seasonal baseline period, and confirm that completed-day targets
are available before forecasting the next day. Review and confirm the meanings
after changing settings. Forecasting is never enabled merely by mentioning a
prediction in the free-text analysis question.

The browser also asks which decision the forecast supports, relative penalties
for underprediction and overprediction, the required relative error reduction,
and an optional maximum MAE. Confirm these criteria before starting. Editing a
setting or loading a new file clears the confirmations. The decision description
is context supplied by the user, not an automatically understood or validated
business objective. Do not enter personal or confidential information.

The server validates the full input before queueing, without fitting a model.
The worker revalidates the input and its binding to the confirmed target, time,
units and settings before training. It rejects a history that would be truncated
by the dashboard preview limit. Forecast jobs use the existing owner-scoped job
workflow, retention and queue limits; CSRF validation is required.

The result separates holdout evaluation from the later future refit. A worse
holdout result remains visible as **No improvement over the baseline**, not a
successful-model badge. Retaining a simple baseline is also a valid outcome.
Input readiness scores describe input checks, not forecast reliability.
The optional forecast section of the audit manifest contains settings and
aggregate metrics, never per-day observations or predictions. The result page
itself contains those values and remains data-bearing.
The user's decision description is omitted from the audit manifest, which keeps
the bound task fingerprint, numeric criteria and aggregate evaluation instead.
The description is retained in the temporary owned job/session and result page;
hashing it is not anonymization.

For a target explicitly confirmed as a count, negative observations are rejected
and negative future point forecasts are flagged and withheld from the displayed
future table. The raw forecast is not clipped or rounded to manufacture a valid
result. Non-negative output alone does not establish domain validity.

### CLI and Python

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

### Predeclared decision policy

Python callers can pass `decision_policy=` directly to `forecast_series`. The
browser submits the same policy under `confirmed-daily-forecast-task-v2`, bound
to confirmed semantics, availability and settings. Existing v1 jobs and CLI
calls keep MAE selection and report business criteria as `not_specified`.

```python
policy = {
    "contract": "forecast-decision-policy-v1",
    "decision": "Plan next week's staffing",
    "underprediction_weight": 5.0,
    "overprediction_weight": 1.0,
    "minimum_improvement": 0.10,  # fraction; the web form displays percent
    "maximum_mae": None,        # target units per day; None is not assessed
}
result = forecast_series(dataframe, task, decision_policy=policy)
```

The weighted loss is the mean of
`underprediction_weight * max(actual - prediction, 0) + overprediction_weight * max(prediction - actual, 0)`.
Weights must be finite, positive and at most 1000. They are relative preference
units, **not currency or realized business costs**. Equal unit weights reduce
the objective to MAE. Candidate fitting/hyperparameters are unchanged; this is
cost-aware candidate selection, not quantile optimization or a general inventory
decision model. Both the selected method and reference baseline minimize this
same objective over the three earlier validation horizons, never the holdout.

The holdout must show strictly lower weighted loss, the predeclared minimum
relative improvement, and the maximum MAE if supplied. Criteria are conjunctive.
Zero baseline loss or a non-representable ratio yields undefined relative gain,
not a pass. If a confirmed count forecast violates the domain check, a successful
historical criteria status is blocked. Ordinary MAE findings remain separately
labelled: a lower MAE can coexist with failure to meet decision criteria.

`criteria_met_on_holdout` is only a descriptive historical result, never model
approval. Production readiness remains `not_established`; financial value is
`not_assessed`. To revise criteria after inspecting results, treat that as a new
analysis requiring fresh future evidence, not a rescue of the old holdout.

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
claimed. The Python/CLI core accepts general numeric point forecasts; the web
adapter adds the explicitly confirmed count-domain check described above.
Other target constraints are not inferred or silently imposed.

Tests include exact naive/seasonal oracles, known trend sensitivity, retaining a
perfect seasonal baseline, full-horizon prefix isolation, scaler-fit isolation,
holdout mutation without reselection, row-order invariance, unused covariates,
malformed tasks, missing/duplicate dates, non-finite targets and bounded CLI
failure paths. The CI quality gate adds independent-row prediction and daily
forecasting scenarios without weakening the historical 0.8 MAE improvement or
0.05 IID permutation thresholds.
