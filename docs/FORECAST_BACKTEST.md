# Historical evaluation for an uploaded daily series

New completed web and `evaluate_forecast.py` runs include a
`daily-rolling-backtest-v1` report under `historical_backtest`. This runs the
existing forecasting procedure at **three fixed past issue dates** on the
uploaded series. It supplements, rather than replaces, the final holdout and
exploratory future forecast. It is not live monitoring or professional acceptance.

## Fixed protocol

Use the confirmed horizon `H`, seasonal period `P`, constant reporting delay `D`,
candidate suite and, when supplied, decision policy. There is no parameter search
over the outer results. After sorting the validated daily series, let `E` be
the exclusive end of released evaluation outcomes: source length `N` in explicit
available-history mode, or `N - D` in retrospective mode. Zero-based outer
target starts are `E - 3H`, `E - 2H`, and `E - H`.

At each start `S`:

1. The historical issue date is the source date at `S - 1`. The available
   training prefix contains only the first `S - D` rows and only date/target
   columns. Neither the unavailable gap nor the outer horizon enters that call.
2. Rerun `forecast_series` in explicit issue-date mode. Its three inner
   validation horizons select the method and reference; its separate inner
   holdout does not reselect either. Refit the selected method on the available
   prefix and forecast the next `H` days after that historical issue date.
3. Check returned inner evidence against that prefix with
   `validate_forecast_result`, including the declared suite. Independently score
   the future predictions against the outer actuals held outside the call.
4. Independently construct every declared simple reference from that same
   prefix, advancing the seasonal cycle through `D` unavailable days. The
   comparator is the reference selected on **inner validation**, never the one
   with the smallest outer error. Record all reference errors and predictions.
5. Report MAE and RMSE, plus the declared weighted absolute loss when present.
   Compare using weighted loss if declared, otherwise MAE. Better/tied/worse
   uses relative tolerance `1e-9`, absolute tolerance `1e-12`; relative improvement
   is undefined for a zero reference loss. Counts of better windows are not an
   accuracy percentage, significance test or business acceptance rate.

Later origins may legitimately use earlier outcomes that have since been
released. The three horizons are nonoverlapping but are not independent samples.
Their outcomes do not select or alter the final future method, its predictions,
the policy, the number of origins or the horizon. The last outer horizon overlaps
the main report's final holdout: it must not be counted as additional unseen
evidence. A per-window method can differ because each origin reruns selection
using its own available history; this evaluates the service procedure, not one
fixed fitted model. There are three additional engine calls per eligible run.

## History and unsuccessful results

Each historical call requires at least `M = max(30, 4P) + 4H + 2D` available rows.
The full protocol therefore requires `M + 3H + D` source rows in explicit
available-history mode, or `M + 3H + 2D` in retrospective mode. For `H=P=7, D=0`,
this is 79 rows. The existing single-holdout procedure can succeed with less.

If any of the three prefixes is too short, **none** is fitted. Keep all three
slots as `not_evaluated`, report `insufficient_history` and required/supplied row
counts. Do not shorten the protocol or silently substitute favourable dates.
This does not erase a valid main holdout or future forecast.

An unsuccessful engine call or inconsistent returned evidence keeps its slot as
`error`, without data-bearing exception text or a replacement run. Valid numeric
results with candidate failures remain scored but flagged. For a web target
explicitly confirmed as `count`, the existing nonnegativity check covers each
historical call's inner method/reference holdout and outer future prediction.
Negative outputs remain visible and block an otherwise met decision-criteria
status. General numeric targets do not acquire an inferred count constraint.
The CLI has no semantic count declaration and does not apply that web-only rule.

`completed` means three scored windows without recorded candidate or domain
failures; `completed_with_issues` retains failures and any scorable results.
Neither establishes predictive acceptance or production readiness. With a policy,
the existing `criteria_met_on_holdout` policy label refers here to that outer
window only; the UI says so. A baseline win, tie, unmet criterion, missing
evaluation or negative count result cannot be promoted to operational approval.

## Evidence, audit and boundaries

Web results show a per-window comparison table, textual failure/criteria labels,
and the evaluation limitation. The table has explicit row/column headers and a
keyboard-focusable horizontal scroll region for narrow screens. CLI JSON also
contains outer daily actuals, predictions and reference predictions: it is
data-bearing and should not be shared as an anonymous audit.

The audit manifest contains only protocol settings, date cutoffs, aggregate
errors, selections and statuses. It excludes outer daily values/predictions and
the free-text business decision. Stored old results are not recomputed or given
a new backtest badge. Raw Python `forecast_series` calls remain unchanged; Python
callers can explicitly use `src.forecast_backtest.historical_forecast_backtest`.

This uses the inspected uploaded file, not a new client or independent unseen
acceptance set. Fixed availability is a user assertion, not verified historical
vintages. Revised targets, variable release delays, multiple entities, external
features, calibrated intervals, persisted model serving, incoming actuals and
live quality monitoring remain unsupported by this protocol. Approved real
cases, agreed business comparators and independent specialist review are still
required before claiming professional-level capability.
