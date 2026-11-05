# Explicit local daily model snapshots

This is the first **local-only** model lifecycle slice. It is not connected to
web uploads, hosted storage, account scopes, model activation, outcome monitoring
or automated retraining. The browser's retention/privacy contract is unchanged.
No model file is written unless the CLI's `--save-model` option or the Python
`save_artifact` function is explicitly called.

## Evaluate, save, predict

For a seven-day forecast issued at the end of 10 January with a two-day target
release delay, the input must contain the complete available daily history only
through 8 January, with enough rows for the existing evaluation protocol:

```bash
python evaluate_forecast.py --csv /path/to/available-history.csv \
  --target orders --time date --horizon 7 --seasonal-period 7 \
  --reporting-delay-days 2 --forecast-origin 2026-01-10 \
  --save-model /private/path/orders.forecast-model.json

# A separate process can reproduce that frozen future forecast:
python predict_forecast.py --model /private/path/orders.forecast-model.json

# History now ends 11 January; issue date is explicitly 13 January:
python predict_forecast.py --model /private/path/orders.forecast-model.json \
  --history-csv /path/to/history-through-january-11.csv \
  --forecast-origin 2026-01-13
```

Paths above are placeholders, not bundled client data. The parent directory must
already exist. A destination file or symlink is never replaced. Save failure
leaves the evaluated main result visible but marks `model_artifact.status=error`
and returns exit code 1. Ordinary evaluations without `--save-model` write no
snapshot. Prediction exits 1 for invalid input or a count-domain failure and 0
for a completed numeric prediction, **not measured quality or operational approval**.

The new-history file must contain the **whole original training prefix unchanged**
plus appended released actuals. Sorting rows is allowed; changing any training
date/value, truncating the prefix, duplicate/missing days, an unavailable tail or
a stale cutoff is rejected. An explicit issue date is mandatory with a new-history
CLI input. There is no automatic use of the wall clock or horizon shortening.
The existing 16 MiB / 64-column / 5,000-row CSV limits apply, including the full
original prefix. Revised targets, entity changes and variable delays are not
silently treated as appended observations.

## What is frozen and what changes

`local-daily-forecast-model-v1` snapshots the method and simple reference already
chosen by the evaluated run. It refits each saved learned method once on the
run's final **available** training prefix, as the original future-refit step does.
Saving verifies that its encoded predictor reproduces the original future refit
within relative `1e-9`, absolute `1e-12` tolerance; it never reselects from the
holdout or historical outer errors. Retrospective input's unreleased final tail
does not enter the model or its history fingerprint.

The format supports all six current candidates:

- Last value and seasonal naive use the latest available lag context.
- Historical mean and median store constants fitted at snapshot creation. They
  **remain fixed**, including when new actuals arrive. This is not the same as
  rerunning evaluation and refitting those constants on an extended series.
- Ridge stores fitted scaler mean/scale, coefficients and intercept.
- Random Forest stores the 60 bounded trees' split/leaf arrays. The interpreter
  matches the fitted estimator's float32 branch-input convention, including
  recursive predicted values. No estimator object is reconstructed or loaded.

`local-frozen-forecast-v1` outputs a fresh prediction ID, immutable model ID/hash,
issue date, available-through and original trained-through cutoffs, model and
reference predictions. New released actuals update lag inputs only; no fit,
candidate selection or hyperparameter search runs during prediction. The fixed
original horizon, seasonal period and release delay cannot be overridden.
Availability remains asserted, not verified historical release timestamps.

Python callers can explicitly use `create_artifact(frame, task, report,
decision_policy=..., target_unit='count')`. Creation binds the policy and verified
source evidence. Count input and the evaluated method/reference/future scopes
must be domain-valid before saving; later negative frozen predictions remain
visible with a failed status, without clipping, fallback or automatic retraining.
The CLI does not infer a count unit or business policy from a column name.
Unused source columns never become predictors or snapshot fields.

## Evidence and privacy boundary

The JSON stores schema/task definitions, candidate suite, fitted parameters,
the final `2 × seasonal_period` target observations needed for recursion, training
dates/row count/fingerprint, policy if supplied, verified historical holdout
aggregate errors, backtest completion status, library versions and source-file
fingerprints. It does not store all original rows or per-day holdout actuals.
Historical holdout errors describe the earlier locked evaluation, **not** the
final refit's in-sample accuracy and not future outcomes. The snapshot retains
the backtest status, not its complete per-window evidence; retain the evaluated
report separately if that full record is needed.

This file is **data-bearing**: target lags, fitted parameters (including tree
leaf values), metadata and free-text decision policy can be confidential. Files
are created atomically with owner-only mode `0600`; this is not encryption,
authentication, cloud durability, secure erasure, an access-control service or
backup. The local operator controls sharing, storage, deletion and retention.
The `*.forecast-model.json` name pattern is ignored by Git to reduce accidental
tracking, but other names and force-add can bypass that convenience.

Loading is bounded to 4 MiB and a strict declarative schema. Duplicate fields,
non-finite values, unknown contracts, invalid tree indices/cycles/depth, malformed
scalers and fingerprint mismatches fail. Pickle/joblib/Python object payloads
are never deserialized. SHA-256 detects accidental changes, **not authenticity**:
someone able to edit a file can also replace its hash and valid parameters.
Use trusted locally created snapshots; a valid file is not proof of honest
training, professional competence or business suitability. Dependency/source
fingerprints record creation provenance, not a signature or automatic check that
the current interpreter source matches it.

## Still required for the hosted lifecycle

The [scoped store and opt-in web download](FORECAST_MODEL_STORE.md) use the
same artifact contract. Web storage remains off by default and does not make
the current ephemeral deployment durable.

An explicitly opt-in local prediction/actual journal is now available through
`predict_forecast.py --ledger`; see [FORECAST_OUTCOME_LEDGER.md](FORECAST_OUTCOME_LEDGER.md).
It stores frozen output/reference pairs, accepts partial reported actuals with
explicit corrections, and computes paired errors. It does not change the
stateless prediction object's `quality_on_new_outcomes=unknown`: only a separate
journal evaluation knows which facts have subsequently been supplied.

Hosted inference, idle-time retention cleanup,
hosted prediction/actual matching, durable hosted measured-quality history, alert rules,
controlled candidate retraining, reviewable model promotion and rollback are not
implemented here. Without the explicit journal option, emitting a new
prediction ID does not persist or monitor that prediction. Missing actuals cannot
be called success. Real approved cases and specialist acceptance also remain
outstanding. These local snapshots are experimental, not operationally approved.
