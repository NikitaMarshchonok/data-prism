# Owner-scoped forecast model store

`scoped-forecast-model-store-v1` stores evaluated declarative daily models.
The browser can explicitly request retention and download an owned model on a
qualified deployment. The web feature is **off by default**. It does not
implicitly retain models, accept model uploads, migrate local journals, or
change the current hosting.

## Scope and storage preconditions

`ForecastModelStore` uses `jobs/analysis_jobs.sqlite3` inside an explicitly
selected runtime-state root. The optional `forecast_models` table is created
only when this store is explicitly constructed. Ordinary application startup
and `AnalysisJobStore` initialization do not create or enable the registry.

Construction requires explicit deployment/durability declarations and an
expected runtime identity. The existing runtime-state checker must report
`production_state_satisfied`: production profile, operator-declared persistent
storage, and a matching state-directory identity. Ephemeral/demo state,
missing/mismatched identity and symlinked root/jobs/database paths are rejected.
Each operation rechecks continuity, including after acquiring the SQLite scope
fence. A missing database fails instead of silently recreating a registry.

These checks **do not prove physical persistence**. An operator can create the
same declarations/identity on a temporary disk. The physical mount, operating
system permissions, encryption, backup/restore procedure and infrastructure
availability remain operator responsibilities. Do not enable this by merely
changing environment labels on the existing free/ephemeral deployment.

All operations require a validated scope. The caller must derive that scope
from authenticated server context; a user-provided scope string is not
authentication. Model IDs alone never authorize a read, inference, listing or
deletion. No team roles or guest retention are provided. The low-level scope
argument does not prove dataset ownership or training provenance by itself.

## Browser flow (opt-in)

Only enable `VIBEDASH_MODEL_STORAGE_ENABLED=true` after providing and
verifying persistent production storage. The web gate requires the declared
runtime root and the configured job database to be the same state tree. It
refuses a missing/mismatched identity, demo/ephemeral state or a different job
database. The shipped Render configuration stays unchanged.

1. Sign in and start a daily forecast for a confirmed number or count target.
   Leave **Do not save** selected for ordinary evaluation, or choose a
   retention period. Replacing the file or leaving forecast mode resets this
   choice. The CSRF-protected queue request binds a nonempty choice to the
   authenticated account scope and the job; it accepts no owner or artifact
   from the client.
2. The worker prepares a frozen artifact from that job's validated data and
   evaluated forecast. Its model ID is the server-issued job ID. The model
   insert runs inside the successful completion transaction. An interrupted
   or failed job cannot leave a model behind. A model-only failure rolls back
   that attempt without losing an otherwise valid forecast report.
3. The result links to the saved model only if it can be read now. Account
   settings lists retained models and can download each JSON via a
   CSRF-protected, owner-scoped POST. Responses are private, no-store
   attachments. The artifact is not embedded in the forecast HTML, standalone
   report or aggregate audit manifest. Downloaded copies remain outside
   server expiry or erasure.

## Immutability, retention and inference

Registration accepts only the existing strict JSON artifact contract from
`src/forecast_artifact.py`. No pickle, joblib or imported Python object is loaded.
The model hash, fitted preprocessing/predictors, reference, schema, training
cutoff, availability, dependencies, policy and evaluation remain in that
artifact. SHA-256 and strict validation detect inconsistencies, not authenticity
or professional predictive quality; use trusted server-generated artifacts.

The caller must explicitly choose **1–30 days** of retention at registration.
Repeated registration of an identical artifact is idempotent and does not
extend expiry. A different artifact with the same owner/model ID is rejected;
there is no in-place replacement, active-model pointer, promotion or automatic
retraining. New candidate training requires a separate model identity and a
future explicit review workflow.

Reads and inference deny an expired artifact immediately, including exactly at
its expiry timestamp. They do not renew it. The existing VibeDash
request-triggered retention sweep deletes expired models from the shared
database, even after the feature flag is switched off. There is no idle
background timer, so an idle instance may still physically retain expired
rows until its next VibeDash request. Job/report retention is separate and
does not extend or shorten a model's selected expiry.

Capacity is bounded to 10 retained models and 16 MiB of serialized artifact
payload per scope; globally 100 models and 128 MiB. The existing 4 MiB limit per
artifact also applies. Expired-but-not-purged rows still consume capacity.
Capacity errors retain existing models instead of silently evicting them.
SQLite page/WAL overhead, backup size and other runtime data are not included
in these payload quotas.

`predict(scope, model_id, ...)` loads and revalidates only that owner's retained
artifact. Ownership and the deletion fence remain held through frozen
inference. The existing history/cutoff/unit rules apply; there is no fitting,
candidate selection or fallback to someone else's model. The output remains
stateless with `quality_on_new_outcomes=unknown` and
`production_readiness=not_established`. It is **not** written to the local
forecast journal or a hosted outcome ledger by this API.

Default list summaries contain only model ID/hash, store contract, byte count
and creation/expiry timestamps, plus the unchanged non-approval status. The
owner's Account page also receives the date and target column names, target
unit, training dates and reporting delay needed for the input form. It does not
receive fitted parameters, policy text or training target values. Explicit
`get` returns the artifact to an already authorized caller.

### Forecast on updated history

In Account settings, expand **Forecast on updated history** on a retained
model. Submit the original training history unchanged plus at least one newly
available daily value, choose an explicit issue date later than the original,
and confirm that the observations were available then. The submitted dates
must end exactly at issue date minus the saved reporting delay. A delay of
two days and issue date of June 13, for example, require history through June 11.
The confirmation is a user assertion, not independently verified release or
issuance provenance.

The form sends a CSRF-protected POST to
`/vibedash/account/models/<model_id>/predict`. It accepts one comma-delimited
UTF-8 CSV (optional BOM), up to 2 MiB, 5,000 records, and 64 uniquely named
columns. Malformed record widths, duplicate headers, missing/duplicate days,
nonfinite targets, changed training prefixes and unavailable tails fail
without imputation or truncation. Authentication, the persistent-store gate
and ownership are checked before reading the CSV; ownership and expiry are
checked again under the scope fence at inference.

The response shows the frozen model and its original reference side by side.
New history updates lag context, not fitted parameters or model selection.
Invalid negative count predictions are withheld. By default the private,
no-store response shows the forecast for that request only. The request parser
may temporarily spool multipart input. The application does not retain the
history file or extend model retention.

### Save a forecast and compare it with daily actuals

Check **Save this forecast to check it against actuals** on the updated-history
form to retain its model and baseline values. Saving uses the same authenticated
persistent-state gate and SQLite deletion fence as the model. An issue date
must be on or before the server's current UTC day. The record distinguishes the
declared issue date from the server timestamp at first save. The saved forecast
list and score view label a forecast **Prospective** when the first save falls on
its declared UTC issue day, before the daily horizon starts. A save after that
day is **Retrospective**, including when some dates in the horizon are still
ahead. The label describes when Data Prism recorded the forecast. Availability
of source observations remains the account owner's declaration.

The first successful save freezes one output per account, model and issue date.
Repeating the same output returns its original ID without changing expiry.
A different output for the same issue date is rejected. Records expire 150 days
after the first save, independently of model or job expiry. This covers the
supported 90-day horizon and 30-day reporting delay with 30 days left to report
the latest actual. Up to 20 forecasts per account and 200 in total can be active.
The frozen JSON payload is capped at 64 KiB. The application does not evict
active forecasts to admit new ones.

Under **Saved forecasts**, select a forecast and upload a UTF-8 CSV with the
header `date,actual`, up to 16 KiB and 90 rows. Each date must be unique within
the file and belong to this forecast. Today in UTC must be on or after the
date plus the frozen reporting delay. Values must be finite decimals in the
supported numeric range. Count values must be non-negative. The server keeps
normalized facts and their receipt timestamps. Repeated identical values
are idempotent. A conflicting value or invalid date rejects the entire batch.
By default, actuals belong to the selected forecast. Uploads are always
validated against the selected forecast's dates.

To revise a saved actual, open **Correct a saved actual**, select its date and
current value, enter the corrected value and provide a brief reason of up to
300 characters. The submission is bound to the version visible in the form.
A stale form fails after a concurrent correction. An identical retry of the
most recent correction is idempotent. The forecast retains up to 90
corrections across its dates. Each prior value,
receipt time and recorded reason stays in the forecast's correction history.
History shows absolute errors for each version against the original model and
baseline. The aggregate scores use the current value for each date. Forecast
values, scope and expiry stay fixed through corrections.

### Share actuals across forecasts of one daily series

While saving a forecast, expand **Share actuals across forecasts**. Choose
an existing account-owned series or create one with a new name, then confirm
that the forecasts represent the same business measure and population. The
service checks an exact match of time column, target column, unit and reporting
delay. A series has a stable ID. Its name is a label and an account-level
case-insensitive uniqueness check. Confirmation of the business meaning and
population comes from the account owner.

The binding is fixed at first save. Existing isolated forecasts keep their
own facts. In a shared series, one versioned fact per date feeds every linked
forecast that covers it. Each forecast compares that fact with its own frozen
model and baseline predictions. A correction submitted in one forecast is
visible in the other forecasts' scores and history for that date. The limit
of 90 retained corrections applies across the shared series. Series actuals
feed evaluation, while inference takes its input from the uploaded history.

Open **Prospective accuracy by lead time** on a linked forecast to compare
its exact saved model with its baseline across the same series. The summary
uses retained prospective issues. Each lead is scored separately, so a
target date can appear once at that lead. Coverage counts the reported
actuals out of dates due under the reporting delay. Both MAEs use the same
matched dates and the latest recorded versions of their actuals. The summary
reports observed errors over retained forecasts and coverage can change with
expiry or new actuals.

When the frozen model has a confirmed maximum MAE, newly retained forecasts
carry that limit. The series view can use a retained limit from any forecast
with the same model hash in that series, including when viewing an older
forecast. A lead is evaluated once at least seven target dates are due and
all due actuals are reported. The account shows a warning when the model's
MAE exceeds the limit, and a status for each lead. These statuses are
recomputed with the current actuals and retained forecasts.

A shared fact and its revisions expire when the last forecast covering that
date is purged. The series label and ID are purged with its last forecast.
SQLite cleanup follows parent forecast deletion inside the same transaction.
Keep serving instances on the same application version so linked forecast
views use the shared actuals store.

The account view shows date coverage and compares MAE, RMSE and, when declared
in the original task, weighted absolute error. Both predictions are scored on
exactly the same reported dates. With no facts, accuracy is unknown. Partial
coverage is explicit and full coverage means that all expected dates have
reported facts. Neither the source of the actuals nor their original release
timing is independently verified. These comparisons do not promote or retrain
the model. The ephemeral deployment does not meet the storage gate.

## Deletion, recovery and data boundary

The table shares the analysis database and `run_if_scope_open` write fence.
Existing account-scope erasure deletes that scope's model, forecast and actual rows atomically with
its jobs/cases/metrics and permanently fences new writes. Foreign owners remain
untouched. Active jobs or a cleanup failure prevent that transaction from
partially deleting model rows. No filesystem model copies are created.

This is logical deletion, not secure erasure of SQLite pages/WAL, historical
backups or downloads. Old backups may contain subsequently deleted/expired data;
backup retention and restoration controls remain essential. The current
metadata-only account export excludes model snapshots, forecasts and facts.
Model snapshots have a separate explicit download. Retained forecast data is
visible through the owner-only account view. Expiry cleanup on VibeDash traffic
also removes forecasts and actuals when the feature is off.

The offline runtime backup includes models, forecasts, shared actuals and
correction history in the analysis database. Local recovery tests verify the
model, frozen predictions, paired scores and original expiry. A fresh
application process also signs in and applies a correction to the restored
series. Follow the [backup and recovery runbook](BACKUP_RECOVERY.md) for
the offline window and a rehearsal on the provisioned infrastructure.
Retention continues from the original expiry timestamps after restore.

Artifacts contain target lags, fitted parameters/leaf values, task metadata and
possibly confidential free-text policy. They are not aggregate-only audit
records. This module adds no encryption, public download, external transfer or
cross-owner sharing. Existing local model files and local outcome journals stay
outside this server deletion/backup boundary.

## Remaining customer-facing work

Idle-time retention scheduling, independent evidence of source availability,
predeclared alerts and controlled model promotion/rollback remain outstanding.
