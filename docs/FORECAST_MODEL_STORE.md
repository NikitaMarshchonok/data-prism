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

List summaries contain only model ID/hash, store contract, byte count and
creation/expiry timestamps, plus the unchanged non-approval status. They exclude
parameters, policy text, source schema names and training tails. Explicit `get`
returns the data-bearing artifact to an already authorized internal caller.

## Deletion, recovery and data boundary

The table shares the analysis database and `run_if_scope_open` write fence.
Existing account-scope erasure deletes that scope's model rows atomically with
its jobs/cases/metrics and permanently fences new writes. Foreign owners remain
untouched. Active jobs or a cleanup failure prevent that transaction from
partially deleting model rows. No filesystem model copies are created.

This is logical deletion, not secure erasure of SQLite pages/WAL, historical
backups or downloads. Old backups may contain subsequently deleted/expired data;
backup retention and restoration controls remain essential. The current
metadata-only account export does **not** add these data-bearing models. They
have a separate explicit download.

The existing offline runtime backup includes this table as part of the shared
database. A round-trip test verifies the artifact, frozen predictions and
original expiry after restore. Backup and restore still require an explicit
offline assertion and all writers stopped; a test in temporary directories is
not evidence that the deployed host has persistent storage or a tested recovery
operation. A model may already be expired when a backup is restored: recovery
does not restart retention.

Artifacts contain target lags, fitted parameters/leaf values, task metadata and
possibly confidential free-text policy. They are not aggregate-only audit
records. This module adds no encryption, public download, external transfer or
cross-owner sharing. Existing local model files and local outcome journals stay
outside this server deletion/backup boundary.

## Remaining customer-facing work

Web inference on new history, idle-time retention scheduling, hosted
prediction/outcome matching with series identity, prospective issuance
provenance, predeclared alerts and controlled model promotion/rollback remain
outstanding. The deployed ephemeral service is not silently upgraded or
represented as a durable model service by this module.
