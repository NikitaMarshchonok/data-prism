# Prepared owner-scoped forecast model store

`scoped-forecast-model-store-v1` prepares server-side storage for evaluated
declarative daily models. It is an **internal Python API, not an enabled web
feature**. It does not automatically retain browser uploads or models, add an
HTTP model-import endpoint, migrate local journals, or change current hosting.

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
authentication and must never be accepted as authority by a future route.
This slice does not implement that route, account opt-in, CSRF handling, team
roles or guest access. Model IDs alone never authorize a read, inference,
listing or deletion. The same ID may be independently retained by two scopes.
A future save route must bind the artifact to the user's owned, evaluated job
and create it from that job's validated data; the low-level scope argument does
not prove dataset ownership or training provenance by itself.

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
its expiry timestamp. They do not renew it. `purge_expired(scope)` explicitly
removes expired rows for that owner; there is **no scheduled cleanup in this
slice**. Until wired to an operator/application lifecycle, expired payloads may
remain physically stored but are inaccessible through this API. Re-registration
after expiry requires explicit purge and a new retention choice.

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
metadata-only account export does **not** add these data-bearing models. A future
customer-facing save flow must implement its own explicitly authorized model
export/retention UX before enabling registration through HTTP.

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

Authenticated opt-in save/load UI/API, authorized model export, a periodic
retention worker, hosted prediction/outcome matching with series identity,
prospective issuance provenance, predeclared alerts and controlled model
promotion/rollback remain outstanding. The deployed ephemeral service is not
silently upgraded or represented as a durable model service by this module.
