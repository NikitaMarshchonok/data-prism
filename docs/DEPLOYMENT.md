# Deployment and operations

This guide describes the repository's current single-instance deployment shape. It does not claim multi-region or high-availability operation.

## Render Blueprint

The root `render.yaml` defines a Docker web service with:

- automatic deployment only after linked CI checks pass and a runtime-relevant
  file changed;
- build filters that skip root Markdown and `docs/**`-only changes;
- `/readyz` as the deployment health check;
- generated session and monitoring API secrets;
- an explicit `demo` deployment profile with `ephemeral` state durability;
- one Gunicorn worker with four threads for the 512 MB free plan;
- one bounded background analysis thread with durable SQLite job states;
- a 25 MB public upload limit and bounded VibeDash previews;
- structured JSON application logs.

To deploy:

1. Sign in to Render and select **New → Blueprint**.
2. Connect `NikitaMarshchonok/data-prism`.
3. Review the proposed free web service and apply the Blueprint.
4. Wait for the Docker build, CI-gated deploy, and readiness check.
5. Open the assigned `onrender.com` URL and run the built-in VibeDash demo.

Render documents the current Blueprint workflow and schema at:

- <https://render.com/docs/infrastructure-as-code>
- <https://render.com/docs/blueprint-spec>
- <https://render.com/docs/monorepo-support#setting-build-filters>

No external AI key is required for the deterministic demo. `OPENAI_API_KEY` can be added later as a secret if optional narrative summaries are needed.

The Blueprint ignores automatic builds when a commit changes only root Markdown
files or files under `docs/`. This avoids restarting the free instance and
discarding its temporary state for documentation-only merges. Render always
processes changes to `render.yaml` itself, and manual deploys or service
configuration changes also bypass build filters. The filter therefore reduces
avoidable restarts; it does not make the filesystem durable or protect state
during an application deployment.

## Persistence boundary

The free Render service uses an ephemeral filesystem. Uploaded datasets, reports, baselines, drift history, and decision cases are therefore lost when the instance is restarted or redeployed. This is acceptable for a public portfolio demo, but not for persistent monitoring or a team decision record.

`/readyz` exposes the bounded `runtime-storage-v1` contract. The checked-in free
Blueprint declares:

```text
DATA_PRISM_DEPLOYMENT_PROFILE=demo
DATA_PRISM_STATE_DURABILITY=ephemeral
```

That combination remains healthy for the public demo, but the response includes
an explicit data-loss warning and reports `production_state_satisfied=false`.
A deployment declaring `DATA_PRISM_DEPLOYMENT_PROFILE=production` fails
readiness unless it also declares `DATA_PRISM_STATE_DURABILITY=persistent`,
sets `DATA_PRISM_STATE_DIR` explicitly, and verifies that directory against
`DATA_PRISM_EXPECTED_STATE_ID`. Unknown values, a missing marker, and a wrong
identity fail closed.

The durability value is an operator declaration, not automatic proof of the
underlying mount. The state identity proves only that the process can read the
same initialized state lineage expected by the operator. A production operator
must still verify provider storage, backup, restore, retention, and deletion
behaviour. Setting these values alone does not satisfy the durable-state release
gate. The marker is not tamper-proof: an actor who can replace state can copy it.

VibeDash pilot accounts use a separate local SQLite file under the configured
state directory. Render sets `SESSION_COOKIE_SECURE=true`; local development
defaults this flag off unless explicitly enabled. Accounts are optional and
do not provide enterprise authentication, email verification, email-based
recovery, team roles, or a durable-account SLA. An authenticated user can
generate eight high-entropy recovery codes after re-entering the current
password. Only their domain-separated hashes are stored; a new set replaces
the old set, and successful recovery consumes the full set, rotates the
password, and invalidates old browser credentials. The codes are displayed
once, so losing both password and saved codes is unrecoverable. The signed Flask cookie carries
the VibeDash account id and opaque credential token; the server checks that pair
against the current password hash atomically on every authenticated request.
Existing cookies without a valid token fail closed. Changing a password rotates
the VibeDash identity/CSRF keys, keeps the current browser authenticated with a
fresh token, and invalidates independently copied old cookies. Ordinary signed
cookie rotation cannot revoke copied cookies before a password change; use a
server-side session/revocation store if that guarantee is required. On the free
host, account data and account-owned history can reset with the filesystem.
Guest analyses remain browser-scoped and are never migrated into an account.

The account database migrates exact legacy v1 and v2 schemas to v3 on open. V3
adds a cascade-deleted security-event table capped at 100 rows per account. Only
fixed event types and UTC timestamps are retained; network addresses,
device/browser details, account email, session material, raw recovery codes,
and free-form metadata are not collected. A malformed or unexpected schema
fails closed rather than being modified.

Authenticated pilot accounts can download their retained metadata from
`POST /vibedash/account/export.json`. The export is limited to the newest 50
analysis jobs, newest 100 decision cases, and at most 100 retained security
events, with truncation flags when older job or case records exist. It excludes
raw datasets, prompts, filenames, schema/session
data, and global pilot metrics. JSON is assembled and serialized in memory;
the request does not create an export artifact or extend retention. The
download is protected by the account CSRF token and private/no-store response
headers. Export size or validation failures are reported generically, without
logging account-owned fields.

The account settings page also exposes `POST /vibedash/account/delete`. A
deletion request requires the account CSRF token, current password, and an
exactly typed `DELETE`. The route refuses to proceed when the account scope has
queued or running jobs; retry after those jobs complete. When the scope is
idle, the local implementation removes the account-owned jobs, decision cases,
opt-in pilot measurements, and safely identifiable scoped VibeDash artifacts.
Classic analysis and monitoring data are outside this account-owned scope.

Deletion spans an account store, an analysis-job store, and filesystem cleanup,
so it is not a cross-store atomic operation. Files with unidentifiable
ownership are left for normal retention cleanup. Local copies/downloads and
historical offline backups are not erased by the request, and a free-host
restart or redeploy can remove state earlier than the application workflow.

For single-instance persistent monitoring, upgrade to a paid service and attach a disk at:

```text
/var/lib/data-prism
```

Keep `DATA_PRISM_STATE_DIR=/var/lib/data-prism`. Render's disk documentation explains the cost and operational constraints: <https://render.com/docs/disks>.

After attaching and verifying the disk, keep the service and all writers stopped
and initialize the mounted directory once:

```bash
python runtime_state.py initialize --state-dir /var/lib/data-prism
```

Save the returned `state_id` in the deployment secret manager as
`DATA_PRISM_EXPECTED_STATE_ID`; do not commit it or keep the only copy on the
mounted disk. Then set:

```text
DATA_PRISM_DEPLOYMENT_PROFILE=production
DATA_PRISM_STATE_DURABILITY=persistent
DATA_PRISM_EXPECTED_STATE_ID=<64-character value kept outside the disk>
```

Restart, then require `/readyz` to report
`runtime_storage.state_identity.status=verified`. Readiness exposes only a
12-character SHA-256 fingerprint for operator correlation. If the marker is
missing or the identity differs, the service returns HTTP 503. Investigate the
mount or restore instead of replacing the expected value.

Do not apply those declarations to the free Blueprint: its filesystem remains
ephemeral regardless of the environment-variable value.

A persistent disk restricts the service to one instance and prevents zero-downtime deploys. A future multi-instance architecture should instead move uploads and reports to object storage and drift history to PostgreSQL.

## Offline runtime backup and recovery

`runtime_backup.py` provides offline snapshot, verification, and restore commands
for a dedicated `DATA_PRISM_STATE_DIR`. It snapshots SQLite with its backup API,
checks file digests, preserves artifact ages, and restores only into a new
directory. The operator must stop all writers; `--offline` is an acknowledgment,
not a stop mechanism. SHA-256 detects corruption, not authenticity, and the
CLI provides no encryption. It does not provision managed persistent hosting,
paid infrastructure, off-host copies, or automatic scheduling. There are no
crash or power-loss durability guarantees beyond the implemented file fsyncs.
Follow the [backup/recovery runbook](BACKUP_RECOVERY.md), including preservation
of signing/API secrets and post-snapshot deletion requests, before using it on
real data.

## Temporary-artifact retention

### Retention settings

VibeDash stores normalized upload copies, dashboard sessions, and generated HTML exports under `DATA_PRISM_STATE_DIR`. Their retention window is controlled by:

```text
VIBEDASH_RETENTION_HOURS=24
```

The accepted range is 1–720 hours. Before each VibeDash request, the application removes expired regular files that match its server-generated naming schemes. It does not recursively traverse directories, follow symbolic links, or remove unrelated files. Cleanup totals and failures are emitted as structured operational events without logging uploaded filenames or session identifiers.

This cleanup is activity-triggered. An inactive service may retain an expired file until the next VibeDash request, and an ephemeral host may remove it earlier during restart or redeployment. Therefore the setting is a bounded application lifecycle policy, not a wall-clock deletion SLA. A deployment requiring strict deletion timing should use a scheduled cleanup job or an object-store lifecycle rule.

Evidence-linked decision cases are stored in the analysis-job SQLite database but
use a separate lifecycle:

```text
VIBEDASH_DECISION_RETENTION_DAYS=90
VIBEDASH_MAX_DECISION_CASES_PER_SCOPE=50
```

Closed cases are removed after the configured 1–730 day window when a VibeDash
request triggers cleanup. Active cases remain until closed. The per-scope limit
accepts 1–500 cases. Guest access depends on the signed browser session; a
signed-in pilot account instead supplies one deterministic account scope across
browsers. These settings do not override the free host's ephemeral storage
behaviour.

## Analysis job lifecycle

JavaScript-enabled VibeDash clients submit work to `POST /vibedash/jobs` and poll the returned status URL. Job records move through `queued`, `running`, `completed`, or `failed` in SQLite. Access to status and results is restricted to the server-signed guest-browser or pilot-account scope that created the job.

`GET /vibedash/history` lists recent runs for that same guest browser or signed-in account scope. Completed jobs expose a downloadable manifest containing the service version, analysis-contract version, request/specification fingerprints, dataset and schema SHA-256 fingerprints, analyzed shape, truncation state, and evidence counts. The manifest contains no source row values. It is temporary metadata: the job record and its manifest are purged with `VIBEDASH_RETENTION_HOURS`, while the associated session and upload follow the same file-retention policy.

The single-instance deployment intentionally runs one in-process analysis worker. The queue defaults to two active jobs per guest-browser or pilot-account scope and 25 across the service. A job that remains `running` longer than 600 seconds is treated as interrupted and reported as failed. These bounds can be adjusted with:

```text
VIBEDASH_JOB_TIMEOUT_SECONDS=600
VIBEDASH_MAX_ACTIVE_JOBS_PER_SCOPE=2
VIBEDASH_MAX_ACTIVE_JOBS=25
```

This topology prevents long analysis from occupying an HTTP request thread, but it does not provide distributed delivery guarantees. A process restart can interrupt active work, and the free Render filesystem can remove queued inputs during restart or redeployment. Multi-instance production deployment requires an external queue, shared object storage, and separate workers.

## Runtime signals

Each response includes an `X-Request-ID`. A valid inbound request ID is preserved; otherwise the application creates a new one. Application request logs contain:

- UTC timestamp and log level;
- service version or deployment commit;
- request ID;
- HTTP method and normalized Flask route;
- response status and duration in milliseconds.

Raw request bodies, uploaded filenames, query strings, IP addresses, and session identifiers are not added to access logs.

Render supplies `RENDER_GIT_COMMIT` automatically. Data Prism exposes its shortened value in `/healthz` and `/readyz` and includes it in structured request logs.

## Operational verification

Run the read-only release verifier before a supervised pilot. Pass the intended
Git SHA; a healthy service on an older revision is a failed release check.

```bash
python release_check.py \
  --base-url https://YOUR-SERVICE.onrender.com \
  --expected-revision "$(git rev-parse HEAD)"
```

The command verifies the liveness and readiness JSON contracts, revision
agreement, request-ID propagation, and the VibeDash landing page. It follows no
redirects, uploads no data, and creates no account or analysis. A successful
result confirms only technical deployment acceptance. Any reported ephemeral
storage warning still applies, and the command does not demonstrate usability,
durability, recovery, or product value. Use `--json` when a machine-readable
`data-prism-release-check-v1` result is needed for private release records.

The individual checks remain useful while diagnosing a failed result:

```bash
curl -i https://YOUR-SERVICE.onrender.com/healthz
curl -i https://YOUR-SERVICE.onrender.com/readyz
curl -i -H "X-Request-ID: portfolio-check-001" \
  https://YOUR-SERVICE.onrender.com/vibedash/
```

Verify that:

- both health endpoints return HTTP 200;
- the response includes the same safe request ID;
- the JSON log event contains the normalized route rather than a raw URL;
- the reported deployment version matches the active Render commit.

The pull-request workflow also builds the production Docker image, starts it with generated test credentials, waits for `/readyz`, verifies request-ID propagation, and confirms that a structured request event reached container logs. This provides the container-level verification even when a contributor does not run Docker locally.

## Rollback

If a deployment fails its readiness check, Render keeps the previous successful version active. For an application-level regression after deployment, select the last known-good deploy in the Render dashboard and redeploy it, then confirm `/readyz` before resuming traffic.
