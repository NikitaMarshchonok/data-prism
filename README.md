# Data Prism

[![CI](https://github.com/NikitaMarshchonok/data-prism/actions/workflows/ci.yml/badge.svg)](https://github.com/NikitaMarshchonok/data-prism/actions/workflows/ci.yml)

**Evidence-first automated data science and drift monitoring for tabular data.**

Data Prism turns CSV, TSV, Excel, JSON, and Parquet files into an interactive analysis workspace. It combines deterministic data-quality checks, statistically validated findings, leakage-safe model evaluation, and persistent drift monitoring in one Flask application.

The project is designed as a decision-support system: every important conclusion should be traceable to a metric, sample size, confidence estimate, baseline, or diagnostic—not just an LLM-generated narrative. A deterministic analytical quality gate measures known-signal detection and false discoveries on synthetic benchmark scenarios before changes are merged.

The product direction is industry-independent: companies bring their own task
and business context to a common intake with specialized, verified executors.
This is not a promise to solve every task from any file. Current confirmed
VibeDash workflows and their limits are exposed by `GET /vibedash/task-capabilities`;
`GET /vibedash/task-intake?task_type=daily_forecast` provides read-only admission
guidance, not dataset approval. The upload form also confirms the question,
row meaning and planned supported result before queueing; this is structured
user context, not automatic understanding of business intent. See the [product plan](docs/DATA_SCIENCE_PRODUCT_PLAN.md)
for the agreed target, current boundaries and acceptance milestones.

![Data Prism evidence-first analytics workspace](docs/assets/data-prism-landing.jpg)

## What the system does

| Area | Capabilities |
| --- | --- |
| Data intake | Validated uploads, safe filenames, normalized working copies, configurable size limits, and row limits |
| Dataset readiness | Pre-analysis score, blocking quality contracts, privacy/schema warnings, and actionable remediation without retaining preflight rows |
| Confirmed semantics | Required review of column roles, units, optional target/time fields, and predictive exclusions before uploaded VibeDash data is analyzed |
| Profiling | Schema summary, missingness, duplicates, constant columns, distributions, correlations, and outlier diagnostics |
| Evidence engine | Ranked findings with supporting metrics, confidence levels, sample sizes, and recommended next steps |
| Decision brief | Up to three ranked priorities that connect a finding to evidence, decision risk, and the next verification action |
| Decision workflow | Evidence snapshots, accountable owners, success metrics, targets, review dates, and measured outcomes |
| Statistical validation | Welch group comparisons, Pearson correlation, effect sizes, 95% confidence intervals, and Benjamini–Hochberg FDR correction |
| Exploratory ML | Fixed-budget anomaly review ranking and gap-statistic-gated, subsample-stability-tested segmentation |
| Predictive ML | Leakage-safe preprocessing, cross-validated model selection, untouched-holdout evaluation, naive-baseline comparison, and a holdout permutation support gate |
| Daily forecasting (opt-in pilot, Python/CLI) | Confirmed single-series task, fixed target reporting delay, available-prefix lag models and outcome-censored temporal selection, simple baselines, reserved-horizon evaluation and separate future refit; no production-readiness or automatic monitoring claim |
| Model reliability | Per-class metrics, calibration, residual analysis, permutation importance, split stability, and supported subgroup checks |
| Monitoring | Aggregate baseline profiles, PSI and categorical drift, missingness/schema changes, persistent history, and deduplicated alerts |
| Interfaces | BI dashboard, prompt-to-dashboard workspace, session-scoped run and decision history, downloadable audit manifests, HTML/PDF reports, authenticated monitoring API, and cron/CI-ready CLI |
| Operations | Durable analysis-job states, bounded background execution, reproducibility fingerprints, request IDs, structured JSON logs, managed temporary-artifact retention, and a CI-gated Render Blueprint |
| Quality evaluation | Versioned synthetic benchmarks for known statistical, anomaly, segmentation, and predictive signals; null-noise guardrails; reproducibility; and machine-readable CI evidence |

## System overview

```mermaid
flowchart LR
    A[Tabular dataset] --> P[Readiness preflight]
    P -->|Ready| S[Confirm roles and units]
    P -->|Blocked| X[Remediation guidance]
    S --> Q[Bounded analysis job]
    Q --> B[Validated loader]
    B --> C[Quality and profiling]
    C --> D[Evidence and statistical engines]
    C --> E[Anomaly and segmentation]
    C --> F[Leakage-safe model evaluation]
    D --> G[Dashboard and reports]
    E --> G
    F --> G
    G --> L[Decision case]
    L --> M[Measured outcome]
    B --> H[Aggregate baseline]
    H --> I[Drift comparison]
    I --> J[History and alerts]
    J --> K[Web UI / API / CLI]
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for component boundaries, [docs/DATASET_READINESS.md](docs/DATASET_READINESS.md) for the pre-analysis contract, [docs/PILOT_DECISION_WORKFLOW.md](docs/PILOT_DECISION_WORKFLOW.md) for the evidence-to-outcome product experiment, and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the supported deployment and persistence model.

For the first bounded forecasting workflow and its evaluation limits, see
[docs/FORECAST_EVALUATION.md](docs/FORECAST_EVALUATION.md). Dated data is no longer
silently treated as independent random rows by the legacy predictive module.

## Reproducible product demo

Start the application, open `http://localhost:5001/vibedash/`, and select **Run the live demo**. No upload or external model is required. The application generates the same privacy-safe SaaS dataset and dashboard specification on every run, then calculates:

- four traceable business KPIs and five visual diagnostics;
- evidence-backed data-quality, correlation, trend, outlier, and concentration findings;
- effect sizes, confidence intervals, p-values, and FDR-adjusted statistical results;
- fixed-budget multivariate anomaly review candidates and exploratory segments that must pass a one-cluster gap-statistic guardrail;
- a dataset-readiness record and a ranked decision brief that separates evidence, risk, and next action.

![Data Prism SaaS evidence dashboard](docs/assets/data-prism-evidence-dashboard.jpg)

The synthetic dataset can also be generated independently for scripts or notebooks:

```bash
python -m src.demo_data --output data/demo/saas_growth_demo.csv
```

The generator is deterministic by default, contains no personal information, and deliberately includes missing values and operational incidents so reviewers can verify that the analysis panels produce meaningful results.

In a JavaScript-enabled browser, VibeDash first performs a non-retained dataset preflight. Blocking issues such as an unusably small sample, ambiguous columns, overwhelming missingness, or material duplicate-row distortion stop the run before a job or retained application working copy is created. Warnings remain visible but allow the user to continue with explicit limitations. An allowed upload then exposes the inferred analytical role and unit for every column. The user must review and confirm this schema-only contract; changing any field removes confirmation. The server validates it again against the uploaded CSV before creating a job.

After confirmation, VibeDash submits analysis through a durable job lifecycle and displays `queued` or `running` progress until the result is ready. Refreshing the page does not remove the SQLite job record. Direct clients of the synchronous upload endpoint must also provide a valid confirmed semantic contract; the built-in synthetic demo remains available without manual schema review because its generated dataset and specification are fixed. Confirmed semantics cover single-dataset VibeDash and its two-period comparison workflow. Standalone drift tools and the standalone predictive API still use the shared conservative heuristic role profiler and remain separate follow-up integrations.

The landing page also provides **Compare two periods** for an independent, observational before/after comparison. Upload two CSV files, label them baseline and current, and keep the row unit and metric definitions aligned. The report checks schema changes and distribution drift, then reports numeric current-minus-baseline mean deltas. For eligible shared numeric columns it runs Welch independent-samples t-tests, 95% confidence intervals, Hedges' g, and Benjamini–Hochberg false-discovery-rate adjustment. These are screening statistics: the comparison is non-causal and does not show that a period change caused an outcome; seasonality, population mix, confounding, and dependence can explain differences.

Choose **Review both files**, edit each file's proposed roles and stored units, and confirm that row units, shared definitions, scales, and currency match. Preflight reads the streams without retaining a Data Prism working copy. Shared columns with different roles or units block comparison; added/removed fields remain visible as schema changes. The server revalidates both contracts before queuing and again in the worker. Numeric tests and distribution drift use the same confirmed roles. There is no automatic scale or currency conversion. For ratio/percent metrics the report distinguishes percentage-point deltas from relative percentage changes. New reports use `period-comparison-v2` and comparison manifest v2, with both schema-only contracts and fingerprints; historical v1 reports remain readable.

Comparison uploads are bounded to 100 MiB per request, 100,000 rows and 100 columns per file, 100,000 combined rows and columns, and 256 MiB of combined in-memory frames. At most 32 candidate metrics receive inferential processing (up to 8 are displayed), with at least 8 finite observations per period required for a test. The two source CSVs are deleted after worker processing; the aggregate result, session record, and aggregate-only audit manifest are retained temporarily under `VIBEDASH_RETENTION_HOURS` (24 hours by default). They are not durable storage or a production availability guarantee.

The browser first calls `POST /vibedash/comparisons/readiness`. After review it submits `POST /vibedash/comparisons/jobs` with `baseline_semantic_contract`, `current_semantic_contract`, and `comparison_semantics_confirmed=yes`, then polls the scoped asynchronous job at `/vibedash/jobs/<job_id>` and opens `/vibedash/jobs/<job_id>/result`. Both POST endpoints require the form CSRF token. The aggregate manifest is available at `/vibedash/jobs/<job_id>/manifest`, and completed comparison results offer an in-memory standalone HTML download at `/vibedash/jobs/<job_id>/comparison-report.html`. Job, history, result, manifest, and report access is restricted to the signed guest-browser or pilot-account scope that created the job.

The comparison HTML download is server-named, generated in memory, and is not a second retained artifact. It reuses the comparison template with trusted CSS inlined, no live dependencies, and a restrictive local-document CSP. The standalone page includes print CSS for browser Print/Save as PDF; server-side PDF generation is not part of this flow.

Completed background runs appear under **History** for the same signed guest-browser or pilot-account scope. Each result includes a versioned audit manifest with the deployment version, source and schema SHA-256 fingerprints, confirmed schema semantics and their fingerprint, request and specification fingerprints, row/column coverage, truncation state, readiness outcome, decision-brief coverage, and evidence counts. Manifests contain no source row values and follow the same temporary retention policy as the result.

From a completed background result, a user can turn one ranked priority into a decision case before the outcome is known. The case freezes a bounded evidence summary and audit fingerprint together with the owner, decision, success metric, target, and review date. Later, the same guest-browser or pilot-account scope records whether the target was validated, invalidated, or cancelled. Validated and invalidated outcomes require a structured baseline value, observed value, shared unit, and observation period; Data Prism deterministically calculates the absolute and relative change. This creates an auditable evidence-to-outcome loop, but it does not infer causality or prove that the action caused the observed result.

An active decision case can download its review date as a private, all-day `.ics` reminder containing the user-entered commitment fields. The reminder is generated in memory, remains protected by the same signed guest-browser or pilot-account scope, and is never sent to a calendar provider by Data Prism.

The decision workspace also provides a scope-protected review queue for active, overdue, due-today, next-seven-day, closed, and all retained cases. Active work is ordered by review date so missed and imminent follow-ups appear before later commitments; this is an in-product queue, not an external notification service.

Each owned decision case can be downloaded as a standalone, print-ready HTML report containing its bounded evidence snapshot, pre-committed owner/metric/target, status, recorded outcome, and any validated structured measurement. The report is generated in memory with no scripts or external assets and excludes source dataset rows. Once downloaded, the copy is controlled by its holder and falls outside server retention and account deletion.

The decision workspace summarizes retained outcomes for the current guest-browser or pilot-account scope: tracking, validated, invalidated, and cancelled cases are shown with explicit denominators. It also shows that scope's retained, opted-in fixed-choice time-saved estimate and next-cycle intent. These signals are self-reported, limited to the 30-day measurement window, and never expose global pilot metrics or another scope's responses.

The same workspace can download a small, aggregate-only pilot evidence receipt for the current signed scope. The in-memory JSON contains the deployed service version, retention windows, decision-status counts, and retained fixed-choice value signals. It excludes scope/account/job/case identifiers, email, decision text, prompts, filenames, and dataset content. This gives a supervised pilot a portable snapshot before free-host state expires, but it does not verify identity, savings, causality, repeat use, adoption, or willingness to pay.

## Pilot onboarding and measurement

Optional VibeDash pilot accounts are available from **Create account**. The
account page (`/vibedash/account`) can change a password and download a bounded
account export. An authenticated user can generate eight high-entropy one-time
recovery codes after re-entering the current password. Only domain-separated
SHA-256 digests are stored; generating a set replaces the previous set, and a
successful recovery consumes the complete set, rotates the password, and
invalidates old signed-in browser credentials. Codes are displayed once and
must be stored independently. There is no email recovery, email verification,
or support bypass for a missing code. The account page also shows the newest 20
of at most 100 retained, allowlisted security events: account creation,
successful sign-in, password change, recovery-code replacement, and recovery.
These events contain no IP address, device/browser detail, email address,
session token, recovery code, or free-form metadata. The export contains the
newest 50 jobs, newest 100 decision cases, and the retained security events for
that account. It excludes raw datasets, prompts, filenames, schema/session
data, and global pilot metrics, and is serialized in memory without creating an
artifact. Account access is represented by
an id plus an opaque credential token inside Flask's signed client-side cookie.
The server validates that pair against the current password hash on every
authenticated request. A password change therefore invalidates copied old
VibeDash cookies, while ordinary cookie rotation cannot revoke a separately
copied cookie. This is a small pilot boundary, not enterprise authentication.
Free-host storage is ephemeral: account records and account-owned history may
disappear after a restart or redeploy, so keep independent copies of important
results.

The account page also provides a destructive deletion form. It requires the
current password, the account CSRF token, and an exact `DELETE` confirmation.
Deletion is refused while the account has queued or running jobs; retry after
those jobs finish. An eligible request purges the account-owned job and case
history, opt-in pilot measurements, and server artifacts that can be safely
identified with the account scope. It does not delete classic analysis or
monitoring data, local copies or downloads, or historical offline backups.
Account-store, job-store, and filesystem cleanup are separate boundaries, so
this is not a cross-store atomic guarantee; unidentifiable orphan files may
remain until normal retention cleanup. Free-host ephemerality can remove state
earlier than either workflow.

The workspace includes a **Weekly SaaS review** guide and prompt preset. Each run
offers an unchecked, optional usage-measurement checkbox. Consenting background
analyses record bounded lifecycle milestones and can submit fixed-choice feedback;
analysis still works without consent. Demo and upload cohorts are reported
separately, and the current guest-browser or pilot-account scope can delete its
pilot measurements without deleting its analyses or cases. This is first-party pseudonymous measurement, not proof of
business value or product-market fit.

Use `python pilot_report.py --database data/jobs/analysis_jobs.sqlite3 --days 7`
to read aggregate activity from the default local database. Local activity is
not Render activity. See [the pilot measurement contract](docs/PILOT_MEASUREMENT.md)
for storage and reporting, [the supervised pilot runbook](docs/PILOT_EXECUTION.md)
and [blank private scorecard template](docs/PILOT_SESSION_SCORECARD.md) for
observed sessions, and [release acceptance gates](docs/RELEASE_READINESS.md) for
remaining work.

## Quick start with Docker

```bash
git clone https://github.com/NikitaMarshchonok/data-prism.git
cd data-prism
cp .env.example .env
```

Replace both placeholder secrets in `.env`, then run:

```bash
docker build -t data-prism .
docker run --rm --env-file .env -p 5001:5001 data-prism
```

Open:

- Main analysis: `http://localhost:5001/`
- Prompt-to-dashboard: `http://localhost:5001/vibedash/`
- Recent analysis history: `http://localhost:5001/vibedash/history`
- Decision cases: `http://localhost:5001/vibedash/decisions`
- Liveness: `http://localhost:5001/healthz`
- Readiness: `http://localhost:5001/readyz`

The container runs Gunicorn as an unprivileged user. `/readyz` returns HTTP 503 if the persistent session key is missing or required runtime directories are not writable. A production profile additionally requires a verified runtime-state identity; the public demo does not claim this production assurance.

## Cloud deployment

The root `render.yaml` defines a free Docker web service with generated secrets, CI-gated deploys, structured logs, and `/readyz` health checks. After this repository is connected as a Render Blueprint, the built-in demo is available from the assigned public URL.

The free service filesystem is ephemeral. This is suitable for the portfolio demo, but drift history and uploaded artifacts require a paid persistent disk or future external storage. Follow [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for exact setup, verification, persistence, and rollback guidance.

VibeDash working copies, session files, and generated HTML exports use a configurable retention window. `VIBEDASH_RETENTION_HOURS` defaults to 24 hours and accepts values from 1 to 720. Expired, application-owned artifacts are removed when VibeDash receives a request; unrelated files and symbolic links are never removed by this cleanup.

Retained VibeDash session records are bound to the signed guest-browser or pilot-account analysis scope that created them. Export, chat-analysis, result, decision, and comparison-report loads fail closed when the owner is missing, malformed, or belongs to another scope. This is a deployment boundary: session JSON written by older releases without `analysis_scope_id` is intentionally unavailable after this hardening, so users must rerun those analyses.

The public single-instance deployment runs one bounded in-process analysis worker. Active work is limited per signed guest-browser or pilot-account scope and across the service; interrupted jobs are reported as failed rather than remaining indefinitely in `running` state.

Decision cases use the same local SQLite database and signed guest-browser or pilot-account scope. Closed cases are retained for 90 days by default, while active cases remain until they are closed. Configure this with `VIBEDASH_DECISION_RETENTION_DAYS` and `VIBEDASH_MAX_DECISION_CASES_PER_SCOPE`. A free Render restart or redeploy can remove them earlier because its filesystem is ephemeral.

## Offline runtime backup and recovery

The local `runtime_backup.py` CLI supports offline `backup`, `verify`, and
`restore` for a dedicated `DATA_PRISM_STATE_DIR`. Before invoking it, an
operator must stop the web service, workers, drift jobs, and every other
writer; `--offline` is only an acknowledgment and does not stop anything.
Snapshots contain sensitive state, are not encrypted, and SHA-256 checks
detect corruption but do not prove authenticity. Restore is fail-closed into
a new directory only; never start the application from a partial restore.
Preserve `FLASK_SECRET_KEY` and `DATA_PRISM_API_KEY` separately. See the
[recovery runbook](docs/BACKUP_RECOVERY.md) for retention, withdrawal, and
trusted-directory requirements. These tools do not provide managed persistent
storage, off-host copies, encryption, or automatic scheduling for the free
Render service.

For a dedicated persistent state directory, `runtime_state.py initialize`
creates a stable random identity exactly once. Store the returned 64-character
ID outside the disk as the secret `DATA_PRISM_EXPECTED_STATE_ID`. A production
profile reports ready only when the mounted directory contains that same valid
identity. The endpoint exposes only a short one-way fingerprint, never the ID.
Offline snapshots include the identity, so a rehearsed restore can prove that
the application opened the intended state lineage. This detects an empty,
wrong, or incomplete mount; it does not prove provider durability, backup
freshness, successful recovery, tamper resistance, or multi-instance safety.
The ID is a continuity value, not an authentication credential.

## Local development

### Development environment

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python web_app.py
```

`OPENAI_API_KEY` is optional. Without it, deterministic analytics and modelling remain available while the external AI summary reports that it is disabled.

## Monitoring API

The monitoring API is disabled until `DATA_PRISM_API_KEY` contains at least 32 characters.

Request an **unconfirmed schema proposal** first (no baseline or raw upload is retained):

```bash
curl -X POST http://localhost:5001/api/v1/drift/semantics \
  -H "X-API-Key: $DATA_PRISM_API_KEY" \
  -F "datafile=@reference.csv"
```

Save the response's `semantic_contract` object as `reference-semantics.json`.
Review each column's role (`measure`, `category`, `identifier`, `temporal`) and
unit (`number`, `count`, `currency`, `ratio`, `percent`, or `none` for non-measures),
then set `confirmed` to `true`. Do not confirm inferred meanings without review.
Create an aggregate baseline using that reviewed object:

```bash
curl -X POST http://localhost:5001/api/v1/drift/baselines \
  -H "X-API-Key: $DATA_PRISM_API_KEY" \
  -F "datafile=@reference.csv" \
  -F "semantic_contract=<reference-semantics.json"
```

Run an idempotent drift check using the returned `baseline_id`:

```bash
curl -X POST http://localhost:5001/api/v1/drift/checks \
  -H "X-API-Key: $DATA_PRISM_API_KEY" \
  -H "Idempotency-Key: batch-2026-09-07" \
  -F "baseline_id=$BASELINE_ID" \
  -F "datafile=@current.csv"
```

Read monitoring state:

```bash
curl http://localhost:5001/api/v1/drift/runs \
  -H "X-API-Key: $DATA_PRISM_API_KEY"

curl http://localhost:5001/api/v1/drift/alerts \
  -H "X-API-Key: $DATA_PRISM_API_KEY"
```

Bearer authentication is also supported. Identical checks are deduplicated using
an `Idempotency-Key` or the uploaded content plus semantic context hash (both
scoped to the baseline ID). Use a new explicit key for a changed batch or meaning.

New API baselines require confirmation and retain profile v3 with the normalized
contract and semantic fingerprint. Checks inherit the fixed baseline definitions;
the report labels this `inherited_from_baseline`, **not** fresh confirmation of
the new batch. Optionally submit `-F "semantic_contract=<current-semantics.json"`
to confirm the current batch explicitly. Shared roles and units must match;
incompatible numeric/count values fail before history is written. Missing/new
columns still produce schema diagnostics. No currency or scale conversion is
performed: upstream definition/scale changes require operator review and a new
baseline. The generic `currency` unit does not identify or verify a currency code.

## Automated drift jobs

Generate a proposal, review its roles/units/flags, and set `confirmed` to `true`:

```bash
python monitor_drift.py propose-semantics \
  --data data/reference.csv \
  --output reference-semantics.json
```

The proposal command refuses to overwrite an existing file. Then create a
reusable aggregate baseline (up to 100 columns; source rows are not stored):

```bash
python monitor_drift.py create-baseline \
  --data data/reference.csv \
  --semantic-contract reference-semantics.json \
  --output data/baselines/reference.json
```

Copy `monitoring_job.example.json`, update its paths, and run:

```bash
python monitor_drift.py run \
  --config monitoring_job.json \
  --batch-id batch-2026-09-07
```

The command emits one JSON document. Exit codes are stable:

- `0`: check completed and the configured threshold was not reached;
- `1`: configuration, input, or execution error;
- `2`: the configured warning or critical threshold was reached.

An optional `current_semantic_contract` config field points to a reviewed JSON
contract for the current batch; paths resolve relative to the job config. Without
it, checks explicitly report inherited baseline definitions. Automatic batch IDs
include the current content hash, baseline revision, and semantic context so a
new baseline does not reuse a stale run. Explicit `--batch-id` retains its
idempotency semantics; change it when the intended check changes.

Legacy v1/v2 baselines remain readable and reports label them
`legacy_without_retained_contract` rather than asserting confirmed provenance.
Recreate them with reviewed semantics to adopt v3; they are never silently
upgraded. The classic web monitoring workflow still creates inferred v2 profiles.
These are screening checks, not certification of business definitions or causality.

## Tests and continuous integration

```bash
python -m unittest discover -s tests -p "test_*.py"
python evaluate_quality.py --config evaluation/quality_gate.json
python test_vibedash.py
```

GitHub Actions runs the full suite on Python 3.11 and 3.12 for every pull request to `main`. The analytical gate checks a known SaaS scenario (including predictive holdout support), a null-noise false-discovery and predictive-support guardrail, and a designed group effect. It emits a machine-readable report that CI preserves as an artifact, including the observed and expected value for every check. See [docs/QUALITY_EVALUATION.md](docs/QUALITY_EVALUATION.md) for the benchmark contract and threshold policy.

The wider suite covers data loading, security boundaries, evidence generation, statistical validation, model evaluation, reliability diagnostics, drift persistence, API behaviour, CLI jobs, and Flask integration.

## Repository structure

```text
data-prism/
├── web_app.py                 # Flask composition root and interactive workflow
├── monitor_drift.py           # Scheduled/CI drift command
├── evaluate_quality.py        # Analytical regression-gate command
├── evaluation/
│   └── quality_gate.json      # Reviewed behavioural acceptance thresholds
├── src/
│   ├── column_semantics.py    # Shared analytical role contract
│   ├── semantic_contract.py   # Confirmed roles, units, exclusions, and fingerprint
│   ├── data_loader.py         # Validated tabular ingestion
│   ├── demo_data.py           # Deterministic, privacy-safe product demo
│   ├── data_analyzer.py       # Profiling and data-quality checks
│   ├── dashboard_generator.py # Dashboard orchestration
│   ├── ml_predictor.py        # Leakage-safe model selection and evaluation
│   ├── model_reliability.py   # Stability and subgroup diagnostics
│   ├── quality_evaluation.py  # Synthetic benchmark scenarios and checks
│   ├── statistical_primitives.py # Shared validated statistical formulas
│   ├── data_drift.py          # Aggregate baseline and drift algorithms
│   ├── drift_store.py         # SQLite history and alert persistence
│   ├── monitoring_api.py      # Authenticated monitoring endpoints
│   ├── runtime_state.py       # Mounted-state continuity identity
│   └── runtime_backup.py      # Offline bounded backup and restore
├── vibedash/                  # Prompt-to-dashboard and evidence engines
│   ├── analysis_jobs.py       # Durable job states, scoped history, bounded dispatcher
│   ├── decision_cases.py      # Evidence-to-outcome cases and bounded retention
│   ├── readiness_engine.py    # Pre-analysis quality and privacy contracts
│   ├── decision_brief.py      # Ranked evidence-to-action priorities
│   └── audit_manifest.py      # Versioned fingerprints and reproducibility metadata
├── templates/                 # Flask/Jinja interfaces and reports
├── tests/                     # Unit and integration tests
├── .github/workflows/ci.yml   # Python 3.11/3.12 CI matrix
├── render.yaml                # CI-gated Render deployment Blueprint
└── Dockerfile                 # Non-root Gunicorn runtime
```

## Data and security boundaries

- Uploaded datasets, generated reports, local baselines, and SQLite history are runtime artifacts and are excluded from version control.
- VibeDash uploads, sessions, and exports are server-named and subject to the configured temporary-artifact retention window.
- Analysis-job status is isolated by a server-signed guest browser or pilot-account scope; job payloads and internal exceptions are not returned by the status API.
- Run history, stored results, and audit-manifest downloads require the same signed scope that created the analysis.
- Decision cases are isolated by that scope and store bounded evidence snapshots rather than source dataset rows.
- Audit manifests include schema metadata and cryptographic fingerprints but never source row values.
- Monitoring API keys are compared using constant-time comparison and are not used directly as storage identifiers.
- API drift uploads are transient; persisted baselines contain aggregate profiles rather than raw rows.
- Server-generated identifiers and filenames are validated before resolving filesystem paths.
- LLM output is supplementary. Core evidence, statistics, model metrics, and drift status are calculated locally.

## Current limitations

This is an actively developed portfolio system, not a managed enterprise platform.

- Classic analysis is synchronous; VibeDash uses a bounded in-process worker intended for the documented single-instance topology.
- Runtime state uses the local filesystem and SQLite rather than managed object storage and a distributed database.
- `/readyz` reports an explicit runtime-storage contract: the public Render Blueprint is `demo`/`ephemeral`, while a `production` profile fails closed unless persistent state, an explicit state directory, and a matching externally retained state identity are configured. Identity continuity detects a missing or wrong mount; it does not verify provider durability or satisfy the recovery gate.
- Guest decision cases are browser-scoped; signed-in pilot accounts share their account scope across browsers, while clearing a session cookie removes only that browser's identity. Free-host restarts can remove the records.
- VibeDash pilot accounts are optional. An account gives the same signed 32-hex analysis scope across browsers, while guest analyses remain browser-scoped and are never migrated into an account. Free-host storage may reset; this is not enterprise authentication or a durable-account guarantee.
- Preserve `FLASK_SECRET_KEY`: it signs the session and derives account scopes. Rotating it requires sign-in again and makes prior account-owned history unavailable under the new scope.
- Measured outcomes are user-entered observations. They support learning and accountability but do not establish that a decision caused the result.
- Temporary-artifact cleanup is request-triggered, so it is not a wall-clock deletion SLA; strict retention guarantees require a scheduler or storage-provider lifecycle policy.
- Account deletion is scoped to account-owned state and identifiable server artifacts; it does not erase user-held copies or historical offline backups, and it does not promise cross-store atomicity or complete removal of unidentifiable orphan files.
- Predictive models are diagnostic baselines. Holdout support means the fitted pipeline beat its naive baseline under the documented permutation gate; it is not proof of temporal validity, causality, fairness, or production readiness.
- Statistical findings are observational and must not be interpreted as causal conclusions.
- External alert delivery, managed scheduling, access-control roles, and production telemetry are not yet implemented.

## Roadmap

- Run 5–10 observed SaaS decision pilots and measure repeat use and willingness to pay
- Managed scheduling and external alert delivery
- External queue and independently scalable background workers
- Object-storage and PostgreSQL adapters
- Role-based access control and audit events
