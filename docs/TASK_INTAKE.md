# Industry-independent task intake

`vibedash-task-intake-v1` is a capability/admission boundary for confirmed
VibeDash workflows. It does not inspect source rows, understand business intent
from free text, approve data, or establish professional equivalence. Industry
does not choose an executor; the requested task and its validated assumptions do.

## Read-only discovery

`GET /vibedash/task-capabilities` returns the public, static catalogue:

- `descriptive_analysis`: existing confirmed evidence workflow;
- `daily_forecast`: one complete daily series, past target lags, explicit
  fixed target availability delay and decision criteria, temporal evaluation;
- `period_comparison`: the existing separate two-CSV comparison workflow;
- `classification`, `regression`, `causal_analysis`: no confirmed standalone
  VibeDash executor in this intake;
- `segmentation`, `anomaly_detection`: embedded exploratory findings, not
  standalone validated task executors or incident/fraud confirmation.

Internal legacy Python predictive functions are not certified or enabled by
this catalogue. Supported high-impact uses, security or regulatory compliance
must not be inferred from `industry_independent: true`.

`GET /vibedash/task-intake?task_type=daily_forecast&workflow=analysis_jobs`
returns static guidance. Both endpoints return HTTP 200 for readable guidance,
do not create jobs or working copies, and do not fit models. The assessment can
be `needs_clarification`, `not_supported`, `unsupported_workflow`,
`workflow_mismatch` or `requires_dataset_review`. Every response keeps
`ready_for_execution: false`: a known task still needs actual validation.

The catalogue/assessment is not permission for a client to bypass upload,
semantic review, CSRF, ownership, capacity or retention controls.

## Explicit submission

Add a `task_type` form field to `POST /vibedash/jobs` alongside the existing
CSV, analysis question, reviewed semantic contract and task-specific settings.
`descriptive_analysis` selects evidence mode; `daily_forecast` selects forecast
mode. If `analysis_mode` is also supplied, it must agree. An explicit forecast
requires CSRF, confirmed daily availability and the reviewed decision policy
described in [FORECAST_EVALUATION.md](FORECAST_EVALUATION.md).

Unknown, empty, unsupported or conflicting explicit types receive HTTP 422 with
`error` and structured `task_intake` guidance before upload storage or CSV
processing. No alternative task is selected. A period comparison is directed
to its own existing workflow, never converted to one-file evidence analysis.
The synchronous preview endpoint also refuses unsupported explicit types and
directs forecasting to the queued workflow.

New queued jobs retain a bounded `task_routing` record: contract, task type,
workflow and whether the type was explicit or derived from legacy mode. The
worker rechecks it against the execution kind before processing the source.
An explicit forecast also retains its confirmed decision policy; the existing
forecast adapter validates its fingerprint and binding before training.

## Confirmed task brief — structured clarification

The main upload form now asks for the expected result and what one row
represents, alongside the existing question and column review. Changing the
question, row meaning, selected task or definitions invalidates confirmation.
Forecast decision criteria and availability remain separate required checks.

`POST /vibedash/task-brief` accepts JSON metadata only, at most 16 KiB, protected
by the page's `X-CSRF-Token`. Fields are `task_type`, `question` (1–4000
characters), `observation_unit` (1–240), `deliverable` and boolean `confirmed`.
The supported deliverables are `evidence_summary` for descriptive analysis and
`forecast_evaluation` for daily forecasting. Duplicate/unknown fields, invalid
text and non-finite JSON are rejected. No uploaded file, retained brief, job,
source processing or model training is created by this endpoint. Responses are
not cached and do not echo the question or row description.

Missing context produces field-specific questions with `needs_clarification`.
A complete confirmed brief returns `ready_for_dataset_review`, never
`ready_for_execution: true`. This deterministic check tests completeness and
the explicitly chosen supported result. It does not interpret arbitrary prose,
verify the truth of the supplied row meaning, discover business requirements,
or detect a contradiction between prose and the selected task.

The browser reviews the brief before queueing and refuses a stale review if the
file or form changes while awaiting it. Submission to `/vibedash/jobs` includes
`task_observation_unit`, `task_deliverable` and `task_brief_confirmed=yes`.
The server independently validates these fields; partial briefs fail before
creating an application working copy. Both review and brief submission require
CSRF protection. After validating column meanings, it binds the complete brief
to their fingerprint with `confirmed-task-brief-v1`. The worker rechecks the
question, task and semantic fingerprint before specification generation or
training. A hash detects inconsistent records; it is not a secret signature or
an independent validation of business meaning.

The owner-scoped analysis and HTML result retain the confirmed question and row
description for the existing retention window. The aggregate audit manifest
contains fixed metadata and the brief fingerprint, not either free-text field.
The result labels context as user-confirmed, not independently validated.

## Compatibility and remaining work

Existing clients without `task_type` retain the meaning of `analysis_mode`
(`evidence` by default). Their routing source is `legacy_analysis_mode`, never
`inferred_business_intent`. Existing stored jobs without a routing record remain
readable. Clients that omit all three task-brief fields retain their existing
path without claiming a reviewed business brief. The new browser form uses
the queued workflow; synchronous preview refuses task-brief fields rather than
silently discarding them.

This boundary does not guarantee that arbitrary free-text requests match the
selected task. Structured questions about observation unit and deliverable are
implemented, but safe conversational interpretation, substantive business
definition review and an adaptive clarification dialogue remain future work.
Do not advertise those capabilities as implemented.

Acceptance tests cover static discovery without training/storage, no silent
substitution, mode conflicts, preservation of legacy modes, forecast CSRF and
decision-policy requirements, bounded brief review, no retained source or
training during clarification, partial-brief refusal, worker binding checks,
escaped result rendering and omission of free text from the audit manifest.
