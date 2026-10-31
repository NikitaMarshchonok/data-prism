# Industry-independent task intake

`vibedash-task-intake-v1` is a capability/admission boundary for confirmed
VibeDash workflows. It does not inspect source rows, understand business intent
from free text, approve data, or establish professional equivalence. Industry
does not choose an executor; the requested task and its validated assumptions do.

## Read-only discovery

`GET /vibedash/task-capabilities` returns the public, static catalogue:

- `descriptive_analysis`: existing confirmed evidence workflow;
- `daily_forecast`: one complete daily series, past target lags, explicit
  availability and decision criteria, temporal evaluation;
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

## Compatibility and remaining work

Existing clients without `task_type` retain the meaning of `analysis_mode`
(`evidence` by default). Their routing source is `legacy_analysis_mode`, never
`inferred_business_intent`. Existing stored jobs without a routing record remain
readable. Ordinary HTML form clients are not automatically migrated to a new
business-intent dialog by this change.

This boundary does not guarantee that arbitrary free-text requests match the
selected task. A future shared intake should propose a task interpretation,
ask essential questions about observation unit, business definitions and
deliverable, and obtain confirmation. Only then should it admit a supported
executor. Do not advertise that future dialog as implemented.

Acceptance tests cover static discovery without training/storage, no silent
substitution, mode conflicts, preservation of legacy modes, forecast CSRF and
decision-policy requirements, and worker refusal of inconsistent routing.
