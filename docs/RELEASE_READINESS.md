# Release readiness: evidence, scope, and pilot controls

As of **2026-09-23**, the public Data Prism service is a portfolio demo and an
invite-only, supervised pilot candidate. It is **not** a durable production SaaS,
an enterprise analytics environment, or proof of product-market fit.

This document separates four different claims that must not be conflated:

1. code and analytical checks pass;
2. the expected revision is deployed and healthy;
3. a supervised participant can complete the intended workflow;
4. repeated real use demonstrates enough value to justify adoption.

Only the first two are confirmed for the current revision. Historical synthetic
workflow evidence supports the third technically, but the current revision still
needs an observed public-pilot pass. The fourth requires real participants and
cannot be established by tests, commits, page views, or self-reported intent.

## Current confirmed baseline

| Evidence | Confirmed result |
| --- | --- |
| Git revision | `2393ec63b0e9976c5583773a4743afb9a91730ee` — merge of PR #33 |
| Main CI | Run `35790946234` passed on Python 3.11, Python 3.12, analytical quality, VibeDash smoke, and the production-container smoke test |
| GitHub deployment | Deployment `6601839291` succeeded for the same revision |
| Public health | `https://data-prism.onrender.com/healthz` returned `status: ok` and version `2393ec63b0e9` |
| Local pre-merge evidence | 354 tests, analytical quality gate 13/13, VibeDash smoke 3/3 |

The current merged scope includes evidence-backed analysis, statistical and
model guardrails, data readiness, anomaly/segment exploration, period
comparison, background jobs, scoped history, decision cases and outcomes,
optional pseudonymous pilot measurement, fixed-choice value feedback, optional
pilot accounts, password rotation, bounded account export, guarded account
deletion, observability, drift tooling, and offline backup/restore tooling.

The latest full public synthetic journey was recorded for the earlier PR #24
baseline. Later changes have unit/integration coverage, CI container coverage,
successful deployments, and matching health revisions, but have not all been
replayed as one public browser journey on PR #33. Do not describe the historical
PR #24 journey as a current real-user validation.

## What the current deployment is suitable for

The free Render deployment may be used for:

- a public portfolio demonstration with the built-in synthetic dataset;
- an invite-only, supervised pilot using approved synthetic or de-identified
  tabular data;
- usability observation of the evidence → decision → outcome workflow;
- collecting optional fixed-choice pilot feedback after explicit opt-in;
- testing whether a participant returns for a second reporting cycle.

It must not be presented as suitable for:

- confidential, personal, regulated, or business-critical datasets;
- unattended production decisions or causal claims;
- durable monitoring, records, accounts, or recovery guarantees;
- multi-user teams, roles, approvals, SSO, billing, or contractual SLAs;
- automatic deployment of predictive models;
- claims of savings, adoption, willingness to pay, or product-market fit.

The free Render filesystem is ephemeral and the service can cold-start slowly.
Accounts, job history, decisions, pilot measurements, uploads, and generated
artifacts can disappear after restart or redeployment. Offline backup tooling
does not change that hosting boundary and is not an active backup service.

## Prototype acceptance gates — 10/10 (100%)

These fixed technical gates describe the narrow prototype, not commercial or
production maturity.

| Gate | Status | Evidence |
| --- | --- | --- |
| Reproducible synthetic demo | Verified | Seeded demo and regression tests |
| CSV readiness validation and actionable errors | Verified | Readiness engine and route tests |
| Calculated evidence, statistical guardrails, and audit manifest | Verified | Unit tests and analytical quality gate |
| Bounded background jobs and failure recovery | Verified | Lifecycle, capacity, timeout, and route tests |
| Scoped history and retained results | Verified | Guest/account authorization regression tests |
| Evidence → decision → outcome workflow | Verified | Decision-case and outcome tests |
| Honest focused onboarding and limitations | Verified | UI contracts and documentation |
| Optional bounded measurement and withdrawal | Verified | Pilot store, route, retention, and withdrawal tests |
| Aggregate reporting with separated demo/upload cohorts | Verified | Read-only report and migration tests |
| CI-gated container deployment and health verification | Verified | Main CI, GitHub deployment, and matching `/healthz` revision |

## First-release acceptance gates — 6/10 (60%)

This denominator remains fixed. A new feature, test, PR, or documentation update
does not by itself satisfy a release gate.

| Gate | Status | Acceptance evidence still required |
| --- | --- | --- |
| Narrow user, problem, and first-use workflow defined | Complete | Weekly SaaS review remains a hypothesis to validate |
| Core end-to-end workflow reproducible | Complete | Automated tests plus historical public synthetic journey |
| Automated analytical/regression controls | Complete | Unit suite, 13-check analytical gate, CI matrix |
| Deployment, health, and diagnostic logging documented | Complete | Render runbook, health endpoints, request correlation |
| Opt-in pilot measurement with stated limits | Complete | Retention, withdrawal, cohort separation, value-feedback migration |
| Release candidate passes CI and technical deployment acceptance | Complete | PR #33 main CI, deployment record, matching health revision |
| Durable state and tested real recovery | Pending | Configure durable storage and complete a timed recovery rehearsal with retained secrets and deletion handling |
| Recoverable identity and full data lifecycle | Partial | Account login/password/export/deletion exist; email recovery, durable hosting, and backup-erasure operations do not |
| Observed real-user usefulness and repeat use | Pending | 5–10 relevant pilots across at least two reporting cycles |
| Release operations and user-facing commitments reviewed | Pending | Rehearse the checklist below; define support, incident, privacy, retention, and cost ownership |

## Go/no-go checklist for every supervised pilot

### Before the session

- [ ] Confirm the participant matches the narrow use case: a recurring SaaS
  metrics decision based on a CSV export.
- [ ] Record the existing workflow, approximate duration, decision to be made,
  and what improvement would justify adoption. Keep interview notes and contact
  details outside this repository and outside pilot telemetry.
- [ ] Obtain permission for the specific data used. Prefer the built-in synthetic
  dataset first; otherwise use an approved de-identified CSV with no secrets,
  personal data, regulated data, or customer identifiers.
- [ ] Open `/healthz` and `/readyz`; confirm HTTP 200 and that the reported
  revision equals the intended deployment.
- [ ] Run the built-in demo once. Stop if upload, job completion, evidence,
  result rendering, or scoped access fails.
- [ ] Explain the ephemeral-host boundary, 24-hour default analysis retention,
  optional measurement, account limitations, and absence of an SLA.
- [ ] Decide who owns session support and who can stop the pilot if an incident
  occurs.

### During the session

- [ ] Let the participant attempt the workflow before coaching; record assistance
  separately from product success.
- [ ] Check that the participant can explain one finding, its evidence, and its
  limitation rather than merely viewing a chart.
- [ ] Ask the participant to create a decision case with an owner, success
  metric, target, and review date only when those fields reflect a real intended
  follow-up.
- [ ] Offer pilot measurement as an unchecked, optional choice. Do not treat a
  missing response as a negative result.
- [ ] Do not copy uploaded rows, prompts, contact details, interview transcripts,
  or private outcome notes into repository issues or public logs.

### After the session and next cycle

- [ ] Record whether the participant completed the workflow, needed assistance,
  understood the evidence, and identified a concrete next action.
- [ ] Treat perceived time saved and next-cycle intent as self-reports only.
- [ ] On the agreed review date, verify whether the participant returned, used
  the evidence, recorded an observed outcome, and would connect a recurring data
  source or make a concrete adoption commitment.
- [ ] Separate built-in demos and operator tests from real participant counts.
- [ ] Withdraw the participant's pilot measurement when requested. Explain that
  account deletion does not erase local downloads or historical offline backups.

## Stop conditions

Pause new pilot sessions and investigate before continuing if any of the
following occurs:

- one guest or account can access another scope's job, result, case, export, or
  account data;
- raw rows, prompts, filenames, credentials, session identifiers, or private
  decision text appear in application logs or aggregate reports;
- the service reports a revision different from the intended release;
- an analysis presents unsupported or materially misleading evidence without an
  explicit limitation;
- repeated jobs remain queued/running, resource limits fail closed incorrectly,
  or uploaded artifacts survive outside the documented retention boundary;
- account deletion reports success while scoped live data remains accessible;
- the participant supplies data outside the approved pilot boundary.

Document the incident privately, preserve only the minimum safe diagnostic
evidence, identify affected data/scope, and do not resume until the failure has a
verified fix or the pilot boundary is narrowed.

## Decision after two reporting cycles

Continue the narrow product experiment only if multiple relevant participants
independently complete a real decision workflow and some return for the next
cycle with evidence of useful follow-up. Investigate before adding broad features
when participants finish a demo but do not return, cannot explain the evidence,
or do not make a concrete decision.

Five to ten supervised pilots can guide iteration; they cannot estimate global
demand. Promotion beyond a supervised pilot additionally requires durable state,
a real recovery rehearsal, reviewed privacy/support/incident commitments, and a
separate decision on paid infrastructure. Passing every technical check still
does not prove causal business impact or product-market fit.
