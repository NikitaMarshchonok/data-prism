# Release readiness: evidence, not a completion promise

As of 2026-09-13. Scope: a narrow SaaS metrics decision-support product, **not an
autonomous replacement for a mid-level data scientist or an enterprise platform**.

Percentages below count equally weighted acceptance gates. They are not estimates
of time, remaining code, market demand, or probability of success. These explicit
gates replace earlier conversational percentages that did not have a fixed rubric.
Do not raise a percentage for a commit alone; attach verification evidence.

## Working prototype — 9/10 gates (90%)

| Gate | Status | Verification |
| --- | --- | --- |
| Reproducible synthetic demo | Verified | Demo tests and public synthetic workflow |
| CSV readiness validation and actionable errors | Verified | Readiness engine and route regression tests |
| Calculated evidence, statistical guardrails, and audit manifest | Verified | Unit tests and analytical quality gate |
| Bounded background jobs and failure recovery | Verified | Lifecycle, capacity, and route tests |
| History with browser-scope isolation | Verified | History tests; public other-browser check |
| Evidence → case → outcome → refresh | Verified | Decision tests and public synthetic check |
| Focused onboarding and honest public-pilot limitations | Verified locally | Landing-page tests and UI review |
| Optional, idempotent measurement and withdrawal | Verified locally | Pilot store and route tests |
| Read-only aggregate reporting with separate demo cohort | Verified locally | Report and CLI tests |
| This milestone's CI, merge, deployment, and public acceptance | Pending | User publishes branch; verify resulting deployed revision |

Public verification for the preceding revision `283f41a841e7` completed on
2026-09-13: health, synthetic background demo, decision creation, a written
cancellation outcome, persistence after refresh, and HTTP 404 for another browser.
The case was explicitly labelled a technical check, not a business outcome.
This does **not** validate the unmerged measurement feature on Render.

Local milestone verification: Python 3.12 ran 168 unit/integration tests; the
analytical gate passed 13/13 checks and VibeDash smoke passed 3/3. Browser review
covered the prompt preset, opt-in demo, saved feedback, and 390px mobile layout.
A separate local HTTP check exercised demo → case → recorded cancellation →
refresh → other-browser denial. The aggregate report separated those synthetic
demo runs from uploads. Python 3.11 and the production container await this PR's
CI; the local `.venv` lacks Flask and was not used for the full suite.

## Official first product release — 5/10 gates (50%)

| Gate | Status | Acceptance |
| --- | --- | --- |
| Narrow user, problem, and first-use workflow defined | Implemented | Weekly SaaS review, documented as a hypothesis |
| Core end-to-end workflow reproducible | Verified | Tests plus public synthetic check |
| Automated analytical/regression quality controls exist | Verified | Unit suite, quality gate, CI configuration |
| Documented deployment, health checks, and diagnostic logs | Implemented | Current Render demo; deployment runbook |
| Opt-in pilot measurement with stated limits | Implemented locally | This milestone's tests and measurement contract |
| Release candidate passes CI and production acceptance | Pending | Latest revision, including enrollment/feedback/withdrawal |
| Durable state and tested backup/restore | Pending | Survive restart/redeploy and rehearse recovery |
| Recoverable identity and data-access/deletion lifecycle | Pending | Account access and recovery; browser scope is not sufficient |
| Observed real-user usefulness and repeat use | Pending | 5–10 relevant pilots, two reporting cycles, documented findings |
| Release operations and user-facing commitments reviewed | Pending | Security/privacy review, cost limits, support, incident handling, onboarding acceptance |

A public portfolio URL is already available. That is different from launching a
reliable product with real customers. Passing all gates permits a release decision,
not a claim of product-market fit. Update this checklist if scope changes and state
that the denominator changed; do not compare revised percentages as if work had
been added or lost.
