# Release readiness: evidence, not a completion promise

As of 2026-09-13. This records the confirmed baseline and the status of the
uncommitted runtime-backup milestone. It is not a claim that the new dirty
working tree is production-ready or that the product is 100% ready.

## Confirmed PR24 baseline

PR24 (`pilot-onboarding-metrics`) is merged at
`050fa75d22f70c8a9a7d6aa50ab6bee219bea040`. Confirmed checks for that revision
include Python 3.11, Python 3.12, the production container, and GitGuardian.
The main CI run `34759292420` passed. The GitHub deployment record for the
Render deployment (`6421961369`) succeeded at `2026-09-13T13:18:27Z`, and
`/healthz` reported version `050fa75d22f7`.

The public synthetic HTTP pass on that version covered: opted-in background
demo; evidence/result; saved feedback; case creation; a cancelled technical
outcome; persistence after refresh; another-browser `404`; and withdrawal
metrics limited to the synthetic scope. The feedback form disappeared while
the analysis and case remained readable. This is a technical synthetic check,
not a real pilot or business benefit claim.

Local PR24 evidence was 168 tests, analytical gate 13/13, and VibeDash smoke
3/3. These figures describe the confirmed PR24 baseline, not the uncommitted
backup changes.

The narrow prototype baseline is **10/10 gates (100%)**. The original
first-release checklist is **6/10 (60%)**, with CI/production acceptance for
PR24 complete. Neither percentage means that durable production state, identity,
real-user usefulness, or the full product is complete. Do not increase a
percentage for a tool, a commit, or a documentation change.

## Prototype acceptance gates — 10/10 (100%)

These are equally weighted gates with a fixed denominator; the percentage is
not a claim about product maturity or production durability.

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
| CI, merge, deployment, and public acceptance for PR24 | Verified | CI/deployment records and public synthetic check |

## First-release acceptance gates — 6/10 (60%)

| Gate | Status | Acceptance |
| --- | --- | --- |
| Narrow user, problem, and first-use workflow defined | Implemented | Weekly SaaS review, documented as a hypothesis |
| Core end-to-end workflow reproducible | Verified | Tests plus public synthetic check |
| Automated analytical/regression quality controls exist | Verified | Unit suite, quality gate, CI configuration |
| Documented deployment, health checks, and diagnostic logs | Implemented | Current Render demo; deployment runbook |
| Opt-in pilot measurement with stated limits | Implemented locally | PR24 tests and measurement contract |
| Release candidate passes CI and production acceptance | Verified for PR24 | Latest merged PR24 revision; backup milestone remains pending |
| Durable state and tested backup/restore | Pending | Requires configured durable storage and real recovery rehearsal |
| Recoverable identity and data-access/deletion lifecycle | Pending | Account access and recovery; browser scope is not sufficient |
| Observed real-user usefulness and repeat use | Pending | 5–10 relevant pilots, two reporting cycles, documented findings |
| Release operations and user-facing commitments reviewed | Pending | Security/privacy review, cost limits, support, incident handling, onboarding acceptance |

## Runtime-backup milestone

The backup branch currently contains uncommitted changes. Local synthetic
verification on 2026-09-13 passed **198/198 tests**; the runtime-backup subset
passed 30/30. This is a local verification record, not CI or deployment
acceptance: the backup changes remain unverified on CI, the deployed host, and
in a real recovery rehearsal. This milestone is separate from the confirmed
PR24 baseline.

The durable-state and real-recovery gate remains **pending**: the current work
is synthetic local tooling only. A successful local backup command does not
demonstrate restart/redeploy durability, managed persistent storage, or a real
recovery rehearsal.

The public deployment is a portfolio/demo baseline. Passing the listed checks
supports a release decision for the defined scope; it does not prove
product-market fit or business value.
