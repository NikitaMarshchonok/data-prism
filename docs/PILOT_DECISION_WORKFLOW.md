# Pilot decision workflow

This document defines the narrow product experiment behind Data Prism's
evidence-to-outcome workflow. It is an implemented pilot contract, not a claim
of product-market fit.

## User problem

The target user has a recurring business question and a table, but lacks a
fast, defensible path from analysis to action. A dashboard alone does not say
who owns the next step, what result would count as success, or whether the
recommendation ultimately helped.

Data Prism tests one proposition: a bounded workflow that connects calculated
evidence to a pre-committed decision and later outcome is more useful than an
analysis that ends with charts or narrative recommendations.

## Case contract

A case can be created only from a completed, session-owned background analysis
and one of its deterministic Decision Brief priorities. At creation time it
records:

- a versioned, bounded snapshot of the selected finding and supporting metrics;
- dataset and analysis fingerprints when present in the audit manifest;
- the accountable owner and the decision to test;
- one success metric, target outcome, and review date.

The evidence snapshot is immutable. The observed outcome and case status may be
updated later. Supported states are:

| State | Meaning |
| --- | --- |
| `tracking` | The decision is active or the result is not final |
| `validated` | The user reports that the pre-defined target was met |
| `invalidated` | The user reports that the target was not met |
| `cancelled` | The decision was not completed or can no longer be evaluated |

A terminal state requires a written actual outcome. Reopening a case to
`tracking` clears its resolution timestamp but preserves the outcome note.

## Privacy and persistence boundary

The application does not copy source dataset rows into a decision case. It
stores only bounded evidence text, audit identifiers, and fields entered by the
user. Users should not put personal, secret, or regulated information into the
owner, decision, target, or outcome fields.

Access is isolated by a random identifier in the server-signed browser session.
This is not user authentication. Clearing the cookie loses access to the case.
The local SQLite store is suitable for a single-instance pilot. On a free Render
instance it is ephemeral and may disappear on restart or redeploy.

Closed cases are removed after `VIBEDASH_DECISION_RETENTION_DAYS` (90 by
default) when a later VibeDash request triggers cleanup. Active cases remain.
The number of cases per browser scope is bounded by
`VIBEDASH_MAX_DECISION_CASES_PER_SCOPE` (50 by default).

The source analysis can expire before the decision case. The immutable evidence
snapshot remains readable, while its source-analysis link can return an expired
result response.

## Pilot success criteria

The next validation step is not another broad feature. Run 5–10 observed pilots
with teams that make a recurring SaaS decision from CSV exports. For each pilot,
measure:

1. time from upload to a decision case with an owner and target;
2. percentage of analyses that produce a case rather than ending at a dashboard;
3. percentage of cases reviewed by their review date;
4. percentage of outcomes with an observed metric and period;
5. whether the user would repeat the workflow on the next reporting cycle;
6. whether the user would connect a real recurring data source or pay for team persistence.

Evidence of value requires repeated use and willingness to adopt or pay. Demo
completion, page views, and positive comments are not sufficient evidence of
product-market fit.

## Not yet supported

- accounts, teams, roles, approvals, or shared ownership;
- notifications and scheduled review reminders;
- managed persistent storage or multi-instance consistency;
- automatic ingestion from business systems;
- causal attribution between the decision and observed outcome;
- portfolio-level reporting across teams or organizations.
