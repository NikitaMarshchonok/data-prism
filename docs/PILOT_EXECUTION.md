# Supervised pilot execution

This runbook tests one narrow product hypothesis; it is not a launch plan or a
claim of product-market fit.

> **Hypothesis:** a SaaS operations or growth lead who reviews recurring metrics
> from a CSV can reach a traceable decision and follow-up faster, with clearer
> evidence and limitations, than in the person's current workflow.

Run 5–10 supervised pilots across at least two reporting cycles. Keep contact
details, recruitment records, interview notes, filled scorecards, downloaded
receipts, and participant data outside this public repository.

## Eligible participant

Include a participant only when all are true:

- they personally review a recurring SaaS metrics table or closely comparable
  tabular business report;
- the review leads to a real decision, investigation, or prioritization;
- they can return for the next natural reporting cycle;
- the first session can use the built-in demo or an approved de-identified CSV;
- they understand the free-host persistence and support boundaries.

Do not count portfolio viewers, friends clicking through a demo, the product
operator, automated tests, or people without a recurring decision workflow as
eligible pilots. Their feedback may still reveal usability issues, but it is a
different evidence class.

## Evidence hierarchy

Interpret evidence from weakest to strongest:

1. page view or positive comment;
2. completed synthetic demo;
3. completed own-data workflow with assistance;
4. concrete decision case with an owner, target, and review date;
5. return at the agreed next cycle without another product tour;
6. recorded outcome with a comparable baseline, observed value, and period;
7. concrete adoption step: recurring data access, team introduction, pilot
   extension, or acceptance of a specific paid offer.

The first three are learning signals. They are not adoption. Self-reported time
saved is not independently measured savings, and an observed outcome does not
prove that the decision caused the change.

## Before recruitment

1. Verify the intended Git revision in `/healthz` and HTTP 200 from `/readyz`.
2. Complete the built-in demo once on the public deployment.
3. Review the stop conditions in [release readiness](RELEASE_READINESS.md).
4. Prepare a private participant register outside the repository. Assign each
   person a non-identifying operator reference such as `P-01`.
5. Copy [the scorecard template](PILOT_SESSION_SCORECARD.md) to that private
   location. Never fill or commit the repository copy.
6. Decide who owns support, incident response, follow-up, and deletion requests.

Suggested invitation, adjusted honestly for the recipient:

> I am testing a supervised prototype for turning recurring SaaS CSV reviews
> into traceable findings, a decision, and a follow-up. It is not a production
> analytics service and should not receive confidential or personal data. Would
> you be willing to try the synthetic demo, then—if appropriate—an approved
> de-identified export, and return for one follow-up cycle? The first session is
> about 35–45 minutes.

Do not advertise guaranteed savings, autonomous decisions, causal analysis,
enterprise security, durable storage, or production availability.

## Cycle 1: observed session

### 1. Baseline interview — 5–10 minutes

Before showing Data Prism, record:

- the recurring decision and who owns it;
- the current tools and approximate elapsed work time;
- the most difficult or risky step;
- what result would make a second use worthwhile;
- whether the person can return at the next reporting cycle.

Do not pitch features until this baseline is captured.

### 2. Unassisted synthetic attempt — up to 10 minutes

Give only the public URL and ask the participant to run the live demo. Record
whether they can:

- start and finish the analysis;
- find one important result;
- explain the evidence behind it;
- identify at least one limitation or warning;
- find the next action without coaching.

Record assistance as `none`, `minor`, or `substantial`. A coached completion is
not an unassisted success.

### 3. Approved data attempt — 10–15 minutes

Only proceed with synthetic or explicitly approved de-identified data. Stop if
the file contains personal, confidential, regulated, secret, or customer-level
identifiers. Check readiness warnings before analysis.

If a real decision exists, let the participant create a decision case with an
owner, success metric, target, and review date. Do not invent these fields to
make the funnel look complete.

### 4. Immediate debrief — 5–10 minutes

Ask in this order:

1. What decision would you make differently, if any?
2. Which evidence do you trust least, and why?
3. What did the tool fail to understand about your context?
4. What step still required manual work?
5. Would you use this in the next reporting cycle?

Offer the in-product fixed-choice feedback form without preselecting consent.
With agreement, download the aggregate pilot receipt before ephemeral state is
lost. Store it privately; it contains a scope snapshot, not proof of identity or
value.

## Cycle 2: observed return

Follow up on the natural review date, not immediately after the demo. Record:

- whether the participant returned without a new product tour;
- whether they used Data Prism on another relevant dataset;
- whether the original decision was reviewed on time;
- whether a structured outcome was recorded;
- whether the evidence affected a real action;
- the main blocker to repeating the workflow;
- whether they will take a concrete adoption step.

Only after understanding the workflow and blocker, present a specific next-step
offer if appropriate—for example, a time-bounded team pilot with persistent
history at an explicit price. Record acceptance, rejection, or no decision. A
general statement such as “I might pay” is not a commitment.

## Precommitted interpretation

These are product-experiment rules, not universal market thresholds:

- **Continue the narrow experiment** when at least five eligible participants
  complete cycle 1, multiple participants can explain the evidence, at least two
  return at the agreed next cycle without another product tour, and at least one
  accepts a concrete adoption step.
- **Narrow or repair the workflow** when participants complete the demo but need
  substantial assistance, cannot identify a decision, or do not return. Rank the
  shared blocker before adding another broad feature.
- **Pause the proposition** when relevant participants consistently prefer their
  existing workflow, the evidence cannot support their decisions, or no one
  accepts a concrete follow-up after the workflow is usable.
- **Stop sessions immediately** for any privacy, authorization, misleading
  analysis, cross-scope access, or unapproved-data incident.

Report numerators and denominators. Do not convert five to ten exploratory
pilots into a market-size estimate, causal impact claim, or product-market-fit
statement.

## Minimum cohort report

After the second cycle, publish or retain only safe aggregate claims:

- eligible participants recruited;
- cycle-1 sessions completed, split by assistance level;
- participants who explained one finding and limitation;
- real decision cases created;
- participants who returned in cycle 2;
- structured outcomes recorded;
- concrete adoption steps accepted;
- top blockers, without private quotes or identifying detail.

Keep synthetic/operator activity separate. If the denominator is small or data
is missing, say so explicitly.
