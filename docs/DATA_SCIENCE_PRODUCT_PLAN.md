# Data Prism: industry-independent data-science service

## Product agreement

Data Prism's target is an industry-independent service: a company uploads
approved data, describes the work it needs, and receives a professionally
evaluated result addressing that request. Users should not need to choose
estimators, validation splits or statistical tests. Industry is business context,
not a product eligibility whitelist; forecasting orders, support volume and
equipment load can share a temporal protocol but have different definitions,
availability assumptions, error costs and acceptable uses.

Universality means a shared intake with specialized, verified task executors.
It does not mean one algorithm for every dataset, a guarantee to solve any
request from any file, or equal autonomy for high-impact and routine decisions.
The service must distinguish a supported task, a need for clarification or
additional data, and an unsupported task. It must follow the client's goal
without adjusting evidence to manufacture a desired answer.

Professional-level work is the acceptance target, not an achieved product claim.
It requires independent review on previously unseen tasks and data, explicit
limitations, and the ability to withhold an unsupported conclusion. Broad
industry access does not establish medical, credit or other high-impact
decision suitability, security certification or regulatory compliance.

## Shared workflow and release gates

1. Intake: understand the requested deliverable and decision; propose the task
   family and ask for essential missing context. Confirm the interpretation.
2. Data review: establish observation unit, definitions, units, entity/time
   structure, target and predictor availability, quality risks and allowed use.
3. Plan: choose a supported executor, baseline, metrics, evaluation design,
   acceptance criteria and compute budget before inspecting holdout outcomes.
4. Execute: prepare and analyze data or train candidates with task-appropriate
   isolation; record reproducibility metadata and failures.
5. Verify: inspect leakage, baseline comparisons, stability and relevant subgroup
   errors. Preserve negative results; do not claim causal effects from association.
6. Deliver: answer the confirmed question, explain support, limits and appropriate
   use, and provide the required output rather than only generic charts.
7. Follow up: retain approved definitions/artifacts, accept new data and actual
   outcomes, measure quality and review retraining before changing active models.

The initial `vibedash-task-intake-v1` capability boundary is deterministic and
uses an explicit task type. It is not yet natural-language business understanding.
The catalogue is scoped to confirmed VibeDash workflows, not every internal
Python experiment. Descriptive analysis, daily single-series forecasting and
two-period comparison have bounded workflows. Standalone classification,
independent-row regression, validated segmentation/anomaly detection and causal
analysis remain unavailable in this intake. Embedded exploratory findings are
not a substitute for these requested deliverables.

Acceptance: catalogue and routing are independent of industry; unknown or
conflicting explicit task types cannot silently fall back to descriptive
analysis; a supported type still requires dataset/semantic review; the worker
rechecks routing. Existing mode-only clients remain compatible but do not gain
an inferred-intent or professional-equivalence claim.

The catalogue, explicit submission rules and compatibility boundary are defined
in [TASK_INTAKE.md](TASK_INTAKE.md). A structured browser brief now asks for
the question, observation unit and expected supported result, and confirms
them with the reviewed column meanings. Missing fields produce clarification
questions; the worker rechecks the bound request before calculation. This is
explicit user context, not inferred business understanding. Conversational
clarification and substantive definition checks remain follow-up work.

## Current implementation and next milestones

The intended product accepts a company's supported task and data, evaluates
candidate methods honestly, delivers a usable result, and eventually checks
that result against newly observed outcomes. Passing CI is necessary software
evidence, not certification as a human data scientist or proof of market demand.

## 1. Explicit task and decision criteria — first implementation

Daily forecasting now binds reviewed target/time/units/availability and an
optional versioned decision policy to one queued evaluation. The browser asks
for the supported decision, error preferences and acceptance criteria before
training. The final holdout cannot select a method or erase a failed criterion.
Defaults are not discovered business requirements; users must review them.

Acceptance: independently calculated error oracles; a change of preferences
can change validation selection; final-target mutation cannot change selection;
missing confirmation or altered bound settings cannot train; a relative gain
cannot override an absolute limit; no monetary-value or deployment claim.

Remaining: adaptive collaborative task clarification, validated business definitions,
nonlinear decision costs and an explicit feedback/review workflow. The service
does not yet infer these safely from a free-text request.

## 2. Broader supported inputs — fixed target delay implemented first

The daily engine now accepts an explicitly confirmed constant target reporting
delay of 0–30 whole calendar days. Each origin gets an embargoed training
prefix and a recursive bridge, and final unreleased target outcomes cannot
influence validation selection or scoring. The report exposes origins and
available-target cutoffs. This is a declared simulation, not reconstructed
release timestamps or historical vintages.

An explicit end-of-day issue date now supports available-history-only input:
the file ends exactly at origin minus delay, and unavailable recent targets
need not be supplied or invented. Future/stale history is rejected before
training. Historical evaluation retains training and selection embargoes; the
forecast starts after the issue date, not after the last available fact. The
previous retrospective convention remains available when no date is supplied.
This is an evaluated forecast run, not persisted-model serving or live monitoring.

Remaining: variable delays, target revisions, external predictors with explicit
availability/as-of joins, entity keys and grouped temporal backtests. Do not
silently pool shops/products or use observed future weather.

Acceptance: fixed-delay prefix/selection isolation and explicit-origin cutoff
tests are implemented, including equivalence with censored retrospective data.
Future extensions still require tests covering as-of joins, unseen entities,
missing days and group boundaries, plus rejection of unavailable predictors.
Each new task type needs its own baselines and evaluation protocol; this is
not permission to treat classification or independent regression as forecasting.

## 3. Diverse, independently reviewed acceptance cases — incomplete

Freeze protocols before reviewing new cases. Include multiple series and
regimes, seasonal and non-seasonal signals, intermittent demand, trend changes,
noise and hard negative/leakage cases. Previously inspected public data is a
regression case, not a new untouched certification dataset.

Acceptance: retain failures and baseline wins; compare against simple methods
per series/window, not only pooled averages; reproduce error calculations with
an independent implementation. Compare bounded task outputs against an
independent specialist's review before claiming a professional capability level.

First software evidence: a separate versioned rolling-origin benchmark now
covers seven synthetic regimes, two stochastic seeds, three outer issue dates
and two fixed reporting delays. It independently scores future forecasts,
retains losses/failures and records source/data/protocol fingerprints. A completed
protocol is not predictive acceptance. Approved real cases and independent
specialist review remain outstanding; see [FORECAST_BENCHMARK.md](FORECAST_BENCHMARK.md).

The default forecast candidate suite now adds historical mean and median as
prefix-only simple baselines, selected by the same predeclared validation loss.
The original four-candidate suite remains explicit for reproducible v1 runs;
the extended suite is evaluated separately as benchmark v2 on the same inspected
synthetic cases. This is not unseen acceptance, and negative changes must remain
visible rather than changing generators or thresholds to manufacture a gain.

The web count-domain check also covers the final holdout method, its reference
baseline and the separate future horizon. Positive future values cannot rescue
negative historical forecasts. Incomplete/non-finite count outputs are rejected;
negative finite outputs remain as failed evidence without clipping or reselection.
An otherwise successful business-criteria status is blocked, not used to justify
an invalid count result. This necessary output check is not domain certification
and does not apply non-negativity to general numeric targets.

New web/CLI results also pass an independent evidence-consistency check before
delivery: source-bound holdout facts/dates, complete future dates, independently
recomputed errors and matching support/business flags. A cached success marker
cannot bypass rechecking. Inconsistent evidence is rejected, not silently
rewritten. This guard does not certify fit isolation, real-world predictive
value or the professional acceptance target; those remain separate work.

New uploaded-series web and CLI runs now also evaluate the service procedure
at three fixed historical issue dates, not just on one final holdout. Each call
receives only its released prefix; outer errors and simple references are scored
independently. Short history, failed windows, candidate failures, baseline wins
and count-domain failures remain explicit. These results do not reselect the
final future method or certify readiness. The last window overlaps the main
holdout and is not additional unseen evidence. This implements the multi-origin
evaluation workflow for the supported single-series task; approved real-client
cases and specialist acceptance are still missing. See
[FORECAST_BACKTEST.md](FORECAST_BACKTEST.md).

## 4. Model and outcome lifecycle — first local slice; hosted lifecycle not implemented

The CLI can now explicitly save a private, versioned declarative JSON snapshot
of an evaluated daily model and its reference, then predict in a separate process
without fitting or method reselection. Appended released actuals update lag
context; scaler/coefficients/trees and mean/median constants remain frozen.
Round-trip tests cover all six current candidate encodings, process restart,
strict history binding and no-overwrite failure behaviour. This is local tooling,
not owner-scoped cloud persistence, a promoted model or measured live quality.
Web storage/retention and automatic saving are unchanged. See
[FORECAST_MODEL_ARTIFACT.md](FORECAST_MODEL_ARTIFACT.md).

Next, persist a versioned, owner-scoped model artifact with preprocessing, input schema,
training cutoff, policy, dependencies and evaluation record. Support safe
prediction on new input without retraining, with storage/retention and deletion
rules. Never load arbitrary user-supplied serialized Python objects.

Record prediction IDs, origin and horizon; accept matched actuals only once they
become available. Score each horizon against its contemporaneous baseline;
distinguish data drift from measured forecast deterioration. Missing actuals are
unknown quality, not a successful prediction. Keep monitoring state durable.

Acceptance: artifact round-trip predictions; restart survival; strict ownership;
duplicate/late/corrected actuals; invalid input rejection; explicit alert policy;
candidate retraining that cannot silently replace the active model, with a
reviewable promotion decision and rollback path.

## Pilot value decision

A working upload-to-result path is not sufficient evidence of usefulness.
Use supported, approved data and ask whether the result changes a real planning
decision, saves review time, or reduces an agreed error relative to the current
process. Record limitations and the cost of failures. Continue or narrow the
product based on those observations; do not report a fabricated readiness
percentage or claim that the universal product vision has already been achieved.
Pilot cases may be narrow to make quality measurable; that does not restrict
the platform to that industry. Expand by demonstrated task capabilities, not
by an accumulating list of industry-specific promises.
