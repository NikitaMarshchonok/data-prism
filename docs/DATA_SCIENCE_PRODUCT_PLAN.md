# From an analytical pilot to a bounded data-science service

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

Remaining: collaborative task clarification, validated business definitions,
nonlinear decision costs and an explicit feedback/review workflow. The service
does not yet infer these safely from a free-text request.

## 2. Broader supported inputs — not implemented

Next introduce a prediction-origin/availability contract before accepting delayed
targets or external predictors. Then add entity keys and grouped temporal
backtests. Do not silently pool shops/products or use observed future weather.

Acceptance: tests covering delayed releases, as-of joins, unseen entities,
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

## 4. Model and outcome lifecycle — not implemented for this forecast engine

Persist a versioned, owner-scoped model artifact with preprocessing, input schema,
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
percentage or promise universal automated data science.
