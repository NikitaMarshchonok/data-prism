# Synthetic rolling-origin forecasting protocol

This benchmark measures the behaviour of the current daily single-series
forecaster on a wider set of **synthetic** data. It is not a client acceptance
test, an untouched certification dataset, or evidence of professional-level
data-science work. It does not replace the existing analytical quality gate.

## Reproduce

```bash
python evaluate_forecast_benchmark.py \
  --config evaluation/forecast_benchmark.json \
  --output forecast-benchmark.json
```

The command prints and optionally writes the same JSON report. Exit `0` means
all scheduled windows and candidates completed the protocol, `2` means a run,
candidate or output-contract failure, and `1` means configuration/file/execution
failure outside a scheduled window. **A model losing to a baseline is a valid
result, not an integrity failure or a reason to delete the case.** Predictive
acceptance is explicitly `not_established`, including when the command exits 0.
CI runs this protocol separately and retains its report, also on failure.

## Frozen v1 design

The checked-in configuration declares seven scenarios, seeds 17 and 83 for the
five stochastic generators, 140 initial history days, three consecutive
non-overlapping seven-day outer windows, and reporting delays of zero and two
days. Constant and exact-weekly oracles are evaluated once, not duplicated per
seed: 12 series and **72 scheduled origin/delay evaluations** in total.

| Scenario | Construction fixed in generator v1 |
| --- | --- |
| Constant | 40 each day; zero-error and undefined-relative-gain oracle |
| Exact weekly | Repeated `[12, 18, 22, 35, 60, 45, 25]`; seasonal reference oracle |
| Noisy weekly | Same weekly cycle plus normal noise, standard deviation 5, floored at zero |
| Trend | `30 + 0.6 × day` plus normal noise, standard deviation 3 |
| Intermittent | Independent 15% occurrence indicator times a Poisson(20) draw |
| White noise | Independent `50 + Normal(0, 10)`; no designed temporal signal |
| Regime change | `50 + 0.3 × day + Normal(0, 3)`, then an unannounced +50 shift and slope adjustment −1.2/day at the first outer target |

These are generic numeric series, not all count-valued demand or physically
meaningful measures. Random noise can contain accidental patterns. Seeds are
reproducibility controls, not independent real-world clients. The protocol,
generator and thresholds must be reviewed **before** inspecting new results;
changes must not be selected just to improve the displayed scores. These
inspected cases become regression evidence, never a new unseen test set.

Each outer issue date is the end of a completed day. All delays use the same
issue dates and actuals; input ends exactly at issue date minus the declared
delay. Neither unreleased recent targets nor any outer outcome is passed to the
forecaster. It receives only two columns: daily date and target. The current
engine's three inner temporal folds choose a method and a reference baseline;
the existing inner holdout does not choose them. The locked method is refitted
on available history to produce the outer forecast. The benchmark scores that
**future forecast**, not the already inspected inner holdout.

At a later issue date, earlier released outer outcomes become legitimate history.
No candidate list, hyperparameter, metric or generator is adapted using outer
errors. Consecutive windows and delay variants are dependent; they must not be
treated as independent trials for confidence intervals or significance tests.

## Independent scoring and review

The scorer does not call the forecaster's private metric or baseline functions.
Its scalar scoring/reference implementation lives in `src/forecast_scoring.py`
and is also used by the uploaded-series historical evaluation. Moving it from
the benchmark module does not change benchmark v1/v2 cases or scoring rules;
the source fingerprint includes this additional module.
It computes MAE and RMSE with scalar arithmetic, and independently constructs
last-value and weekly-seasonal forecasts, including the unavailable-day bridge.
The reference used for the displayed comparison is the one selected by inner
validation, **not** whichever baseline wins after viewing the outer actuals.
Both references remain available for review. Relative improvement against a
zero-MAE reference is `null`, not an infinite gain, zero gain, or discarded case.
Better/tie/worse counts use MAE with a numerical tie tolerance of `1e-9` absolute
and relative; exact errors are retained alongside these descriptive labels.

These two references are the v1 engine's baselines, not the strongest
possible simple methods. In particular, beating last-value/seasonal-naive on
independent noise can reflect smoothing a noisy reference, not discovering a
temporal signal. Historical mean/median are now included in the separately
versioned v2 protocol below, not inserted retrospectively into v1. Gains against
these references still do not establish useful predictability. Stochastic scenarios reuse seed streams; their errors
are not independent evidence either.

Every scheduled window retains its issue date, cutoff, input fingerprint,
actuals (synthetic only), forecasts, both reference errors, method selection and
candidate validation scores/failures. A failed forecast keeps its slot and
independent references. A failed candidate also fails the integrity gate even
if the selected method returns a usable prediction. There is no success-only
average and no pooled cross-series accuracy percentage. Report summaries count
better/tie/worse windows and undefined relative improvements; they are not a
business acceptance criterion or a probability of future success.

The report includes the protocol, protocol/data/input SHA-256 fingerprints,
source fingerprints and dependency versions, with no clock or random run ID.
Reproducibility is conditional on the recorded implementation and environment;
floating-point results can vary across dependency/platform versions. A hash
identifies bytes; it is not proof that the methodology is correct.

Tests include independent arithmetic/phase oracles, complete schedule accounting,
future-target mutation, baseline selection without outer hindsight, malformed
outputs, real engine calls, and preservation of losses and failures. Metadata
validation alone cannot establish that fits respected their cutoffs; existing
prefix-isolation tests remain necessary. No model hyperparameters or earlier
quality-gate thresholds are changed by this benchmark.

## Separate v2 reference set

```bash
python evaluate_forecast_benchmark.py \
  --config evaluation/forecast_benchmark_v2.json \
  --output forecast-benchmark-v2.json
```

`forecast-benchmark-v1` explicitly requests the unchanged
`daily-lag-candidates-v1` engine suite, even though the service's default suite
has advanced. `forecast-benchmark-v2` requests `daily-lag-candidates-v2`: it adds
whole-available-prefix mean and median as two more baselines before Ridge and
Random Forest. No source generator, seed, origin, horizon, availability rule,
selection metric or error calculation changes. Each version records all of its
baselines and candidates; the script's default remains v1 for compatibility.
The v2 independent scorer uses scalar mean/median arithmetic, not the engine's
predictor. Unknown/mismatched suite outputs fail instead of silently downgrading.
CI retains both reports. Neither version establishes predictive acceptance.

These cases have already been inspected while developing v1. The addition of
more appropriate simple references is an engineering change, not a fresh
unseen evaluation or proof of improvement on real customer data. Compare both
versions against a **common** reference when assessing changes in method error:
their own validation-selected references can differ, so the better/tie/worse
counts are not directly comparable percentages. Preserve regressions, including
cases where the new constants or the expanded selector perform worse. Do not
change a seed, generator or threshold to erase an adverse result.

### Initial local observation, not an acceptance threshold

With Python 3.11.9, NumPy 2.2.6, pandas 2.3.1 and scikit-learn 1.7.2, all 72
v1 windows reproduced the earlier forecasts, selected methods and errors exactly.
Both protocols completed with no failed candidates or windows. Comparing the
v2 selected forecast's MAE directly with the v1 selected forecast on identical
actuals gave **5 lower, 65 tied and 2 higher** errors (the same numerical tie
tolerance as above). Four improvements occurred in intermittent seed 17; another
in white-noise seed 83. A zero median's lower MAE on sparse data does not establish
that it estimates expected volume or meets the client's cost of underprediction.

The two regressions are retained: white-noise seed 17, window 0, delay 2 changed
from Ridge MAE approximately 9.9865 to historical-mean MAE 10.2692; seed 83,
window 1, delay 2 changed from Ridge 4.7145 to historical mean 6.1572. These are
inspected synthetic cases and not evidence of universal improvement. Against
their **own** validation-selected reference, v1 had 25 better/35 tied/12 worse
windows and v2 had 18 better/41 tied/13 worse. The references differ; do not
turn these counts into directly comparable accuracy or success percentages.

## What remains unproven

There is no pass threshold asserting useful predictive performance in v1.
Review adverse windows before choosing any expansion or new algorithm. The
next evidence must include approved real series, a protocol and acceptance
criteria set before examining their holdouts, comparison to the client's current
process, and independent specialist review. This benchmark does not evaluate
variable release delays, revisions, external as-of joins, multiple entities,
calibrated prediction intervals, model persistence, real-outcome monitoring,
fairness, financial benefit or high-impact decision suitability.
