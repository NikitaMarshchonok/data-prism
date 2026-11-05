# Analytical quality evaluation

Data Prism has a deterministic regression gate for analytical behaviour. Unit
tests verify individual functions; this gate verifies that the assembled
evidence, statistical, anomaly, segmentation, and predictive engines still behave
sensibly on synthetic datasets whose properties are known in advance.

## Run the gate

```bash
python evaluate_quality.py \
  --config evaluation/quality_gate.json \
  --output quality-report.json
```

The command prints the complete JSON report and optionally writes the same
report to a file. Exit codes are stable:

- `0`: every configured quality requirement passed;
- `1`: configuration or execution error;
- `2`: at least one analytical quality requirement failed.

GitHub Actions runs the gate on Python 3.11 and 3.12. Its JSON output is kept
as a workflow artifact even when the gate fails, so a reviewer can inspect the
observed value, expected threshold, and description for every check.

## Benchmark scenarios

Quality contract/configuration v2 requires the five scenarios below. Historical
v1 configurations must be migrated rather than silently skipping the new
temporal checks. Analytical dashboard manifest contracts are unaffected.

| Scenario | Known property | Regression guarded against |
| --- | --- | --- |
| `saas_known_signals` | Designed customer/revenue relationship, three operational incidents, quality issues, segment structure, and dated rows | Lost evidence, lower signal sensitivity, missed incidents, unstable segmentation, or silently applying random-split prediction to dated data |
| `null_noise_guardrail` | Five independent numeric variables and an independent categorical target with no designed relationship or segment structure | Strong claims, FDR discoveries, defensible-looking segments, or predictive support manufactured from seeded random noise; anomaly candidates promoted beyond exploratory review |
| `known_group_effect` | Balanced control/treatment groups with a fixed outcome shift | Failure to detect a large group effect or report uncertainty correctly |
| `independent_predictive_signal` | Independent rows with a known numeric regression signal | Failure to beat a mean baseline with the unchanged 0.8 MAE-improvement and 0.05 permutation thresholds |
| `daily_forecast_contract` | Daily trend and exact weekly repetition, plus a mutation of final holdout actuals | Temporal leakage, failure to retain an exact seasonal baseline, or IID significance claims about forecasting |

The dated `saas_known_signals` scenario now checks `temporal-inference-withheld`
instead of requiring a minimum number of IID statistical discoveries. This is
an eligibility-policy correction, not a lower p-value/power threshold: the
independent-sample `known_group_effect` sensitivity check and the null-noise FDR
check remain in place, together with frozen mathematical oracles. Descriptive
correlation strength and incident/segmentation checks are unchanged. The dated
SaaS predictive check now requires withholding unconfirmed random evaluation;
positive IID predictive sensitivity is tested on a separate independent-row
fixture rather than declaring dated rows independent to keep the gate green.
Its historical acceptance thresholds are unchanged. Daily forecasting adds a
separate known-trend sensitivity threshold and causal-isolation software tests,
not an IID significance or causal-effect claim.

The datasets are generated in memory from fixed seeds and contain no personal
or uploaded data. The report intentionally has no timestamp, host identifier,
or random run identifier, making it reproducible and safe to share.

## Threshold policy

Thresholds live in `evaluation/quality_gate.json` and are reviewed like source
code. They express behavioural tolerances—for example a minimum correlation,
minimum silhouette/subsample-stability score, and maximum false discoveries—
instead of pinning exact floating-point output. The null scenario additionally
requires the gap statistic to select one cluster and keeps fixed-budget anomaly
rankings explicitly exploratory. Predictive checks require a material
improvement over a naive baseline and a holdout target-permutation p-value at
or below the reviewed threshold for the IID predictor; the null scenario must
remain unsupported. Forecasting uses temporal validation MAE or explicitly
predeclared asymmetric error weights, with descriptive future-holdout error,
never the IID permutation test. Additional checks verify a hand-calculated
asymmetric-loss oracle, rejection of an unmet absolute-error criterion despite
relative improvement, and holdout isolation under weighted selection.
This avoids fragile snapshots while still failing on material analytical
regressions.

Changing an algorithm and changing its acceptance threshold should be treated
as separate decisions whenever possible. A lower threshold needs an explicit
reason in the pull request; it must not be used merely to make a failing build
green.

## Scope and limitations

This benchmark is a software quality control, not proof that every result is
correct for every domain. It measures repeatability, known-signal detection,
and one seeded false-discovery guardrail, including a predictive null case.
Broader validation should add
domain-specific golden datasets, repeated null simulations, fairness checks,
and externally reviewed acceptance criteria before operational decisions are
automated.

## Separate rolling-origin forecasting report

Local outcome lifecycle checks are separate from predictive acceptance:
`tests/test_forecast_ledger.py` covers immutable issuance, matched-date arithmetic,
unknown/partial quality, fixed-delay availability, atomic imports, duplicate facts,
explicit revisions, shared overlapping facts, logical deletion and process restart.
Controlled declarative predictors provide independent error oracles; they are
not evidence that those parameters were selected on unseen client data. The
journal does not certify historical issuance or release timestamps; see
[FORECAST_OUTCOME_LEDGER.md](FORECAST_OUTCOME_LEDGER.md).

The server registry has separate scope/fence/retention/recovery tests in
`tests/test_forecast_models.py`, plus an account-deletion HTTP regression case.
The opt-in web path is tested in `tests/test_vibedash_forecast_model_web.py`,
including no-consent, failed completion, rollback, owner-only download and
expiry independent of job-history retention.
These exercise a declared persistent test root with a known identity, not a
physical hosting mount. They do not establish deployed durability or enable
customer-facing model saving; see [FORECAST_MODEL_STORE.md](FORECAST_MODEL_STORE.md).

`evaluate_forecast_benchmark.py` additionally evaluates seven synthetic regimes
over predeclared outer issue dates, independently scoring future forecasts and
all declared simple references. The v1 suite keeps last-value and seasonal
naive; v2 additionally includes historical mean and median. It retains failed runs and baseline wins, without
changing this gate's five scenarios or thresholds. Its green integrity result
does **not** establish predictive acceptance. The versioned protocol, failure
semantics and limitations are in [FORECAST_BENCHMARK.md](FORECAST_BENCHMARK.md).

Uploaded-series web/CLI runs now also use three fixed historical issue dates;
see [FORECAST_BACKTEST.md](FORECAST_BACKTEST.md). They share the independent scalar
scorer/reference builder in `src/forecast_scoring.py` with the synthetic benchmark,
not the fitted engine's private metric/baseline implementation. Source-isolation,
short-history, failure retention and per-window evidence have separate regression
tests. This integration does not change gate thresholds or supply real-client
acceptance evidence.
