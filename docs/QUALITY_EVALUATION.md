# Analytical quality evaluation

Data Prism has a deterministic regression gate for analytical behaviour. Unit
tests verify individual functions; this gate verifies that the assembled
evidence, statistical, anomaly, and segmentation engines still behave
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

| Scenario | Known property | Regression guarded against |
| --- | --- | --- |
| `saas_known_signals` | Designed customer/revenue relationship, three operational incidents, quality issues, and segment structure | Lost evidence, lower signal sensitivity, missed incidents, unstable or weak segmentation |
| `null_noise_guardrail` | Five independent numeric variables and no designed relationship | Strong claims or FDR discoveries manufactured from seeded random noise |
| `known_group_effect` | Balanced control/treatment groups with a fixed outcome shift | Failure to detect a large group effect or report uncertainty correctly |

The datasets are generated in memory from fixed seeds and contain no personal
or uploaded data. The report intentionally has no timestamp, host identifier,
or random run identifier, making it reproducible and safe to share.

## Threshold policy

Thresholds live in `evaluation/quality_gate.json` and are reviewed like source
code. They express behavioural tolerances—for example a minimum correlation,
minimum silhouette/stability score, and maximum false discoveries—instead of
pinning exact floating-point output. This avoids fragile snapshots while still
failing on material analytical regressions.

Changing an algorithm and changing its acceptance threshold should be treated
as separate decisions whenever possible. A lower threshold needs an explicit
reason in the pull request; it must not be used merely to make a failing build
green.

## Scope and limitations

This benchmark is a software quality control, not proof that every result is
correct for every domain. It measures repeatability, known-signal detection,
and one seeded false-discovery guardrail. Broader validation should add
domain-specific golden datasets, repeated null simulations, fairness checks,
and externally reviewed acceptance criteria before operational decisions are
automated.
