# Dataset readiness contract

Data Prism evaluates whether an uploaded CSV can support its automated analytical contracts before creating a retained working copy or background job. The check is deterministic and returns aggregate metadata only.

## Outcomes

- `ready`: all automated checks passed.
- `ready_with_warnings`: analysis may continue, but the dashboard and audit manifest preserve documented limitations.
- `blocked`: one or more conditions make automatic analysis unsafe; the user receives evidence and remediation instead of a misleading dashboard.

The readiness score is a communication aid, not a probability or certification. Blocking conditions always take precedence over the numeric score.

## Checks

| Check | Blocking condition | Warning condition |
| --- | --- | --- |
| Dataset coverage | Empty table or fewer than 8 rows | Fewer than 30 rows |
| Column identity | Duplicate column names | Empty or placeholder names |
| Usable signal | No varying fields | One varying field or constant fields present |
| Missing data | At least 50% of all cells missing | At least 10% missing, or fully empty fields |
| Duplicate rows | At least 25% duplicates | Any duplicated rows |
| Finite numeric values | — | Positive or negative infinity present |
| Potentially sensitive fields | — | Common personal or regulated field names detected |
| High-cardinality dimensions | — | Text fields behave like identifiers or free text; parseable time dimensions are excluded |

Each non-passing result includes observed evidence and a concrete next step. The checks intentionally avoid claiming that a dataset is legally approved, causally valid, representative, or suitable for an unreviewed business decision.

## Decision brief

For accepted datasets, the completed dashboard includes a `decision-brief-v1` record. It ranks at most three items from already calculated evidence:

1. statistically validated signals;
2. anomaly candidates requiring operational review;
3. readiness limitations or other high-value evidence-backed findings.

Every item separates the finding, supporting evidence, decision risk, and next verification action. The brief does not use an LLM to invent conclusions and never represents observational evidence as causal proof.

## API behaviour

`POST /vibedash/readiness` accepts one `.csv` file in the `datafile` form field. It reads the request stream and returns a readiness report without creating a retained Data Prism working copy of the source rows.

`POST /vibedash/jobs` repeats the readiness assessment at the trusted server boundary. A blocked upload returns HTTP 422 with the same report, removes the temporary file, and does not create or dispatch a job.
