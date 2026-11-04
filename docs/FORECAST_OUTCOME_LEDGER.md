# Local forecast and reported-outcome journal

`local-forecast-outcome-ledger-v1` is an **opt-in local tooling slice**, not a
hosted model registry, authenticated multi-user service or production monitor.
It records frozen predictions and their contemporaneous reference before
matching operator-supplied actuals. It does not alter training, method selection,
existing web retention or the default stateless prediction command.

## Explicit workflow

First create a model with the evaluated local workflow described in
[FORECAST_MODEL_ARTIFACT.md](FORECAST_MODEL_ARTIFACT.md). Then initialize a private
journal and explicitly record a forecast:

```bash
python forecast_ledger.py --ledger /private/path/orders.forecast-ledger init
python predict_forecast.py --model /private/path/orders.forecast-model.json \
  --ledger /private/path/orders.forecast-ledger

# With newly released history: preserve the original training prefix unchanged.
python predict_forecast.py --model /private/path/orders.forecast-model.json \
  --history-csv /path/to/available-history.csv --forecast-origin 2026-01-13 \
  --ledger /private/path/orders.forecast-ledger
```

Paths and dates are examples, not approved customer data. The parent directory
must already exist. Initialization refuses existing paths rather than adopting
or overwriting them. The journal directory is created with mode `0700`, its
SQLite database with `0600`; opening non-private directories/files or symlinked
directory/database paths is rejected. This is a local operator boundary, not
protection against a malicious local administrator or filesystem races.

`predict_forecast.py` returns a fresh `prediction_id`, a registration timestamp
and `ledger.status=recorded` only after the transaction commits. The saved output
includes every original model and baseline prediction, model ID/hash, issue date,
available/trained cutoffs, task, units and predeclared policy. Neither predictor
is fitted or selected again. Without `--ledger`, no journal is opened or created.
The journal does not save model parameters or the complete source dataset; keep
the model snapshot separately if further inference is needed.

Every invocation issues a **new run**, even if model/origin are identical. This
is not a scheduled exactly-once issuer. Future issue dates are rejected against
the local UTC calendar. Past issue dates are allowed for replay. A claimed
business issue date is **not verified historical issuance**; all reports retain
`historical_issuance_verified=false`. A replay must not be sold as a prospectively
registered or untouched live forecast, even if its measured errors are small.

## Actuals and fixed-delay availability

Supply an explicit UTF-8 JSON document (maximum 64 KiB):

```json
{
  "as_of_date": "2026-01-18",
  "actuals": [
    {"date": "2026-01-14", "actual": 105},
    {"date": "2026-01-15", "actual": 99}
  ]
}
```

```bash
python forecast_ledger.py --ledger /private/path/orders.forecast-ledger actuals \
  --prediction-id PREDICTION_ID --input /private/path/actuals.json
python forecast_ledger.py --ledger /private/path/orders.forecast-ledger report \
  --prediction-id PREDICTION_ID
```

Replace `PREDICTION_ID` with the actual returned ID. A batch contains 1–90 unique
dates from that prediction's issued horizon, not arbitrary dates or another
entity/series. No unmatched rows are silently dropped. Each date plus the
model's fixed reporting delay must be no later than the declared `as_of_date`;
this date cannot exceed today's UTC date. These are operator assertions, **not
verified historical release timestamps or target vintages**. There is no
timezone inference from a dataset or use of future facts to train a new model.
Receipt timestamps use UTC; target dates retain the existing timezone-free
business-date contract. Variable/partial-day delays remain unsupported.

Booleans, strings, nulls, non-finite values, unsafe integer precision, duplicate
JSON fields, extra row fields and invalid dates are rejected. Negative actuals
are allowed for general numeric targets but not explicit count targets. Inputs
are never clipped, rounded or imputed. All row validation and inserts are in one
transaction: a conflicting or invalid row rolls back the **whole batch**.

## Partial scoring and revisions

- No actuals: `quality_status=unknown`, metrics and criteria are null, not zero.
- Some actuals: `partial`; errors use **the exact same matched dates** for the
  model and its frozen reference. Pending dates remain visible. Business
  criteria are not assessed on an incomplete horizon.
- Complete actuals: `measured_on_reported_outcomes`; compute MAE, RMSE and,
  where predeclared, asymmetric weighted error. Each matched date includes its
  lead, actual, both frozen predictions and absolute errors.

A zero reference error makes relative improvement undefined (`null`), including
a perfect tie. Losses and baseline wins remain visible. Criteria require all
predeclared checks and a domain-valid original forecast; a negative count
forecast remains recorded as failed evidence and cannot pass the criteria.
Without a policy, no business threshold is invented. Passing criteria here
means only the reported matched outcomes passed, not expected future accuracy,
financial savings, significance or operational approval.

Reimporting the same date/value is a no-op. A different value requires an explicit
`--correction-reason` of 1–500 characters; old revisions, receipt timestamps and
reasons remain available through `revisions --prediction-id PREDICTION_ID`.
Updates cannot backdate the previous asserted availability date. Reports use
the **latest revision** and therefore may change after a correction; earlier
computed metrics are not separately snapshotted. New facts are allowed to
arrive late or out of order within a batch. SQLite serializes concurrent writers.
Correcting a reported future outcome does not refit the frozen model or permit
changes to its original training prefix in a subsequent inference call.

Facts are shared by **exact model SHA-256 and target date**, so overlapping
origins of that same artifact see the same corrected fact. Different models
are not automatically treated as the same series, even with identical column
names; multi-model series reconciliation remains future work. Overlapping runs
are not independent trials, and this slice does not pool errors across them,
infer drift, create alerts or promote/retrain a model.

## Storage, deletion and limits

The directory/database contains predictions, reported target outcomes, policy
text, task definitions and correction notes. Treat it as **confidential,
data-bearing storage**. No automatic upload, web route, account export,
server backup or account-deletion integration is added. The local operator
controls physical storage, copying, retention and deletion. Restarting the
process reopens the same SQLite file; this is not proof that a host retains
the directory after restart/replacement or that a backup can be recovered.

Each journal is capped at 1,000 forecasts and 100,000 outcome revisions. Capacity
refuses new records rather than silently expiring unscored forecasts or erasing
revision evidence. The explicit `delete` command requires `--confirm-delete`:

```bash
python forecast_ledger.py --ledger /private/path/orders.forecast-ledger delete \
  --prediction-id PREDICTION_ID --confirm-delete
```

Deletion removes that forecast and its points; fact revisions remain while
another retained forecast of the same model references the same date. Orphaned
revisions are logically deleted. This is **not secure erasure** of disks,
SQLite journals, snapshots or operator backups. Database transactions and full
synchronous writes are not a cloud durability or encryption guarantee. Forecast
payload hashes and point consistency checks detect accidental changes, not
authenticity: an editor of the database can also rewrite hashes and actuals.
Do not open arbitrary untrusted SQLite files as an authentication mechanism.
The `*.forecast-ledger/` Git ignore pattern is only an accidental-tracking guard.

## Remaining product gate

The hosted workflow still needs server-derived ownership, a defined series
identity, authorized model retention/deletion/export, durable storage/recovery,
prospective issuance provenance, measured-quality history and predeclared alert
rules. Controlled retraining, promotion/rollback, real approved cases and
independent specialist acceptance are still missing. This local journal is
testable infrastructure toward that workflow, not its completed web delivery.
