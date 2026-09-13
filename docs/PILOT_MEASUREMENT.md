# Pilot onboarding and measurement

This is an instrumented product experiment, not validated demand. Start with one
user: a SaaS operations or growth lead conducting a weekly metrics review from a
CSV export. The hypothesis is that traceable findings plus a concrete owner,
target, and follow-up reduce the work of deciding what to investigate.

## First session

1. Ask what decision the person made last week, which table they used, how long it
   took, and what was difficult. Do not pitch before hearing their current process.
2. Let them try the synthetic demo without instruction. Then use approved,
   de-identified data only: consistent row units and periods, a date column, and
   numeric business metrics. Churn must already be defined/calculated; the prompt
   preset does not turn arbitrary CSV fields into a correct churn calculation.
3. Use **Weekly SaaS review** if helpful. Observe whether they understand warnings
   and can explain one finding using its evidence, including its limitations.
4. Ask them to select a priority and commit to a next action, owner, success
   metric, target, and review date. Record assistance and time separately.
5. Offer the optional usefulness/blocker form. At the next reporting cycle, ask
   what they actually used and what outcome they observed. Do not treat a demo
   run or a compliment as adoption.

Recruit 5–10 relevant participants through a separate, consented process. Do not
put interview transcripts, contact information, customer CSVs, or private notes
in this repository. Ask about paid adoption with a concrete offer only after
understanding the problem; stated interest is weaker evidence than a commitment.

## Opt-in contract

Both landing-page forms have an unchecked checkbox chosen separately per run.
Without it, the normal analysis still works but no pilot measurement is created.
Opting in additionally requires a session-bound CSRF token. Synchronous preview
and rejected preflight requests are not measured. There is no external analytics
service and no public metrics/admin endpoint.

`pilot_analyses` lives in the analysis-job SQLite database. It contains only:

- a job ID and an HMAC-derived pseudonymous browser token;
- the server-selected source cohort (`demo` or `upload`);
- UTC times for acceptance, start, completion/failure, first decision, and first
  recorded outcome (`validated` or `invalidated`, not cancellation);
- optional fixed-choice usefulness and blocker values.

It contains no dataset rows, prompts, filenames, IP addresses, owner names, or
decision text. This is pseudonymization, **not anonymization**: the operational
job table can still link a job to its browser scope. Operational analysis and
case storage are separate purposes and retain their existing data contracts.

Milestones are written in the same transaction as their corresponding lifecycle
change. Polling, refreshing, multiple cases for one analysis, and repeated edits
do not increment them. Feedback is replaced, not appended. An outcome timestamp
means an outcome was recorded at least once; reopening a case does not erase
that event. It does not represent the latest case state or causal effectiveness.

Measurement records expire 30 days after analysis acceptance; subsequent
application activity removes them. This is not a timer-based deletion SLA. A
service-wide cap of 10,000 records bounds storage; a full table skips measurement
for new analyses, without blocking analysis. Consent withdrawal removes records
for the current signed browser scope but leaves analyses and cases intact.
Later lifecycle changes never recreate removed records. Future runs remain
opt-in. Cookie loss or a changed session key loses access to the previous scope.

## Read the report on the measured host

From the repository root, with its dependencies installed:

```bash
# Default local-development database; run after a consenting background analysis.
python pilot_report.py --database data/jobs/analysis_jobs.sqlite3 --days 7
```

If `DATA_PRISM_STATE_DIR` is configured, pass the absolute path to
`jobs/analysis_jobs.sqlite3` under that directory instead. For example, **only if
that deployment is configured to use `/var/lib/data-prism`**:

```bash
python pilot_report.py --database /var/lib/data-prism/jobs/analysis_jobs.sqlite3 --days 7
```

The CLI opens an existing database read-only; it neither starts the web app nor
creates or migrates a database. It prints aggregate JSON to standard output,
never per-browser tokens, job IDs, or source text. `--days` accepts 1–30. A legacy
database without the measurement table reports `collection_installed: false`;
a wrong path produces an error rather than a misleading empty report.

**Your local database does not contain Render activity.** The free Render demo
has ephemeral storage and is not a durable pilot ledger. Reports must run on the
host holding the data (where shell access is available), or on a secured snapshot
under your control. Do not expose the raw SQLite file as a public download. A
durable multi-week pilot needs persistent storage and an operational reporting
path; this milestone does not provision either or change paid hosting plans.

## Interpret counts before percentages

The report splits demo and upload cohorts. Cohorts are based on analysis
acceptance time, not the time of the later decision or outcome.

| Field | Definition |
| --- | --- |
| `completion_rate` | Completed / accepted opted-in analyses |
| `decision_rate_among_completed` | Completed analyses with ≥1 case / completed analyses |
| `outcome_rate_among_decision_analyses` | Analyses with ≥1 recorded outcome / analyses with ≥1 case |
| `median_seconds_to_first_decision` | Acceptance to first case, among analyses with a case |
| `browser_scopes_with_repeat_completed_analysis` | Browser scopes with ≥2 completed runs in the cohort |
| `browser_scopes_active_on_multiple_utc_dates` | Browser scopes with completed runs accepted on ≥2 UTC dates |
| `feedback_responses` | Completed analyses with one current response |

Ratios have range 0–1; zero denominator yields `null`, not a failure score.
Always show denominators. Consent selection, cookie clearing, recent unfinished
work, deletion, the storage cap, and redeployment bias coverage. A browser is
neither a person nor a company. An uploaded synthetic CSV is still in the upload
cohort, so operators must keep their own test uploads out of real-pilot results.

Review-date adherence, actual time saved versus the old workflow, measurement
quality of outcome notes, repeat use by a real person, and willingness to pay
remain **interview/manual follow-up measures**, not automatically measured by
this table. Existing Decision Brief priorities are heuristic; the prompt does not
guarantee a tailored business recommendation.

## Make a decision after two reporting cycles

Before recruiting, write down the baseline workflow and what material improvement
would justify adoption for that participant. After two cycles, report counts of
participants, completed journeys, assistance, repeated use, and concrete adoption
commitments together with blockers. Separate synthetic checks from real users.

Continue the narrow use case if several participants independently return and
can identify a real decision or meaningful time saved. If they enjoy charts but
do not use the workflow again, investigate the blocker before adding broad
features. Five to ten interviews guide iteration; they cannot estimate global
demand or establish product-market fit.
