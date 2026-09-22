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
5. Offer the optional fixed-choice feedback form: usefulness, primary blocker,
   estimated time saved, and intent to use the workflow in the next review
   cycle. At that next cycle, ask what they actually used and what outcome they
   observed. A self-reported estimate or intention is not verified savings,
   repeat use, or adoption.

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

- a job ID and an HMAC-derived pseudonymous guest-browser or pilot-account
  scope token;
- the server-selected source cohort (`demo` or `upload`);
- UTC times for acceptance, start, completion/failure, first decision, and first
  recorded outcome (`validated` or `invalidated`, not cancellation);
- optional fixed-choice usefulness, blocker, perceived-time-saved, and
  next-cycle-intent values.

It contains no dataset rows, prompts, filenames, IP addresses, owner names, or
decision text. This is pseudonymization, **not anonymization**: the operational
job table can still link a job to its VibeDash scope. Operational analysis and
case storage are separate purposes and retain their existing data contracts.

Milestones are written in the same transaction as their corresponding lifecycle
change. Polling, refreshing, multiple cases for one analysis, and repeated edits
do not increment them. Feedback is replaced, not appended. An outcome timestamp
means an outcome was recorded at least once; reopening a case does not erase
that event. It does not represent the latest case state or causal effectiveness.
The time-saved ranges are subjective estimates. Next-cycle intent is stated
intent, not observed return behavior or an adoption commitment. Databases created
before these two fields remain readable and are migrated without rewriting prior
responses; the report identifies those legacy responses instead of interpreting
missing values as negative answers.

Measurement records expire 30 days after analysis acceptance; subsequent
application activity removes them. This is not a timer-based deletion SLA. A
service-wide cap of 10,000 records bounds storage; a full table skips measurement
for new analyses, without blocking analysis. Consent withdrawal removes records
for the current signed guest-browser or pilot-account scope but leaves analyses
and cases intact. Later lifecycle changes never recreate removed records. Future
runs remain opt-in. Guest cookie loss loses access to the previous guest scope;
a pilot account can restore its account scope across browsers while a changed
session key still makes prior signed-cookie identity unavailable. Because the
Flask secret also derives account scopes, rotating it makes prior account-owned
job history unavailable after the account signs in again.

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
| `browser_scopes_with_repeat_completed_analysis` | Guest-browser or pilot-account scopes with ≥2 completed runs in the cohort (the stable field name is retained for report compatibility) |
| `browser_scopes_active_on_multiple_utc_dates` | Guest-browser or pilot-account scopes with completed runs accepted on ≥2 UTC dates (the stable field name is retained for report compatibility) |
| `feedback_eligible_analyses` | Completed opted-in analyses that could provide feedback; denominator for feedback rates |
| `feedback_responses` | Completed analyses with one current response |
| `feedback_response_rate_among_completed` | Responses / completed opted-in analyses |
| `value_feedback_responses` | Responses containing both a perceived-time-saved range and next-cycle intent |
| `value_feedback_response_rate_among_completed` | Complete value responses / completed opted-in analyses |
| `legacy_feedback_responses_without_value_signals` | Earlier responses that lack one or both additive value fields; these are not counted as zero time saved or no intent |
| `perceived_time_saved` | Counts by fixed self-reported time range, among complete value responses |
| `next_cycle_intent` | Counts by fixed stated-intent choice, among complete value responses |

Ratios have range 0–1; zero denominator yields `null`, not a failure score.
Always show denominators. Consent selection, cookie clearing, recent unfinished
work, deletion, the storage cap, and redeployment bias coverage. A scope token
is neither a person nor a company. An uploaded synthetic CSV is still in the
upload cohort, so operators must keep their own test uploads out of real-pilot
results.

Review-date adherence, objectively measured time saved versus the old workflow,
measurement quality of outcome notes, observed repeat use by a real person, and
willingness to adopt or pay remain **interview/manual follow-up measures**, not
automatically measured by this table. The new fixed-choice fields record only an
estimate and intention. Existing Decision Brief priorities are heuristic; the
prompt does not guarantee a tailored business recommendation.

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
