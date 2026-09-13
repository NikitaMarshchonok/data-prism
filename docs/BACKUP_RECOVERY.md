# Offline runtime backup and recovery

This milestone provides **local operator tools**, not managed backup hosting,
automatic scheduling, encryption, or a persistent Render disk. The free Render
demo is still ephemeral. A copy kept on that same ephemeral filesystem does not
protect against a restart or redeploy.

## Scope and prerequisites

Use one dedicated `DATA_PRISM_STATE_DIR`, with the layout already used by the
Docker deployment:

```text
runtime-state/
├── jobs/analysis_jobs.sqlite3     # Jobs, decision cases, opt-in pilot measurement
├── drift/drift_history.sqlite3   # Monitoring runs and alerts, when present
├── baselines/                    # Aggregate reference profiles
├── uploads/                      # Temporary input files
├── sessions/vibedash/            # Stored dashboard results
├── exports/vibedash/             # Generated exports, when present
└── reports/                      # Reports, when present
```

The job database is required; other components may be absent. This tool does not
silently collect the repository's legacy split layout (`data/`, `tmp/`, and
`exports/`), change its storage configuration, or copy files outside the selected
state directory. A running deployment using that legacy layout needs a separate,
reviewed migration. Do not point `--state-dir` at your home directory or repository.

Before taking a backup:

1. Stop accepting new work; let queued/running analyses finish. Check History.
2. Stop the web service, background workers, drift CLI jobs, and every other
   writer. The operator, not the CLI, is responsible for this quiescence.
   `--offline` is an operator acknowledgment, **not** a command that stops them.
   The tool rejects active job records and detects ordinary file/WAL changes,
   but it cannot prove no unrelated process is writing.
3. Use the application OS user and a trusted parent directory. The source must
   permit SQLite's WAL bookkeeping; a read-only mount without pre-existing WAL
   metadata may fail even though the main database is opened read-only.
4. Choose a **new** backup directory outside the state directory. Keep enough
   free space for the snapshot and a separate restore rehearsal.

SQLite files use Python's [SQLite backup API](https://docs.python.org/3.12/library/sqlite3.html#sqlite3.Connection.backup),
which includes committed WAL pages. Database snapshots are checked for integrity
and expected tables. Database files and companion JSON/uploads still need the
offline window for a coherent whole-application snapshot.

## Commands

Examples below assume a dedicated local runtime at
`/srv/data-prism/runtime-state` and an existing private backup parent
`/srv/data-prism/backups`. **Replace these example paths with the actual host's
paths.** They are not the location of your Mac project or a claim that backups
already exist. Run the CLI from the repository directory.

```bash
python runtime_backup.py backup \
  --state-dir /srv/data-prism/runtime-state \
  --output /srv/data-prism/backups/snapshot-2026-09-13 \
  --revision 050fa75 \
  --offline

python runtime_backup.py verify \
  --backup /srv/data-prism/backups/snapshot-2026-09-13

python runtime_backup.py restore \
  --backup /srv/data-prism/backups/snapshot-2026-09-13 \
  --destination /srv/data-prism/runtime-restored-2026-09-13 \
  --offline
```

Record the revision of the application being backed up, from its `/healthz` or
deployment record, not an unrelated checkout. `--revision` accepts a 7–40 character
Git hash, or defaults to `unknown`. Recording a revision does not migrate a
database or guarantee compatibility with a newer application version.

Every successful command returns aggregate JSON with operation, contract,
revision, timestamp, file count, and byte count. It does not print source rows,
case text, browser IDs, or file inventories. Failures return exit code 2.

The output is a directory containing `manifest.json` and `state/`. The manifest
records file sizes, SHA-256 digests, and original modification times. Verification
checks exact file membership, digests, database integrity, and schema prerequisites.
It rejects symbolic/hard links, special files, hidden/unknown paths, path traversal,
extra SQLite sidecars, malformed metadata, and unsupported contracts. Limits are
20,000 files, 2 GiB of payload, an 8 MiB manifest, and eight path components.
Individual SQLite operations have a 30-second budget; this is not a total-run SLA.

**No existing destination is overwritten, even an empty directory.** Restore
verifies the snapshot before creating its destination and verifies each file again
while copying. A disk-full or interrupted operation can leave a new incomplete
directory; the original state is untouched. Do not point the application at that
partial restore: the operator must use only a fully verified, new destination
after the recovery checks below. Select a fresh destination for a retry. A backup is complete only when
its final manifest exists and `verify` succeeds; also rehearse opening it in the
application. Do not edit or concurrently write to a snapshot during verification
or restore. Parent directories must be operator-controlled, not shared scratch space.

## Recovery rehearsal and cutover

1. Restore into a new directory while the live service stays stopped. Keep the
   original and backup intact so cutover is reversible.
2. Supply the **same `FLASK_SECRET_KEY`** through your secret manager/environment.
   Existing browser cookies are signed with it. The same `DATA_PRISM_API_KEY`
   is also required to retain the existing monitoring API scope. Secrets are not
   part of this backup; preserve them separately and never commit them.
3. Start the matching application revision with `DATA_PRISM_STATE_DIR` set to the
   restored directory, initially inaccessible to other users. Keep the browser
   hostname/cookie context unchanged when verifying existing access; a different
   domain does not automatically receive the old cookie.
4. Check `/healthz`, `/readyz`, History, an unexpired dashboard, a decision case,
   monitoring history, and `pilot_report.py` on the restored job database. Confirm
   a separate browser cannot read an existing case. An expired source dashboard
   may be unavailable while its longer-lived case remains readable.
5. Original modification times and database timestamps are preserved. Normal
   retention cleanup applies on the next application request, so restoration
   must not be used to extend the life of expired uploads or measurements.
6. Apply any deletion/withdrawal requests made **after** the snapshot before
   reopening access. An older backup can otherwise reintroduce removed data.
   This tool has no external deletion ledger and does not erase older backups
   when a user deletes pilot measurement.
7. Record the snapshot age (potential lost work), measured recovery duration,
   revision, verification results, and responsible operator. Resume traffic only
   after acceptance; on failure, return to the intact original state/configuration.

Automated regression coverage uses synthetic state: completed jobs, decision
evidence, pilot counts, a drift alert, profiles, sessions, and a retained upload.
It tests committed WAL content, corruption, overwrite refusal, unsafe paths,
partial-copy failure, permissions, and a fresh application process that accepts
the original signed cookie. Expired uploads are cleaned without resetting their
age, and a changed signing key cannot access the old case. These tests are not
a rehearsal of your real Render disk, credentials, or backup infrastructure.

## Security, retention, and hosting limits

- Full snapshots contain uploaded rows, prompts, decision text, and pseudonymous
  measurement. They are sensitive operational data, unlike the aggregate pilot
  report. The tools do not upload them anywhere.
- New directories use POSIX mode `0700` and files `0600`; these permissions are
  not encryption. Store copies on an encrypted, access-controlled device or
  approved storage service. SHA-256 detects corruption, not authenticity: an
  attacker who replaces both payload and manifest can recompute hashes. Restore
  only trusted backups and matching application code.
- Set an off-host backup schedule, retention limit, deletion procedure, and
  restore-drill cadence with the data owner. None is enabled automatically.
  Avoid backups inside a public web directory or repository. Conventional
  `backups/`, `runtime-state/`, and `runtime-restored*/` directories are excluded
  from Git and Docker context, but that is not a general data-loss-prevention rule.
- A persistent disk protects state only under its configured mount path. Render
  documents [disk persistence and constraints](https://render.com/docs/disks).
  It does not turn a snapshot on the same disk into an off-host backup or provide
  multi-instance consistency. Attaching paid infrastructure requires a separate
  budget decision; this change leaves `render.yaml` on the free plan.
- The implementation fsyncs copied files and the completion marker, but it makes
  no broader crash- or power-loss-durability guarantee for the host or filesystem.
- The next release gate is **configured durable storage plus a real recovery
  rehearsal**, not merely merging these CLI tools. Account recovery and team
  permissions remain separate work.
