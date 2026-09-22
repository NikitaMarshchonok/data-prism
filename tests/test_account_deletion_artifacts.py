import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import vibedash.account_deletion_artifacts as artifacts
from vibedash.account_deletion_artifacts import (
    AccountArtifactCleanupError,
    purge_account_artifacts,
)


def _upload(scope: str) -> str:
    return f"vibedash-{scope}.csv"


def _session_path(directory: Path, session_id: str, owner: str, **extra) -> Path:
    path = directory / f"{session_id}.json"
    payload = {"analysis_scope_id": owner, **extra}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class AccountDeletionArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.uploads = root / "uploads"
        self.sessions = root / "sessions"
        self.exports = root / "exports"
        for directory in (self.uploads, self.sessions, self.exports):
            directory.mkdir()
        self.owner = "a" * 32
        self.foreign = "b" * 32

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_upload(self, name: str):
        (self.uploads / name).write_text("data", encoding="utf-8")

    def _write_export(self, session_id: str):
        path = self.exports / (
            f"vibedash_export_{session_id}_20260922_010203.html"
        )
        path.write_text("html", encoding="utf-8")
        return path

    def test_deletes_owned_session_job_upload_and_export_but_preserves_foreign(self):
        owned_session = str(uuid.uuid4())
        foreign_session = str(uuid.uuid4())
        owned_name = _upload("c" * 32)
        job_name = _upload("d" * 32)
        foreign_name = _upload("e" * 32)
        _session_path(
            self.sessions,
            owned_session,
            self.owner,
            stored_filename=owned_name,
        )
        _session_path(
            self.sessions,
            foreign_session,
            self.foreign,
            stored_filename=foreign_name,
        )
        for name in (owned_name, job_name, foreign_name):
            self._write_upload(name)
        owned_export = self._write_export(owned_session)
        foreign_export = self._write_export(foreign_session)

        result = purge_account_artifacts(
            self.owner,
            [
                {
                    "scope_id": self.owner,
                    "session_id": None,
                    "payload": {"stored_filename": job_name},
                },
                {
                    "scope_id": self.foreign,
                    "payload": {"stored_filename": foreign_name},
                },
            ],
            upload_dir=self.uploads,
            sessions_dir=self.sessions,
            exports_dir=self.exports,
        )

        self.assertEqual(
            result,
            {"removed_sessions": 1, "removed_uploads": 2, "removed_exports": 1},
        )
        self.assertFalse((self.sessions / f"{owned_session}.json").exists())
        self.assertTrue((self.sessions / f"{foreign_session}.json").exists())
        self.assertFalse((self.uploads / owned_name).exists())
        self.assertFalse((self.uploads / job_name).exists())
        self.assertTrue((self.uploads / foreign_name).exists())
        self.assertFalse(owned_export.exists())
        self.assertTrue(foreign_export.exists())

    def test_preview_session_without_job_and_comparison_inputs(self):
        preview_session = str(uuid.uuid4())
        preview_name = _upload("c" * 32)
        _session_path(
            self.sessions,
            preview_session,
            self.owner,
            stored_filename=preview_name,
        )
        baseline_name = _upload("d" * 32)
        current_name = _upload("e" * 32)
        for name in (preview_name, baseline_name, current_name):
            self._write_upload(name)

        result = purge_account_artifacts(
            self.owner,
            [
                {
                    "scope_id": self.owner,
                    "payload": {
                        "analysis_kind": "period_comparison",
                        "baseline": {"stored_filename": baseline_name},
                        "current": {"stored_filename": current_name},
                    },
                }
            ],
            upload_dir=self.uploads,
            sessions_dir=self.sessions,
            exports_dir=self.exports,
        )

        self.assertEqual(result["removed_sessions"], 1)
        self.assertEqual(result["removed_uploads"], 3)
        self.assertEqual(list(self.sessions.iterdir()), [])

    def test_foreign_reference_and_explicit_protection_win_over_owned_reference(self):
        session_id = str(uuid.uuid4())
        shared_name = _upload("c" * 32)
        _session_path(
            self.sessions,
            session_id,
            self.owner,
            stored_filename=shared_name,
        )
        self._write_upload(shared_name)

        result = purge_account_artifacts(
            self.owner,
            [{"scope_id": self.owner, "payload": {"stored_filename": shared_name}}],
            upload_dir=self.uploads,
            sessions_dir=self.sessions,
            exports_dir=self.exports,
            protected_upload_names=[shared_name],
        )

        self.assertEqual(result["removed_uploads"], 0)
        self.assertTrue((self.uploads / shared_name).exists())
        self.assertFalse((self.sessions / f"{session_id}.json").exists())

    def test_job_session_reference_must_match_owned_session_before_export_delete(self):
        owned_job_session = str(uuid.uuid4())
        foreign_session = str(uuid.uuid4())
        _session_path(self.sessions, foreign_session, self.foreign)
        foreign_export = self._write_export(foreign_session)

        # A corrupt/contradictory owned job must not let its session_id claim
        # delete an export whose persisted session belongs to another scope.
        with self.assertRaises(AccountArtifactCleanupError):
            purge_account_artifacts(
                self.owner,
                [
                    {
                        "scope_id": self.owner,
                        "session_id": foreign_session,
                        "payload": {},
                    }
                ],
                upload_dir=self.uploads,
                sessions_dir=self.sessions,
                exports_dir=self.exports,
            )
        self.assertTrue(foreign_export.exists())

        # A foreign job reference is protected even when the direct helper is
        # called with a mixed prefiltered list.
        result = purge_account_artifacts(
            self.owner,
            [
                {"scope_id": self.owner, "session_id": owned_job_session, "payload": {}},
                {"scope_id": self.foreign, "session_id": foreign_session, "payload": {}},
            ],
            upload_dir=self.uploads,
            sessions_dir=self.sessions,
            exports_dir=self.exports,
        )
        self.assertEqual(result["removed_exports"], 0)
        self.assertTrue(foreign_export.exists())

    def test_symlinks_and_arbitrary_paths_are_preserved(self):
        session_id = str(uuid.uuid4())
        owned_name = _upload("c" * 32)
        outside_upload = Path(self.tempdir.name) / "outside.csv"
        outside_upload.write_text("must remain", encoding="utf-8")
        (self.uploads / owned_name).symlink_to(outside_upload)
        outside_session = Path(self.tempdir.name) / "outside.json"
        outside_session.write_text("must remain", encoding="utf-8")
        (self.sessions / f"{session_id}.json").symlink_to(outside_session)
        outside_export = Path(self.tempdir.name) / "outside.html"
        outside_export.write_text("must remain", encoding="utf-8")
        export_name = f"vibedash_export_{session_id}_20260922_010203.html"
        (self.exports / export_name).symlink_to(outside_export)
        unrelated = self.uploads / "customer-owned.csv"
        unrelated.write_text("must remain", encoding="utf-8")

        result = purge_account_artifacts(
            self.owner,
            [],
            upload_dir=self.uploads,
            sessions_dir=self.sessions,
            exports_dir=self.exports,
        )

        self.assertEqual(
            result,
            {"removed_sessions": 0, "removed_uploads": 0, "removed_exports": 0},
        )
        self.assertTrue((self.sessions / f"{session_id}.json").is_symlink())
        self.assertTrue((self.uploads / owned_name).is_symlink())
        self.assertEqual(outside_upload.read_text(), "must remain")
        self.assertEqual(outside_session.read_text(), "must remain")
        self.assertEqual(outside_export.read_text(), "must remain")
        self.assertTrue(unrelated.exists())

    def test_session_metadata_is_retained_when_dependent_cleanup_fails(self):
        session_id = str(uuid.uuid4())
        owned_name = _upload("c" * 32)
        _session_path(
            self.sessions,
            session_id,
            self.owner,
            stored_filename=owned_name,
        )
        self._write_upload(owned_name)
        owned_export = self._write_export(session_id)

        original_unlink = artifacts._safe_unlink

        def fail_upload(path, **kwargs):
            if kwargs.get("name") == owned_name:
                raise AccountArtifactCleanupError("synthetic cleanup failure")
            return original_unlink(path, **kwargs)

        with patch.object(artifacts, "_safe_unlink", side_effect=fail_upload):
            with self.assertRaises(AccountArtifactCleanupError):
                purge_account_artifacts(
                    self.owner,
                    [],
                    upload_dir=self.uploads,
                    sessions_dir=self.sessions,
                    exports_dir=self.exports,
                )

        # A retry still has the ownership proof needed to remove the export.
        self.assertTrue((self.sessions / f"{session_id}.json").exists())
        self.assertTrue((self.uploads / owned_name).exists())
        self.assertTrue(owned_export.exists())

        result = purge_account_artifacts(
            self.owner,
            [],
            upload_dir=self.uploads,
            sessions_dir=self.sessions,
            exports_dir=self.exports,
        )
        self.assertEqual(result["removed_sessions"], 1)
        self.assertEqual(result["removed_uploads"], 1)
        self.assertEqual(result["removed_exports"], 1)

    def test_session_directory_swap_after_scan_cannot_redirect_unlink(self):
        session_id = str(uuid.uuid4())
        session_name = f"{session_id}.json"
        _session_path(self.sessions, session_id, self.owner)
        original_directory = self.sessions.with_name("sessions-original")
        swapped = False
        list_directory_names = artifacts._directory_names

        def swap_after_first_scan(directory_fd):
            nonlocal swapped
            names = list_directory_names(directory_fd)
            if not swapped:
                swapped = True
                self.sessions.rename(original_directory)
                self.sessions.mkdir()
                _session_path(self.sessions, session_id, self.owner)
            return names

        with patch.object(
            artifacts,
            "_directory_names",
            side_effect=swap_after_first_scan,
        ):
            result = purge_account_artifacts(
                self.owner,
                [],
                upload_dir=self.uploads,
                sessions_dir=self.sessions,
                exports_dir=self.exports,
            )

        self.assertEqual(result["removed_sessions"], 1)
        self.assertFalse((original_directory / session_name).exists())
        self.assertTrue((self.sessions / session_name).exists())

    def test_malformed_candidate_fails_closed_with_retryable_error(self):
        owned_session = str(uuid.uuid4())
        owned_name = _upload("c" * 32)
        _session_path(
            self.sessions,
            owned_session,
            self.owner,
            stored_filename=owned_name,
        )
        malformed = self.sessions / f"{uuid.uuid4()}.json"
        malformed.write_text("{not json", encoding="utf-8")
        self._write_upload(owned_name)

        with self.assertRaises(AccountArtifactCleanupError) as raised:
            purge_account_artifacts(
                self.owner,
                [],
                upload_dir=self.uploads,
                sessions_dir=self.sessions,
                exports_dir=self.exports,
            )

        self.assertEqual(
            str(raised.exception),
            "Account artifact cleanup could not establish artifact ownership.",
        )
        self.assertTrue((self.sessions / f"{owned_session}.json").exists())
        self.assertTrue((self.uploads / owned_name).exists())
        self.assertTrue(malformed.exists())

    def test_deletion_failure_raises_generic_safe_error(self):
        session_id = str(uuid.uuid4())
        _session_path(self.sessions, session_id, self.owner)
        with patch.object(os, "unlink", side_effect=PermissionError("secret path")):
            with self.assertRaises(AccountArtifactCleanupError) as raised:
                purge_account_artifacts(
                    self.owner,
                    [],
                    upload_dir=self.uploads,
                    sessions_dir=self.sessions,
                    exports_dir=self.exports,
                )
        self.assertEqual(
            str(raised.exception),
            "Account artifact cleanup could not remove a proven artifact.",
        )
        self.assertNotIn(str(self.sessions), str(raised.exception))


if __name__ == "__main__":
    unittest.main()
