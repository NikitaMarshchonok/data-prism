import os
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from vibedash.exporter import (
    cleanup_expired_artifacts,
    load_session_data,
    save_export,
    save_session_data,
)


class VibeDashSessionStorageTests(unittest.TestCase):
    def test_scoped_session_round_trip_rejects_foreign_and_malformed_owners(self):
        session_id = str(uuid.uuid4())
        owner = "a" * 32
        stranger = "b" * 32

        with TemporaryDirectory() as state_directory:
            self.assertTrue(
                save_session_data(
                    session_id,
                    {"dashboard": "private"},
                    state_directory,
                    owner_id=owner,
                )
            )
            self.assertEqual(
                load_session_data(
                    session_id,
                    state_directory,
                    owner_id=owner,
                )["analysis_scope_id"],
                owner,
            )
            self.assertIsNone(load_session_data(session_id, state_directory))
            self.assertFalse(
                save_session_data(
                    str(uuid.uuid4()),
                    {"dashboard": "unowned"},
                    state_directory,
                )
            )
            self.assertIsNone(
                load_session_data(
                    session_id,
                    state_directory,
                    owner_id=stranger,
                )
            )
            self.assertIsNone(
                load_session_data(
                    session_id,
                    state_directory,
                    owner_id="not-a-scope",
                )
            )

            session_path = (
                Path(state_directory) / f"{session_id}.json"
            )
            session_path.write_text('{"dashboard": "ownerless"}', encoding="utf-8")
            self.assertIsNone(
                load_session_data(
                    session_id,
                    state_directory,
                    owner_id=owner,
                )
            )

    def test_scoped_save_does_not_follow_predictable_temp_symlink(self):
        session_id = str(uuid.uuid4())
        owner = "a" * 32

        with TemporaryDirectory() as state_directory:
            state_path = Path(state_directory)
            outside = state_path / "outside.json"
            outside.write_text('{"must": "remain"}', encoding="utf-8")
            temporary_path = state_path / f"{session_id}.json.tmp"
            temporary_path.symlink_to(outside)

            self.assertTrue(
                save_session_data(
                    session_id,
                    {"dashboard": "private"},
                    state_directory,
                    owner_id=owner,
                )
            )
            self.assertEqual(
                outside.read_text(encoding="utf-8"),
                '{"must": "remain"}',
            )
            self.assertTrue(temporary_path.is_symlink())
            self.assertEqual(
                load_session_data(
                    session_id,
                    state_directory,
                    owner_id=owner,
                )["dashboard"],
                "private",
            )

    def test_scoped_save_does_not_follow_replaced_sessions_directory_symlink(self):
        session_id = str(uuid.uuid4())
        owner = "a" * 32

        with TemporaryDirectory() as state_directory:
            state_path = Path(state_directory)
            real_sessions = state_path / "real-sessions"
            real_sessions.mkdir()
            sessions_link = state_path / "sessions"
            sessions_link.symlink_to(real_sessions, target_is_directory=True)

            self.assertFalse(
                save_session_data(
                    session_id,
                    {"dashboard": "private"},
                    sessions_link,
                    owner_id=owner,
                )
            )
            self.assertEqual(list(real_sessions.iterdir()), [])
            self.assertTrue(sessions_link.is_symlink())

    def test_scoped_load_does_not_follow_final_symlink_after_check(self):
        """A swapped session link must not be read outside the state dir."""
        session_id = str(uuid.uuid4())
        owner = "a" * 32

        with TemporaryDirectory() as state_directory:
            state_path = Path(state_directory)
            outside = state_path / "outside.json"
            outside.write_text(
                '{"dashboard": "private", "analysis_scope_id": "' + owner + '"}',
                encoding="utf-8",
            )
            session_path = state_path / f"{session_id}.json"
            session_path.symlink_to(outside)

            # Simulate an attacker winning the check/open replacement window
            # in the old pathname-based implementation. The descriptor-based
            # reader must still reject the link itself.
            with patch.object(Path, "is_symlink", return_value=False), patch.object(
                Path, "is_file", return_value=True
            ):
                self.assertIsNone(
                    load_session_data(session_id, state_directory, owner_id=owner)
                )

    def test_runtime_state_directory_is_used_for_session_round_trip(self):
        session_id = str(uuid.uuid4())
        owner = "a" * 32
        session_data = {
            "created_at": pd.Timestamp("2026-09-09T00:00:00Z"),
            "row_count": np.int64(360),
            "values": np.array([1.0, 2.0, 3.0]),
        }

        with TemporaryDirectory() as state_directory:
            with patch.dict(
                os.environ,
                {"DATA_PRISM_STATE_DIR": state_directory},
            ):
                self.assertTrue(
                    save_session_data(session_id, session_data, owner_id=owner)
                )
                stored_path = (
                    Path(state_directory)
                    / "sessions"
                    / "vibedash"
                    / f"{session_id}.json"
                )
                self.assertTrue(stored_path.is_file())
                loaded = load_session_data(session_id, owner_id=owner)

        self.assertEqual(loaded["created_at"], "2026-09-09T00:00:00+00:00")
        self.assertEqual(loaded["row_count"], 360)
        self.assertEqual(loaded["values"], [1.0, 2.0, 3.0])

    def test_invalid_session_identifier_is_rejected(self):
        with TemporaryDirectory() as state_directory:
            owner = "a" * 32
            self.assertIsNone(
                load_session_data(
                    "../outside", state_directory, owner_id=owner
                )
            )
            self.assertFalse(
                save_session_data(
                    "../outside", {}, state_directory, owner_id=owner
                )
            )

    def test_runtime_state_directory_is_used_for_export(self):
        session_id = str(uuid.uuid4())

        with TemporaryDirectory() as state_directory:
            with patch.dict(
                os.environ,
                {"DATA_PRISM_STATE_DIR": state_directory},
            ):
                export_path = Path(save_export("<h1>Dashboard</h1>", session_id))

            self.assertTrue(export_path.is_file())
            self.assertEqual(
                export_path.parent,
                Path(state_directory) / "exports" / "vibedash",
            )
            self.assertEqual(export_path.read_text(), "<h1>Dashboard</h1>")

    def test_export_replace_does_not_follow_existing_destination_symlink(self):
        session_id = str(uuid.uuid4())
        fixed_now = datetime(2026, 9, 22, 1, 2, 3)

        with TemporaryDirectory() as state_directory:
            exports_dir = Path(state_directory) / "exports"
            exports_dir.mkdir()
            destination = exports_dir / (
                f"vibedash_export_{session_id}_20260922_010203.html"
            )
            outside = Path(state_directory) / "outside.html"
            outside.write_text("must remain", encoding="utf-8")
            destination.symlink_to(outside)

            with patch("vibedash.exporter.datetime") as mocked_datetime:
                mocked_datetime.now.return_value = fixed_now
                saved = Path(save_export("private", session_id, exports_dir))

            self.assertEqual(saved, destination)
            self.assertEqual(saved.read_text(encoding="utf-8"), "private")
            self.assertEqual(outside.read_text(encoding="utf-8"), "must remain")

    def test_export_does_not_follow_replaced_export_directory_symlink(self):
        session_id = str(uuid.uuid4())

        with TemporaryDirectory() as state_directory:
            state_path = Path(state_directory)
            real_exports = state_path / "real-exports"
            real_exports.mkdir()
            exports_link = state_path / "exports"
            exports_link.symlink_to(real_exports, target_is_directory=True)

            with self.assertRaises(OSError):
                save_export("private", session_id, exports_link)

            self.assertEqual(list(real_exports.iterdir()), [])
            self.assertTrue(exports_link.is_symlink())

    def test_cleanup_removes_only_expired_vibedash_artifacts(self):
        now = 1_800_000_000.0
        expired_timestamp = now - (25 * 60 * 60)
        fresh_timestamp = now - (2 * 60 * 60)

        with TemporaryDirectory() as state_directory:
            state_path = Path(state_directory)
            sessions_dir = state_path / "sessions"
            uploads_dir = state_path / "uploads"
            exports_dir = state_path / "exports"
            for directory in (sessions_dir, uploads_dir, exports_dir):
                directory.mkdir()

            expired_session = sessions_dir / f"{uuid.uuid4()}.json"
            fresh_session = sessions_dir / f"{uuid.uuid4()}.json"
            expired_upload = uploads_dir / f"vibedash-{'a' * 32}.csv"
            fresh_upload = uploads_dir / f"vibedash-{'b' * 32}.csv"
            expired_export = (
                exports_dir
                / f"vibedash_export_{uuid.uuid4()}_20260910_010203.html"
            )
            unrelated_file = uploads_dir / "customer-owned.csv"

            for path in (
                expired_session,
                fresh_session,
                expired_upload,
                fresh_upload,
                expired_export,
                unrelated_file,
            ):
                path.write_text("test", encoding="utf-8")
            for path in (expired_session, expired_upload, expired_export, unrelated_file):
                os.utime(path, (expired_timestamp, expired_timestamp))
            for path in (fresh_session, fresh_upload):
                os.utime(path, (fresh_timestamp, fresh_timestamp))

            result = cleanup_expired_artifacts(
                uploads_dir,
                24,
                sessions_dir=sessions_dir,
                exports_dir=exports_dir,
                now=now,
            )

            self.assertEqual(
                result,
                {
                    "removed_sessions": 1,
                    "removed_uploads": 1,
                    "removed_exports": 1,
                    "errors": 0,
                },
            )
            self.assertFalse(expired_session.exists())
            self.assertFalse(expired_upload.exists())
            self.assertFalse(expired_export.exists())
            self.assertTrue(fresh_session.exists())
            self.assertTrue(fresh_upload.exists())
            self.assertTrue(unrelated_file.exists())

    def test_cleanup_rejects_invalid_retention_and_skips_symlinks(self):
        with TemporaryDirectory() as state_directory:
            state_path = Path(state_directory)
            uploads_dir = state_path / "uploads"
            uploads_dir.mkdir()
            target = state_path / "outside.csv"
            target.write_text("must remain", encoding="utf-8")
            symlink = uploads_dir / f"vibedash-{'c' * 32}.csv"
            symlink.symlink_to(target)

            with self.assertRaises(ValueError):
                cleanup_expired_artifacts(uploads_dir, 0)
            result = cleanup_expired_artifacts(
                uploads_dir,
                24,
                sessions_dir=state_path / "sessions",
                exports_dir=state_path / "exports",
                now=1_800_000_000.0,
            )

            self.assertEqual(result["removed_uploads"], 0)
            self.assertTrue(target.exists())
            self.assertTrue(symlink.is_symlink())

    def test_cleanup_does_not_follow_an_artifact_directory_symlink(self):
        with TemporaryDirectory() as state_directory:
            state_path = Path(state_directory)
            uploads_dir = state_path / "uploads"
            outside_dir = state_path / "outside"
            outside_dir.mkdir()
            outside = outside_dir / f"vibedash-{'d' * 32}.csv"
            outside.write_text("must remain", encoding="utf-8")
            os.utime(outside, (1_700_000_000.0, 1_700_000_000.0))
            uploads_dir.symlink_to(outside_dir, target_is_directory=True)

            result = cleanup_expired_artifacts(
                uploads_dir,
                24,
                sessions_dir=state_path / "sessions",
                exports_dir=state_path / "exports",
                now=1_800_000_000.0,
            )

            self.assertEqual(result["removed_uploads"], 0)
            self.assertGreaterEqual(result["errors"], 1)
            self.assertTrue(outside.exists())
            self.assertTrue(uploads_dir.is_symlink())


if __name__ == "__main__":
    unittest.main()
