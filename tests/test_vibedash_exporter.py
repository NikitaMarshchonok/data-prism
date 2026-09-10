import os
import unittest
import uuid
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
    def test_runtime_state_directory_is_used_for_session_round_trip(self):
        session_id = str(uuid.uuid4())
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
                self.assertTrue(save_session_data(session_id, session_data))
                stored_path = (
                    Path(state_directory)
                    / "sessions"
                    / "vibedash"
                    / f"{session_id}.json"
                )
                self.assertTrue(stored_path.is_file())
                loaded = load_session_data(session_id)

        self.assertEqual(loaded["created_at"], "2026-09-09T00:00:00+00:00")
        self.assertEqual(loaded["row_count"], 360)
        self.assertEqual(loaded["values"], [1.0, 2.0, 3.0])

    def test_invalid_session_identifier_is_rejected(self):
        with TemporaryDirectory() as state_directory:
            self.assertIsNone(load_session_data("../outside", state_directory))
            self.assertFalse(
                save_session_data("../outside", {}, state_directory)
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


if __name__ == "__main__":
    unittest.main()
