import os
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from vibedash.exporter import load_session_data, save_export, save_session_data


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


if __name__ == "__main__":
    unittest.main()
