import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from src.runtime_state import (
    IDENTITY_RELATIVE_PATH,
    RuntimeStateError,
    initialize_runtime_state,
    inspect_runtime_state,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class RuntimeStateIdentityTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.state = Path(self.directory.name) / "runtime"
        self.state.mkdir()

    def test_initialization_is_stable_and_private(self):
        first = initialize_runtime_state(self.state)
        second = initialize_runtime_state(self.state)

        self.assertEqual(first, second)
        self.assertRegex(first["state_id"], r"^[0-9a-f]{64}$")
        self.assertEqual(first["contract"], "runtime-state-identity-v1")
        marker = self.state / IDENTITY_RELATIVE_PATH
        self.assertEqual(marker.stat().st_mode & 0o777, 0o600)
        self.assertEqual(marker.parent.stat().st_mode & 0o777, 0o700)

    def test_matching_identity_is_verified_without_returning_secret(self):
        identity = initialize_runtime_state(self.state)

        result = inspect_runtime_state(self.state, identity["state_id"])

        self.assertTrue(result["verified"])
        self.assertEqual(result["status"], "verified")
        self.assertRegex(result["fingerprint"], r"^[0-9a-f]{12}$")
        self.assertNotIn(identity["state_id"], json.dumps(result))

    def test_mismatch_and_invalid_expected_identity_fail_closed(self):
        initialize_runtime_state(self.state)

        mismatch = inspect_runtime_state(self.state, "0" * 64)
        invalid = inspect_runtime_state(self.state, "operator-secret")

        self.assertFalse(mismatch["verified"])
        self.assertEqual(mismatch["status"], "identity_mismatch")
        self.assertFalse(invalid["verified"])
        self.assertEqual(invalid["status"], "invalid_expected_identity")
        self.assertNotIn("operator-secret", json.dumps(invalid))

    def test_missing_malformed_and_oversized_identity_fail_closed(self):
        missing = inspect_runtime_state(self.state, "1" * 64)
        self.assertEqual(missing["status"], "missing_or_invalid_identity")

        marker = self.state / IDENTITY_RELATIVE_PATH
        marker.parent.mkdir()
        marker.write_text("not-json", encoding="utf-8")
        self.assertEqual(
            inspect_runtime_state(self.state, "1" * 64)["status"],
            "missing_or_invalid_identity",
        )
        marker.write_bytes(b"x" * 4097)
        self.assertEqual(
            inspect_runtime_state(self.state, "1" * 64)["status"],
            "missing_or_invalid_identity",
        )

    def test_identity_symlinks_are_rejected(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symbolic links are unavailable")
        outside = Path(self.directory.name) / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        marker = self.state / IDENTITY_RELATIVE_PATH
        marker.parent.mkdir()
        marker.symlink_to(outside)

        with self.assertRaises(RuntimeStateError):
            initialize_runtime_state(self.state)
        self.assertEqual(
            inspect_runtime_state(self.state, "1" * 64)["status"],
            "missing_or_invalid_identity",
        )

    def test_cli_initializes_then_verifies_without_disclosing_id(self):
        initialized = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "runtime_state.py"),
                "initialize",
                "--state-dir",
                str(self.state),
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        identity = json.loads(initialized.stdout)
        verified = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "runtime_state.py"),
                "verify",
                "--state-dir",
                str(self.state),
                "--expected-state-id",
                identity["state_id"],
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        payload = json.loads(verified.stdout)
        self.assertTrue(payload["verified"])
        self.assertNotIn(identity["state_id"], verified.stdout)

    def test_cli_verify_failure_is_nonzero_and_secret_free(self):
        initialize_runtime_state(self.state)
        supplied = "f" * 64

        result = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "runtime_state.py"),
                "verify",
                "--state-dir",
                str(self.state),
                "--expected-state-id",
                supplied,
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertNotIn(supplied, result.stderr)


if __name__ == "__main__":
    unittest.main()
