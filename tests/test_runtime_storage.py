import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.runtime_storage import evaluate_runtime_storage_contract
from src.runtime_state import initialize_runtime_state


class RuntimeStorageContractTests(unittest.TestCase):
    def test_demo_ephemeral_state_is_ready_with_an_explicit_warning(self):
        contract = evaluate_runtime_storage_contract(
            deployment_profile="demo",
            state_durability="ephemeral",
            state_directory_configured=True,
        )

        self.assertEqual(contract["issues"], [])
        self.assertEqual(contract["deployment_profile"], "demo")
        self.assertEqual(contract["state_durability"], "ephemeral")
        self.assertFalse(contract["production_state_required"])
        self.assertFalse(contract["production_state_satisfied"])
        self.assertIn("can be lost", contract["warnings"][0])

    def test_production_rejects_ephemeral_state(self):
        contract = evaluate_runtime_storage_contract(
            deployment_profile="production",
            state_durability="ephemeral",
            state_directory_configured=True,
        )

        self.assertIn(
            "Production deployment requires DATA_PRISM_STATE_DURABILITY=persistent.",
            contract["issues"],
        )
        self.assertTrue(contract["production_state_required"])
        self.assertFalse(contract["production_state_satisfied"])

    def test_persistent_state_requires_an_explicit_state_directory(self):
        contract = evaluate_runtime_storage_contract(
            deployment_profile="development",
            state_durability="persistent",
            state_directory_configured=False,
        )

        self.assertEqual(
            contract["issues"],
            ["Persistent state requires an explicit DATA_PRISM_STATE_DIR."],
        )

    def test_production_accepts_an_explicit_persistent_declaration(self):
        with TemporaryDirectory() as directory:
            identity = initialize_runtime_state(Path(directory))
            contract = evaluate_runtime_storage_contract(
                deployment_profile=" production ",
                state_durability=" PERSISTENT ",
                state_directory_configured=True,
                state_directory=directory,
                expected_state_id=identity["state_id"],
            )

        self.assertEqual(contract["issues"], [])
        self.assertTrue(contract["production_state_satisfied"])
        self.assertEqual(contract["assurance"], "continuity-verified")
        self.assertTrue(contract["state_identity"]["verified"])
        self.assertIn("operator-declared", contract["warnings"][0])

    def test_production_rejects_missing_or_mismatched_state_identity(self):
        with TemporaryDirectory() as directory:
            identity = initialize_runtime_state(Path(directory))
            missing_expected = evaluate_runtime_storage_contract(
                deployment_profile="production",
                state_durability="persistent",
                state_directory_configured=True,
                state_directory=directory,
                expected_state_id=None,
            )
            mismatch = evaluate_runtime_storage_contract(
                deployment_profile="production",
                state_durability="persistent",
                state_directory_configured=True,
                state_directory=directory,
                expected_state_id="0" * 64,
            )

        self.assertFalse(missing_expected["production_state_satisfied"])
        self.assertIn("DATA_PRISM_EXPECTED_STATE_ID", missing_expected["issues"][0])
        self.assertFalse(mismatch["production_state_satisfied"])
        self.assertIn("does not match", mismatch["issues"][0])
        self.assertNotIn(identity["state_id"], str(mismatch))

    def test_unknown_declarations_fail_closed_without_echoing_input(self):
        contract = evaluate_runtime_storage_contract(
            deployment_profile="customer-secret-profile",
            state_durability="network-share-token",
            state_directory_configured=True,
        )

        self.assertEqual(contract["deployment_profile"], "invalid")
        self.assertEqual(contract["state_durability"], "invalid")
        self.assertEqual(len(contract["issues"]), 2)
        self.assertNotIn("customer-secret-profile", str(contract))
        self.assertNotIn("network-share-token", str(contract))


if __name__ == "__main__":
    unittest.main()
