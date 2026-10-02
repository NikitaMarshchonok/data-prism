import unittest

from src.runtime_storage import evaluate_runtime_storage_contract


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
        contract = evaluate_runtime_storage_contract(
            deployment_profile=" production ",
            state_durability=" PERSISTENT ",
            state_directory_configured=True,
        )

        self.assertEqual(contract["issues"], [])
        self.assertTrue(contract["production_state_satisfied"])
        self.assertEqual(contract["assurance"], "operator-declared")
        self.assertIn("operator-declared", contract["warnings"][0])

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
