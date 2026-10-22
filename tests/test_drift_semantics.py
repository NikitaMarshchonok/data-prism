"""Confirmed baseline invariants, independent of inferred names/cardinality."""

import copy
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data_drift import (
    compare_to_baseline, create_baseline_profile, load_baseline_profile,
    save_baseline_profile,
)
from src.semantic_contract import infer_semantic_contract


class ConfirmedDriftSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "code": [1, 2] * 50, "value": list(range(100)),
            "customer_id": [f"private-row-{i}" for i in range(100)],
        })
        self.contract = infer_semantic_contract(self.data, confirmed=True)
        self.contract["columns"][0].update(role="category", unit="none", exclude_from_predictors=True)
        self.profile = create_baseline_profile(self.data, semantic_contract=self.contract)

    def test_v3_retains_schema_fingerprint_not_raw_identifiers(self):
        self.assertEqual(self.profile["profile_version"], 3)
        self.assertTrue(self.profile["semantic_contract"]["confirmed"])
        self.assertNotIn("private-row-", json.dumps(self.profile))
        with TemporaryDirectory() as root:
            path = Path(root) / "baseline.json"
            save_baseline_profile(self.profile, path)
            self.assertEqual(load_baseline_profile(path), self.profile)
            path.write_text(json.dumps(self.profile, sort_keys=True), encoding="utf-8")
            self.assertEqual(load_baseline_profile(path), self.profile)

    def test_current_cardinality_does_not_reinfer_confirmed_category(self):
        current = self.data.copy()
        current["code"] = range(1000, 1100)
        report = compare_to_baseline(current, self.profile)
        feature = next(x for x in report["feature_drift"] if x["feature"] == "code")
        self.assertEqual(feature["metric"], "TV distance")
        self.assertEqual(feature["score"], 1.0)
        self.assertEqual(report["schema_changes"]["type_changes"], [])
        self.assertEqual(report["semantics"]["current"], "inherited_from_baseline")

    def test_low_cardinality_measure_stays_numeric_despite_name(self):
        data = pd.DataFrame({"status_code": [1, 2] * 50})
        contract = infer_semantic_contract(data, confirmed=True)
        contract["columns"][0].update(role="measure", unit="number")
        profile = create_baseline_profile(data, semantic_contract=contract)
        report = compare_to_baseline(data, profile)
        self.assertEqual(report["feature_drift"][0]["metric"], "PSI")
        self.assertEqual(report["status"], "stable")

    def test_current_confirmation_requires_matching_units_and_roles(self):
        for change in ({"unit": "currency"}, {"role": "identifier", "unit": "none", "exclude_from_predictors": True}):
            contract = copy.deepcopy(self.contract)
            contract["columns"][1].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "differ from the baseline"):
                compare_to_baseline(self.data, self.profile, semantic_contract=contract)
        report = compare_to_baseline(self.data, self.profile, semantic_contract=self.contract)
        self.assertEqual(report["semantics"]["current"], "confirmed_for_batch")
        self.assertEqual(report["semantics"]["current_sha256"], self.profile["semantic_contract"]["sha256"])

    def test_incompatible_measure_rejected_but_missingness_still_monitored(self):
        current = self.data.copy()
        current["value"] = "not-numeric"
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            compare_to_baseline(current, self.profile)
        current["value"] = None
        report = compare_to_baseline(current, self.profile)
        self.assertEqual(report["status"], "critical")
        self.assertEqual(next(x for x in report["feature_drift"] if x["feature"] == "value")["missing_rate_delta"], 1.0)

    def test_count_rejects_fractional_batch(self):
        contract = copy.deepcopy(self.contract)
        contract["columns"][1]["unit"] = "count"
        profile = create_baseline_profile(self.data, semantic_contract=contract)
        current = self.data.copy()
        current["value"] = current["value"] + 0.25
        with self.assertRaisesRegex(ValueError, "integer"):
            compare_to_baseline(current, profile)
        current["value"] = None
        self.assertEqual(compare_to_baseline(current, profile)["status"], "critical")

    def test_missing_and_new_columns_remain_explicit(self):
        current = self.data.drop(columns="code").assign(new_column=1)
        report = compare_to_baseline(current, self.profile)
        self.assertEqual(report["schema_changes"]["missing_columns"], ["code"])
        self.assertEqual(report["schema_changes"]["new_columns"], ["new_column"])
        self.assertEqual(report["status"], "critical")

    def test_temporal_role_stays_pinned_when_dates_stop_parsing(self):
        data = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=30).astype(str)})
        profile = create_baseline_profile(data, semantic_contract=infer_semantic_contract(data, confirmed=True))
        future = pd.DataFrame({"Date": pd.date_range("2027-01-01", periods=30).astype(str)})
        self.assertEqual(compare_to_baseline(future, profile)["status"], "stable")
        broken = pd.DataFrame({"Date": ["not-a-date"] * 30})
        report = compare_to_baseline(broken, profile)
        self.assertEqual(report["schema_changes"]["type_changes"], [])
        self.assertEqual(report["feature_drift"][0]["metric"], "parseable-rate delta")
        self.assertEqual(report["feature_drift"][0]["score"], 1.0)
        self.assertEqual(report["status"], "critical")

    def test_profile_metadata_tampering_rejected_before_comparison(self):
        mutations = [
            lambda p: p["semantic_contract"].update(sha256="bad"),
            lambda p: p["semantic_contract"].update(confirmed=False),
            lambda p: p["columns"]["value"].update(unit="currency"),
            lambda p: p["columns"]["value"].update(feature_type="identifier"),
            lambda p: p.update(column_count=999),
            lambda p: p["columns"].pop("value"),
            lambda p: p.update(profile_version=[]),
        ]
        for mutate in mutations:
            profile = copy.deepcopy(self.profile)
            mutate(profile)
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                compare_to_baseline(self.data, profile)

    def test_override_cannot_bypass_retained_semantics(self):
        with self.assertRaisesRegex(ValueError, "overridden"):
            compare_to_baseline(self.data, self.profile, column_profiles={})
        with self.assertRaisesRegex(ValueError, "simultaneous"):
            create_baseline_profile(self.data, semantic_contract=self.contract, column_profiles={})

    def test_legacy_profiles_readable_and_honestly_labeled(self):
        for version in (1, 2):
            profile = create_baseline_profile(self.data)
            profile["profile_version"] = version
            report = compare_to_baseline(self.data, profile)
            self.assertEqual(report["semantics"]["baseline"], "legacy_without_retained_contract")
            with self.assertRaisesRegex(ValueError, "recreate"):
                compare_to_baseline(self.data, profile, semantic_contract=self.contract)

    def test_confirmed_baselines_do_not_silently_truncate_columns(self):
        data = pd.DataFrame({f"v{i}": range(30) for i in range(101)})
        with self.assertRaisesRegex(ValueError, "at most 100"):
            create_baseline_profile(data, semantic_contract=infer_semantic_contract(data, confirmed=True))

    def test_concurrent_profile_writers_use_separate_temporary_files(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "baseline.json"
            profiles = [{**self.profile, "dataset_name": f"writer-{i}"} for i in range(16)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda profile: save_baseline_profile(profile, path), profiles))
            self.assertIn(load_baseline_profile(path), profiles)
            self.assertEqual(list(Path(root).iterdir()), [path])
