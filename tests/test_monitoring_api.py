import unittest
import json
import pandas as pd
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import web_app
from src.semantic_contract import infer_semantic_contract


class MonitoringApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.upload_folder = root / "uploads"
        self.baseline_folder = root / "baselines"
        self.database_path = root / "drift.sqlite3"
        self.api_key = "test-monitoring-key-0123456789abcdef"
        self.previous_config = {
            "TESTING": web_app.app.config.get("TESTING"),
            "UPLOAD_FOLDER": web_app.app.config["UPLOAD_FOLDER"],
            "BASELINE_FOLDER": web_app.app.config["BASELINE_FOLDER"],
            "DRIFT_STORE_PATH": web_app.app.config["DRIFT_STORE_PATH"],
            "DRIFT_HISTORY_RETENTION": web_app.app.config["DRIFT_HISTORY_RETENTION"],
            "DATA_PRISM_API_KEY": web_app.app.config["DATA_PRISM_API_KEY"],
        }
        web_app.app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=str(self.upload_folder),
            BASELINE_FOLDER=str(self.baseline_folder),
            DRIFT_STORE_PATH=str(self.database_path),
            DRIFT_HISTORY_RETENTION=100,
            DATA_PRISM_API_KEY=self.api_key,
        )

    def tearDown(self):
        web_app.app.config.update(self.previous_config)
        self.temporary_directory.cleanup()

    def test_api_is_disabled_without_key_and_rejects_wrong_key(self):
        with web_app.app.test_client() as client:
            web_app.app.config["DATA_PRISM_API_KEY"] = None
            disabled = client.get("/api/v1/drift/runs")
            web_app.app.config["DATA_PRISM_API_KEY"] = "short-key"
            weak_key = client.get(
                "/api/v1/drift/runs",
                headers={"X-API-Key": "short-key"},
            )
            web_app.app.config["DATA_PRISM_API_KEY"] = self.api_key
            unauthorized = client.get(
                "/api/v1/drift/runs",
                headers={"X-API-Key": "wrong-key"},
            )
            bearer = client.get(
                "/api/v1/drift/runs",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )

        self.assertEqual(disabled.status_code, 503)
        self.assertEqual(disabled.get_json()["error"]["code"], "api_not_configured")
        self.assertEqual(weak_key.status_code, 503)
        self.assertEqual(weak_key.get_json()["error"]["code"], "api_key_too_short")
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(bearer.status_code, 200)

    def test_creates_aggregate_baseline_and_removes_raw_upload(self):
        with web_app.app.test_client() as client:
            response = self._create_baseline(client)

        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertFalse(payload["stores_raw_rows"])
        self.assertEqual(payload["row_count"], 100)
        self.assertEqual(list(self.upload_folder.iterdir()), [])
        profiles = list(self.baseline_folder.rglob("*.json"))
        self.assertEqual(len(profiles), 1)

    def test_check_is_deduplicated_and_exposed_through_history_api(self):
        with web_app.app.test_client() as client:
            baseline_id = self._create_baseline(client).get_json()["baseline_id"]
            first = self._create_check(client, baseline_id)
            duplicate = self._create_check(client, baseline_id)
            runs = client.get("/api/v1/drift/runs", headers=self._headers())
            alerts = client.get("/api/v1/drift/alerts", headers=self._headers())
            run_id = first.get_json()["run"]["run_id"]
            full_run = client.get(
                f"/api/v1/drift/runs/{run_id}",
                headers=self._headers(),
            )

        self.assertEqual(first.status_code, 201)
        self.assertEqual(first.get_json()["report"]["status"], "critical")
        self.assertTrue(first.get_json()["run"]["alert_created"])
        self.assertEqual(duplicate.status_code, 200)
        self.assertFalse(duplicate.get_json()["run"]["created"])
        self.assertEqual(len(runs.get_json()["runs"]), 1)
        self.assertEqual(len(alerts.get_json()["alerts"]), 1)
        self.assertEqual(full_run.get_json()["run"]["report"]["status"], "critical")
        self.assertEqual(list(self.upload_folder.iterdir()), [])

    def test_alert_can_be_acknowledged_through_api(self):
        with web_app.app.test_client() as client:
            baseline_id = self._create_baseline(client).get_json()["baseline_id"]
            self._create_check(client, baseline_id)
            alerts = client.get(
                "/api/v1/drift/alerts",
                headers=self._headers(),
            ).get_json()["alerts"]
            response = client.post(
                f"/api/v1/drift/alerts/{alerts[0]['id']}/acknowledge",
                headers=self._headers(),
            )
            remaining = client.get(
                "/api/v1/drift/alerts",
                headers=self._headers(),
            ).get_json()["alerts"]

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["acknowledged"])
        self.assertEqual(remaining, [])

    def test_api_key_rotation_isolates_existing_baselines(self):
        with web_app.app.test_client() as client:
            baseline_id = self._create_baseline(client).get_json()["baseline_id"]
            rotated_key = "rotated-monitoring-key-0123456789abcd"
            web_app.app.config["DATA_PRISM_API_KEY"] = rotated_key
            response = client.post(
                "/api/v1/drift/checks",
                data={"baseline_id": baseline_id},
                headers={"X-API-Key": rotated_key},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"]["code"], "baseline_not_found")

    def test_proposal_is_schema_only_unconfirmed_and_requires_authentication(self):
        with web_app.app.test_client() as client:
            denied = client.post("/api/v1/drift/semantics")
            response = client.post("/api/v1/drift/semantics", headers=self._headers(), data={
                "datafile": (BytesIO(_csv_values(range(100))), "baseline.csv"),
            })
            contract = response.get_json()["semantic_contract"]
            rejected = client.post("/api/v1/drift/baselines", headers=self._headers(), data={
                "datafile": (BytesIO(_csv_values(range(100))), "baseline.csv"),
                "semantic_contract": json.dumps(contract),
            })
            contract["confirmed"] = True
            accepted = client.post("/api/v1/drift/baselines", headers=self._headers(), data={
                "datafile": (BytesIO(_csv_values(range(100))), "baseline.csv"),
                "semantic_contract": json.dumps(contract),
            })
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["semantic_contract"]["confirmed"])
        self.assertEqual(set(contract["columns"][0]), {
            "name", "role", "unit", "is_target", "is_time", "exclude_from_predictors",
        })
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(accepted.status_code, 201)
        self.assertEqual(accepted.get_json()["profile_version"], 3)
        self.assertEqual(len(accepted.get_json()["semantic_sha256"]), 64)
        self.assertEqual(list(self.upload_folder.iterdir()), [])

    def test_missing_malformed_or_deep_contract_fails_without_retaining_upload(self):
        with web_app.app.test_client() as client:
            for value in (None, "[]", "{" , "[" * 2000 + "]" * 2000):
                data = {"datafile": (BytesIO(_csv_values(range(100))), "baseline.csv")}
                if value is not None:
                    data["semantic_contract"] = value
                with self.subTest(value=value):
                    response = client.post("/api/v1/drift/baselines", headers=self._headers(), data=data)
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(list(self.upload_folder.iterdir()), [])
        self.assertEqual(list(self.baseline_folder.rglob("*.json")), [])

    def test_current_unit_conflict_rejected_before_history_write(self):
        contract = infer_semantic_contract(pd.DataFrame({"value": range(1000, 1100)}), confirmed=True)
        contract["columns"][0]["unit"] = "currency"
        with web_app.app.test_client() as client:
            baseline_id = self._create_baseline(client).get_json()["baseline_id"]
            response = client.post("/api/v1/drift/checks", headers=self._headers(), data={
                "baseline_id": baseline_id,
                "datafile": (BytesIO(_csv_values(range(1000, 1100))), "current.csv"),
                "semantic_contract": json.dumps(contract),
            })
            runs = client.get("/api/v1/drift/runs", headers=self._headers()).get_json()["runs"]
        self.assertEqual(response.status_code, 400)
        self.assertEqual(runs, [])
        self.assertEqual(list(self.upload_folder.iterdir()), [])

    def test_optional_current_confirmation_has_distinct_auto_deduplication_context(self):
        contract = infer_semantic_contract(pd.DataFrame({"value": range(1000, 1100)}), confirmed=True)
        with web_app.app.test_client() as client:
            baseline_id = self._create_baseline(client).get_json()["baseline_id"]
            inherited = self._create_check(client, baseline_id).get_json()
            confirmed = client.post("/api/v1/drift/checks", headers=self._headers(), data={
                "baseline_id": baseline_id,
                "datafile": (BytesIO(_csv_values(range(1000, 1100))), "current.csv"),
                "semantic_contract": json.dumps(contract),
            })
        self.assertEqual(inherited["report"]["semantics"]["current"], "inherited_from_baseline")
        self.assertEqual(confirmed.status_code, 201)
        self.assertEqual(confirmed.get_json()["report"]["semantics"]["current"], "confirmed_for_batch")
        self.assertNotEqual(inherited["run"]["run_id"], confirmed.get_json()["run"]["run_id"])

    def _create_baseline(self, client):
        return client.post(
            "/api/v1/drift/baselines",
            data={"datafile": (BytesIO(_csv_values(range(100))), "baseline.csv"),
                  "semantic_contract": json.dumps(infer_semantic_contract(
                      pd.DataFrame({"value": range(100)}), confirmed=True,
                  ))},
            headers=self._headers(),
            content_type="multipart/form-data",
        )

    def _create_check(self, client, baseline_id):
        return client.post(
            "/api/v1/drift/checks",
            data={
                "baseline_id": baseline_id,
                "datafile": (BytesIO(_csv_values(range(1000, 1100))), "current.csv"),
            },
            headers=self._headers(),
            content_type="multipart/form-data",
        )

    def _headers(self):
        return {"X-API-Key": self.api_key}


def _csv_values(values):
    return ("value\n" + "\n".join(str(value) for value in values) + "\n").encode("utf-8")


if __name__ == "__main__":
    unittest.main()
