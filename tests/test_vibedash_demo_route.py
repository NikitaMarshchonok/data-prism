import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import web_app


EMPTY_DASHBOARD = {
    "kpis": [],
    "charts": [],
    "tables": [],
    "ai_summary": "",
    "insights": [],
    "statistical_validation": {"tests": []},
    "pattern_analysis": {"status": "insufficient_data"},
    "df_shape": (360, 11),
}


class VibeDashDemoRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.previous_upload_folder = web_app.app.config["UPLOAD_FOLDER"]
        self.previous_retention_hours = web_app.app.config[
            "VIBEDASH_RETENTION_HOURS"
        ]
        web_app.app.config.update(
            TESTING=True,
            UPLOAD_FOLDER=self.temporary_directory.name,
            VIBEDASH_RETENTION_HOURS=24,
        )

    def tearDown(self):
        web_app.app.config["UPLOAD_FOLDER"] = self.previous_upload_folder
        web_app.app.config[
            "VIBEDASH_RETENTION_HOURS"
        ] = self.previous_retention_hours
        self.temporary_directory.cleanup()

    @patch("vibedash.routes.save_session_data")
    @patch("vibedash.routes.bridge_generate_dashboard_data", return_value=EMPTY_DASHBOARD)
    @patch("vibedash.routes.parse_prompt_to_viz_spec")
    def test_builtin_demo_opens_without_a_file_upload(
        self,
        parse_prompt,
        generate_dashboard,
        save_session,
    ):
        with web_app.app.test_client() as client:
            response = client.post(
                "/vibedash/preview",
                data={
                    "demo_dataset": "saas_growth",
                    "prompt": "Show evidence-backed SaaS performance.",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("saas_growth_demo.csv", response.get_data(as_text=True))
        generated_data = generate_dashboard.call_args.args[0]
        self.assertEqual(generated_data.shape, (360, 11))
        parse_prompt.assert_not_called()
        generated_spec = generate_dashboard.call_args.args[1]
        self.assertEqual(generated_spec.title, "SaaS Growth Evidence Dashboard")
        self.assertEqual(len(generated_spec.metrics), 4)
        self.assertEqual(len(generated_spec.charts), 5)
        save_session.assert_called_once()
        stored_files = list(Path(self.temporary_directory.name).glob("vibedash-*.csv"))
        self.assertEqual(len(stored_files), 1)

    def test_unknown_demo_is_rejected(self):
        with web_app.app.test_client() as client:
            response = client.post(
                "/vibedash/preview",
                data={"demo_dataset": "unknown", "prompt": "Demo"},
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(list(Path(self.temporary_directory.name).iterdir()), [])

    def test_builtin_demo_and_export_use_runtime_storage(self):
        with TemporaryDirectory() as state_directory:
            with patch.dict(
                os.environ,
                {"DATA_PRISM_STATE_DIR": state_directory},
            ):
                with web_app.app.test_client() as client:
                    preview_response = client.post(
                        "/vibedash/preview",
                        data={
                            "demo_dataset": "saas_growth",
                            "prompt": "Show evidence-backed SaaS performance.",
                        },
                    )

                    session_files = list(
                        (
                            Path(state_directory)
                            / "sessions"
                            / "vibedash"
                        ).glob("*.json")
                    )
                    self.assertEqual(preview_response.status_code, 200)
                    self.assertEqual(len(session_files), 1)

                    export_response = client.get(
                        f"/vibedash/export/{session_files[0].stem}"
                    )

            self.assertEqual(export_response.status_code, 200)
            self.assertIn(
                "attachment",
                export_response.headers["Content-Disposition"],
            )
            self.assertIn(
                b"SaaS Growth Evidence Dashboard",
                export_response.data,
            )
            export_response.close()

    @patch("vibedash.routes.is_ollama_available", return_value=False)
    def test_landing_page_exposes_the_one_click_demo(self, _ollama_status):
        with web_app.app.test_client() as client:
            response = client.get("/vibedash/")

        page = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Run the live demo", page)
        self.assertIn('name="demo_dataset" value="saas_growth"', page)
        self.assertIn('id="demo-loading-status"', page)
        self.assertIn("Building the evidence dashboard", page)
        self.assertIn("Temporary storage · 24-hour retention window.", page)
        self.assertNotIn("Data Prism v2", page)


if __name__ == "__main__":
    unittest.main()
