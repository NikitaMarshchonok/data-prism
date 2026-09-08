import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import web_app


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DeploymentContractTests(unittest.TestCase):
    def test_external_state_directory_keeps_runtime_artifacts_together(self):
        with TemporaryDirectory() as temporary_directory:
            paths = web_app.resolve_runtime_paths(
                REPOSITORY_ROOT,
                temporary_directory,
            )

            state_path = Path(temporary_directory).resolve()
            self.assertEqual(Path(paths["uploads"]).parent, state_path)
            self.assertEqual(Path(paths["reports"]).parent, state_path)
            self.assertEqual(Path(paths["baselines"]).parent, state_path)
            self.assertEqual(Path(paths["drift_store"]).parents[1], state_path)

    def test_render_blueprint_is_ci_gated_and_contains_no_secret_values(self):
        blueprint = (REPOSITORY_ROOT / "render.yaml").read_text(encoding="utf-8")

        self.assertIn("runtime: docker", blueprint)
        self.assertIn("plan: free", blueprint)
        self.assertIn("healthCheckPath: /readyz", blueprint)
        self.assertIn("autoDeployTrigger: checksPass", blueprint)
        self.assertEqual(blueprint.count("generateValue: true"), 2)
        self.assertNotIn("replace-with", blueprint)

    def test_container_uses_mounted_state_and_application_request_logs(self):
        dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("DATA_PRISM_STATE_DIR=/var/lib/data-prism", dockerfile)
        self.assertIn("LOG_FORMAT=json", dockerfile)
        self.assertNotIn("--access-logfile", dockerfile)

    def test_ci_builds_and_exercises_the_production_container(self):
        workflow = (
            REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("docker build --tag data-prism:ci", workflow)
        self.assertIn("http://127.0.0.1:5001/readyz", workflow)
        self.assertIn("X-Request-ID: ci-health-check", workflow)


if __name__ == "__main__":
    unittest.main()
