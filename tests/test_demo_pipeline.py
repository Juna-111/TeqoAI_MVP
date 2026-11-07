import subprocess
import unittest
from unittest.mock import patch

from src import demo_pipeline


class DemoPipelineTests(unittest.TestCase):
    @patch("src.demo_pipeline.subprocess.run")
    @patch("src.demo_pipeline.os.path.isfile", return_value=True)
    def test_runs_all_steps_with_synthetic_profile(self, isfile, run):
        run.return_value.returncode = 0
        run.return_value.stdout = ""
        run.return_value.stderr = ""
        completed_steps = []

        demo_pipeline.run_synthetic_demo(completed_steps.append)

        self.assertEqual(completed_steps, list(demo_pipeline.DEMO_STEPS))
        self.assertEqual(run.call_count, len(demo_pipeline.DEMO_STEPS))
        for call, step in zip(run.call_args_list, demo_pipeline.DEMO_STEPS):
            self.assertEqual(call.args[0][-2:], ["-m", step])
            self.assertEqual(call.kwargs["cwd"], demo_pipeline.BASE_DIR)
            self.assertEqual(
                call.kwargs["env"]["TEQO_DATASET"],
                "synthetic",
            )
        self.assertEqual(isfile.call_count, len(demo_pipeline.DEMO_OUTPUTS))

    @patch("src.demo_pipeline.subprocess.run")
    def test_reports_failed_pipeline_step_and_logs(self, run):
        run.return_value.returncode = 1
        run.return_value.stdout = "pipeline output"
        run.return_value.stderr = "pipeline error"

        with self.assertRaisesRegex(
            RuntimeError,
            "src.generate_synthetic_data failed with exit code 1",
        ) as error:
            demo_pipeline.run_synthetic_demo()

        self.assertIn("pipeline output", str(error.exception))
        self.assertIn("pipeline error", str(error.exception))
        self.assertEqual(run.call_count, 1)

    @patch(
        "src.demo_pipeline.subprocess.run",
        side_effect=subprocess.TimeoutExpired("python", 600),
    )
    def test_reports_pipeline_timeout(self, run):
        with self.assertRaisesRegex(
            TimeoutError,
            "src.generate_synthetic_data exceeded the 10-minute time limit",
        ):
            demo_pipeline.run_synthetic_demo()

    @patch("src.demo_pipeline.subprocess.run")
    @patch("src.demo_pipeline.os.path.isfile", return_value=False)
    def test_reports_missing_pipeline_outputs(self, isfile, run):
        run.return_value.returncode = 0
        run.return_value.stdout = ""
        run.return_value.stderr = ""

        with self.assertRaisesRegex(
            FileNotFoundError,
            "auth_events_clean.csv",
        ):
            demo_pipeline.run_synthetic_demo()

        self.assertEqual(run.call_count, len(demo_pipeline.DEMO_STEPS))
        self.assertEqual(isfile.call_count, len(demo_pipeline.DEMO_OUTPUTS))


if __name__ == "__main__":
    unittest.main()
