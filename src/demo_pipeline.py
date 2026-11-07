"""Run the reproducible synthetic demo pipeline."""

import os
import subprocess
import sys

from src.dataset_paths import BASE_DIR

DEMO_STEPS = (
    "src.generate_synthetic_data",
    "src.data_prep",
    "src.model",
    "src.risk_engine",
)

DEMO_OUTPUTS = (
    "data/processed/auth_events_clean.csv",
    "data/processed/auth_events_risk.csv",
    "data/processed/evaluation_report.json",
)


def run_synthetic_demo(on_step=None):
    """Generate synthetic data and run each pipeline stage in order."""
    environment = os.environ.copy()
    environment["TEQO_DATASET"] = "synthetic"

    for step in DEMO_STEPS:
        if on_step is not None:
            on_step(step)
        try:
            result = subprocess.run(
                [sys.executable, "-m", step],
                cwd=BASE_DIR,
                env=environment,
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                f"Demo pipeline step {step} exceeded the 10-minute time limit."
            ) from error
        if result.returncode:
            output = "\n".join(
                part.strip()
                for part in (result.stdout, result.stderr)
                if part.strip()
            )
            raise RuntimeError(
                f"Demo pipeline step {step} failed with exit code "
                f"{result.returncode}.\n{output}"
            )

    missing_outputs = [
        path for path in DEMO_OUTPUTS
        if not os.path.isfile(os.path.join(BASE_DIR, path))
    ]
    if missing_outputs:
        raise FileNotFoundError(
            "Demo pipeline completed without creating expected outputs: "
            + ", ".join(missing_outputs)
        )
