"""Run the reproducible synthetic demo pipeline."""

import hashlib
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

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

LOCK_STALE_SECONDS = 3600
LOCK_FILE = os.path.join(
    tempfile.gettempdir(),
    "teqoai-demo-"
    + hashlib.sha256(BASE_DIR.encode("utf-8")).hexdigest()[:12]
    + ".lock",
)


@contextmanager
def _demo_lock():
    """Prevent two demo generations from writing the same artifacts at once."""
    if os.path.exists(LOCK_FILE):
        age = time.time() - os.path.getmtime(LOCK_FILE)
        if age < LOCK_STALE_SECONDS:
            raise RuntimeError(
                "A demo generation is already running on this computer. "
                "Wait for it to finish and try again."
            )
        try:
            os.unlink(LOCK_FILE)
        except OSError:
            pass

    try:
        descriptor = os.open(
            LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
    except FileExistsError as error:
        raise RuntimeError(
            "A demo generation is already running on this computer. "
            "Wait for it to finish and try again."
        ) from error

    try:
        os.write(descriptor, f"{os.getpid()} {time.time()}".encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(LOCK_FILE)
        except OSError:
            pass


def run_synthetic_demo(on_step=None):
    """Generate synthetic data and run each pipeline stage in order."""
    with _demo_lock():
        _run_steps(on_step)


def _run_steps(on_step=None):
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
