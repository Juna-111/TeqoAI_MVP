"""Dataset-specific paths shared by the local pipeline."""

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET = os.environ.get("TEQO_DATASET", "synthetic").strip().lower()

if DATASET not in {"synthetic", "lanl"}:
    raise ValueError(
        "TEQO_DATASET must be either 'synthetic' or 'lanl', "
        f"not {DATASET!r}."
    )

DATASET_SUBDIR = "" if DATASET == "synthetic" else DATASET
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", DATASET_SUBDIR)
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed", DATASET_SUBDIR)
MODELS_DIR = os.path.join(BASE_DIR, "models", DATASET_SUBDIR)

CLEAN_FILE = os.path.join(PROCESSED_DIR, "auth_events_clean.csv")
SCORED_FILE = os.path.join(PROCESSED_DIR, "auth_events_scored.csv")
RISK_FILE = os.path.join(PROCESSED_DIR, "auth_events_risk.csv")
EVALUATION_FILE = os.path.join(PROCESSED_DIR, "evaluation_report.json")
