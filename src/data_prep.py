"""
TeqoAI — Step 1: Data Preparation

Streams the LANL auth.txt.gz file (too large to load fully into memory),
takes a manageable slice, maps it into TeqoAI's common event schema
(Source -> Time -> User -> Device -> Action -> Result), and merges in
red-team labels for evaluation purposes only (not for training).

Usage:
    python -m src.data_prep

Input (place in data/raw/):
    auth.txt.gz      https://csr.lanl.gov/data/cyber1/
    redteam.txt.gz    https://csr.lanl.gov/data/cyber1/

Output:
    data/processed/auth_events_clean.csv
"""

import gzip
import csv
import os

from src.dataset_paths import CLEAN_FILE, DATASET, RAW_DIR

# ---- Config ----------------------------------------------------------
AUTH_FILE = os.path.join(RAW_DIR, "auth.txt.gz")
REDTEAM_FILE = os.path.join(RAW_DIR, "redteam.txt.gz")
OUTPUT_FILE = CLEAN_FILE

# The full auth file has 1.6B+ rows. Keep processing bounded by default, while
# allowing a larger prefix to be selected without editing this source file.
try:
    MAX_ROWS = int(os.environ.get("TEQO_MAX_AUTH_ROWS", "2000000"))
except ValueError as error:
    raise ValueError("TEQO_MAX_AUTH_ROWS must be a positive integer.") from error
if MAX_ROWS < 1:
    raise ValueError("TEQO_MAX_AUTH_ROWS must be a positive integer.")

# LANL auth.txt.gz columns (no header row in the raw file):
# time, source_user@domain, destination_user@domain, source_computer,
# destination_computer, authentication_type, logon_type,
# authentication_orientation, success_or_failure
AUTH_COLUMNS = [
    "time",
    "source_user",
    "destination_user",
    "source_computer",
    "destination_computer",
    "auth_type",
    "logon_type",
    "auth_orientation",
    "result_raw",
]

# redteam.txt.gz columns: time, user@domain, source_computer, destination_computer
REDTEAM_COLUMNS = ["time", "user", "source_computer", "destination_computer"]


def load_redteam_labels(path, max_rows=None):
    """Load red-team (malicious) events into a set of keys for fast lookup.

    Key = (time, user, source_computer, destination_computer)
    This is used ONLY to label rows for evaluation after the fact —
    never fed into the anomaly detection model as a feature.
    """
    labels = set()
    min_time = None
    max_time = None
    if not os.path.exists(path):
        print(f"[warn] {path} not found - proceeding without red-team labels. "
              f"is_malicious will be 0 for all rows.")
        return labels

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            if len(row) < 4:
                continue
            time_, user, src, dst = row[0], row[1], row[2], row[3]
            labels.add((time_, user, src, dst))
            try:
                event_time = int(time_)
            except ValueError:
                continue
            min_time = event_time if min_time is None else min(min_time, event_time)
            max_time = event_time if max_time is None else max(max_time, event_time)
    print(f"[info] loaded {len(labels):,} red-team labeled events")
    if min_time is not None:
        print(f"[info] red-team elapsed-time range: {min_time}..{max_time} seconds")
    return labels


def clean_and_map(auth_path, redteam_labels, output_path, max_rows=MAX_ROWS):
    """Stream auth.txt.gz, map each row into the common schema, and write CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(auth_path):
        raise FileNotFoundError(
            f"Could not find {auth_path}. Download auth.txt.gz from "
            f"https://csr.lanl.gov/data/cyber1/ and place it in {RAW_DIR}."
        )

    rows_written = 0
    rows_malicious = 0
    min_time = None
    max_time = None

    with gzip.open(auth_path, "rt", encoding="utf-8", errors="replace") as infile, \
         open(output_path, "w", newline="", encoding="utf-8") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(
            ["source", "time", "user", "device", "target_device",
             "action", "result", "is_malicious"]
        )

        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            if len(row) < 9:
                continue  # skip malformed rows

            (time_, src_user, dst_user, src_computer, dst_computer,
             auth_type, logon_type, auth_orientation, result_raw) = row
            try:
                event_time = int(time_)
            except ValueError:
                continue

            # We model the LOGIN itself: the source_user acting from
            # src_computer, authenticating toward dst_computer.
            user = src_user
            device = src_computer
            target_device = dst_computer
            action = f"{auth_type}/{logon_type}"
            result = "Success" if result_raw.strip().lower() == "success" else "Failure"

            # Check against red-team labels (best-effort key match)
            key = (time_, src_user, src_computer, dst_computer)
            is_malicious = 1 if key in redteam_labels else 0
            if is_malicious:
                rows_malicious += 1

            writer.writerow(
                ["windows_auth", time_, user, device, target_device,
                 action, result, is_malicious]
            )
            rows_written += 1
            min_time = event_time if min_time is None else min(min_time, event_time)
            max_time = event_time if max_time is None else max(max_time, event_time)

            if rows_written % 250_000 == 0:
                print(f"[info] processed {rows_written:,} rows...")

    print(f"\n[done] wrote {rows_written:,} rows to {output_path}")
    if min_time is not None:
        print(f"[info] auth elapsed-time range: {min_time}..{max_time} seconds")
    print(f"[done] {rows_malicious:,} rows matched a red-team label "
          f"({rows_malicious / max(rows_written,1) * 100:.4f}% of the slice)")
    if rows_malicious == 0:
        print("[note] 0 malicious matches is common in an early slice of the "
              "dataset - red-team activity may occur later in the 58-day span. "
              "You can raise MAX_ROWS, or check redteam.txt.gz timestamps "
              "against the range of `time` values in your output to confirm.")


if __name__ == "__main__":
    print(f"[info] selected dataset: {DATASET}")
    print("[info] loading red-team labels...")
    redteam_labels = load_redteam_labels(REDTEAM_FILE)

    print(f"[info] processing up to {MAX_ROWS:,} rows from {AUTH_FILE}...")
    clean_and_map(AUTH_FILE, redteam_labels, OUTPUT_FILE, MAX_ROWS)
