import csv
import gzip
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_prep import clean_and_map  # noqa: E402


class DataPreparationTests(unittest.TestCase):
    def test_auth_events_are_mapped_and_exact_labels_are_matched(self):
        with tempfile.TemporaryDirectory() as directory:
            auth_path = Path(directory) / "auth.txt.gz"
            output_path = Path(directory) / "clean.csv"
            rows = [
                ["10", "U1@D", "U1@D", "C1", "SRV1", "Kerberos", "Network", "LogOn", "Success"],
                ["11", "U1@D", "U1@D", "C1", "SRV2", "NTLM", "Network", "LogOn", "Fail"],
            ]
            with gzip.open(auth_path, "wt", newline="", encoding="utf-8") as file:
                csv.writer(file).writerows(rows)

            clean_and_map(
                str(auth_path),
                {("10", "U1@D", "C1", "SRV1")},
                str(output_path),
                max_rows=10,
            )
            clean = pd.read_csv(output_path)

        self.assertEqual(clean["is_malicious"].tolist(), [1, 0])
        self.assertEqual(clean["user"].tolist(), ["U1@D", "U1@D"])
        self.assertEqual(clean["target_device"].tolist(), ["SRV1", "SRV2"])
        self.assertEqual(clean["result"].tolist(), ["Success", "Failure"])


if __name__ == "__main__":
    unittest.main()
