import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import case_store  # noqa: E402


class CaseStoreTests(unittest.TestCase):
    def test_case_save_persists_status_and_audit_entry(self):
        original_database = case_store.CASE_DB
        try:
            with tempfile.TemporaryDirectory() as directory:
                case_store.CASE_DB = os.path.join(directory, "cases.sqlite3")
                case_store.save_case(
                    "synthetic",
                    "event-1",
                    "Investigating",
                    "analyst",
                    "Reviewed surrounding activity.",
                    {"Validate the activity with the account owner or identity team": True},
                )

                cases = case_store.get_cases("synthetic")
                history = case_store.get_case_history("synthetic", "event-1")

                self.assertEqual(cases["event-1"]["status"], "Investigating")
                self.assertEqual(cases["event-1"]["owner"], "analyst")
                self.assertEqual(len(history), 1)
                self.assertEqual(history[0]["action"], "case_created")
        finally:
            case_store.CASE_DB = original_database


if __name__ == "__main__":
    unittest.main()
