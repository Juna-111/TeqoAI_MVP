import os
import tempfile
import unittest

from src import case_store


class CaseStoreTests(unittest.TestCase):
    def test_csv_export_text_is_safe_for_spreadsheets(self):
        for value in ("=1+1", " +SUM(A1:A2)", "\t@SUM(A1:A2)", "-1+2"):
            with self.subTest(value=value):
                self.assertEqual(case_store.csv_safe_text(value), "'" + value)

        self.assertEqual(case_store.csv_safe_text("ordinary note"), "ordinary note")
        self.assertEqual(case_store.csv_safe_text(42), 42)

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
