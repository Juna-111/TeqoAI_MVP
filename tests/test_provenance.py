import unittest

from src.provenance import create_pipeline_run_id, validate_pipeline_provenance


def valid_report():
    return {
        "schema_version": 3,
        "dataset_profile": "synthetic",
        "run_id": "run-123",
        "input": {"sha256": "input-hash"},
        "split": {
            "training_events": 3,
            "test_events": 1,
            "training_known_positive_labels": 1,
            "test_known_positive_labels": 0,
        },
        "evaluation": {
            "alert_budgets": [30, 100],
            "models": {"isolation_forest": {"alert_budgets": {}}},
        },
        "walk_forward_validation": {"folds": []},
    }


class PipelineProvenanceTests(unittest.TestCase):
    def test_run_id_is_stable_and_changes_with_pipeline_inputs(self):
        run_id = create_pipeline_run_id(
            "synthetic", "input-hash", 100, 8, {"seed": 42}
        )

        self.assertEqual(
            run_id,
            create_pipeline_run_id(
                "synthetic", "input-hash", 100, 8, {"seed": 42}
            ),
        )
        self.assertNotEqual(
            run_id,
            create_pipeline_run_id(
                "synthetic", "different-input", 100, 8, {"seed": 42}
            ),
        )

    def test_matching_report_and_risk_scores_have_no_provenance_issues(self):
        report = valid_report()

        self.assertEqual(
            validate_pipeline_provenance(
                report, "synthetic", "input-hash", ["run-123", "run-123"]
            ),
            [],
        )

    def test_stale_input_and_mismatched_scores_are_reported(self):
        report = valid_report()
        report["input"]["sha256"] = "old-input"

        issues = validate_pipeline_provenance(
            report, "lanl", "new-input", ["old-run"]
        )

        self.assertTrue(any("different dataset" in issue for issue in issues))
        self.assertTrue(any("different input" in issue for issue in issues))
        self.assertTrue(any("do not match" in issue for issue in issues))

    def test_missing_run_identifier_is_reported(self):
        report = valid_report()
        del report["run_id"]

        issues = validate_pipeline_provenance(
            report, "synthetic", "input-hash", []
        )

        self.assertTrue(any("run identifier" in issue for issue in issues))
        self.assertTrue(any("Risk scores" in issue for issue in issues))

    def test_unavailable_optional_model_may_omit_alert_budget_metrics(self):
        report = valid_report()
        report["evaluation"]["models"]["autoencoder"] = {
            "status": "not_available"
        }

        self.assertEqual(
            validate_pipeline_provenance(
                report, "synthetic", "input-hash", ["run-123"]
            ),
            [],
        )

    def test_available_model_requires_well_formed_alert_budget_metrics(self):
        report = valid_report()
        report["evaluation"]["models"]["isolation_forest"] = {
            "status": "evaluated"
        }

        issues = validate_pipeline_provenance(
            report, "synthetic", "input-hash", ["run-123"]
        )

        self.assertTrue(any("isolation_forest" in issue for issue in issues))

    def test_outdated_pipeline_schema_requires_rerunning_model_pipeline(self):
        report = valid_report()
        report["schema_version"] = 2

        issues = validate_pipeline_provenance(
            report, "synthetic", "input-hash", ["run-123"]
        )

        self.assertTrue(
            any("outdated or unsupported pipeline schema" in issue for issue in issues)
        )


if __name__ == "__main__":
    unittest.main()
