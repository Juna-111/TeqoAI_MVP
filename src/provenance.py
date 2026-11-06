"""Helpers for identifying and validating a dataset-specific pipeline run."""

import hashlib
import json

PIPELINE_SCHEMA_VERSION = 3


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_pipeline_run_id(dataset_profile, input_sha256, cutoff_time, feature_count, config):
    manifest = {
        "dataset_profile": dataset_profile,
        "input_sha256": input_sha256,
        "cutoff_time": int(cutoff_time),
        "feature_count": int(feature_count),
        "config": config,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def validate_pipeline_provenance(report, dataset_profile, input_sha256, risk_run_ids):
    """Return user-facing reasons why report, input, and scores do not match."""
    issues = []
    if not isinstance(report, dict):
        return ["Evaluation report is missing or is not a JSON object."]

    if report.get("schema_version") != PIPELINE_SCHEMA_VERSION:
        issues.append(
            "Evaluation report uses an outdated or unsupported pipeline schema; "
            "rerun the model pipeline."
        )

    if report.get("dataset_profile") != dataset_profile:
        issues.append("Evaluation report belongs to a different dataset profile.")

    input_details = report.get("input")
    if not isinstance(input_details, dict) or not isinstance(
        input_details.get("sha256"), str
    ):
        issues.append("Evaluation report is missing its input fingerprint.")
    elif input_details.get("sha256") != input_sha256:
        issues.append("Evaluation report was generated from a different input file.")

    run_id = report.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        issues.append("Evaluation report does not include a pipeline run identifier.")

    risk_run_ids = list(risk_run_ids)
    if any(not isinstance(value, str) or not value for value in risk_run_ids):
        issues.append("Risk scores contain missing or malformed run identifiers.")
    observed_run_ids = {value for value in risk_run_ids if isinstance(value, str) and value}
    if not isinstance(run_id, str) or not run_id or observed_run_ids != {run_id}:
        issues.append("Risk scores do not match the evaluation report's pipeline run.")

    split = report.get("split")
    if not isinstance(split, dict):
        issues.append("Evaluation report is missing split details.")
    else:
        split_counts = (
            "training_events",
            "test_events",
            "training_known_positive_labels",
            "test_known_positive_labels",
        )
        if any(
            not isinstance(split.get(key), int) or split[key] < 0
            for key in split_counts
        ):
            issues.append("Evaluation report split counts are malformed.")

    evaluation = report.get("evaluation")
    if not isinstance(evaluation, dict) or not isinstance(
        evaluation.get("models"), dict
    ) or not isinstance(evaluation.get("alert_budgets"), list):
        issues.append("Evaluation report model metrics are missing or malformed.")
    else:
        for model_name, metrics in evaluation["models"].items():
            if not isinstance(metrics, dict) or not isinstance(
                metrics.get("alert_budgets", {}), dict
            ):
                issues.append(
                    f"Evaluation metrics for {model_name!r} are malformed."
                )
                break
            if any(
                not isinstance(result, dict)
                for result in metrics["alert_budgets"].values()
            ):
                issues.append(
                    f"Alert-budget metrics for {model_name!r} are malformed."
                )
                break

    walk_forward = report.get("walk_forward_validation")
    if not isinstance(walk_forward, dict) or not isinstance(
        walk_forward.get("folds"), list
    ):
        issues.append("Evaluation report walk-forward details are missing or malformed.")
    elif any(
        not isinstance(fold, dict)
        or not isinstance(fold.get("metrics"), dict)
        for fold in walk_forward["folds"]
    ):
        issues.append("Evaluation report contains malformed walk-forward folds.")

    return issues
