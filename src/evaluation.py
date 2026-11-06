"""Metrics for chronological, label-aware anomaly detection evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_ranked_scores(labels, scores, alert_budgets=(30, 100)):
    """Evaluate rankings against known labels at fixed analyst alert budgets."""
    alert_budgets = tuple(alert_budgets)
    y_true = np.asarray(labels)
    y_score = np.asarray(scores, dtype=float)
    if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
        raise ValueError("labels and scores must be one-dimensional and have equal length.")
    if len(y_true) == 0:
        return {
            "status": "not_evaluable_no_test_events",
            "test_events": 0,
            "known_positive_events": 0,
            "unmatched_events": 0,
            "average_precision": None,
            "roc_auc": None,
            "alert_budgets": {},
        }
    if pd.isna(y_true).any() or not np.isin(y_true, [0, 1]).all():
        raise ValueError("labels must contain only non-missing binary values 0 or 1.")
    if not np.isfinite(y_score).all():
        raise ValueError("scores must contain only finite numeric values.")
    if any(
        not isinstance(budget, (int, np.integer))
        or isinstance(budget, (bool, np.bool_))
        or budget < 1
        for budget in alert_budgets
    ):
        raise ValueError("alert budgets must be positive integers.")

    y_true = y_true.astype(int)
    positives = int(y_true.sum())
    unmatched = len(y_true) - positives
    status = "evaluated" if positives else "not_evaluable_no_test_positives"
    result = {
        "status": status,
        "test_events": int(len(y_true)),
        "known_positive_events": positives,
        "unmatched_events": int(unmatched),
        "average_precision": (
            float(average_precision_score(y_true, y_score)) if positives else None
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, y_score)) if positives and unmatched else None
        ),
        "alert_budgets": {},
    }

    if not positives:
        return result

    order = np.argsort(-y_score, kind="stable")
    prevalence = positives / len(y_true)
    for budget in alert_budgets:
        selected_count = min(int(budget), len(y_true))
        hits = int(y_true[order[:selected_count]].sum())
        result["alert_budgets"][str(budget)] = {
            "selected_events": selected_count,
            "known_positive_hits": hits,
            "precision": hits / selected_count,
            "recall": hits / positives,
            "random_baseline_expected_precision": prevalence,
        }
    return result
