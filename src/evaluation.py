"""Metrics for chronological, label-aware anomaly detection evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ALERT_BUDGETS = (30, 100)


def _validate_budgets(alert_budgets):
    alert_budgets = tuple(alert_budgets)
    if any(
        not isinstance(budget, (int, np.integer))
        or isinstance(budget, (bool, np.bool_))
        or budget < 1
        for budget in alert_budgets
    ):
        raise ValueError("alert budgets must be positive integers.")
    return alert_budgets


def _budget_metrics(y_true, order, alert_budgets):
    """Summarise the first k ranked events for each review budget."""
    positives = int(y_true.sum())
    unmatched = len(y_true) - positives
    prevalence = positives / len(y_true)
    budgets = {}
    for budget in alert_budgets:
        selected_count = min(int(budget), len(y_true))
        selected = order[:selected_count]
        hits = int(y_true[selected].sum())
        false_positives = selected_count - hits
        false_negatives = positives - hits
        true_negatives = unmatched - false_positives
        precision = hits / selected_count
        recall = hits / positives if positives else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if recall is not None and precision + recall
            else 0.0 if recall is not None else None
        )
        budgets[str(budget)] = {
            "selected_events": selected_count,
            "known_positive_hits": hits,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": (
                false_positives / unmatched if unmatched else None
            ),
            "confusion_matrix": {
                "true_positive": hits,
                "false_positive": false_positives,
                "false_negative": false_negatives,
                "true_negative": true_negatives,
            },
            "random_baseline_expected_precision": prevalence,
        }
    return budgets


def evaluate_ranked_scores(labels, scores, alert_budgets=ALERT_BUDGETS):
    """Evaluate rankings against known labels at fixed analyst alert budgets."""
    alert_budgets = _validate_budgets(alert_budgets)
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

    order = np.argsort(-y_score, kind="stable")
    result["alert_budgets"] = _budget_metrics(y_true, order, alert_budgets)
    return result


def evaluate_ranked_order(labels, order, alert_budgets=ALERT_BUDGETS):
    """Evaluate an explicit event ordering, such as the dashboard review queue."""
    alert_budgets = _validate_budgets(alert_budgets)
    y_true = np.asarray(labels)
    order = np.asarray(order)
    if y_true.ndim != 1 or order.ndim != 1 or len(y_true) != len(order):
        raise ValueError("labels and order must be one-dimensional and have equal length.")
    if len(y_true) == 0:
        return {
            "status": "not_evaluable_no_test_events",
            "test_events": 0,
            "known_positive_events": 0,
            "unmatched_events": 0,
            "alert_budgets": {},
        }
    if pd.isna(y_true).any() or not np.isin(y_true, [0, 1]).all():
        raise ValueError("labels must contain only non-missing binary values 0 or 1.")
    if (
        not np.issubdtype(order.dtype, np.integer)
        or order.min() < 0
        or order.max() >= len(y_true)
        or len(np.unique(order)) != len(order)
    ):
        raise ValueError("order must be a permutation of the event positions.")

    y_true = y_true.astype(int)
    positives = int(y_true.sum())
    return {
        "status": "evaluated" if positives else "not_evaluable_no_test_positives",
        "test_events": int(len(y_true)),
        "known_positive_events": positives,
        "unmatched_events": int(len(y_true) - positives),
        "alert_budgets": _budget_metrics(y_true, order, alert_budgets),
    }
