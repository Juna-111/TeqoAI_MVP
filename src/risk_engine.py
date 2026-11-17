"""
TeqoAI — Step 3: Simplified Risk Engine

Combines the Isolation Forest anomaly score with a small number of
hand-coded, transparent risk factors (new device, sensitive asset,
failed login, burst of activity) into ONE human-readable risk level:
Low, Medium, or High.

This is intentionally simple and rule-assisted rather than another
model — the point of the MVP risk engine is transparency: anyone should
be able to look at why an event got a High rating without needing to
trust a black box. Phase Two replaces this with a proper multi-model
risk engine.

Calibration:
    Percentiles are measured against the training-period score
    distribution rather than the full dataset, so an event's displayed
    priority does not depend on future events. High and Medium tiers are
    defined by training-period quantiles, which keeps the alert volume
    tied to a documented review capacity instead of a fixed 0-100 guess.

Usage:
    python -m src.risk_engine

Input:
    data/processed/auth_events_scored.csv   (from src/model.py)

Output:
    data/processed/auth_events_risk.csv     (adds risk_score, risk_level,
                                             risk_reasons; excludes labels)
"""

import json
import os

import numpy as np
import pandas as pd

from src.dataset_paths import EVALUATION_FILE
from src.dataset_paths import RISK_FILE as OUTPUT_FILE
from src.dataset_paths import SCORED_FILE as INPUT_FILE
from src.evaluation import (
    ALERT_BUDGETS,
    evaluate_ranked_order,
    evaluate_ranked_scores,
)

# Assets considered high-value / sensitive. Reaching one of these is treated
# as more serious than a routine workstation login. Adjust this list to
# match whatever your Digital Twin (next step) treats as critical assets.
SENSITIVE_ASSETS = {"DB-CORE", "APP-FIN", "APP-HR"}

# Hand-coded bonus points added to the anomaly percentile. These are the
# "small number of hand coded risk factors" from the proposal — kept
# deliberately simple and explainable rather than tuned/optimized.
RISK_FACTOR_POINTS = {
    "new_device": 15,
    "sensitive_asset": 20,
    "failed_login": 10,
    "burst_activity": 10,
}

BURST_THRESHOLD = 5  # rolling_count_1h at/above this counts as a burst

# Tier targets as training-period quantiles of the unclipped risk score.
# High targets roughly the top 5% of the training period and Medium the
# next 15%. The resulting raw thresholds are stored with the run report.
HIGH_CAPACITY_QUANTILE = 0.95
MEDIUM_CAPACITY_QUANTILE = 0.80

# Labels and evaluation split are needed while calibrating and measuring,
# but they must not be shipped in the analyst-facing serving artifact.
SERVING_EXCLUDED_COLUMNS = ("is_malicious", "evaluation_split")


def _bonus_from_raw(frame):
    """Compute factor points from the raw event columns."""
    new_device = frame["is_new_target_device"].astype(int)
    sensitive = (
        frame["target_device"]
        .astype("string")
        .isin(SENSITIVE_ASSETS)
        .fillna(False)
        .astype(int)
    )
    failed = frame["is_failed"].astype(int)
    burst = (frame["rolling_count_1h"] >= BURST_THRESHOLD).astype(int)
    return (
        new_device * RISK_FACTOR_POINTS["new_device"]
        + sensitive * RISK_FACTOR_POINTS["sensitive_asset"]
        + failed * RISK_FACTOR_POINTS["failed_login"]
        + burst * RISK_FACTOR_POINTS["burst_activity"]
    )


def _percentiles(reference_scores, scores):
    """Percentile of each score against a fixed reference distribution."""
    reference_scores = np.sort(np.asarray(reference_scores, dtype=float))
    if len(reference_scores) == 0:
        raise ValueError("At least one reference score is required.")
    scores = np.asarray(scores, dtype=float)
    return np.searchsorted(reference_scores, scores, side="right") / len(
        reference_scores
    ) * 100


def compute_risk(df):
    df = df.copy()
    scores = pd.to_numeric(df["anomaly_score_iforest"], errors="coerce")
    if scores.isna().any():
        raise ValueError("anomaly_score_iforest must contain only numeric values.")

    # 1) Percentile against the training period, not the full dataset. This
    # keeps a historical event's priority independent of future events.
    using_training_reference = False
    reference = df
    if "evaluation_split" in df.columns:
        train_mask = df["evaluation_split"].astype("string").str.lower().eq("train")
        if train_mask.any():
            reference = df.loc[train_mask]
            using_training_reference = True

    reference_scores = reference["anomaly_score_iforest"].to_numpy(dtype=float)
    df["anomaly_percentile"] = _percentiles(reference_scores, scores.to_numpy())

    # 2) Hand-coded risk factors — each one is independently checkable,
    # which is exactly what makes this engine transparent.
    df["factor_new_device"] = df["is_new_target_device"] == 1
    df["factor_sensitive_asset"] = (
        df["target_device"].astype("string").isin(SENSITIVE_ASSETS).fillna(False)
    )
    df["factor_failed_login"] = df["is_failed"] == 1
    df["factor_burst_activity"] = df["rolling_count_1h"] >= BURST_THRESHOLD

    # 3) Rank on the unclipped score. The displayed score is still capped at
    # 100 for readability, but clipping must not decide queue order.
    df["risk_rank_score"] = (
        df["anomaly_percentile"] + _bonus_from_raw(df).to_numpy()
    ).round(6)
    df["risk_score"] = df["risk_rank_score"].clip(lower=0, upper=100).round(1)

    # 4) Calibrate tiers on the same reference period used for percentiles.
    reference_percentiles = _percentiles(reference_scores, reference_scores)
    reference_raw = reference_percentiles + _bonus_from_raw(reference).to_numpy()
    high_threshold = float(
        np.quantile(reference_raw, HIGH_CAPACITY_QUANTILE)
    )
    medium_threshold = float(
        np.quantile(reference_raw, MEDIUM_CAPACITY_QUANTILE)
    )
    df["risk_level"] = pd.cut(
        df["risk_rank_score"],
        bins=[-np.inf, medium_threshold, high_threshold, np.inf],
        labels=["Low", "Medium", "High"],
        right=False,
    )

    # 5) A short, human-readable reason string — a preview of what the
    # SHAP explainability step will formalize next.
    def build_reason(row):
        reasons = []
        if row["factor_new_device"]:
            reasons.append("first-time access to this destination")
        if row["factor_sensitive_asset"]:
            reasons.append(f"reached a sensitive asset ({row['target_device']})")
        if row["factor_failed_login"]:
            reasons.append("failed login attempt")
        if row["factor_burst_activity"]:
            reasons.append(f"{int(row['rolling_count_1h'])} logins in the past hour")
        if not reasons:
            reasons.append("elevated anomaly score, no specific risk factor triggered")
        return "; ".join(reasons)

    df["risk_reasons"] = df.apply(build_reason, axis=1)

    df.attrs["risk_calibration"] = {
        "percentile_reference": (
            "training period" if using_training_reference else "all loaded events"
        ),
        "reference_events": int(len(reference)),
        "high_capacity_quantile": HIGH_CAPACITY_QUANTILE,
        "medium_capacity_quantile": MEDIUM_CAPACITY_QUANTILE,
        "high_threshold": round(high_threshold, 6),
        "medium_threshold": round(medium_threshold, 6),
        "tier_strategy": (
            "High and Medium are training-period quantiles of the unclipped "
            "risk score; tiers are relative review capacity, not probabilities."
        ),
    }
    return df


def evaluate(df, label_col="is_malicious"):
    """Sanity check: how does risk_level line up with known malicious events?"""
    if label_col not in df.columns:
        print("\n[evaluation] ground-truth labels are not present; skipping printout.")
        return
    evaluation_df = df
    if "evaluation_split" in df:
        evaluation_df = df[df["evaluation_split"] == "test"]

    if evaluation_df[label_col].sum() == 0:
        print("\n[evaluation] no labeled events in the held-out period.")
        return
    print("\n[evaluation] held-out risk_level breakdown for known malicious events:")
    print(evaluation_df[evaluation_df[label_col] == 1]["risk_level"].value_counts().to_string())
    print("\n[evaluation] held-out risk_level breakdown for ALL events:")
    print(evaluation_df["risk_level"].value_counts().to_string())


def evaluate_decision_queue(df, calibration):
    """Measure the score and tier ordering that analysts actually see."""
    required = {"evaluation_split", "is_malicious", "risk_level", "risk_rank_score"}
    if not required.issubset(df.columns):
        return {
            "status": "not_available",
            "reason": "Ground-truth labels or queue fields are not present.",
        }

    holdout = df[
        df["evaluation_split"].astype("string").str.lower().eq("test")
    ].reset_index(drop=True)
    if holdout.empty:
        return {
            "status": "not_evaluable_no_test_events",
            "test_events": 0,
            "known_positive_events": 0,
        }
    labels = holdout["is_malicious"].astype(int).to_numpy()
    positives = int(labels.sum())
    if positives == 0:
        return {
            "status": "not_evaluable_no_test_positives",
            "test_events": int(len(holdout)),
            "known_positive_events": 0,
        }

    score_metrics = evaluate_ranked_scores(
        labels, holdout["risk_rank_score"].to_numpy(), ALERT_BUDGETS
    )

    level_order = {"High": 0, "Medium": 1, "Low": 2}
    ordered = holdout.assign(
        _level_order=holdout["risk_level"]
        .map(level_order)
        .fillna(3)
        .astype(int)
    )
    sort_columns = ["_level_order", "risk_rank_score"]
    ascending = [True, False]
    if "time" in ordered.columns:
        sort_columns.append("time")
        ascending.append(False)
    ordered = ordered.sort_values(
        sort_columns, ascending=ascending, kind="stable"
    )
    queue_metrics = evaluate_ranked_order(
        labels, ordered.index.to_numpy(), ALERT_BUDGETS
    )

    tiers = {}
    for level in ("High", "Medium", "Low"):
        mask = holdout["risk_level"].eq(level).to_numpy()
        count = int(mask.sum())
        hits = int(labels[mask].sum())
        tiers[level] = {
            "events": count,
            "share_of_holdout": count / len(holdout),
            "known_positive_events": hits,
            "precision": hits / count if count else None,
            "recall": hits / positives if positives else None,
        }

    return {
        "status": "evaluated",
        "method": (
            "Risk level first (High, Medium, Low), then unclipped risk score "
            "descending, then time descending. Matches the dashboard queue sort."
        ),
        "score_ranking": score_metrics,
        "queue_at_budget": queue_metrics,
        "tiers": tiers,
        "calibration": calibration,
    }


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)
        file.write("\n")
    os.replace(temporary, path)


def update_evaluation_report(df, calibration):
    """Attach the deployed-queue evaluation and thresholds to the run report."""
    if not os.path.exists(EVALUATION_FILE):
        print("[warning] evaluation report is missing; queue metrics were not stored.")
        return None
    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        report = json.load(file)
    if not isinstance(report, dict):
        raise ValueError("Evaluation report is not a JSON object.")

    run_id = str(df["pipeline_run_id"].iloc[0])
    if report.get("run_id") != run_id:
        print(
            "[warning] evaluation report run identifier does not match the scored "
            "events; queue metrics were not stored."
        )
        return None

    report.setdefault("risk_engine", {}).update(
        {
            "high_threshold": calibration["high_threshold"],
            "medium_threshold": calibration["medium_threshold"],
            "high_capacity_quantile": calibration["high_capacity_quantile"],
            "medium_capacity_quantile": calibration["medium_capacity_quantile"],
            "percentile_reference": calibration["percentile_reference"],
            "tier_strategy": calibration["tier_strategy"],
        }
    )
    report["decision_queue"] = evaluate_decision_queue(df, calibration)
    _write_json_atomic(EVALUATION_FILE, report)
    return report


def serving_frame(df):
    """Return the analyst-facing frame with evaluation labels removed."""
    return df.drop(
        columns=[
            column
            for column in SERVING_EXCLUDED_COLUMNS
            if column in df.columns
        ]
    )


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Run src/model.py first."
        )

    print("[info] loading scored events...")
    df = pd.read_csv(INPUT_FILE)
    if (
        "pipeline_run_id" not in df
        or df["pipeline_run_id"].isna().any()
        or df["pipeline_run_id"].nunique() != 1
    ):
        raise ValueError(
            "Scored events have no single pipeline run identifier. "
            "Rerun src/model.py before risk scoring."
        )

    print("[info] computing risk scores and levels...")
    df = compute_risk(df)
    calibration = dict(df.attrs.get("risk_calibration", {}))

    evaluate(df)
    report = update_evaluation_report(df, calibration)
    if report is not None:
        queue = report.get("decision_queue", {})
        score_ranking = queue.get("score_ranking", {})
        if score_ranking.get("status") == "evaluated":
            print(
                "[evaluation] deployed risk score: "
                f"AP={score_ranking['average_precision']:.4f}, "
                f"top-30 precision="
                f"{queue['queue_at_budget']['alert_budgets']['30']['precision']:.4f}"
            )

    serving = serving_frame(df)
    temporary = f"{OUTPUT_FILE}.tmp"
    serving.to_csv(temporary, index=False)
    os.replace(temporary, OUTPUT_FILE)
    print(f"\n[done] risk-scored events written to {OUTPUT_FILE}")
    print(
        "[info] serving artifact excludes ground-truth labels and the "
        "evaluation split."
    )

    print("\n[info] sample High-priority events:")
    high_risk = df[df["risk_level"] == "High"].sort_values(
        "risk_rank_score", ascending=False
    )
    cols = ["user", "device", "target_device", "risk_score", "risk_level", "risk_reasons"]
    print(high_risk[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
