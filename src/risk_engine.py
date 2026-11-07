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

Usage:
    python -m src.risk_engine

Input:
    data/processed/auth_events_scored.csv   (from src/model.py)

Output:
    data/processed/auth_events_risk.csv     (adds risk_score, risk_level, risk_reasons)
"""

import os
import pandas as pd

from src.dataset_paths import RISK_FILE as OUTPUT_FILE
from src.dataset_paths import SCORED_FILE as INPUT_FILE

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

# Final risk level thresholds (0-100 scale)
HIGH_THRESHOLD = 70
MEDIUM_THRESHOLD = 40


def compute_risk(df):
    df = df.copy()

    # 1) Normalize the anomaly score into a 0-100 percentile. Percentile
    # rank (rather than raw min-max scaling) keeps this stable even if the
    # anomaly score's raw range shifts as you retrain the model later.
    df["anomaly_percentile"] = df["anomaly_score_iforest"].rank(pct=True) * 100

    # 2) Hand-coded risk factors — each one is independently checkable,
    # which is exactly what makes this engine transparent.
    df["factor_new_device"] = df["is_new_target_device"] == 1
    df["factor_sensitive_asset"] = df["target_device"].isin(SENSITIVE_ASSETS)
    df["factor_failed_login"] = df["is_failed"] == 1
    df["factor_burst_activity"] = df["rolling_count_1h"] >= BURST_THRESHOLD

    bonus = (
        df["factor_new_device"] * RISK_FACTOR_POINTS["new_device"]
        + df["factor_sensitive_asset"] * RISK_FACTOR_POINTS["sensitive_asset"]
        + df["factor_failed_login"] * RISK_FACTOR_POINTS["failed_login"]
        + df["factor_burst_activity"] * RISK_FACTOR_POINTS["burst_activity"]
    )

    # 3) Combine anomaly percentile with the factor bonus, capped at 100.
    df["risk_score"] = (df["anomaly_percentile"] + bonus).clip(upper=100).round(1)

    # 4) Bucket into Low / Medium / High
    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, MEDIUM_THRESHOLD, HIGH_THRESHOLD, 101],
        labels=["Low", "Medium", "High"],
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

    return df


def evaluate(df, label_col="is_malicious"):
    """Sanity check: how does risk_level line up with known malicious events?"""
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

    evaluate(df)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[done] risk-scored events written to {OUTPUT_FILE}")

    print("\n[info] sample High-risk events:")
    high_risk = df[df["risk_level"] == "High"].sort_values("risk_score", ascending=False)
    cols = ["user", "device", "target_device", "risk_score", "risk_level", "risk_reasons"]
    print(high_risk[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()