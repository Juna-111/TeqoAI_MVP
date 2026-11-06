"""
TeqoAI — Step 2: Anomaly Detection Model

Builds features from the cleaned auth event data (Step 1's output) and
trains an Isolation Forest to flag unusual login/authentication behavior.
A PyTorch autoencoder is also included as a second approach for
comparison, per the proposal's "either or both" methodology.

Both models are UNSUPERVISED — they never see the is_malicious label
during training. That label is used only afterward, to evaluate how well
each model's anomaly ranking lines up with known red-team events in a
chronologically held-out test period.

Usage:
    python -m src.model

Input:
    data/processed/auth_events_clean.csv   (from src/data_prep.py)

Output:
    data/processed/auth_events_scored.csv  (adds anomaly scores per event)
    models/isolation_forest.pkl
    models/scaler.pkl
    models/autoencoder.pt                  (if PyTorch available)
    models/autoencoder_scaler.pkl          (if PyTorch available)
"""

import importlib.metadata
import json
import os
import platform
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.dataset_paths import CLEAN_FILE as INPUT_FILE
from src.dataset_paths import (
    DATASET,
    EVALUATION_FILE,
    MODELS_DIR,
    SCORED_FILE as OUTPUT_FILE,
)
from src.evaluation import evaluate_ranked_scores
from src.provenance import (
    PIPELINE_SCHEMA_VERSION,
    create_pipeline_run_id,
    sha256_file,
)
from src.risk_engine import (
    BURST_THRESHOLD,
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    RISK_FACTOR_POINTS,
    SENSITIVE_ASSETS,
)

# Evaluate only the latest chronological portion against ground truth.
TEST_FRACTION = 0.3
ALERT_BUDGETS = (30, 100)
RANDOM_STATE = 42
CONTAMINATION = 0.1
AUTOENCODER_EPOCHS = 30
WALK_FORWARD_FOLDS = 3
AUTOENCODER_BATCH_SIZE = 256
AUTOENCODER_SCORE_BATCH_SIZE = 4096


# ---- Feature engineering ---------------------------------------------------

def action_categories(df):
    """Return categorical values observed in the supplied training events."""
    split_action = df["action"].fillna("").astype(str).str.split("/", n=1, expand=True)
    if split_action.shape[1] < 2:
        split_action[1] = "Unknown"
    auth_types = sorted(split_action[0].replace("", "Unknown").unique().tolist())
    logon_types = sorted(
        split_action[1].fillna("Unknown").replace("", "Unknown").unique().tolist()
    )
    return auth_types, logon_types


def build_features(df, categories=None):
    """Turn the common-schema event log into a numeric feature matrix.

    Features:
      - is_failed                            (failed login attempt)
      - is_new_target_device                 (first time this user reached this device)
      - rolling_count_1h                     (this user's login volume in the past hour)
      - one-hot auth_type / logon_type        (from the combined `action` field)

    Event timestamps are elapsed seconds, not wall-clock timestamps. In
    particular, LANL does not disclose the calendar epoch, so hour-of-day
    and weekday features would be fabricated. Elapsed time is used only
    for duration-based features such as the rolling one-hour count.
    """
    df = df.copy()
    df["time"] = df["time"].astype(int)
    df = df.sort_values(
        ["user", "time", "device", "target_device", "action", "result"],
        kind="stable",
    ).reset_index(drop=True)

    # Result / device novelty features
    df["is_failed"] = (df["result"].str.lower() != "success").astype(int)
    first_seen_time = df.groupby(
        ["user", "target_device"], sort=False, dropna=False
    )["time"].transform("min")
    df["is_new_target_device"] = (df["time"] == first_seen_time).astype(int)

    # Exclude simultaneous events from the history, then count the current
    # event once. Peers at the same timestamp must not influence one another.
    df["elapsed_time"] = pd.to_timedelta(df["time"], unit="s")
    df = df.set_index("elapsed_time")
    df["rolling_count_1h"] = (
        df.groupby("user", sort=False)["time"]
        .rolling("1h", closed="left")
        .count()
        .reset_index(level=0, drop=True)
        .fillna(0)
        .add(1)
    )
    df = df.reset_index()

    # Split the combined action field ("AuthType/LogonType") and one-hot encode
    split_action = df["action"].fillna("").astype(str).str.split("/", n=1, expand=True)
    if split_action.shape[1] < 2:
        split_action[1] = "Unknown"
    df["auth_type"] = split_action[0].replace("", "Unknown")
    df["logon_type"] = split_action[1].fillna("Unknown").replace("", "Unknown")
    auth_categories, logon_categories = categories or action_categories(df)
    known_auth = df["auth_type"].where(df["auth_type"].isin(auth_categories))
    known_logon = df["logon_type"].where(df["logon_type"].isin(logon_categories))
    auth_dummies = pd.get_dummies(
        pd.Categorical(known_auth, categories=auth_categories),
        prefix="auth",
        dtype=int,
    )
    auth_dummies.index = df.index
    logon_dummies = pd.get_dummies(
        pd.Categorical(known_logon, categories=logon_categories),
        prefix="logon",
        dtype=int,
    )
    logon_dummies.index = df.index
    action_dummies = pd.concat([auth_dummies, logon_dummies], axis=1)

    numeric_features = df[
        ["is_failed", "is_new_target_device", "rolling_count_1h"]
    ]

    X = pd.concat([numeric_features, action_dummies], axis=1).fillna(0)
    return X, df


def align_features(features, feature_names):
    """Match scoring features to the model's training-time columns."""
    return features.reindex(columns=list(feature_names), fill_value=0)


# ---- Isolation Forest -------------------------------------------------------

def temporal_split(df, test_fraction=TEST_FRACTION):
    """Split by timestamp, keeping equal timestamps on the same side."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    unique_times = np.sort(df["time"].unique())
    if len(unique_times) < 2:
        raise ValueError("At least two distinct event times are required for a temporal split.")

    cutoff_index = int(len(unique_times) * (1 - test_fraction))
    cutoff_index = min(max(cutoff_index, 1), len(unique_times) - 1)
    cutoff_time = unique_times[cutoff_index]
    train_mask = df["time"] < cutoff_time
    test_mask = ~train_mask
    return train_mask, test_mask, int(cutoff_time)


def chronological_validation_splits(df, training_mask, n_splits=WALK_FORWARD_FOLDS):
    """Create expanding-window validation folds strictly before the final holdout."""
    if n_splits < 1:
        raise ValueError("n_splits must be a positive integer.")

    training_times = np.sort(df.loc[training_mask, "time"].unique())
    first_validation_index = max(2, int(len(training_times) * 0.5))
    validation_times = training_times[first_validation_index:]
    if len(validation_times) == 0:
        return []

    windows = np.array_split(validation_times, min(n_splits, len(validation_times)))
    folds = []
    for window in windows:
        if len(window) == 0:
            continue
        train_times = training_times[training_times < window[0]]
        train_mask = df["time"].isin(train_times)
        validation_mask = df["time"].isin(window)
        if int(train_mask.sum()) > 1 and int(validation_mask.sum()) > 0:
            folds.append((train_mask, validation_mask))
    return folds


def evaluate_walk_forward(df, n_splits=WALK_FORWARD_FOLDS):
    """Compare every available ranking on identical expanding time windows."""
    outer_train_mask, _, _ = temporal_split(df)
    folds = chronological_validation_splits(df, outer_train_mask, n_splits)
    fold_results = []

    for fold_number, (train_mask, validation_mask) in enumerate(folds, start=1):
        train_events = df.loc[train_mask]
        validation_events = df.loc[validation_mask]
        fold_categories = action_categories(train_events)

        prefix = df.loc[df["time"] <= validation_events["time"].max()]
        prefix_features, prepared_prefix = build_features(prefix, fold_categories)
        prepared_train_mask = prepared_prefix["time"] < validation_events["time"].min()
        prepared_validation_mask = prepared_prefix["time"].isin(
            validation_events["time"].unique()
        )
        X_train = prefix_features.loc[prepared_train_mask]
        X_validation = prefix_features.loc[prepared_validation_mask]
        y_validation = prepared_prefix.loc[
            prepared_validation_mask, "is_malicious"
        ].to_numpy()

        model_metrics = {
            "first_seen_destination_baseline": evaluate_ranked_scores(
                y_validation,
                prepared_prefix.loc[
                    prepared_validation_mask, "is_new_target_device"
                ].to_numpy(dtype=float),
                ALERT_BUDGETS,
            )
        }

        _, _, scores, _ = train_isolation_forest(
            X_train,
            X_validation,
            contamination=CONTAMINATION,
            random_state=RANDOM_STATE,
        )
        model_metrics["isolation_forest"] = evaluate_ranked_scores(
            y_validation, scores, ALERT_BUDGETS
        )
        autoencoder, autoencoder_scores, _ = train_autoencoder(
            X_train,
            X_validation,
            epochs=AUTOENCODER_EPOCHS,
            random_state=RANDOM_STATE,
        )
        model_metrics["autoencoder"] = (
            evaluate_ranked_scores(y_validation, autoencoder_scores, ALERT_BUDGETS)
            if autoencoder_scores is not None
            else {"status": "not_available"}
        )
        fold_results.append(
            {
                "fold": fold_number,
                "training_events": int(len(X_train)),
                "validation_events": int(len(X_validation)),
                "validation_start_elapsed_seconds": int(validation_events["time"].min()),
                "validation_end_elapsed_seconds": int(validation_events["time"].max()),
                "metrics": model_metrics["isolation_forest"],
                "models": model_metrics,
            }
        )

    return {
        "strategy": (
            "expanding chronological windows within the final training period; "
            "all models use identical train and validation events"
        ),
        "requested_folds": int(n_splits),
        "completed_folds": len(fold_results),
        "model": "isolation_forest",
        "models": [
            "first_seen_destination_baseline",
            "isolation_forest",
            "autoencoder",
        ],
        "folds": fold_results,
    }


def train_isolation_forest(X_train, X_score=None, contamination=0.1, random_state=42):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_score_scaled = scaler.transform(X_train if X_score is None else X_score)
    X_train_scaled = scaler.transform(X_train)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_train_scaled)

    # sklearn convention: higher decision_function = more normal.
    # We flip the sign so higher score = more anomalous, which is more
    # intuitive for the risk engine and dashboard built in later steps.
    anomaly_score = -model.decision_function(X_score_scaled)
    is_flagged = (model.predict(X_score_scaled) == -1).astype(int)

    return model, scaler, anomaly_score, is_flagged


# ---- Autoencoder (optional second approach) --------------------------------

def build_autoencoder(input_dim):
    """Construct the small, deterministic dense reconstruction baseline."""
    import torch.nn as nn

    hidden_dim = max(4, input_dim // 2)

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, max(2, hidden_dim // 2)), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(max(2, hidden_dim // 2), hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return Autoencoder(), hidden_dim


def score_autoencoder(
    model, scaler, X_score, batch_size=AUTOENCODER_SCORE_BATCH_SIZE
):
    """Calculate reconstruction errors in bounded batches."""
    import torch

    if (
        not isinstance(batch_size, (int, np.integer))
        or isinstance(batch_size, (bool, np.bool_))
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X_score), batch_size):
            batch = X_score.iloc[start:start + batch_size]
            scaled_batch = scaler.transform(batch).astype("float32")
            tensor = torch.from_numpy(scaled_batch)
            reconstructed = model(tensor)
            scores.append(
                (tensor - reconstructed).square().mean(dim=1).cpu().numpy()
            )
    return np.concatenate(scores) if scores else np.empty(0, dtype=float)


def save_autoencoder_artifacts(model, scaler, model_path, scaler_path):
    import torch

    input_dim = int(scaler.n_features_in_)
    hidden_dim = max(4, input_dim // 2)
    torch.save(
        {"input_dim": input_dim, "hidden_dim": hidden_dim, "state_dict": model.state_dict()},
        model_path,
    )
    joblib.dump(scaler, scaler_path)


def load_autoencoder_artifacts(model_path, scaler_path):
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model, hidden_dim = build_autoencoder(int(checkpoint["input_dim"]))
    if hidden_dim != int(checkpoint["hidden_dim"]):
        raise ValueError("Autoencoder artifact architecture metadata is inconsistent.")
    scaler = joblib.load(scaler_path)
    if int(scaler.n_features_in_) != int(checkpoint["input_dim"]):
        raise ValueError("Autoencoder scaler and model feature counts do not match.")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, scaler


def train_autoencoder(
    X_train,
    X_score=None,
    epochs=AUTOENCODER_EPOCHS,
    random_state=42,
    batch_size=AUTOENCODER_BATCH_SIZE,
):
    """A simple autoencoder: trained to reconstruct normal-looking events.
    High reconstruction error = anomaly. X_score is transformed with a
    scaler fitted only on X_train. Training and scoring use bounded batches.
    Returns None if PyTorch isn't installed, so this step is optional rather
    than a hard dependency."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as error:
        print(
            f"[info] PyTorch unavailable ({type(error).__name__}) - skipping "
            "autoencoder, Isolation Forest results are still complete on their own."
        )
        return None, None, None

    if len(X_train) == 0:
        raise ValueError("X_train must contain at least one event.")
    if (
        not isinstance(epochs, (int, np.integer))
        or isinstance(epochs, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or isinstance(batch_size, (bool, np.bool_))
        or epochs < 1
        or batch_size < 1
    ):
        raise ValueError("epochs and batch_size must be positive integers.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype("float32")
    X_tensor = torch.tensor(X_train_scaled)
    generator = torch.Generator(device="cpu").manual_seed(random_state)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(random_state)
        model, _ = build_autoencoder(X_tensor.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(len(X_tensor), generator=generator)
            epoch_loss = 0.0
            for start in range(0, len(X_tensor), batch_size):
                indices = permutation[start:start + batch_size]
                batch = X_tensor[indices]
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = loss_fn(reconstructed, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            epoch_loss /= len(X_tensor)
            if (epoch + 1) % 10 == 0:
                print(
                    f"[info] autoencoder epoch {epoch+1}/{epochs} "
                    f"- loss: {epoch_loss:.4f}"
                )

    scores = score_autoencoder(
        model, scaler, X_train if X_score is None else X_score
    )
    return model, scores, scaler


# ---- Evaluation --------------------------------------------------------------

def evaluate(df, score_col, label_col="is_malicious"):
    """Compute held-out ranking metrics against known label matches."""
    return evaluate_ranked_scores(
        df[label_col].to_numpy(),
        df[score_col].to_numpy(),
        alert_budgets=ALERT_BUDGETS,
    )


def write_evaluation_report(
    df,
    train_mask,
    test_mask,
    cutoff_time,
    feature_count,
    model_metrics,
    walk_forward_metrics,
):
    """Write a reproducible manifest and held-out metric report."""
    train_labels = int(df.loc[train_mask, "is_malicious"].sum())
    test_labels = int(df.loc[test_mask, "is_malicious"].sum())
    input_sha256 = sha256_file(INPUT_FILE)
    package_versions = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "pandas", "scikit-learn")
    }
    try:
        package_versions["torch"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        package_versions["torch"] = None
    run_config = {
        "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
        "python": platform.python_version(),
        "packages": package_versions,
        "random_state": RANDOM_STATE,
        "isolation_forest_contamination": CONTAMINATION,
        "autoencoder_epochs": AUTOENCODER_EPOCHS,
        "autoencoder_batch_size": AUTOENCODER_BATCH_SIZE,
        "autoencoder_score_batch_size": AUTOENCODER_SCORE_BATCH_SIZE,
        "test_fraction": TEST_FRACTION,
        "walk_forward_folds": WALK_FORWARD_FOLDS,
        "risk_engine": {
            "factor_points": RISK_FACTOR_POINTS,
            "sensitive_assets": sorted(SENSITIVE_ASSETS),
            "burst_threshold": BURST_THRESHOLD,
            "high_threshold": HIGH_THRESHOLD,
            "medium_threshold": MEDIUM_THRESHOLD,
        },
    }
    run_id = create_pipeline_run_id(
        DATASET, input_sha256, cutoff_time, feature_count, run_config
    )
    report = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "run_id": run_id,
        "dataset_profile": DATASET,
        "input": {
            "file": os.path.relpath(INPUT_FILE, os.path.dirname(EVALUATION_FILE)),
            "sha256": input_sha256,
        },
        "split": {
            "strategy": "chronological unique timestamps; earlier events train, latest events test",
            "test_fraction": TEST_FRACTION,
            "cutoff_elapsed_seconds": int(cutoff_time),
            "training_events": int(train_mask.sum()),
            "test_events": int(test_mask.sum()),
            "training_known_positive_labels": train_labels,
            "test_known_positive_labels": test_labels,
        },
        "evaluation": {
            "label_definition": "Exact matches to red-team ground-truth records.",
            "negative_label_caveat": (
                "Non-matching events are treated as negative for these metrics, "
                "but may include malicious activity not present in the ground-truth labels."
            ),
            "alert_budgets": list(ALERT_BUDGETS),
            "feature_count": int(feature_count),
            "models": model_metrics,
        },
        "walk_forward_model_comparison": (
            "The first-seen baseline, Isolation Forest, and optional autoencoder "
            "are evaluated on the same chronological folds. Fold metrics are "
            "reported per model."
        ),
        "walk_forward_validation": walk_forward_metrics,
        "score_semantics": (
            "Anomaly rankings and risk scores are prioritization signals, not "
            "calibrated probabilities of compromise."
        ),
        "reproducibility": {
            **run_config,
        },
        "limitations": [
            "Metrics measure agreement with known labels, not proof of malicious activity.",
            "A test window with zero known positives is not suitable for detection-quality claims.",
            "Unsupervised models are trained without labels and may include malicious events in their training data.",
            "The autoencoder uses fixed-epoch reconstruction ranking; its scores are not calibrated probabilities.",
        ],
    }
    os.makedirs(os.path.dirname(EVALUATION_FILE), exist_ok=True)
    with open(EVALUATION_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=True)
        file.write("\n")

    return report


# ---- Main --------------------------------------------------------------------

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Run src/data_prep.py first."
        )
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[info] loading cleaned event data...")
    raw_df = pd.read_csv(INPUT_FILE)

    raw_train_mask, _, cutoff_time = temporal_split(raw_df)
    training_categories = action_categories(raw_df.loc[raw_train_mask])

    print("[info] building features...")
    X, df = build_features(raw_df, training_categories)
    print(f"[info] feature matrix shape: {X.shape}")

    train_mask, test_mask, _ = temporal_split(df)
    X_train = X.loc[train_mask]
    X_all = X
    print(
        f"[info] chronological split at elapsed second {cutoff_time}: "
        f"{len(X_train):,} training events, {int(test_mask.sum()):,} held-out events "
        f"({int(df.loc[train_mask, 'is_malicious'].sum()):,} and "
        f"{int(df.loc[test_mask, 'is_malicious'].sum()):,} labeled events)"
    )
    df["evaluation_split"] = np.where(test_mask, "test", "train")
    print("[info] running expanding-window validation within the training period...")
    walk_forward_metrics = evaluate_walk_forward(raw_df)
    print(
        f"[evaluation] walk-forward windows: "
        f"{walk_forward_metrics['completed_folds']}/"
        f"{walk_forward_metrics['requested_folds']} completed"
    )
    model_metrics = {}

    print("[info] evaluating first-seen destination baseline...")
    df["anomaly_score_first_seen_baseline"] = df["is_new_target_device"].astype(float)
    model_metrics["first_seen_destination_baseline"] = evaluate(
        df.loc[test_mask], "anomaly_score_first_seen_baseline"
    )

    print("[info] training Isolation Forest...")
    if_model, if_scaler, if_scores, if_flagged = train_isolation_forest(
        X_train, X_all, contamination=CONTAMINATION, random_state=RANDOM_STATE
    )
    df["anomaly_score_iforest"] = if_scores
    df["is_flagged_iforest"] = if_flagged
    model_metrics["isolation_forest"] = evaluate(
        df.loc[test_mask], "anomaly_score_iforest"
    )

    print("[info] training autoencoder (optional second model)...")
    ae_model, ae_scores, ae_scaler = train_autoencoder(
        X_train, X_all, epochs=AUTOENCODER_EPOCHS, random_state=RANDOM_STATE
    )
    if ae_scores is not None:
        df["anomaly_score_autoencoder"] = ae_scores
        model_metrics["autoencoder"] = evaluate(
            df.loc[test_mask], "anomaly_score_autoencoder"
        )
    else:
        model_metrics["autoencoder"] = {"status": "not_available"}

    for model_name, metrics in model_metrics.items():
        print(f"[evaluation] {model_name}: {metrics['status']}")
        if metrics["status"] == "evaluated":
            print(
                f"  AP={metrics['average_precision']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']!s}"
            )
            for budget, result in metrics["alert_budgets"].items():
                print(
                    f"  top-{budget}: precision={result['precision']:.4f}, "
                    f"recall={result['recall']:.4f}, "
                    f"known hits={result['known_positive_hits']}/"
                    f"{result['selected_events']}, "
                    f"random expected precision="
                    f"{result['random_baseline_expected_precision']:.4f}"
                )
        else:
            print(
                "  Held-out metrics are unavailable; "
                "the report includes event and known-label coverage."
            )

    report = write_evaluation_report(
        df,
        train_mask,
        test_mask,
        cutoff_time,
        X.shape[1],
        model_metrics,
        walk_forward_metrics,
    )
    df["pipeline_run_id"] = report["run_id"]
    df_out = df.drop(columns=["elapsed_time"], errors="ignore")
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[done] scored events written to {OUTPUT_FILE}")
    print(f"[done] evaluation report written to {EVALUATION_FILE}")

    # Save model artifacts
    joblib.dump(if_model, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    joblib.dump(if_scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    if ae_model is not None:
        try:
            save_autoencoder_artifacts(
                ae_model,
                ae_scaler,
                os.path.join(MODELS_DIR, "autoencoder.pt"),
                os.path.join(MODELS_DIR, "autoencoder_scaler.pkl"),
            )
        except Exception as e:
            print(f"[warn] could not save autoencoder weights ({type(e).__name__}) - "
                  f"Isolation Forest artifacts were saved successfully regardless.")
    print(f"[done] model artifacts saved to {os.path.normpath(MODELS_DIR)}")


if __name__ == "__main__":
    main()
