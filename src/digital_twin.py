"""
TeqoAI — Step 4: Cyber Digital Twin, Attack-Path Analysis & Explainability

This module covers the rest of the MVP's analytical core in one place:

  1. Build a small Cyber Digital Twin: a directed graph of assets (users,
     laptops, servers, applications, databases) from successful auth events,
     with each asset tagged by type and criticality.
  2. Attack-path analysis: given a starting point (e.g. a compromised
     user's laptop), find which assets are reachable and how risky each
     path is, based on asset criticality.
  3. Explainability: SHAP-based, human-readable justification for why a
     specific event got the anomaly score it did.

Usage (as a demo):
    python -m src.digital_twin

Input:
    data/processed/auth_events_clean.csv    (from src/data_prep.py)
    data/processed/auth_events_risk.csv     (from src/risk_engine.py)
    models/isolation_forest.pkl, models/scaler.pkl

The dashboard (Step 5) imports build_graph(), get_attack_paths(), and
explain_event() directly from this file rather than duplicating logic.
"""

import json
import os
import joblib
import pandas as pd
import networkx as nx

from src.dataset_paths import CLEAN_FILE, MODELS_DIR, RISK_FILE

# Reuse the exact same feature engineering used for training, so SHAP
# explanations are computed on identical features to the live model.
from src.model import align_features, build_features

# Asset criticality — used to weight how "bad" a reachable asset is.
# Higher = more valuable / sensitive if reached by a compromised account.
CRITICALITY = {
    "workstation": 1,
    "server": 2,
    "application": 3,
    "database": 4,
    "user": 0,
}


def classify_asset(node_name: str) -> str:
    """Infer an asset's type from its naming convention in the data."""
    if node_name.startswith("DB"):
        return "database"
    if node_name.startswith("APP"):
        return "application"
    if node_name.startswith("SRV"):
        return "server"
    if node_name.startswith("C") and node_name[1:].isdigit():
        return "workstation"
    return "other"


# ---- 1) Build the Cyber Digital Twin ---------------------------------------

def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build a graph of device relationships confirmed by successful auth events.

    Failed attempts are not evidence that a source reached a destination and
    must not contribute graph edges or attack paths.
    """
    required_columns = {"device", "target_device", "result"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Authentication outcomes are required to build observed paths; "
            f"missing columns: {', '.join(sorted(missing_columns))}."
        )

    G = nx.DiGraph()

    successful_events = df.loc[
        df["result"].astype("string").str.strip().str.casefold().eq("success").fillna(False)
    ]
    for row in successful_events.itertuples(index=False):
        src, dst = row.device, row.target_device
        for node in (src, dst):
            if node not in G:
                asset_type = classify_asset(node)
                G.add_node(node, asset_type=asset_type, criticality=CRITICALITY.get(asset_type, 1))
        if G.has_edge(src, dst):
            G[src][dst]["weight"] += 1
        else:
            G.add_edge(src, dst, weight=1)

    return G


# ---- 2) Attack-path analysis ------------------------------------------------

def get_attack_paths(
    G: nx.DiGraph, start_node: str, max_hops: int = 4
) -> list[dict[str, object]]:
    """From a starting asset (typically a compromised user's laptop),
    find every asset reachable within max_hops, the path to reach it, and
    a risk score for that path based on the criticality of assets touched
    along the way."""
    if start_node not in G:
        return []

    results = []
    # One bounded breadth-first search yields the shortest path to every
    # reachable asset, so large observed graphs stay cheap to query.
    shortest_paths = nx.single_source_shortest_path(
        G, start_node, cutoff=max_hops
    )

    for target, path in shortest_paths.items():
        if target == start_node:
            continue
        hops = len(path) - 1

        path_criticality = sum(G.nodes[n].get("criticality", 1) for n in path)
        max_criticality = max(G.nodes[n].get("criticality", 1) for n in path)

        results.append({
            "target": target,
            "target_type": G.nodes[target].get("asset_type", "other"),
            "hops": hops,
            "path": path,
            "path_risk": path_criticality,
            "max_asset_criticality": max_criticality,
        })

    # Most concerning paths first: highest-value asset reached, fewest hops
    results.sort(key=lambda r: (-r["max_asset_criticality"], r["hops"]))
    return results


# ---- 3) Explainability (SHAP) -----------------------------------------------

def load_model_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return model, scaler


def format_feature_name(name):
    """Translate encoded model columns into concise analyst-facing labels."""
    labels = {
        "is_failed": "Failed authentication activity",
        "is_new_target_device": "First-seen user-to-destination pairing",
        "rolling_count_1h": "Authentication volume in the prior hour",
    }
    if name in labels:
        return labels[name]
    if name.startswith("auth_"):
        return f"Authentication method: {name.removeprefix('auth_')}"
    if name.startswith("logon_"):
        return f"Logon category: {name.removeprefix('logon_')}"
    return name.replace("_", " ")


def explain_event(
    model,
    scaler,
    features,
    row_index,
    background,
    max_background=128,
    max_features=12,
):
    """Compute exact interventional Shapley values for TeqoAI's anomaly score."""
    import math

    import numpy as np

    if not isinstance(features, pd.DataFrame) or not isinstance(background, pd.DataFrame):
        raise TypeError("features and background must be pandas DataFrames.")
    if features.empty or background.empty:
        raise ValueError("features and background must contain at least one row.")
    if not features.columns.is_unique or not background.columns.is_unique:
        raise ValueError("Feature and background column names must be unique.")
    if list(features.columns) != list(background.columns):
        raise ValueError("Background columns must match feature columns in order.")
    if (
        not isinstance(row_index, (int, np.integer))
        or isinstance(row_index, (bool, np.bool_))
        or row_index not in features.index
    ):
        raise ValueError("row_index must identify a row in features.")
    if (
        not isinstance(max_background, (int, np.integer))
        or isinstance(max_background, (bool, np.bool_))
        or max_background < 1
    ):
        raise ValueError("max_background must be a positive integer.")

    grouped_columns = {}
    grouped_labels = {}
    for column_index, column_name in enumerate(features.columns):
        if column_name.startswith("auth_"):
            group_key, group_label = "authentication_method", "Authentication method"
        elif column_name.startswith("logon_"):
            group_key, group_label = "logon_category", "Logon category"
        else:
            group_key = column_name
            group_label = format_feature_name(column_name)
        grouped_columns.setdefault(group_key, []).append(column_index)
        grouped_labels[group_key] = group_label

    group_keys = list(grouped_columns)
    feature_count = len(group_keys)
    if feature_count > max_features:
        raise ValueError(
            f"Exact Shapley enumeration supports at most {max_features} semantic "
            f"features; received {feature_count}."
        )

    try:
        feature_values = features.to_numpy(dtype=float)
        background_values = background.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Model features and background values must be numeric.") from error
    if not np.isfinite(feature_values).all() or not np.isfinite(background_values).all():
        raise ValueError("Model features and background values must be finite.")

    sample_count = min(len(background_values), int(max_background))
    sample_positions = np.linspace(
        0, len(background_values) - 1, num=sample_count, dtype=int
    )
    target_values = features.loc[[row_index]].to_numpy(dtype=float)
    scaled_target = scaler.transform(
        pd.DataFrame(target_values, columns=features.columns)
    )
    scaled_background = scaler.transform(
        pd.DataFrame(background_values[sample_positions], columns=features.columns)
    )

    def anomaly_scores(matrix):
        scores = -np.asarray(model.decision_function(matrix), dtype=float).reshape(-1)
        if not np.isfinite(scores).all():
            raise ValueError("Isolation Forest returned non-finite decision scores.")
        return scores

    coalition_count = 1 << feature_count
    coalition_values = np.empty(coalition_count, dtype=float)
    for coalition in range(coalition_count):
        masked_background = scaled_background.copy()
        for feature_index, group_key in enumerate(group_keys):
            if coalition & (1 << feature_index):
                feature_columns = grouped_columns[group_key]
                masked_background[:, feature_columns] = scaled_target[
                    0, feature_columns
                ]
        coalition_values[coalition] = anomaly_scores(masked_background).mean()

    factorials = [math.factorial(value) for value in range(feature_count + 1)]
    shap_values = np.zeros(feature_count, dtype=float)
    interactions = np.zeros((feature_count, feature_count), dtype=float)
    for feature_index in range(feature_count):
        feature_bit = 1 << feature_index
        for coalition in range(coalition_count):
            if coalition & feature_bit:
                continue
            size = coalition.bit_count()
            weight = (
                factorials[size]
                * factorials[feature_count - size - 1]
                / factorials[feature_count]
            )
            shap_values[feature_index] += weight * (
                coalition_values[coalition | feature_bit] - coalition_values[coalition]
            )

    for first in range(feature_count):
        first_bit = 1 << first
        for second in range(first + 1, feature_count):
            second_bit = 1 << second
            pair_value = 0.0
            for coalition in range(coalition_count):
                if coalition & (first_bit | second_bit):
                    continue
                size = coalition.bit_count()
                weight = (
                    factorials[size]
                    * factorials[feature_count - size - 2]
                    / (2 * factorials[feature_count - 1])
                )
                pair_value += weight * (
                    coalition_values[coalition | first_bit | second_bit]
                    - coalition_values[coalition | first_bit]
                    - coalition_values[coalition | second_bit]
                    + coalition_values[coalition]
                )
            interactions[first, second] = pair_value
            interactions[second, first] = pair_value
    np.fill_diagonal(interactions, shap_values - interactions.sum(axis=1))

    base_value = float(coalition_values[0])
    prediction = float(anomaly_scores(scaled_target)[0])
    additivity_error = float(base_value + shap_values.sum() - prediction)
    if not np.isclose(base_value + shap_values.sum(), prediction, rtol=1e-9, atol=1e-10):
        raise ArithmeticError("Shapley values failed the anomaly-score additivity check.")
    if not np.allclose(interactions.sum(axis=1), shap_values, rtol=1e-9, atol=1e-10):
        raise ArithmeticError("Shapley interaction values do not sum to feature values.")

    sampled_background = background_values[sample_positions]
    group_encoded_names = {
        group_key: [features.columns[column] for column in columns]
        for group_key, columns in grouped_columns.items()
    }
    background_comparison = {}
    for group_key, columns in grouped_columns.items():
        encoded_names = group_encoded_names[group_key]
        if group_key in {"authentication_method", "logon_category"}:
            category_counts = {}
            for background_row in sampled_background:
                active = [
                    name.split("_", 1)[1]
                    for name, column in zip(encoded_names, columns)
                    if background_row[column] >= 0.5
                ]
                category = ", ".join(active) if active else "Other / unknown category"
                category_counts[category] = category_counts.get(category, 0) + 1
            category, count = max(
                category_counts.items(), key=lambda item: item[1]
            )
            background_comparison[group_key] = {
                "kind": "category",
                "value": category,
                "share": float(count / sample_count),
            }
        else:
            background_comparison[group_key] = {
                "kind": "numeric",
                "value": float(sampled_background[:, columns[0]].mean()),
                "share": None,
            }

    feature_rows = []
    for index, group_key in enumerate(group_keys):
        columns = grouped_columns[group_key]
        encoded_names = group_encoded_names[group_key]
        if group_key in {"authentication_method", "logon_category"}:
            active = [
                name.split("_", 1)[1]
                for name, column in zip(encoded_names, columns)
                if target_values[0, column] >= 0.5
            ]
            value = ", ".join(active) if active else "Other / unknown category"
        else:
            value = float(target_values[0, columns[0]])
        feature_rows.append({
            "feature": group_key,
            "label": grouped_labels[group_key],
            "encoded_features": encoded_names,
            "value": value,
            "background": background_comparison[group_key],
            "shap_value": float(shap_values[index]),
        })
    feature_rows.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    pair_rows = []
    for first in range(feature_count):
        for second in range(first + 1, feature_count):
            contribution = float(2 * interactions[first, second])
            if abs(contribution) > 1e-12:
                pair_rows.append({
                    "feature_a": grouped_labels[group_keys[first]],
                    "feature_b": grouped_labels[group_keys[second]],
                    "contribution": contribution,
                })
    pair_rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    return {
        "output": "Isolation Forest anomaly score (-decision_function)",
        "output_semantics": "Higher values indicate more anomalous events.",
        "method": (
            "Exact interventional Shapley values with exhaustive semantic-feature "
            "coalition enumeration; one-hot categories are grouped to preserve "
            "valid category states."
        ),
        "background": {
            "reference": "Training-period rows only, selected at evenly spaced positions.",
            "rows_used": int(sample_count),
            "available_training_rows": int(len(background_values)),
        },
        "base_value": base_value,
        "prediction": prediction,
        "drift": float(prediction - base_value),
        "shap_sum": float(shap_values.sum()),
        "additivity_error": additivity_error,
        "features": feature_rows,
        "positive_drivers": [item for item in feature_rows if item["shap_value"] > 0],
        "negative_drivers": [item for item in feature_rows if item["shap_value"] < 0],
        "interactions": pair_rows,
        "interaction_convention": (
            "Each unique pair is twice its symmetric off-diagonal interaction "
            "value. Interaction values are descriptive and must not be added "
            "again to the main-effect Shapley sum."
        ),
        "dependence_note": (
            "Authentication-method and logon-category one-hot columns are grouped "
            "as single features to avoid invalid multi-category states."
        ),
    }


# ---- Demo / self-test --------------------------------------------------------

def main():
    print("[info] loading cleaned events and rebuilding model features...")
    df_clean = pd.read_csv(CLEAN_FILE)
    X, _ = build_features(df_clean)

    print("[info] loading risk-scored events...")
    df_risk = pd.read_csv(RISK_FILE)

    print("[info] building the Cyber Digital Twin graph...")
    G = build_graph(df_clean)
    print(f"[info] graph built: {G.number_of_nodes()} assets, {G.number_of_edges()} relationships")

    print("[info] loading Isolation Forest for exact Shapley explanation...")
    model, scaler = load_model_artifacts()
    X = align_features(X, scaler.feature_names_in_)

    # Pick one known-malicious event (if any) to demo attack path + explanation
    malicious = df_risk[df_risk["is_malicious"] == 1].sort_values("risk_score", ascending=False)
    if malicious.empty:
        print("[info] no malicious events in this dataset to demo — showing top risk event instead.")
        demo_row = df_risk.sort_values("risk_score", ascending=False).iloc[0]
    else:
        demo_row = malicious.iloc[0]

    print(f"\n[demo] Selected event: user={demo_row['user']}, "
          f"device={demo_row['device']} -> target_device={demo_row['target_device']}, "
          f"risk_level={demo_row['risk_level']}")

    print("\n[demo] Attack paths from this device:")
    paths = get_attack_paths(G, demo_row["device"])
    for p in paths[:5]:
        print(f"  -> {p['target']} ({p['target_type']}, criticality peak "
              f"{p['max_asset_criticality']}) in {p['hops']} hop(s): "
              f"{' -> '.join(p['path'])}")

    training_rows = (
        df_risk["evaluation_split"].eq("train")
        if "evaluation_split" in df_risk
        else pd.Series(True, index=df_risk.index)
    )
    explanation = explain_event(
        model, scaler, X, int(demo_row.name), X.loc[training_rows]
    )
    print("\n[demo] Exact Shapley breakdown:")
    print(json.dumps(explanation, indent=2, ensure_ascii=True))

    print(f"\n[demo] Rule-based reason (from risk_engine.py): {demo_row['risk_reasons']}")


if __name__ == "__main__":
    main()