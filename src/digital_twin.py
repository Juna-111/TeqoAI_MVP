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
    lengths = nx.single_source_shortest_path_length(G, start_node, cutoff=max_hops)

    for target, hops in lengths.items():
        if target == start_node:
            continue
        try:
            path = nx.shortest_path(G, start_node, target)
        except nx.NetworkXNoPath:
            continue

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


def build_explainer(model):
    """Wraps SHAP's TreeExplainer, with a clear failure mode if the
    installed SHAP/sklearn version pairing doesn't support IsolationForest
    directly — the rest of the pipeline still works without it."""
    try:
        import shap
        return shap.TreeExplainer(model)
    except ImportError as error:
        print(
            f"[warn] SHAP is unavailable ({type(error).__name__}); "
            "falling back to rule-based reasons."
        )
        return None
    except Exception as e:
        print(f"[warn] SHAP TreeExplainer unavailable ({type(e).__name__}: {e}). "
              f"Falling back to rule-based reasons only (see risk_reasons column).")
        return None


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


def explain_event(explainer, scaler, X, feature_names, row_index, top_n=3):
    """Return a short, human-readable explanation for one event's score,
    naming the top contributing features and their direction."""
    if explainer is None:
        return "SHAP explainer unavailable — see risk_reasons for a rule-based justification."

    X_scaled = scaler.transform(X.iloc[[row_index]])
    shap_values = explainer.shap_values(X_scaled)

    # shap_values shape can vary by SHAP version; normalize to 1D array
    values = shap_values[0] if hasattr(shap_values, "__len__") and len(shap_values) == 1 else shap_values
    values = values.flatten()

    # SHAP explains the model's raw decision_function, where HIGHER = more
    # normal. Our anomaly_score is the negated version (HIGHER = more
    # anomalous), so we flip the sign here to keep the wording consistent
    # with the anomaly_score everyone actually looks at.
    contributions = list(zip(feature_names, -values))
    contributions.sort(key=lambda c: abs(c[1]), reverse=True)

    parts = []
    for name, value in contributions[:top_n]:
        direction = "increased" if value > 0 else "decreased"
        parts.append(f"{format_feature_name(name)} {direction} the anomaly ranking")

    return "; ".join(parts)


# ---- Demo / self-test --------------------------------------------------------

def main():
    print("[info] loading cleaned events and rebuilding features for SHAP alignment...")
    df_clean = pd.read_csv(CLEAN_FILE)
    X, df_features = build_features(df_clean)

    print("[info] loading risk-scored events...")
    df_risk = pd.read_csv(RISK_FILE)

    print("[info] building the Cyber Digital Twin graph...")
    G = build_graph(df_clean)
    print(f"[info] graph built: {G.number_of_nodes()} assets, {G.number_of_edges()} relationships")

    print("[info] loading model + building SHAP explainer...")
    model, scaler = load_model_artifacts()
    X = align_features(X, scaler.feature_names_in_)
    explainer = build_explainer(model)
    feature_names = X.columns.tolist()

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

    # Find the matching row in df_features for SHAP (align by user/time/target_device)
    match = df_features[
        (df_features["user"] == demo_row["user"])
        & (df_features["time"] == demo_row["time"])
        & (df_features["target_device"] == demo_row["target_device"])
    ]
    if not match.empty:
        row_index = match.index[0]
        explanation = explain_event(explainer, scaler, X, feature_names, row_index)
        print(f"\n[demo] SHAP explanation: {explanation}")
    else:
        print("\n[demo] Could not align this event for SHAP explanation "
              "(unexpected — check that clean/risk files are from the same run).")

    print(f"\n[demo] Rule-based reason (from risk_engine.py): {demo_row['risk_reasons']}")


if __name__ == "__main__":
    main()