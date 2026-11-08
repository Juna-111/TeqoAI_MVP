"""Cached data, graph, and model access shared by dashboard pages."""

import hashlib
import json
import os

import pandas as pd
import streamlit as st

from src.dataset_paths import CLEAN_FILE, MODELS_DIR, RISK_FILE
from src.digital_twin import build_graph, load_model_artifacts
from src.model import (
    align_features,
    build_features,
    load_autoencoder_artifacts,
)
from src.provenance import sha256_file


@st.cache_data
def load_data(clean_mtime_ns, clean_size, risk_mtime_ns, risk_size):
    clean_stat = os.stat(CLEAN_FILE)
    risk_stat = os.stat(RISK_FILE)
    if (
        clean_stat.st_mtime_ns != clean_mtime_ns
        or clean_stat.st_size != clean_size
        or risk_stat.st_mtime_ns != risk_mtime_ns
        or risk_stat.st_size != risk_size
    ):
        raise OSError("Processed data changed while it was being loaded; retry the page.")
    clean = pd.read_csv(CLEAN_FILE)
    risk = pd.read_csv(RISK_FILE)
    clean_stat = os.stat(CLEAN_FILE)
    risk_stat = os.stat(RISK_FILE)
    if (
        clean_stat.st_mtime_ns != clean_mtime_ns
        or clean_stat.st_size != clean_size
        or risk_stat.st_mtime_ns != risk_mtime_ns
        or risk_stat.st_size != risk_size
    ):
        raise OSError("Processed data changed while it was being loaded; retry the page.")
    if "time" in risk:
        risk["time"] = pd.to_numeric(risk["time"], errors="coerce")
    if "risk_score" in risk:
        risk["risk_score"] = pd.to_numeric(risk["risk_score"], errors="coerce")
    identity_columns = [
        column
        for column in ["source", "time", "user", "device", "target_device", "action", "result"]
        if column in risk.columns
    ]
    identity_frame = risk[identity_columns].fillna("").astype(str)
    duplicates = identity_frame.groupby(identity_columns, dropna=False).cumcount()
    risk["_event_id"] = [
        hashlib.sha256(
            json.dumps([*values, int(occurrence)], ensure_ascii=True).encode("utf-8")
        ).hexdigest()[:24]
        for values, occurrence in zip(
            identity_frame.itertuples(index=False, name=None), duplicates
        )
    ]
    return clean, risk


@st.cache_data
def cached_file_sha256(path, modified_ns, file_size):
    stat = os.stat(path)
    if stat.st_mtime_ns != modified_ns or stat.st_size != file_size:
        raise OSError("Input file changed while checking evaluation provenance.")
    digest = sha256_file(path)
    stat = os.stat(path)
    if stat.st_mtime_ns != modified_ns or stat.st_size != file_size:
        raise OSError("Input file changed while checking evaluation provenance.")
    return digest


@st.cache_resource
def load_graph(clean):
    return build_graph(clean)


@st.cache_resource
def load_analysis_resources(clean):
    X, _ = build_features(clean)
    model, scaler = load_model_artifacts()
    X = align_features(X, scaler.feature_names_in_)
    return X, model, scaler


@st.cache_resource
def probe_model_artifacts():
    """Load runtime model artifacts once to establish their actual availability."""
    status = {
        "isolation_forest": {"loaded": False, "error": None, "reason": None},
        "autoencoder": {"loaded": False, "error": None, "reason": None},
    }
    try:
        model, scaler = load_model_artifacts()
        status["isolation_forest"] = {
            "loaded": bool(hasattr(model, "decision_function")),
            "error": None,
            "reason": None,
            "model_type": type(model).__name__,
            "feature_count": int(len(getattr(scaler, "feature_names_in_", []))),
        }
    except ImportError as error:
        status["isolation_forest"]["error"] = str(error)
        status["isolation_forest"]["reason"] = "dependency"
    except (OSError, RuntimeError, ValueError) as error:
        status["isolation_forest"]["error"] = str(error)
        status["isolation_forest"]["reason"] = "artifact"

    autoencoder_model = os.path.join(MODELS_DIR, "autoencoder.pt")
    autoencoder_scaler = os.path.join(MODELS_DIR, "autoencoder_scaler.pkl")
    if not (
        os.path.exists(autoencoder_model) and os.path.exists(autoencoder_scaler)
    ):
        status["autoencoder"]["error"] = (
            "Optional autoencoder artifacts are not present in the models directory."
        )
        status["autoencoder"]["reason"] = "artifact"
        return status
    try:
        model, scaler = load_autoencoder_artifacts(
            autoencoder_model, autoencoder_scaler
        )
        status["autoencoder"] = {
            "loaded": True,
            "error": None,
            "reason": None,
            "model_type": type(model).__name__,
            "feature_count": int(len(getattr(scaler, "feature_names_in_", []))),
        }
    except ImportError as error:
        status["autoencoder"]["error"] = str(error)
        status["autoencoder"]["reason"] = "dependency"
    except (OSError, RuntimeError, ValueError) as error:
        status["autoencoder"]["error"] = str(error)
        status["autoencoder"]["reason"] = "artifact"
    return status