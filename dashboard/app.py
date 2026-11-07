"""TeqoAI analyst dashboard for authentication anomaly investigation."""

import base64
import hashlib
import html
import json
import os
import sqlite3
import sys

import networkx as nx
import pandas as pd
import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.digital_twin import (
    build_graph,
    explain_event,
    get_attack_paths,
    load_model_artifacts,
)
from src.demo_pipeline import run_synthetic_demo
from src.dataset_paths import (
    CLEAN_FILE,
    DATASET,
    EVALUATION_FILE,
    RISK_FILE,
)
from src.provenance import sha256_file, validate_pipeline_provenance
from src.case_store import (
    CASE_STATUSES,
    CHECKLIST_ITEMS,
    csv_safe_text,
    export_cases,
    get_case_history,
    get_cases,
    save_case,
)
from src.model import align_features, build_features

BRAND = "#38BDF8"
LOGO_FILE = os.path.join(BASE_DIR, "assets", "teqoai-logo.png")

st.set_page_config(
    page_title="TeqoAI | Security Operations",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
        --teqo-accent: #38BDF8;
        --teqo-bg: #07111F;
        --teqo-panel: #0E1B2A;
        --teqo-border: rgba(148, 163, 184, 0.18);
        --teqo-muted: #94A3B8;
    }
    .stApp { background: var(--teqo-bg); }
    [data-testid="stHeader"] { background: rgba(7, 17, 31, 0.92); }
    [data-testid="stSidebar"] {
        background: #0A1726;
        border-right: 1px solid var(--teqo-border);
    }
    .stMainBlockContainer {
        max-width: 1440px;
        padding: 2rem clamp(1rem, 3vw, 3rem) 3rem;
    }
    :where(a, button, input, textarea, select, [tabindex]):focus-visible {
        outline: 3px solid #7DD3FC !important;
        outline-offset: 3px !important;
        box-shadow: 0 0 0 2px #07111F !important;
    }
    [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: rgba(56, 189, 248, .14);
        border-left: 3px solid var(--teqo-accent);
    }
    [data-testid="stSidebarNav"] a,
    .stButton > button,
    [data-testid="stPageLink-NavLink"] {
        min-height: 44px;
    }
    .teqo-brand { font-size: 1.25rem; font-weight: 750; letter-spacing: .04em; }
    .teqo-brand span { color: var(--teqo-accent); }
    .teqo-eyebrow {
        color: #B3C1CC; font-size: .73rem; font-weight: 650;
        letter-spacing: .12em; text-transform: uppercase;
    }
    .teqo-banner {
        border: 1px solid rgba(56, 189, 248, .25);
        border-left: 3px solid var(--teqo-accent);
        border-radius: 8px; padding: 12px 16px;
        background: rgba(56, 189, 248, .06); margin: 8px 0 20px;
    }
    .teqo-banner.warning {
        border-color: rgba(251, 191, 36, .34);
        border-left-color: #FBBF24;
        background: rgba(251, 191, 36, .07);
    }
    .teqo-overview-intro {
        border: 1px solid var(--teqo-border);
        border-radius: 10px;
        background: linear-gradient(110deg, rgba(56,189,248,.09), rgba(14,27,42,.92) 58%);
        padding: 16px 18px;
        margin: 4px 0 16px;
    }
    .teqo-overview-intro h3 { margin: 2px 0 4px; font-size: 1.15rem; }
    .teqo-overview-intro p { margin: 0; color: #A9BBC9; font-size: .88rem; }
    .teqo-home-panel {
        height: 100%;
        border: 1px solid var(--teqo-border);
        border-radius: 10px;
        background: var(--teqo-panel);
        padding: 15px 17px;
        margin: 4px 0 18px;
    }
    .teqo-home-panel h4 {
        margin: 0 0 12px;
        color: #E5F0F7;
        font-size: .92rem;
    }
    .teqo-process {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(min(100%, 190px), 1fr));
        gap: 12px;
        margin: 12px 0 24px;
        padding: 0;
        list-style: none;
    }
    .teqo-process li {
        position: relative;
        min-width: 0;
        min-height: 104px;
        padding: 16px;
        border: 1px solid var(--teqo-border);
        border-radius: 12px;
        background: linear-gradient(145deg, #0E1B2A, #0A1726);
    }
    .teqo-process strong { display: block; margin-bottom: 7px; color: #E5F0F7; }
    .teqo-process span { color: #B3C1CC; font-size: .84rem; line-height: 1.55; }
    .teqo-contribution {
        height: 100%;
        min-height: 150px;
        padding: 16px;
        border: 1px solid var(--teqo-border);
        border-radius: 12px;
        background: var(--teqo-panel);
    }
    .teqo-contribution strong { display: block; margin-bottom: 8px; color: #E5F0F7; }
    .teqo-contribution p { margin: 0; color: #B3C1CC; font-size: .84rem; line-height: 1.6; }
    .teqo-contribution small { display: block; margin-top: 10px; color: #7DD3FC; }
    .teqo-research-note {
        padding: 14px 16px;
        border: 1px solid rgba(251, 191, 36, .34);
        border-left: 3px solid #FBBF24;
        border-radius: 9px;
        background: rgba(251, 191, 36, .07);
        color: #F3E4BE;
        line-height: 1.6;
    }
    .teqo-risk-row {
        display: grid;
        grid-template-columns: 72px 1fr 52px;
        align-items: center;
        gap: 10px;
        margin: 11px 0;
        color: #C7D5E0;
        font-size: .78rem;
    }
    .teqo-risk-track {
        height: 7px;
        overflow: hidden;
        border-radius: 99px;
        background: #1B2B3A;
    }
    .teqo-risk-fill { height: 100%; border-radius: inherit; }
    .teqo-queue-stat {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 8px 0;
        border-bottom: 1px solid var(--teqo-border);
        color: #A9BBC9;
        font-size: .8rem;
    }
    .teqo-queue-stat:last-child { border-bottom: 0; }
    .teqo-queue-stat strong { color: #E5F0F7; font-variant-numeric: tabular-nums; }
    div[data-testid="stMetric"] {
        background: var(--teqo-panel); border: 1px solid var(--teqo-border);
        padding: 14px 16px; border-radius: 10px;
    }
    div[data-testid="stMetricLabel"] { color: var(--teqo-muted); }
    div[data-testid="stMetricValue"] { color: #E5F0F7; }
    .stButton > button[kind="primary"] {
        background: var(--teqo-accent); color: #04111E; border: 0;
        font-weight: 700;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label {
        border-radius: 7px;
        padding: 8px 14px;
        border: 1px solid var(--teqo-border);
        background: var(--teqo-panel);
    }
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        border-color: var(--teqo-accent);
        background: rgba(56, 189, 248, .12);
    }
    .teqo-foot {
        color: var(--teqo-muted); font-size: .78rem;
        border-top: 1px solid var(--teqo-border);
        margin-top: 30px; padding-top: 14px;
    }
    .teqo-data-badge {
        display: inline-block;
        margin-top: 12px;
        padding: 4px 9px;
        border: 1px solid rgba(251, 191, 36, .34);
        border-radius: 999px;
        color: #FCD887;
        background: rgba(251, 191, 36, .07);
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .06em;
        white-space: nowrap;
    }
    .teqo-section-kicker {
        color: var(--teqo-accent);
        font-size: .73rem;
        font-weight: 750;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .st-key-teqo-home {
        background:
            radial-gradient(ellipse at 86% 3%, rgba(56,189,248,.09), transparent 31%);
    }
    .teqo-home-hero {
        max-width: 920px;
        padding: clamp(1.5rem, 5vw, 4rem) 0 2.5rem;
    }
    .teqo-home-hero-row {
        display: grid;
        grid-template-columns: minmax(0, 1.6fr) minmax(220px, .7fr);
        align-items: center;
        gap: clamp(1rem, 4vw, 4rem);
    }
    .teqo-home-hero-logo {
        display: flex;
        justify-content: center;
        align-items: center;
        min-width: 0;
    }
    .teqo-home-hero-logo img {
        width: min(100%, 340px);
        height: auto;
        filter: drop-shadow(0 18px 34px rgba(0, 120, 255, .16));
    }
    .teqo-home-hero h1 {
        margin: 14px 0 18px;
        font-size: clamp(2.7rem, 7vw, 5.4rem);
        line-height: 1.02;
        letter-spacing: -.055em;
    }
    .teqo-home-hero p {
        max-width: 760px;
        color: #B5C5D2;
        font-size: clamp(1rem, 1.5vw, 1.16rem);
        line-height: 1.8;
    }
    .teqo-home-section {
        margin-top: clamp(2.5rem, 6vw, 5rem);
    }
    .teqo-home-section h2 {
        margin: 8px 0 12px;
        font-size: clamp(1.5rem, 3vw, 2rem);
        letter-spacing: -.025em;
    }
    .teqo-home-copy {
        max-width: 760px;
        color: #B5C5D2;
        font-size: 1rem;
        line-height: 1.8;
    }
    .teqo-home-facts {
        display: flex;
        flex-wrap: wrap;
        gap: 10px 30px;
        margin-top: 22px;
        color: #D8E5EE;
        font-size: .88rem;
    }
    .teqo-home-facts span { color: #8FA5B5; }
    .teqo-home-feature {
        padding: 12px 0;
    }
    .teqo-home-features {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr));
        gap: clamp(20px, 4vw, 48px);
        margin-top: 20px;
    }
    .teqo-home-feature .teqo-section-kicker { margin-bottom: 12px; }
    .teqo-home-feature h3 { margin: 0 0 8px; font-size: 1.08rem; }
    .teqo-home-feature p {
        margin: 0;
        color: #B5C5D2;
        font-size: .9rem;
        line-height: 1.7;
    }
    .teqo-home-steps {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(min(100%, 190px), 1fr));
        gap: 20px;
        margin: 20px 0 0;
        padding: 0;
        list-style: none;
    }
    .teqo-home-steps li { min-width: 0; padding: 8px 0; }
    .teqo-home-steps strong {
        display: block;
        margin: 10px 0 6px;
        color: #E5F0F7;
        font-size: 1rem;
    }
    .teqo-home-steps span { color: #B5C5D2; font-size: .88rem; line-height: 1.6; }
    .teqo-home-number { color: var(--teqo-accent); font-size: .76rem; font-weight: 750; }
    .teqo-home-about { max-width: 880px; color: #B5C5D2; line-height: 1.8; }
    .teqo-home-contacts {
        display: flex;
        flex-wrap: wrap;
        gap: 10px 24px;
        margin-top: 18px;
    }
    .teqo-home-contacts a {
        color: #DDF5FF !important;
        text-decoration: underline;
        text-decoration-color: rgba(56,189,248,.55);
        text-underline-offset: 4px;
    }
    .teqo-home-contacts a:hover { color: #7DD3FC !important; }
    .teqo-home-note {
        max-width: 920px;
        margin-top: 2rem;
        color: #C4D5E1;
        font-size: .9rem;
        line-height: 1.75;
    }
    .st-key-teqo-home-footer {
        margin-top: clamp(3rem, 7vw, 6rem);
        padding: clamp(1.5rem, 4vw, 2.5rem) 0 1rem;
        border-top: 1px solid var(--teqo-border);
        background: linear-gradient(180deg, rgba(14,27,42,.35), transparent 75%);
    }
    .teqo-home-footer-brand {
        max-width: 230px;
        color: #B5C5D2;
        font-size: .85rem;
        line-height: 1.65;
    }
    .teqo-home-footer-brand img {
        width: min(100%, 190px);
        height: auto;
        margin-bottom: 12px;
    }
    .teqo-home-footer-content h2 {
        margin: 4px 0 8px;
        color: #E5F0F7;
        font-size: 1.35rem;
    }
    .teqo-home-footer-content p {
        margin: 0;
        color: #B5C5D2;
        font-size: .9rem;
        line-height: 1.7;
    }
    .teqo-home-footer-contacts {
        display: flex;
        flex-wrap: wrap;
        gap: 12px 24px;
        margin: 18px 0 22px;
    }
    .teqo-home-footer-contacts a {
        color: #DDF5FF !important;
        font-size: .88rem;
        text-decoration: underline;
        text-decoration-color: rgba(56,189,248,.55);
        text-underline-offset: 4px;
    }
    .teqo-home-footer-contacts a:hover { color: #7DD3FC !important; }
    .teqo-home-footer-contacts a {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .teqo-home-contact-icon {
        display: inline-flex;
        width: 18px;
        height: 18px;
        align-items: center;
        justify-content: center;
        color: #38BDF8;
    }
    .teqo-home-contact-icon svg {
        width: 17px;
        height: 17px;
        fill: none;
        stroke: currentColor;
        stroke-width: 1.7;
        stroke-linecap: round;
        stroke-linejoin: round;
    }
    .teqo-home-footer-note {
        max-width: 800px;
        margin-top: 14px !important;
        color: #9FB1BF !important;
        font-size: .8rem !important;
    }
    .teqo-home-footer-bottom {
        margin-top: 28px;
        padding-top: 14px;
        border-top: 1px solid var(--teqo-border);
        color: #94A3B8;
        font-size: .76rem;
    }
    .stMain .st-key-teqo-home [data-testid="stPageLink-NavLink"] {
        display: inline-flex;
        width: auto;
        justify-content: flex-start;
        min-height: 44px;
        margin-right: 1rem;
        padding: 8px 0;
        border: 0;
        border-radius: 0;
        background: transparent;
        color: #DDF5FF;
        text-decoration: none;
    }
    .stMain .st-key-teqo-home [data-testid="stPageLink-NavLink"]:hover {
        border: 0;
        background: transparent;
        color: #7DD3FC;
        text-decoration: underline;
        text-underline-offset: 5px;
    }
    [data-testid="stMain"] [data-testid="stPageLink-NavLink"] {
        display: flex;
        justify-content: center;
        min-height: 42px;
        padding: 8px 12px;
        border: 1px solid rgba(56, 189, 248, .32);
        border-radius: 9px;
        background: rgba(56, 189, 248, .08);
        color: #DDF5FF;
        text-decoration: none;
        transition: background-color .18s ease, border-color .18s ease;
    }
    [data-testid="stMain"] [data-testid="stPageLink-NavLink"]:hover {
        border-color: var(--teqo-accent);
        background: rgba(56, 189, 248, .16);
    }
    .teqo-graph-shell {
        border: 1px solid rgba(56, 189, 248, .2);
        border-radius: 14px;
        overflow: hidden;
        background:
            radial-gradient(ellipse at 50% 45%, rgba(15, 54, 79, .55), transparent 70%),
            #07111F;
        box-shadow: 0 18px 48px rgba(0, 0, 0, .28), inset 0 0 40px rgba(56, 189, 248, .04);
    }
    .teqo-graph-shell svg { display: block; width: 100%; height: auto; }
    .teqo-graph-edge {
        stroke-dasharray: 8 10;
        animation: teqo-flow 3.2s linear infinite;
    }
    .teqo-graph-node { transform-box: fill-box; transform-origin: center; }
    .teqo-graph-node.source { animation: teqo-pulse 2.4s ease-in-out infinite; }
    .teqo-graph-node.critical { animation: teqo-pulse 2.8s ease-in-out infinite; }
    @keyframes teqo-flow { to { stroke-dashoffset: -72; } }
    @keyframes teqo-pulse {
        0%, 100% { opacity: .84; }
        50% { opacity: 1; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: .01ms !important;
        }
        .teqo-graph-edge, .teqo-graph-node { animation: none; }
    }
    @media (max-width: 760px) {
        .stMainBlockContainer { padding: 1.25rem 1rem 2rem; }
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
            gap: .75rem;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            min-width: min(100%, 260px);
            flex: 1 1 260px;
        }
        .teqo-home-hero { padding-top: 1.5rem; }
        .teqo-home-hero-row { grid-template-columns: 1fr; }
        .teqo-home-hero-logo { justify-content: flex-start; }
        .teqo-home-hero-logo img { width: min(58vw, 240px); }
        .teqo-home-facts { display: grid; gap: 10px; }
        .teqo-home-steps { grid-template-columns: 1fr; gap: 12px; }
        .teqo-home-footer-brand { max-width: 180px; }
    }
    @media (max-width: 420px) {
        .stMainBlockContainer { padding-right: .75rem; padding-left: .75rem; }
        .stMain .st-key-teqo-home [data-testid="stPageLink-NavLink"] { margin-right: .5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        column for column in
        ["source", "time", "user", "device", "target_device", "action", "result"]
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


def render_shap_explanation(explanation):
    st.markdown("#### Exact Shapley decomposition")
    st.caption(
        "Output: Isolation Forest anomaly score (negative decision_function); "
        "higher scores indicate more anomalous events."
    )
    base_column, prediction_column, drift_column = st.columns(3)
    base_column.metric(
        "Expected value (training background)",
        f"{explanation['base_value']:.6f}",
    )
    prediction_column.metric(
        "Event prediction", f"{explanation['prediction']:.6f}"
    )
    drift_column.metric(
        "Feature-attributed drift",
        f"{explanation['drift']:+.6f}",
        help="Prediction minus expected value. Signed feature Shapley values sum to this drift.",
    )
    st.caption(
        f"{explanation['method']} Background: "
        f"{explanation['background']['rows_used']} of "
        f"{explanation['background']['available_training_rows']} training-period "
        f"events. Additivity residual: "
        f"{explanation['additivity_error']:.3g}."
    )

    positive_column, negative_column = st.columns(2)
    for column, heading, key in (
        (positive_column, "Positive drivers · increase anomaly score", "positive_drivers"),
        (negative_column, "Negative drivers · decrease anomaly score", "negative_drivers"),
    ):
        with column:
            st.markdown(f"**{heading}**")
            drivers = explanation[key]
            if drivers:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Feature": item["label"],
                                "Event value": item["value"],
                                "Shapley contribution": item["shap_value"],
                            }
                            for item in drivers
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("No contributions in this direction.")

    st.markdown("**Largest pairwise interactions**")
    interactions = explanation["interactions"][:10]
    if interactions:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature pair": f"{item['feature_a']} × {item['feature_b']}",
                        "Interaction contribution": item["contribution"],
                    }
                    for item in interactions
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No pairwise interaction contributions were measured.")
    st.caption(explanation["interaction_convention"])
    st.caption(explanation["dependence_note"])
    st.caption(
        "Attributions describe this model output relative to the selected training "
        "background; they are not causal effects or probabilities."
    )


def recommend_action(row):
    if row["risk_level"] == "High":
        if bool(row.get("factor_sensitive_asset", False)):
            return (
                f"Prioritize review. This event reached {row['target_device']}, "
                "a configured sensitive asset. Verify the session and identity "
                "before deciding on containment."
            )
        return (
            "Prioritize this event for analyst review. Verify the activity with "
            "the account owner and surrounding authentication events."
        )
    if row["risk_level"] == "Medium":
        return "Review in the normal queue and correlate with related activity."
    return "No immediate action is suggested by this prototype."


def event_label(row):
    event_time = row.get("time")
    time_text = "time unavailable" if pd.isna(event_time) else f"t={int(event_time)}"
    return (
        f"{row['risk_level']} · {row['user']} · {row['device']} → "
        f"{row['target_device']} · {time_text}"
    )


def apply_filters(data, cases):
    st.sidebar.markdown('<div class="teqo-eyebrow">Refine alerts · optional</div>', unsafe_allow_html=True)
    case_view = st.sidebar.segmented_control(
        "Case stage",
        options=["All", "Open", "Resolved", "Benign"],
        default="All",
        label_visibility="visible",
        key="case_view",
        width="stretch",
    )
    status_filter = {
        "All": list(CASE_STATUSES),
        "Open": ["New", "Investigating", "Escalated"],
        "Resolved": ["Resolved"],
        "Benign": ["Benign"],
    }[case_view or "All"]
    levels = st.sidebar.pills(
        "Severity",
        options=["High", "Medium", "Low"],
        selection_mode="multi",
        default=["High", "Medium", "Low"],
        key="severity_filter",
        width="stretch",
    )
    query = st.sidebar.text_input(
        "Find an account or device",
        placeholder="Type to search",
    ).strip().casefold()
    sort_order = st.sidebar.selectbox(
        "Order by",
        ["Priority", "Highest score", "Newest first", "Oldest first"],
    )

    filtered = data.copy()
    filtered["case_status"] = filtered["_event_id"].map(
        lambda event_id: cases.get(event_id, {}).get("status", "New")
    )
    filtered["case_owner"] = filtered["_event_id"].map(
        lambda event_id: cases.get(event_id, {}).get("owner", "")
    )
    filtered = filtered[
        filtered["case_status"].isin(status_filter)
        & filtered["risk_level"].isin(levels)
    ].copy()
    if query:
        searchable = (
            filtered[["user", "device", "target_device", "risk_reasons"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.casefold()
        )
        filtered = filtered[searchable.str.contains(query, regex=False)]
    if sort_order == "Priority":
        status_order = {"Escalated": 0, "New": 1, "Investigating": 2, "Resolved": 3, "Benign": 4}
        risk_order = {"High": 0, "Medium": 1, "Low": 2}
        filtered["_status_order"] = filtered["case_status"].map(status_order)
        filtered["_risk_order"] = filtered["risk_level"].map(risk_order)
        filtered = filtered.sort_values(
            ["_status_order", "_risk_order", "risk_score", "time"],
            ascending=[True, True, False, False],
            na_position="last",
        ).drop(columns=["_status_order", "_risk_order"])
    else:
        sort_column = "risk_score" if sort_order == "Highest score" else "time"
        filtered = filtered.sort_values(
            sort_column,
            ascending=sort_order == "Oldest first",
            na_position="last",
        )
    st.sidebar.caption(f"{len(filtered):,} alerts shown · {len(data):,} detected")
    return filtered


def render_overview(data, queue, cases):
    case_statuses = data["_event_id"].map(
        lambda event_id: cases.get(event_id, {}).get("status", "New")
    )
    active = int(case_statuses.isin({"New", "Investigating", "Escalated"}).sum())
    escalated = sum(case["status"] == "Escalated" for case in cases.values())
    high_active = int(
        (
            (data["risk_level"] == "High")
            & ~data["_event_id"].isin(
                [
                    event_id for event_id, case in cases.items()
                    if case["status"] in {"Resolved", "Benign"}
                ]
            )
        ).sum()
    )
    cards = st.columns(4)
    cards[0].metric("Active alerts", f"{active:,}")
    cards[1].metric("High-risk · active", f"{high_active:,}")
    cards[2].metric("Escalated", f"{escalated:,}")
    cards[3].metric("Events analyzed", f"{len(data):,}")

    st.markdown(
        '<div class="teqo-overview-intro">'
        '<div class="teqo-eyebrow">Shift overview</div>'
        '<h3>Know what needs attention first</h3>'
        '<p>Review the priority queue below, then open <b>Investigate</b> to '
        'validate an alert and record a case decision. TeqoAI does not take '
        'automated response actions.</p></div>',
        unsafe_allow_html=True,
    )

    posture, snapshot = st.columns([1.15, 1])
    with posture:
        severity_counts = data["risk_level"].value_counts()
        severity_colors = {"High": "#FB7185", "Medium": "#FBBF24", "Low": "#34D399"}
        total_events = max(len(data), 1)
        severity_rows = []
        for level in ("High", "Medium", "Low"):
            count = int(severity_counts.get(level, 0))
            percent = count / total_events * 100
            severity_rows.append(
                f'<div class="teqo-risk-row"><span>{level}</span>'
                f'<div class="teqo-risk-track"><div class="teqo-risk-fill" '
                f'style="width:{percent:.1f}%;background:{severity_colors[level]}">'
                f'</div></div><strong>{count:,}</strong></div>'
            )
        st.markdown(
            '<div class="teqo-home-panel"><h4>Detection severity</h4>'
            + "".join(severity_rows)
            + '<div class="teqo-eyebrow" style="margin-top:12px">'
            + f'{len(data):,} analyzed events</div></div>',
            unsafe_allow_html=True,
        )
    with snapshot:
        queue_high = int((queue["risk_level"] == "High").sum()) if not queue.empty else 0
        queue_new = int((queue["case_status"] == "New").sum()) if not queue.empty else 0
        queue_unassigned = (
            int(queue["case_owner"].fillna("").astype(str).str.strip().eq("").sum())
            if not queue.empty else 0
        )
        st.markdown(
            '<div class="teqo-home-panel"><h4>Current queue</h4>'
            f'<div class="teqo-queue-stat"><span>Matching your filters</span>'
            f'<strong>{len(queue):,}</strong></div>'
            f'<div class="teqo-queue-stat"><span>High severity</span>'
            f'<strong>{queue_high:,}</strong></div>'
            f'<div class="teqo-queue-stat"><span>New cases</span>'
            f'<strong>{queue_new:,}</strong></div>'
            f'<div class="teqo-queue-stat"><span>Unassigned</span>'
            f'<strong>{queue_unassigned:,}</strong></div></div>',
            unsafe_allow_html=True,
        )

    heading, export = st.columns([5, 1])
    heading.markdown("### Priority alerts")
    heading.caption(
        "Sorted by urgency using the sidebar filters. Open **Investigate** "
        "to review evidence and update a case."
    )
    saved_cases = export_cases(DATASET)
    if saved_cases:
        case_export = pd.DataFrame(saved_cases).merge(
            data[
                [
                    "_event_id", "time", "user", "device", "target_device",
                    "risk_level", "risk_score", "risk_reasons",
                ]
            ],
            left_on="event_id",
            right_on="_event_id",
            how="left",
        ).drop(columns=["_event_id"])
        case_export = case_export.apply(
            lambda column: column.map(csv_safe_text)
        )
        export.caption(
            "The export includes analyst notes. Store it securely; text is escaped "
            "to reduce spreadsheet formula risks."
        )
        export.download_button(
            "Export cases",
            data=case_export.to_csv(index=False),
            file_name="teqoai_cases.csv",
            mime="text/csv",
            width="stretch",
        )

    if queue.empty:
        st.info("No alerts match this queue view. Change the status or risk filters.")
        return
    display = queue.head(25).copy()
    columns = [
        "case_status", "risk_level", "risk_score", "time", "user", "device",
        "target_device", "case_owner", "risk_reasons",
    ]
    st.dataframe(
        display[columns],
        width="stretch",
        hide_index=True,
        height=430,
        column_config={
            "case_status": st.column_config.TextColumn("Case"),
            "risk_level": st.column_config.TextColumn("Severity"),
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0, max_value=100, format="%.1f"
            ),
            "time": st.column_config.NumberColumn("Elapsed seconds", format="%d"),
            "case_owner": st.column_config.TextColumn("Analyst"),
        },
    )
    remaining = queue.iloc[25:300]
    if not remaining.empty:
        with st.expander(f"Show {len(remaining):,} more filtered alerts"):
            st.dataframe(
                remaining[columns],
                width="stretch",
                hide_index=True,
                height=430,
                column_config={
                    "case_status": st.column_config.TextColumn("Case"),
                    "risk_level": st.column_config.TextColumn("Severity"),
                    "risk_score": st.column_config.ProgressColumn(
                        "Risk score", min_value=0, max_value=100, format="%.1f"
                    ),
                    "time": st.column_config.NumberColumn("Elapsed seconds", format="%d"),
                    "case_owner": st.column_config.TextColumn("Analyst"),
                },
            )
    if len(queue) > 300:
        st.caption(
            "The Overview shows the first 300 alerts. Use Investigate to browse "
            "the full filtered queue. Event times are elapsed seconds, not local clock times."
        )


def render_investigation(queue, data, graph, cases, clean_events):
    st.subheader("Case investigation")
    if queue.empty:
        st.info("No alerts match this queue view. Change the status or risk filters.")
        return

    event_ids = queue["_event_id"].tolist()[:500]
    selected_event_id = st.selectbox(
        "Alert to investigate",
        options=event_ids,
        format_func=lambda event_id: event_label(
            queue.loc[queue["_event_id"] == event_id].iloc[0]
        ),
    )
    row = queue.loc[queue["_event_id"] == selected_event_id].iloc[0]
    event_id = str(row["_event_id"])
    case = cases.get(event_id, {})
    status_default = case.get("status", "New")

    summary = st.columns([1, 1, 1, 2])
    summary[0].metric("Severity", str(row["risk_level"]))
    summary[1].metric("Priority score", f"{row['risk_score']:.1f}/100")
    summary[2].metric("Case status", status_default)
    summary[3].metric("Assigned analyst", case.get("owner") or "Unassigned")

    context, evidence = st.columns([1, 1])
    with context:
        st.markdown("#### Alert summary")
        st.write(f"**Account:** `{row['user']}`")
        st.write(f"**Source:** `{row['device']}`")
        st.write(f"**Destination:** `{row['target_device']}`")
        st.write(f"**Authentication:** {row['result']} · {row['action']}")
        st.write(f"**Elapsed time:** {int(row['time'])} seconds")
        st.write(f"**Detection reason:** {row.get('risk_reasons', 'No reason recorded')}")
        st.caption("Priority score is a relative ranking signal, not a probability of compromise.")
        st.info(recommend_action(row))
    with evidence:
        st.markdown("#### Detection evidence")
        factor_labels = {
            "factor_new_device": "First observed user-to-destination pair",
            "factor_sensitive_asset": "Destination is configured as sensitive",
            "factor_failed_login": "Authentication failed",
            "factor_burst_activity": "User activity exceeded burst threshold",
        }
        triggered = [
            label for key, label in factor_labels.items() if bool(row.get(key, False))
        ]
        if triggered:
            for label in triggered:
                st.markdown(f"- {label}")
        else:
            st.write("No rule factor was recorded; anomaly ranking contributed to this score.")
        st.caption(
            "Prototype signals require analyst validation; they are not proof of compromise."
        )
        if st.button("Generate SHAP explanation", key=f"shap-{event_id}"):
            try:
                X, model, scaler = load_analysis_resources(clean_events)
                training_rows = (
                    data["evaluation_split"].eq("train")
                    if "evaluation_split" in data
                    else pd.Series(True, index=data.index)
                )
                explanation = explain_event(
                    model,
                    scaler,
                    X,
                    int(row.name),
                    X.loc[training_rows],
                )
                render_shap_explanation(explanation)
            except (
                ArithmeticError,
                IndexError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                st.error(f"The Shapley explanation could not be generated: {error}")

    timeline, twin = st.columns([1.2, 1])
    with timeline:
        st.markdown("#### Related authentication activity")
        time_value = row["time"]
        related = data[
            ((data["user"] == row["user"]) | (data["device"] == row["device"]))
            & (data["time"].between(time_value - 300, time_value + 300))
        ].sort_values("time")
        if related.empty:
            st.info("No related events in the five-minute context window.")
        else:
            st.dataframe(
                related[["time", "user", "device", "target_device", "result", "action"]]
                .head(100),
                width="stretch",
                hide_index=True,
                height=250,
            )
        st.caption("Related events use a ±300-second window and matching account or source device.")
    with twin:
        st.markdown("#### Assets along observed paths")
        paths = get_attack_paths(graph, str(row["device"]))
        if paths:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Asset": path["target"],
                            "Type": path["target_type"],
                            "Hops": path["hops"],
                            "Observed route": " → ".join(path["path"]),
                        }
                        for path in paths[:8]
                    ]
                ),
                width="stretch",
                hide_index=True,
                height=250,
            )
        else:
            st.info("No observed outbound authentication paths for this source device.")
        st.caption("Observed event links are not verified network reachability or permissions.")

    st.markdown("#### Analyst action plan")
    st.caption("Checklist items are reminders only. TeqoAI does not execute containment actions.")
    with st.form(f"case-form-{event_id}"):
        form_left, form_right = st.columns([1, 2])
        with form_left:
            status = st.selectbox(
                "Case status",
                options=list(CASE_STATUSES),
                index=list(CASE_STATUSES).index(status_default),
                key=f"status-{event_id}",
            )
            owner = st.text_input(
                "Assigned analyst",
                value=case.get("owner", ""),
                max_chars=120,
                key=f"owner-{event_id}",
            )
        with form_right:
            notes = st.text_area(
                "Investigation notes",
                value=case.get("notes", ""),
                max_chars=10_000,
                height=130,
                key=f"notes-{event_id}",
            )
        checklist_current = case.get("checklist", {})
        checklist = {}
        checklist_cols = st.columns(2)
        for index, item in enumerate(CHECKLIST_ITEMS):
            with checklist_cols[index % 2]:
                checklist[item] = st.checkbox(
                    item,
                    value=bool(checklist_current.get(item, False)),
                    key=f"check-{event_id}-{index}",
                )
        submitted = st.form_submit_button("Save case and audit entry", type="primary")
        if submitted:
            try:
                save_case(DATASET, event_id, status, owner, notes, checklist)
            except (OSError, sqlite3.Error, ValueError):
                st.error(
                    "This case could not be saved. Your latest changes were not recorded."
                )
            else:
                st.success("Case saved locally.")
                st.rerun()

    history = get_case_history(DATASET, event_id)
    if history:
        st.markdown("#### Case activity")
        history_frame = pd.DataFrame(
            [
                {
                    "When (UTC)": item["occurred_at"],
                    "Action": item["action"].replace("_", " ").title(),
                    "Analyst": item["actor"],
                    "Details": json.dumps(item["details"], ensure_ascii=True),
                }
                for item in history
            ]
        )
        st.dataframe(history_frame, width="stretch", hide_index=True)


def render_graph_svg(graph, selected_source, highlighted_nodes, highlighted_edges):
    """Render a compact, animated, accessible SVG relationship view."""
    nodes = sorted(graph.nodes())
    subgraph = graph.subgraph(nodes)
    positions = nx.spring_layout(subgraph, seed=42, k=1.1)
    coordinates = {
        node: (
            55 + (float(positions[node][0]) + 1) * 425,
            45 + (float(positions[node][1]) + 1) * 220,
        )
        for node in nodes
    }

    svg_parts = [
        '<style>'
        '.teqo-graph-shell{border:1px solid rgba(56,189,248,.2);border-radius:14px;'
        'overflow:hidden;background:radial-gradient(ellipse at 50% 45%,'
        'rgba(15,54,79,.55),transparent 70%),#07111F;'
        'box-shadow:0 18px 48px rgba(0,0,0,.28),inset 0 0 40px rgba(56,189,248,.04)}'
        '.teqo-graph-shell svg{display:block;width:100%;height:auto}'
        '.teqo-graph-edge{stroke-dasharray:8 10;animation:teqo-flow 3.2s linear infinite}'
        '.teqo-graph-node.source,.teqo-graph-node.critical{transform-box:fill-box;'
        'transform-origin:center;animation:teqo-pulse 2.6s ease-in-out infinite}'
        '@keyframes teqo-flow{to{stroke-dashoffset:-72}}'
        '@keyframes teqo-pulse{0%,100%{opacity:.84}50%{opacity:1}}'
        '@media(prefers-reduced-motion:reduce){.teqo-graph-edge,'
        '.teqo-graph-node{animation:none}}'
        '</style>'
        '<div class="teqo-graph-shell"><svg viewBox="0 0 960 530" role="img" '
        'aria-labelledby="teqo-graph-title teqo-graph-desc" '
        'xmlns="http://www.w3.org/2000/svg">',
        '<title id="teqo-graph-title">Observed authentication relationships</title>',
        '<desc id="teqo-graph-desc">Animated directional links show authentication '
        'relationships found in the loaded events. They do not prove network access.</desc>',
        '<defs><marker id="teqo-arrow" markerWidth="8" markerHeight="8" '
        'refX="6" refY="3" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L0,6 L7,3 z" fill="#38BDF8"/></marker>'
        '<marker id="teqo-arrow-muted" markerWidth="8" markerHeight="8" '
        'refX="6" refY="3" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L0,6 L7,3 z" fill="#31516A"/></marker>'
        '<filter id="teqo-glow" x="-80%" y="-80%" width="260%" height="260%">'
        '<feGaussianBlur stdDeviation="5" result="blur"/>'
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
        '</defs>',
    ]

    for source, target in sorted(graph.edges()):
        source_x, source_y = coordinates[source]
        target_x, target_y = coordinates[target]
        delta_x, delta_y = target_x - source_x, target_y - source_y
        distance = max((delta_x**2 + delta_y**2) ** 0.5, 1)
        unit_x, unit_y = delta_x / distance, delta_y / distance
        start_x, start_y = source_x + unit_x * 20, source_y + unit_y * 20
        end_x, end_y = target_x - unit_x * 23, target_y - unit_y * 23
        highlighted = (source, target) in highlighted_edges
        edge_class = "teqo-graph-edge" if highlighted else ""
        color = "#38BDF8" if highlighted else "#31516A"
        marker = "teqo-arrow" if highlighted else "teqo-arrow-muted"
        opacity = "0.95" if highlighted else "0.66"
        width = "2.8" if highlighted else "1.5"
        svg_parts.append(
            f'<path class="{edge_class}" d="M {start_x:.1f} {start_y:.1f} '
            f'L {end_x:.1f} {end_y:.1f}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" opacity="{opacity}" marker-end="url(#{marker})"/>'
        )

    for node in nodes:
        x, y = coordinates[node]
        is_source = node == selected_source
        is_critical = node in highlighted_nodes and not is_source
        kind = "source" if is_source else "critical" if is_critical else "regular"
        color = "#38BDF8" if is_source else "#FFB454" if is_critical else "#0EA5A8"
        glow = ' filter="url(#teqo-glow)"' if is_source or is_critical else ""
        label = html.escape(str(node), quote=True)
        asset_type = html.escape(
            str(graph.nodes[node].get("asset_type", "device")).replace("_", " ").title(),
            quote=True,
        )
        svg_parts.append(
            f'<g class="teqo-graph-node {kind}" role="group" '
            f'aria-label="{asset_type}: {label}">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="18" fill="{color}" '
            f'fill-opacity="0.18" stroke="{color}" stroke-width="2"{glow}/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>'
            f'<text x="{x:.1f}" y="{y + 36:.1f}" text-anchor="middle" '
            'fill="#DCEAF5" font-family="Inter,Segoe UI,sans-serif" '
            f'font-size="12">{label}</text>'
            f'<title>{asset_type}: {label}</title></g>'
        )
    svg_parts.append("</svg></div>")
    return "".join(svg_parts)


def render_twin(graph, queue):
    if graph.number_of_nodes() == 0:
        st.info("No connections are available to explore.")
        return

    source_options = sorted(graph.nodes())
    default_source = (
        str(queue.iloc[0]["device"])
        if not queue.empty and str(queue.iloc[0]["device"]) in graph
        else source_options[0]
    )
    selected_source = st.selectbox(
        "Start from a device",
        options=source_options,
        index=source_options.index(default_source),
    )
    paths = get_attack_paths(graph, selected_source)
    highlighted_nodes = {selected_source}
    highlighted_edges = set()
    for path in paths[:5]:
        highlighted_nodes.update(path["path"])
        highlighted_edges.update(zip(path["path"][:-1], path["path"][1:]))

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Devices and systems", f"{graph.number_of_nodes():,}")
    metric_b.metric("Recorded connections", f"{graph.number_of_edges():,}")
    metric_c.metric("Systems along highlighted routes", f"{len(paths):,}")
    visible_nodes = set(highlighted_nodes)
    visible_nodes.update(graph.successors(selected_source))
    visible_graph = graph.subgraph(visible_nodes)
    st.iframe(
        render_graph_svg(
            visible_graph, selected_source, highlighted_nodes, highlighted_edges
        ),
        height=550,
        width="stretch",
    )

    legend_source, legend_path, legend_other = st.columns(3)
    legend_source.markdown("🔹 **Selected starting device**")
    legend_path.markdown("🟠 **Priority route**")
    legend_other.markdown("◌ Other recorded connections")

    if paths:
        st.markdown("#### Where activity was observed next")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "System": path["target"],
                        "Steps": path["hops"],
                        "Observed route": " → ".join(path["path"]),
                    }
                    for path in paths[:20]
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(f"No onward connections were recorded from {selected_source}.")


MODEL_LABELS = {
    "first_seen_destination_baseline": "First-seen destination baseline",
    "isolation_forest": "Isolation Forest",
    "autoencoder": "Optional autoencoder",
}


def _format_metric(value, *, percent=False):
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.1%}" if percent else f"{number:.3f}"


def render_process_diagram(title, description, steps):
    st.markdown(f"#### {title}")
    st.caption(description)
    items = "".join(
        "<li><strong>{}</strong><span>{}</span></li>".format(
            html.escape(str(label)), html.escape(str(detail))
        )
        for label, detail in steps
    )
    st.markdown(
        f'<ol class="teqo-process" aria-label="{html.escape(title)}">{items}</ol>',
        unsafe_allow_html=True,
    )


def _format_model_results(models, known_positives, budget_number):
    rows = []
    for model_key, label in MODEL_LABELS.items():
        metrics = models.get(model_key, {})
        if not isinstance(metrics, dict):
            continue
        budget = metrics.get("alert_budgets", {}).get(str(budget_number), {})
        evaluated = metrics.get("status") == "evaluated"
        rows.append(
            {
                "Approach": label,
                "Status": metrics.get("status", "not available").replace("_", " "),
                "Known matches": (
                    f"{budget.get('known_positive_hits', 0)} / {known_positives}"
                    if evaluated else "Not evaluable"
                ),
                "Precision": _format_metric(budget.get("precision"), percent=True),
                "Recall": _format_metric(budget.get("recall"), percent=True),
                "F1": _format_metric(budget.get("f1_score"), percent=True),
                "False-positive rate": _format_metric(
                    budget.get("false_positive_rate"), percent=True
                ),
            }
        )
    return rows


def render_data_quality(clean, data, report, provenance_issues):
    rows, users, assets = st.columns(3)
    rows.metric("Activity records reviewed", f"{len(data):,}")
    users.metric("Accounts represented", f"{data['user'].nunique():,}")
    assets.metric(
        "Devices and systems represented",
        f"{pd.concat([data['device'], data['target_device']]).nunique():,}",
    )

    if provenance_issues:
        st.warning(
            "Results are unavailable because the current activity and evaluation "
            "report do not match. Rerun the local demo pipeline before interpreting them."
        )
        return
    if report is None:
        st.info("Evaluation evidence is not available yet. Run the local demo pipeline first.")
        return

    split = report.get("split", {})
    evaluation = report.get("evaluation", {})
    known_positives = int(split.get("test_known_positive_labels", 0))
    st.markdown("#### Chronological holdout")
    st.caption(
        f"{int(split.get('training_events', 0)):,} earlier events were used for "
        f"training and {int(split.get('test_events', 0)):,} later events were held "
        f"out. The held-out period contains {known_positives} known positive labels."
    )
    model_metrics = evaluation.get("models", {})
    ranking_rows = [
        {
            "Approach": label,
            "Status": model_metrics.get(model_key, {}).get("status", "not available").replace("_", " "),
            "Average precision": _format_metric(
                model_metrics.get(model_key, {}).get("average_precision"),
                percent=True,
            ),
            "ROC-AUC": _format_metric(
                model_metrics.get(model_key, {}).get("roc_auc"),
                percent=True,
            ),
        }
        for model_key, label in MODEL_LABELS.items()
        if isinstance(model_metrics.get(model_key, {}), dict)
    ]
    if ranking_rows:
        st.markdown("#### Ranking metrics")
        st.dataframe(pd.DataFrame(ranking_rows), width="stretch", hide_index=True)
    for budget_number in (30, 100):
        findings = _format_model_results(
            model_metrics, known_positives, budget_number
        )
        if findings:
            st.markdown(f"#### Top-{budget_number} review budget")
            st.dataframe(pd.DataFrame(findings), width="stretch", hide_index=True)
    st.caption(
        "False-positive rates treat events without an exact label match as "
        "negative for calculation; some may be malicious but unlabeled."
    )

    if known_positives < 20:
        st.markdown(
            f'<div class="teqo-research-note"><strong>Small label sample:</strong> '
            f'the holdout contains only {known_positives} known positive labels. '
            "These results are illustrative, not a stable measure of detection "
            "performance. Events without a ground-truth match are not confirmed benign."
            "</div>",
            unsafe_allow_html=True,
        )
    elif report.get("dataset_profile") == "synthetic":
        st.warning(
            "This is generated demonstration activity; its metrics do not establish "
            "detection quality on LANL data or real organization networks."
        )
    st.info(
        "The report evaluates raw ranking scores against known labels, not the "
        "rule-adjusted risk score. A high score is a review priority—not a "
        "probability of compromise."
    )

    walk_forward = report.get("walk_forward_validation", {})
    folds = walk_forward.get("folds", []) if isinstance(walk_forward, dict) else []
    if folds:
        st.markdown("#### Earlier chronological validation folds")
        fold_rows = []
        for fold in folds:
            fold_models = fold.get("models", {})
            for model_key, label in MODEL_LABELS.items():
                metrics = fold_models.get(model_key, {})
                if not isinstance(metrics, dict):
                    continue
                budget = metrics.get("alert_budgets", {}).get("30", {})
                fold_rows.append(
                    {
                        "Fold": fold.get("fold"),
                        "Approach": label,
                        "Validation events": metrics.get("test_events", 0),
                        "Known positive labels": metrics.get("known_positive_events", 0),
                        "Status": metrics.get("status", "not available").replace("_", " "),
                        "Average precision": _format_metric(
                            metrics.get("average_precision"), percent=True
                        ),
                        "Known matches in top 30": budget.get("known_positive_hits", 0),
                    }
                )
        if fold_rows:
            st.dataframe(pd.DataFrame(fold_rows), width="stretch", hide_index=True)
            st.caption(
                "Each fold uses the same chronological windows across the baseline "
                "and available models. Folds without known positives are not "
                "evaluable for detection quality."
            )

    st.markdown("#### Activity overview")
    distribution = data["risk_level"].value_counts().reindex(
        ["High", "Medium", "Low"], fill_value=0
    )
    distribution.index = ["Priority review", "Review", "Routine"]
    st.bar_chart(distribution, color=BRAND)


def render_model_intelligence():
    st.markdown("### What the models can—and cannot—say")
    st.markdown(
        "The detector learns a ranking of unusual authentication events without "
        "training on the malicious-activity labels. Ground-truth labels are kept "
        "for evaluation after the chronological split."
    )
    st.markdown(
        '<div class="teqo-research-note"><strong>Interpretation:</strong> an '
        "anomaly score is not a probability that an account is compromised. "
        "Novelty can be legitimate, and suspicious activity can look familiar.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Event signals")
    signals = [
        (
            "Authentication outcome",
            "Whether the attempt succeeded or failed. Failures are a model feature "
            "and risk factor, but never create observed device-graph edges.",
        ),
        (
            "First-seen destination",
            "Whether this user-to-destination pairing appears for the first time "
            "in the available chronology.",
        ),
        (
            "Prior-hour activity",
            "The user's rolling authentication count over the preceding hour; "
            "events at the same timestamp do not inflate one another.",
        ),
        (
            "Authentication and logon type",
            "One-hot encoded categories derived from the event's action field. "
            "The model uses elapsed time, not invented calendar-hour features.",
        ),
    ]
    columns = st.columns(2, gap="medium")
    for index, (title, description) in enumerate(signals):
        with columns[index % 2]:
            st.markdown(
                f'<div class="teqo-contribution"><strong>{html.escape(title)}</strong>'
                f"<p>{html.escape(description)}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Compared ranking approaches")
    models = [
        (
            "First-seen destination baseline",
            "A simple behavior baseline that prioritizes first-time user-to-destination "
            "pairs. It is included to make the value of more complex models testable.",
        ),
        (
            "Isolation Forest",
            "The primary unsupervised model. It is fitted on training-period event "
            "features and scores events by isolation-based anomaly ranking.",
        ),
        (
            "Optional autoencoder",
            "A small reconstruction model trained in bounded batches. Higher "
            "reconstruction error contributes to the anomaly ranking; the scores "
            "are not calibrated probabilities.",
        ),
    ]
    columns = st.columns(3, gap="medium")
    for column, (title, description) in zip(columns, models):
        with column:
            st.markdown(
                f'<div class="teqo-contribution"><strong>{html.escape(title)}</strong>'
                f"<p>{html.escape(description)}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Turning scores into review context")
    st.markdown(
        "A transparent risk layer adds independently checkable factors—new "
        "destination, sensitive asset, failed authentication, and burst activity—"
        "to the normalized anomaly percentile. The analyst view shows those "
        "reasons and exact interventional Shapley contributions for the "
        "Isolation Forest anomaly score. Neither step automates a response."
    )
    render_process_diagram(
        "From event to human review",
        "Each stage adds context; it does not make an automated access or containment decision.",
        [
            ("Authentication event", "Common fields and observed outcome."),
            ("Event features", "Novelty, prior-hour volume, method and outcome."),
            ("Model ranking", "Isolation Forest and optional reconstruction error."),
            ("Risk context", "Percentile plus transparent, named factors."),
            ("Analyst review", "Event context, observed paths and case decision."),
        ],
    )


RESEARCH_CONTRIBUTIONS = [
    (
        "01 · Consistent event schema",
        "Supported synthetic and LANL input profiles map to a common authentication-event shape, keeping later pipeline stages consistent.",
        "src/data_prep.py",
    ),
    (
        "02 · Bounded log processing",
        "Compressed authentication logs are streamed with a configurable row bound rather than loaded in full, making local experiments more manageable.",
        "src/data_prep.py",
    ),
    (
        "03 · Authentication-specific features",
        "Features include failed outcomes, first-seen user/destination pairs, prior-hour activity, and one-hot action categories.",
        "src/model.py",
    ),
    (
        "04 · Time-safe feature behavior",
        "Chronological ordering and tied-timestamp handling prevent simultaneous records from influencing each other's rolling history.",
        "src/model.py",
    ),
    (
        "05 · Unsupervised model comparison",
        "Isolation Forest and an optional bounded-batch autoencoder are compared with a transparent first-seen-destination baseline.",
        "src/model.py",
    ),
    (
        "06 · Chronological validation",
        "The latest time period is held out; expanding-window folds compare the same approaches without training on future events.",
        "src/model.py",
    ),
    (
        "07 · Review-budget evaluation",
        "The evaluation report includes ranking and fixed top-30/top-100 review-budget metrics, confusion counts, and known-label coverage.",
        "src/evaluation.py",
    ),
    (
        "08 · Transparent risk factors",
        "Named factors and documented score contributions complement the anomaly percentile so analysts can inspect why an event ranks highly.",
        "src/risk_engine.py",
    ),
    (
        "09 · Evidence-bounded relationship paths",
        "The device graph counts successful authentication observations only; paths describe the sample, not general network reachability.",
        "src/digital_twin.py",
    ),
    (
        "10 · Reproducible, analyst-owned review",
        "Run identifiers, hashes and provenance checks accompany results; case status, notes and audit entries remain local and analyst-controlled.",
        "src/provenance.py · src/case_store.py",
    ),
]


def render_research_and_evaluation(report, data, provenance_issues):
    st.markdown("### Research question")
    st.markdown(
        "Can unsupervised authentication-anomaly rankings help prioritize known "
        "red-team events in a later chronological period, compared with a "
        "first-seen-destination baseline?"
    )
    st.markdown("### Evaluation design")
    render_process_diagram(
        "Train and evaluate without time leakage",
        "Labels are evaluation evidence only; they are not inputs to unsupervised model fitting.",
        [
            ("Earlier events", "Fit feature transforms and unsupervised model(s)."),
            ("Later holdout", "Keep a chronological period separate from training."),
            ("Rank events", "Score holdout events using models fitted on earlier data."),
            ("Compare", "Measure agreement with known labels at fixed review budgets."),
        ],
    )

    if report is None or provenance_issues:
        render_data_quality(pd.DataFrame(), data, None, provenance_issues)
    else:
        split = report.get("split", {})
        st.markdown("### Evidence in the current run")
        metadata_a, metadata_b, metadata_c = st.columns(3)
        metadata_a.metric("Dataset profile", str(report.get("dataset_profile", "unknown")))
        metadata_b.metric(
            "Training events",
            f"{int(split.get('training_events', 0)):,}",
            help="Earlier chronological portion used to fit the models.",
        )
        metadata_c.metric(
            "Holdout positive labels",
            f"{int(split.get('test_known_positive_labels', 0)):,}",
            help="Exact ground-truth matches in the later test period.",
        )
        render_data_quality(pd.DataFrame(), data, report, provenance_issues)

    st.markdown("### Limitations and responsible interpretation")
    st.markdown(
        "- Ground truth is exact-match red-team labeling; an event without a label "
        "match may still be malicious.\n"
        "- A small number of known positives cannot support stable performance claims.\n"
        "- Synthetic demonstrations validate the pipeline and evaluation code, "
        "not real-world detection quality.\n"
        "- Risk scores are prioritization values, not calibrated probabilities; "
        "analysts remain responsible for case decisions.\n"
        "- The current ingestion profiles are synthetic and LANL; arbitrary "
        "organization exports need an approved, validated adapter."
    )

def render_architecture():
    st.title("System architecture")
    st.caption(
        "A view of the current local research implementation and how its components connect."
    )
    st.markdown(
        "The diagrams below describe the current local research implementation, "
        "not a hosted or live-monitoring deployment."
    )
    render_process_diagram(
        "Implemented data and review flow",
        "Artifacts and processing follow the current pipeline modules and dashboard workflow.",
        [
            ("Profile input", "Synthetic demo or an authorized LANL sample."),
            ("Data preparation", "Stream, normalize and label known matches."),
            ("Chronological split", "Earlier training period; later held-out period."),
            ("Model and risk", "Unsupervised ranking with transparent risk factors."),
            ("Dashboard", "Filter, investigate, explore observed paths and manage cases."),
        ],
    )
    render_process_diagram(
        "Observed-path data flow",
        "Only successful authentication events create device-to-device observations.",
        [
            ("Clean events", "Read device, destination and outcome."),
            ("Successful only", "Failed attempts are excluded from graph edges."),
            ("Directed graph", "Repeated successful observations increase edge weight."),
            ("Bounded path query", "Shortest paths are limited to four hops and ranked by asset criticality."),
            ("Analyst context", "Observed sample paths—not proof of reachability or authorization."),
        ],
    )
    render_process_diagram(
        "Local analyst case workflow",
        "The analyst owns status, notes and decisions; TeqoAI does not execute containment.",
        [
            ("Select an event", "Review its authentication and risk context."),
            ("Record decision", "Set status, owner, notes and checklist progress."),
            ("Persist locally", "SQLite stores current case records on this computer."),
            ("Audit changes", "Changed case state is recorded in a local audit history."),
        ],
    )
    st.markdown("### Research and engineering contributions")
    st.caption(
        "These are implementation contributions of the prototype, not claims of "
        "validated operational effectiveness."
    )
    columns = st.columns(2, gap="medium")
    for index, (title, description, source) in enumerate(RESEARCH_CONTRIBUTIONS):
        with columns[index % 2]:
            st.markdown(
                f'<div class="teqo-contribution"><strong>{html.escape(title)}</strong>'
                f"<p>{html.escape(description)}</p>"
                f"<small>{html.escape(source)}</small></div>",
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="teqo-research-note"><strong>Local prototype:</strong> '
        "case notes are stored unencrypted on this computer. Do not enter "
        "sensitive production information. The project is not a production "
        "security product and does not provide live monitoring or automated response."
        "</div>",
        unsafe_allow_html=True,
    )


PAGE_TITLES = {
    "Overview": "Operations overview",
    "Investigate": "Alert investigation",
    "Observed Paths": "Observed paths",
    "AI Intelligence": "AI model intelligence",
    "Research & Evaluation": "Research and evaluation",
}

PAGE_DESCRIPTIONS = {
    "Overview": "A concise view of the current event queue and review workload.",
    "Investigate": "Inspect event evidence and record an analyst-owned case decision.",
    "Observed Paths": "Explore successful authentication relationships in the loaded sample.",
    "AI Intelligence": "Understand the implemented features, ranking approaches and score interpretation.",
    "Research & Evaluation": "Review the chronological evaluation design, measured evidence and limitations.",
}


def render_demo_setup(button_key):
    if DATASET != "synthetic":
        st.info(
            "The hosted evaluation demo uses the synthetic profile. Set "
            "TEQO_DATASET=synthetic and restart the app to generate it."
        )
        return
    if os.path.exists(CLEAN_FILE) and os.path.exists(RISK_FILE):
        return

    st.markdown("### Try the reproducible model evaluation")
    st.markdown(
        "Generate a small, deterministic synthetic authentication dataset with "
        "known red-team labels, then run the same chronological model pipeline "
        "used by the project. Nothing needs to be uploaded."
    )
    st.caption(
        "This is a software demonstration, not evidence of performance on real "
        "organizations. Generated files and case records on Community Cloud "
        "are temporary."
    )
    if st.button(
        "Generate demo data and run evaluation",
        key=button_key,
        type="primary",
    ):
        try:
            with st.status("Preparing the synthetic evaluation...", expanded=True) as status:
                run_synthetic_demo(
                    lambda step: st.write(f"Running `python -m {step}`")
                )
                status.update(
                    label="Synthetic evaluation is ready",
                    state="complete",
                    expanded=False,
                )
        except (RuntimeError, FileNotFoundError, TimeoutError) as error:
            st.error("The synthetic evaluation could not be prepared.")
            st.code(str(error))
            return

        st.success(
            "Demo data and evaluation results are ready. Open Research & Evaluation "
            "to inspect the metrics, then use Overview and Investigate to explore events."
        )
        st.rerun()


def render_homepage():
    with open(LOGO_FILE, "rb") as logo_file:
        logo_data = base64.b64encode(logo_file.read()).decode("ascii")
    logo_markup = (
        f'<img src="data:image/png;base64,{logo_data}" alt="TeqoAI logo">'
    )
    features = [
        (
            "01 · Detect",
            "Prioritize unusual activity",
            "Anomaly rankings and a simple behavior baseline help bring potentially unusual authentication events into focus.",
        ),
        (
            "02 · Investigate",
            "Understand each alert",
            "Review event context, contributing risk signals, related activity, and an analyst-owned case decision.",
        ),
        (
            "03 · Connect",
            "Explore observed relationships",
            "See device-to-device links found in the event sample. These links are observations—not proof of reachability or access.",
        ),
    ]
    feature_markup = "".join(
        f'<article class="teqo-home-feature">'
        f'<div class="teqo-section-kicker">{html.escape(kicker)}</div>'
        f'<h3>{html.escape(title)}</h3>'
        f'<p>{html.escape(description)}</p></article>'
        for kicker, title, description in features
    )
    creator_paragraphs = [
        "I am a passionate and technology-driven student with a strong interest in "
        "AI, cybersecurity, and software development. I am particularly interested "
        "in exploring how emerging technologies can address real-world challenges "
        "and contribute to building a safer digital environment. This passion "
        "motivated me to participate in the INSA Summer Camp, even while balancing "
        "the academic responsibilities of being a Grade 12 student preparing for "
        "the national examination.",
        "I enjoy researching new technologies, developing innovative solutions, "
        "and transforming complex ideas into practical applications. Through the "
        "INSA Summer Camp, I have been expanding my technical knowledge and "
        "exploring the role of artificial intelligence in modern cybersecurity.",
        "My project, TeqoAI, focuses on intelligent cybersecurity through "
        "specialized AI models, risk analysis, and explainable artificial "
        "intelligence. I aspire to develop innovative solutions that strengthen "
        "digital security across sectors such as education, telecommunications, "
        "finance, healthcare, and critical infrastructure. My long-term goal "
        "is to become a skilled technology innovator who contributes to Ethiopia's "
        "digital transformation through research, creativity, and impactful "
        "technological solutions.",
    ]
    creator_markup = "".join(
        f"<p>{html.escape(paragraph)}</p>" for paragraph in creator_paragraphs
    )

    with st.container(key="teqo-home"):
        hero_text, hero_logo = st.columns([1.7, .8], gap="large")
        with hero_text:
            st.markdown(
                '<section class="teqo-home-hero">'
                '<div class="teqo-section-kicker">INSA SUMMER CAMP · GRADUATION PROJECT</div>'
                '<h1>Welcome to <span style="color:#38BDF8">TeqoAI</span></h1>'
                '<p>Making authentication security easier to understand, investigate, '
                'and improve with explainable artificial intelligence.</p>'
                '<p>TeqoAI explores how unusual login behavior can be prioritized for '
                'human review—connecting anomaly signals with clear evidence and '
                'observed device relationships.</p>'
                '</section>',
                unsafe_allow_html=True,
            )
        with hero_logo:
            st.markdown(
                f'<div class="teqo-home-hero-logo">{logo_markup}</div>',
                unsafe_allow_html=True,
            )

        cta_workspace, cta_results = st.columns([1, 1], gap="small")
        cta_workspace.page_link(
            overview_page,
            label="Explore the dashboard",
            icon=":material/arrow_forward:",
        )
        cta_results.page_link(
            results_page,
            label="See project results",
            icon=":material/analytics:",
        )
        render_demo_setup("teqo-demo-build-home")

        st.markdown(
            '<section class="teqo-home-section">'
            '<div class="teqo-section-kicker">THE PROJECT</div>'
            '<h2>Explainable authentication triage</h2>'
            '<p class="teqo-home-copy">A research prototype that ranks unusual '
            'authentication activity, explains contributing signals, and supports '
            'analyst-led investigation.</p>'
            '<div class="teqo-home-facts">'
            '<div><span>Focus</span> · Authentication activity</div>'
            '<div><span>Approach</span> · Unsupervised anomaly ranking</div>'
            '<div><span>Decision-maker</span> · Human analyst</div>'
            '</div></section>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<section class="teqo-home-section">'
            '<div class="teqo-section-kicker">WHAT YOU CAN EXPLORE</div>'
            '<div class="teqo-home-features">'
            + feature_markup
            + "</div></section>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<section class="teqo-home-section">'
            '<div class="teqo-section-kicker">THE APPROACH</div>'
            '<h2>How the prototype works</h2>'
            '<ol class="teqo-home-steps">'
            '<li><span class="teqo-home-number">01</span><strong>Activity</strong>'
            '<span>Read authentication events.</span></li>'
            '<li><span class="teqo-home-number">02</span><strong>Prioritize</strong>'
            '<span>Rank unusual patterns and add transparent risk factors.</span></li>'
            '<li><span class="teqo-home-number">03</span><strong>Investigate</strong>'
            '<span>Give analysts context and observed relationships.</span></li>'
            '<li><span class="teqo-home-number">04</span><strong>Evaluate</strong>'
            '<span>Compare rankings with known labels when available.</span></li>'
            '</ol></section>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<section class="teqo-home-section">'
            '<div class="teqo-section-kicker">MEET THE CREATOR</div>'
            '<h2>Hobby Yano</h2>'
            f'<div class="teqo-home-about">{creator_markup}</div>'
            '</section>',
            unsafe_allow_html=True,
        )

        with st.container(key="teqo-home-footer"):
            footer_brand, footer_content = st.columns([.8, 2.2], gap="large")
            with footer_brand:
                st.markdown(
                    '<div class="teqo-home-footer-brand">'
                    + logo_markup
                    + "</div>",
                    unsafe_allow_html=True,
                )
            with footer_content:
                st.markdown(
                    '<div class="teqo-home-footer-content">'
                    '<div class="teqo-section-kicker">CONNECT</div>'
                    '<h2>Let’s build safer technology</h2>'
                    '<p>Reach out for questions, feedback, or collaboration.</p>'
                    '<div class="teqo-home-footer-contacts">'
                    '<a href="mailto:hobbieyano@gmail.com">'
                    '<span class="teqo-home-contact-icon" aria-hidden="true">'
                    '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/>'
                    '<path d="m3 7 9 6 9-6"/></svg></span>'
                    'Email · hobbieyano@gmail.com</a>'
                    '<a href="https://t.me/hobbie111" target="_blank" rel="noopener noreferrer">'
                    '<span class="teqo-home-contact-icon" aria-hidden="true">'
                    '<svg viewBox="0 0 24 24"><path d="m22 2-7 20-4-9-9-4Z"/>'
                    '<path d="M22 2 11 13"/></svg></span>'
                    'Telegram · @hobbie111</a>'
                    '<a href="https://github.com/Juna-111" target="_blank" '
                    'rel="noopener noreferrer">'
                    '<span class="teqo-home-contact-icon" aria-hidden="true">'
                    '<svg viewBox="0 0 24 24"><path d="m8 9-3 3 3 3M16 9l3 3-3 3"/>'
                    '<path d="m14 5-4 14"/></svg></span>'
                    'GitHub · github.com/Juna-111</a>'
                    '<a href="tel:+251962703558">'
                    '<span class="teqo-home-contact-icon" aria-hidden="true">'
                    '<svg viewBox="0 0 24 24"><path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 '
                    '19.8 0 0 1-8.6-3.1 19.4 19.4 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 '
                    '2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7l.5 2.8a2 2 0 0 1-.6 1.9L7.7 '
                    '9.7a16 16 0 0 0 6 6l1.3-1.3a2 2 0 0 1 1.9-.6l2.8.5a2 2 0 0 1 '
                    '1.7 2.6Z"/></svg></span>'
                    'Phone · (+251962703558)</a>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
                st.page_link(
                    architecture_page,
                    label="Explore the system architecture and research contributions",
                    icon=":material/schema:",
                )
                st.markdown(
                    '<p class="teqo-home-footer-note"><strong>Research prototype, '
                    'not a production security product.</strong> '
                    + (
                        "This build uses generated demonstration activity; its results do not "
                        "establish accuracy on real networks. TeqoAI supports analyst review "
                        "and does not perform automated response."
                        if DATASET == "synthetic"
                        else
                        "Results depend on the selected dataset and available confirmed labels. "
                        "Scores are review priorities, not probabilities, and TeqoAI does not "
                        "perform automated response."
                    )
                    + "</p>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="teqo-home-footer-bottom">'
                'TEQOAI · Created by Hobby Yano · INSA Summer Camp</div>',
                unsafe_allow_html=True,
            )


def render_analyst_page(selected_page):
    if not os.path.exists(CLEAN_FILE) or not os.path.exists(RISK_FILE):
        st.title(PAGE_TITLES[selected_page])
        st.caption(PAGE_DESCRIPTIONS[selected_page])
        render_demo_setup(f"teqo-demo-build-{selected_page.lower().replace(' ', '-')}")
        if selected_page == "AI Intelligence":
            render_model_intelligence()
        elif selected_page == "Research & Evaluation":
            render_research_and_evaluation(None, pd.DataFrame(), [])
        else:
            st.info(
                "The analyst workspace has no processed events yet. Run the local "
                "demo pipeline, then return to this page."
            )
        return

    clean_file_stat = os.stat(CLEAN_FILE)
    risk_file_stat = os.stat(RISK_FILE)
    try:
        clean_events, risk_events = load_data(
            clean_file_stat.st_mtime_ns,
            clean_file_stat.st_size,
            risk_file_stat.st_mtime_ns,
            risk_file_stat.st_size,
        )
    except (OSError, pd.errors.ParserError, ValueError):
        st.error("TeqoAI could not load the activity results. Please rerun the local demo.")
        return

    required_risk_columns = {
        "risk_level", "risk_score", "time", "user", "device",
        "target_device", "result", "action",
    }
    if required_risk_columns.difference(risk_events.columns):
        st.error("The activity results are incomplete. Please rerun the local demo.")
        return

    case_records = {}
    if selected_page in {"Overview", "Investigate", "Observed Paths"}:
        try:
            case_records = get_cases(DATASET)
        except sqlite3.Error:
            st.error("The case workspace is unavailable. Restart the local TeqoAI demo.")
            return

    report = None
    provenance_issues = []
    if os.path.exists(EVALUATION_FILE):
        try:
            with open(EVALUATION_FILE, encoding="utf-8") as report_file:
                report = json.load(report_file)
        except (OSError, json.JSONDecodeError):
            provenance_issues.append("Evaluation results could not be read.")

    if report is not None and not isinstance(report, dict):
        provenance_issues.append("Evaluation results are not in a usable format.")
        report = None

    if report is not None:
        clean_sha256 = cached_file_sha256(
            CLEAN_FILE, clean_file_stat.st_mtime_ns, clean_file_stat.st_size
        )
        run_ids = (
            risk_events["pipeline_run_id"].tolist()
            if "pipeline_run_id" in risk_events else []
        )
        provenance_issues.extend(
            validate_pipeline_provenance(report, DATASET, clean_sha256, run_ids)
        )
        if len(clean_events) != len(risk_events):
            provenance_issues.append(
                "Activity and risk results contain different event counts."
            )

    filtered_events = risk_events
    if selected_page in {"Overview", "Investigate", "Observed Paths"}:
        with st.sidebar:
            filtered_events = apply_filters(risk_events, case_records)
            with st.expander("Privacy note"):
                st.caption(
                    "Case notes are stored unencrypted on this computer. Do not enter "
                    "sensitive production information."
                )

    title_column, badge_column = st.columns([5, 1])
    with title_column:
        st.title(PAGE_TITLES[selected_page])
        st.caption(PAGE_DESCRIPTIONS[selected_page])
    with badge_column:
        if DATASET == "synthetic":
            st.markdown(
                '<div class="teqo-data-badge">SYNTHETIC DEMO</div>',
                unsafe_allow_html=True,
            )

    if selected_page == "Overview":
        render_overview(risk_events, filtered_events, case_records)
    elif selected_page == "Investigate":
        try:
            graph = load_graph(clean_events)
            render_investigation(
                filtered_events, risk_events, graph, case_records, clean_events
            )
        except (KeyError, ValueError, sqlite3.Error):
            st.error(
                "The alert investigation could not be prepared. Refresh the demo "
                "or try again later."
            )
    elif selected_page == "Observed Paths":
        try:
            graph = load_graph(clean_events)
            render_twin(graph, filtered_events)
        except (KeyError, ValueError, nx.NetworkXException):
            st.error(
                "The observed connections could not be prepared. Refresh the demo "
                "or try again later."
            )
    elif selected_page == "AI Intelligence":
        render_model_intelligence()
    elif selected_page == "Research & Evaluation":
        render_research_and_evaluation(report, risk_events, provenance_issues)
    else:
        render_data_quality(clean_events, risk_events, report, provenance_issues)

    st.markdown(
        '<div class="teqo-foot">TEQOAI · Analyst-led security triage</div>',
        unsafe_allow_html=True,
    )


home_page = st.Page(
    render_homepage,
    title="Home",
    icon=":material/home:",
    url_path="home",
    default=True,
)
overview_page = st.Page(
    lambda: render_analyst_page("Overview"),
    title="Overview",
    icon=":material/space_dashboard:",
    url_path="overview",
)
investigation_page = st.Page(
    lambda: render_analyst_page("Investigate"),
    title="Investigate",
    icon=":material/search:",
    url_path="investigate",
)
paths_page = st.Page(
    lambda: render_analyst_page("Observed Paths"),
    title="Observed Paths",
    icon=":material/hub:",
    url_path="observed-paths",
)
intelligence_page = st.Page(
    lambda: render_analyst_page("AI Intelligence"),
    title="AI Intelligence",
    icon=":material/psychology:",
    url_path="ai-intelligence",
)
results_page = st.Page(
    lambda: render_analyst_page("Research & Evaluation"),
    title="Research & Evaluation",
    icon=":material/analytics:",
    url_path="research",
)
architecture_page = st.Page(
    render_architecture,
    title="System Architecture",
    icon=":material/schema:",
    url_path="architecture",
)

current_page = st.navigation(
    {
        "Project": [home_page],
        "Operations": [overview_page, investigation_page],
        "Intelligence": [paths_page, intelligence_page],
        "Research": [results_page, architecture_page],
    },
    position="sidebar",
)
with st.sidebar:
    st.divider()
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, width=46)
    st.markdown(
        '<div class="teqo-brand">TEQO<span>AI</span></div>'
        '<div style="color:#B3C1CC;font-size:.76rem;line-height:1.5">'
        "Explainable authentication triage</div>",
        unsafe_allow_html=True,
    )
    st.caption("Local research prototype")
current_page.run()
