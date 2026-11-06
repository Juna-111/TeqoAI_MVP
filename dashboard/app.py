"""TeqoAI analyst dashboard for authentication anomaly investigation."""

import hashlib
import html
import json
import os
import sqlite3

import networkx as nx
import pandas as pd
import streamlit as st

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")

from src.digital_twin import (
    build_explainer,
    build_graph,
    explain_event,
    get_attack_paths,
    load_model_artifacts,
)
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
    initial_sidebar_state="expanded",
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
    .teqo-brand { font-size: 1.25rem; font-weight: 750; letter-spacing: .04em; }
    .teqo-brand span { color: var(--teqo-accent); }
    .teqo-eyebrow {
        color: var(--teqo-muted); font-size: .73rem; font-weight: 650;
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
    .teqo-welcome-hero {
        position: relative;
        overflow: hidden;
        padding: 34px;
        border: 1px solid rgba(56, 189, 248, .24);
        border-radius: 18px;
        background:
            radial-gradient(ellipse at 86% 12%, rgba(56,189,248,.18), transparent 38%),
            linear-gradient(125deg, #0E1B2A 5%, #0A1726 70%);
        box-shadow: 0 22px 60px rgba(0,0,0,.22);
    }
    .teqo-welcome-hero h1 {
        margin: 8px 0 12px;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.08;
        letter-spacing: -.04em;
    }
    .teqo-welcome-hero p {
        max-width: 720px;
        color: #B5C5D2;
        font-size: 1.04rem;
        line-height: 1.7;
    }
    .teqo-section-kicker {
        color: var(--teqo-accent);
        font-size: .73rem;
        font-weight: 750;
        letter-spacing: .13em;
        text-transform: uppercase;
    }
    .teqo-feature-card {
        min-height: 168px;
        padding: 20px;
        border: 1px solid var(--teqo-border);
        border-radius: 12px;
        background: linear-gradient(145deg, #0E1B2A, #0A1726);
    }
    .teqo-feature-card h3 { margin: 0 0 8px; font-size: 1rem; }
    .teqo-feature-card p {
        margin: 0;
        color: #A9BBC9;
        font-size: .88rem;
        line-height: 1.6;
    }
    .teqo-project-note {
        border-left: 3px solid var(--teqo-accent);
        border-radius: 8px;
        padding: 14px 18px;
        background: rgba(56,189,248,.07);
        color: #C4D5E1;
    }
    .teqo-contact-links {
        display: grid;
        gap: 10px;
        margin-top: 14px;
    }
    .teqo-contact-link {
        display: flex;
        align-items: center;
        gap: 12px;
        min-height: 62px;
        padding: 10px 13px;
        border: 1px solid var(--teqo-border);
        border-radius: 11px;
        background: linear-gradient(120deg, rgba(14,27,42,.98), rgba(10,23,38,.96));
        color: #E5F0F7 !important;
        text-decoration: none !important;
        transition: border-color .18s ease, transform .18s ease, background .18s ease;
    }
    .teqo-contact-link:hover {
        border-color: rgba(56,189,248,.65);
        background: rgba(56,189,248,.08);
        transform: translateY(-1px);
    }
    .teqo-contact-icon {
        display: grid;
        flex: 0 0 38px;
        width: 38px;
        height: 38px;
        place-items: center;
        border-radius: 10px;
        background: rgba(56,189,248,.12);
        color: #38BDF8;
    }
    .teqo-contact-icon.telegram {
        background: #229ED9;
        color: #FFFFFF;
    }
    .teqo-contact-icon.github {
        background: #F0F6FC;
        color: #161B22;
    }
    .teqo-contact-icon svg { width: 21px; height: 21px; display: block; }
    .teqo-contact-copy { min-width: 0; flex: 1; }
    .teqo-contact-copy strong { display: block; font-size: .88rem; }
    .teqo-contact-copy span {
        display: block;
        overflow: hidden;
        color: #94A3B8;
        font-size: .75rem;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .teqo-contact-arrow { color: #7F94A6; font-size: 1rem; }
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
        .teqo-graph-edge, .teqo-graph-node { animation: none; }
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


@st.cache_resource
def load_explainer(_model):
    return build_explainer(_model)


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


def render_investigation(queue, data, graph, cases):
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
                explainer = load_explainer(model)
                if explainer is None:
                    st.warning("SHAP is unavailable for this model; review the evidence above.")
                else:
                    event_position = int(
                        queue.loc[queue["_event_id"] == selected_event_id].index[0]
                    )
                    explanation = explain_event(
                        explainer, scaler, X, X.columns.tolist(),
                        int(row.name) if isinstance(row.name, int) else event_position,
                    )
                    st.success(explanation)
            except (IndexError, KeyError, ValueError, RuntimeError):
                st.warning(
                    "The explanation could not be generated for this alert. "
                    "Review the detection evidence above or try another alert."
                )

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


def render_data_quality(clean, data, report, provenance_issues):
    st.subheader("Project results")
    st.caption(
        "A clear view of what this prototype reviewed and how it handled known examples."
    )
    rows, users, assets = st.columns(3)
    rows.metric("Activity records reviewed", f"{len(data):,}")
    users.metric("Accounts represented", f"{data['user'].nunique():,}")
    assets.metric(
        "Devices and systems represented",
        f"{pd.concat([data['device'], data['target_device']]).nunique():,}",
    )

    if provenance_issues:
        st.warning(
            "Results are unavailable because the current activity and results "
            "do not match. Please refresh the TeqoAI demo."
        )
        return
    if report is None:
        st.info("Results are not available yet. Please run the TeqoAI demo first.")
        return

    split = report["split"]
    evaluation = report["evaluation"]
    test_labels = int(split.get("test_known_positive_labels", 0))
    st.markdown("#### Known examples found")
    st.caption(
        "The system reviews the most unusual activity first. These results count "
        "how many examples marked in advance appeared in the first 30 items."
    )
    findings = []
    labels = {
        "first_seen_destination_baseline": "Simple behavior comparison",
        "isolation_forest": "TeqoAI detection model",
        "autoencoder": "Alternative model",
    }
    for model_name, metrics in evaluation.get("models", {}).items():
        if not isinstance(metrics, dict):
            continue
        top_30 = metrics.get("alert_budgets", {}).get("30", {})
        if metrics.get("status") != "evaluated":
            found = "Not available"
        else:
            found = (
                f"{int(top_30.get('known_positive_hits', 0))} of {test_labels}"
            )
        findings.append(
            {"Approach": labels.get(model_name, "Additional comparison"), "Marked examples in first 30": found}
        )
    if findings:
        st.dataframe(pd.DataFrame(findings), width="stretch", hide_index=True)

    if test_labels < 20:
        st.warning(
            f"This demonstration contains only {test_labels} marked examples in "
            "the separate test period. Results are illustrative and are not a "
            "stable measure of real-world detection performance."
        )
    st.info(
        "This run uses generated demonstration activity, not live organization "
        "data. Unmarked activity is not guaranteed harmless, and a high score is "
        "a review priority—not a probability of compromise."
    )

    walk_forward = report.get("walk_forward_validation", {})
    folds = walk_forward.get("folds", []) if isinstance(walk_forward, dict) else []
    if folds:
        examples_by_period = [
            int(fold.get("metrics", {}).get("known_positive_events", 0))
            for fold in folds
        ]
        st.markdown("#### Earlier consistency checks")
        st.caption(
            f"{len(examples_by_period)} earlier periods contained "
            f"{', '.join(map(str, examples_by_period))} "
            "marked examples respectively. That is too few to draw a reliable "
            "conclusion; real-data testing is still needed."
        )

    st.markdown("#### Activity overview")
    distribution = data["risk_level"].value_counts().reindex(
        ["High", "Medium", "Low"], fill_value=0
    )
    distribution.index = ["Priority review", "Review", "Routine"]
    st.bar_chart(distribution, color=BRAND)


PAGE_TITLES = {
    "Overview": "Shift overview",
    "Investigate": "Alert investigation",
    "Observed Paths": "Observed connections",
    "Results & Method": "Results and approach",
}


def render_homepage():
    st.markdown(
        "<style>[data-testid='stSidebar'],"
        "[data-testid='stSidebarCollapsedControl']{display:none!important}</style>",
        unsafe_allow_html=True,
    )

    hero, summary = st.columns([1.55, 1], gap="large", vertical_alignment="center")
    with hero:
        st.markdown('<div class="teqo-section-kicker">INSA SUMMER CAMP · GRADUATION PROJECT</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="teqo-welcome-hero">'
            '<h1>Welcome to <span style="color:#38BDF8">TeqoAI</span></h1>'
            '<p>Making authentication security easier to understand, investigate, '
            'and improve with explainable artificial intelligence.</p>'
            '<p>TeqoAI explores how unusual login behavior can be prioritized for '
            'human review—connecting anomaly signals with clear evidence and '
            'observed device relationships.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        cta_workspace, cta_results = st.columns(2)
        cta_workspace.page_link(
            overview_page,
            label="Explore the dashboard",
            icon=":material/arrow_forward:",
            width="stretch",
        )
        cta_results.page_link(
            results_page,
            label="See project results",
            icon=":material/analytics:",
            width="stretch",
        )

    with summary:
        st.markdown(
            '<div class="teqo-home-panel">'
            '<div class="teqo-section-kicker">THE PROJECT</div>'
            '<h3 style="margin-top:10px">Explainable authentication triage</h3>'
            '<p style="color:#A9BBC9;line-height:1.65">A research prototype that '
            'ranks unusual authentication activity, explains contributing signals, '
            'and supports analyst-led investigation.</p>'
            '<div class="teqo-queue-stat"><span>Focus</span><strong>Authentication activity</strong></div>'
            '<div class="teqo-queue-stat"><span>Approach</span><strong>Unsupervised anomaly ranking</strong></div>'
            '<div class="teqo-queue-stat"><span>Decision-maker</span><strong>Human analyst</strong></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown('<div class="teqo-section-kicker">WHAT YOU CAN EXPLORE</div>', unsafe_allow_html=True)
    feature_columns = st.columns(3, gap="medium")
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
    for column, (kicker, title, description) in zip(feature_columns, features):
        with column:
            st.markdown(
                f'<div class="teqo-feature-card">'
                f'<div class="teqo-section-kicker">{kicker}</div>'
                f'<h3 style="margin-top:12px">{title}</h3>'
                f'<p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown("### How the prototype works")
    steps = st.columns(4, gap="small")
    for number, (title, description) in enumerate(
        [
            ("Activity", "Read authentication events."),
            ("Prioritize", "Rank unusual patterns and add transparent risk factors."),
            ("Investigate", "Give analysts context and observed relationships."),
            ("Evaluate", "Compare rankings with known labels when available."),
        ],
        start=1,
    ):
        with steps[number - 1]:
            st.markdown(
                f'<div class="teqo-home-panel">'
                f'<div class="teqo-section-kicker">STEP {number:02d}</div>'
                f'<h4 style="margin-top:10px">{title}</h4>'
                f'<p style="color:#A9BBC9;font-size:.86rem">{description}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown('<div class="teqo-section-kicker">MEET THE CREATOR</div>', unsafe_allow_html=True)
    about_column, contact_column = st.columns([2, 1], gap="large")
    with about_column:
        st.markdown("### Hobby Yano")
        st.markdown(
            "I am a passionate and technology-driven student with a strong interest in "
            "AI, cybersecurity, and software development. I am particularly interested "
            "in exploring how emerging technologies can address real-world challenges "
            "and contribute to building a safer digital environment. This passion "
            "motivated me to participate in the INSA Summer Camp, even while balancing "
            "the academic responsibilities of being a Grade 12 student preparing for "
            "the national examination.\n\n"
            "I enjoy researching new technologies, developing innovative solutions, "
            "and transforming complex ideas into practical applications. Through the "
            "INSA Summer Camp, I have been expanding my technical knowledge and "
            "exploring the role of artificial intelligence in modern cybersecurity.\n\n"
            "My project, TeqoAI, focuses on intelligent cybersecurity through "
            "specialized AI models, risk analysis, and explainable artificial "
            "intelligence. I aspire to develop innovative solutions that strengthen "
            "digital security across sectors such as education, telecommunications, "
            "finance, healthcare, and critical infrastructure. My long-term goal "
            "is to become a skilled technology innovator who contributes to Ethiopia's "
            "digital transformation through research, creativity, and impactful "
            "technological solutions."
        )

    with contact_column:
        st.markdown(
            '<div class="teqo-home-panel">'
            '<div class="teqo-section-kicker">CONNECT</div>'
            '<h3 style="margin-top:10px">Let’s build safer technology</h3>'
            '<p style="color:#A9BBC9;margin-bottom:0">Reach out for questions, feedback, or collaboration.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="teqo-contact-links">'
            '<a class="teqo-contact-link" href="mailto:hobbieyano@gmail.com">'
            '<span class="teqo-contact-icon"><svg viewBox="0 0 24 24" fill="none" '
            'aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2" '
            'stroke="currentColor" stroke-width="1.8"/><path d="m4 7 8 6 8-6" '
            'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
            'stroke-linejoin="round"/></svg></span>'
            '<span class="teqo-contact-copy"><strong>Email</strong>'
            '<span>hobbieyano@gmail.com</span></span>'
            '<span class="teqo-contact-arrow" aria-hidden="true">↗</span></a>'
            '<a class="teqo-contact-link" href="https://t.me/hobbie111" '
            'target="_blank" rel="noopener noreferrer">'
            '<span class="teqo-contact-icon telegram"><svg viewBox="0 0 24 24" '
            'fill="currentColor" aria-hidden="true"><path d="M21.8 3.2 18.5 20c-.25 '
            '1.18-.9 1.47-1.83.91l-5.05-3.72-2.44 2.35c-.27.27-.5.5-1.02.5l.36-5.14 '
            '9.36-8.45c.41-.36-.09-.56-.64-.2L5.67 13.42.7 11.87c-1.08-.34-1.1-1.08 '
            '.23-1.6L20.37 2.7c.9-.33 1.68.22 1.43.5Z"/></svg></span>'
            '<span class="teqo-contact-copy"><strong>Telegram</strong>'
            '<span>@hobbie111</span></span>'
            '<span class="teqo-contact-arrow" aria-hidden="true">↗</span></a>'
            '<a class="teqo-contact-link" href="https://github.com/Juna-111" '
            'target="_blank" rel="noopener noreferrer">'
            '<span class="teqo-contact-icon github"><svg viewBox="0 0 24 24" '
            'fill="currentColor" aria-hidden="true"><path d="M12 .9a11.1 11.1 0 0 0-3.51 '
            '21.63c.56.1.76-.24.76-.54v-2.08c-3.1.67-3.76-1.31-3.76-1.31-.5-1.28-1.24-'
            '1.62-1.24-1.62-1.01-.69.08-.68.08-.68 1.12.08 1.71 1.15 1.71 1.15 1 1.7 '
            '2.61 1.2 3.25.92.1-.72.39-1.2.71-1.48-2.47-.28-5.07-1.24-5.07-5.5 0-1.22 '
            '.44-2.22 1.15-3-.12-.28-.5-1.43.11-2.97 0 0 .94-.3 3.05 1.15a10.6 10.6 '
            '0 0 1 5.55 0c2.12-1.45 3.05-1.15 3.05-1.15.61 1.54.23 2.69.12 2.97.72.78 '
            '1.15 1.78 1.15 3 0 4.27-2.61 5.22-5.1 5.49.4.35.76 1.02.76 2.06v3.05c0 '
            '.3.2.65.77.54A11.1 11.1 0 0 0 12 .9Z"/></svg></span>'
            '<span class="teqo-contact-copy"><strong>GitHub</strong>'
            '<span>github.com/Juna-111</span></span>'
            '<span class="teqo-contact-arrow" aria-hidden="true">↗</span></a>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        '<div class="teqo-project-note"><b>Research prototype, not a production '
        'security product.</b> '
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
        + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="teqo-foot">TEQOAI · Created by Hobby Yano · INSA Summer Camp</div>',
        unsafe_allow_html=True,
    )


def render_analyst_page(selected_page):
    if not os.path.exists(CLEAN_FILE) or not os.path.exists(RISK_FILE):
        st.title(PAGE_TITLES[selected_page])
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
            render_investigation(filtered_events, risk_events, graph, case_records)
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
    else:
        render_data_quality(clean_events, risk_events, report, provenance_issues)

    st.markdown(
        '<div class="teqo-foot">TEQOAI · Analyst-led security triage</div>',
        unsafe_allow_html=True,
    )


brand_mark, brand_title = st.columns([0.65, 5])
with brand_mark:
    if os.path.exists(LOGO_FILE):
        st.image(LOGO_FILE, width=68)
    else:
        st.markdown("◈")
with brand_title:
    st.markdown(
        '<div style="padding-top:8px">'
        '<div style="font-size:1.55rem;font-weight:800;letter-spacing:.04em">'
        'TEQO<span style="color:#38BDF8">AI</span></div>'
        '<div style="color:#94A3B8;font-size:.75rem">Explainable AI for authentication security</div>'
        '</div>',
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
results_page = st.Page(
    lambda: render_analyst_page("Results & Method"),
    title="Results & Method",
    icon=":material/analytics:",
    url_path="results",
)

current_page = st.navigation(
    [
        home_page,
        overview_page,
        investigation_page,
        paths_page,
        results_page,
    ],
    position="top",
)
current_page.run()
