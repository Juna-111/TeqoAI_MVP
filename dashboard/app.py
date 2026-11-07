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
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
for project_path in (BASE_DIR, DASHBOARD_DIR):
    if project_path not in sys.path:
        sys.path.insert(0, project_path)

from components import (  # noqa: E402
    badge,
    callout,
    contact_list,
    contribution_grid,
    empty_state,
    environment_badge,
    factor_list,
    feature_grid,
    kv_card,
    kv_rows,
    legend,
    limitations,
    metric_grid,
    page_header,
    provenance_block,
    risk_distribution,
    section_header,
    status_badge,
)
from theme import load_styles  # noqa: E402
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

BRAND = "#36B9D6"
LOGO_FILE = os.path.join(BASE_DIR, "assets", "teqoai-logo.png")

st.set_page_config(
    page_title="TeqoAI | Security Operations",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="auto",
)

if os.path.exists(LOGO_FILE):
    st.logo(LOGO_FILE, size="large")

load_styles()


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


def format_event_value(value):
    """Render mixed numeric/categorical event values as display text."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"{float(value):g}"


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
                                "Event value": format_event_value(item["value"]),
                                "Shapley contribution": item["shap_value"],
                            }
                            for item in drivers
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Shapley contribution": st.column_config.NumberColumn(
                            format="%.6f"
                        ),
                    },
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
            column_config={
                "Interaction contribution": st.column_config.NumberColumn(
                    format="%.6f"
                ),
            },
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

    st.markdown(
        metric_grid(
            [
                {
                    "label": "Active alerts",
                    "value": f"{active:,}",
                    "hint": "Open or under investigation",
                    "tone": "info",
                },
                {
                    "label": "High-risk active",
                    "value": f"{high_active:,}",
                    "hint": "Priority review required",
                    "tone": "critical",
                },
                {
                    "label": "Escalated cases",
                    "value": f"{escalated:,}",
                    "hint": "Escalated by an analyst",
                    "tone": "warning",
                },
                {
                    "label": "Events analyzed",
                    "value": f"{len(data):,}",
                    "hint": "Events in the loaded sample",
                    "tone": "blue",
                },
            ]
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        callout(
            "Review sequence",
            "Start with the priority queue, select an alert, and open Alert Investigation. "
            "Evidence and case decisions stay with the analyst; TeqoAI does not take "
            "automated response actions.",
        ),
        unsafe_allow_html=True,
    )

    posture, snapshot = st.columns([1.2, 1], gap="medium")
    with posture:
        severity_counts = data["risk_level"].value_counts()
        st.markdown(
            risk_distribution(
                severity_counts,
                len(data),
                "Risk levels are transparent review priorities, not confirmed incidents "
                "or probabilities of compromise.",
            ),
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
            '<div class="teqo-panel"><div class="teqo-panel-head">'
            '<h3 class="teqo-panel-title">Current queue</h3>'
            '<span class="teqo-panel-meta">Sidebar filters</span></div>'
            + kv_rows(
                [
                    ("Matching filters", f"{len(queue):,}"),
                    ("High severity", f"{queue_high:,}"),
                    ("New cases", f"{queue_new:,}"),
                    ("Unassigned", f"{queue_unassigned:,}"),
                    ("Case stage", str(st.session_state.get("case_view") or "All")),
                ]
            )
            + '<p class="teqo-panel-note">Case stage and severity filters apply to '
            "the queue below. Filters are retained while you investigate an alert.</p></div>",
            unsafe_allow_html=True,
        )

    heading, export = st.columns([5, 1])
    with heading:
        st.markdown(
            section_header(
                "PRIORITY QUEUE",
                "Alerts to review first",
                "Sorted by case stage and severity. Select a row to open it in "
                "Alert Investigation.",
            ),
            unsafe_allow_html=True,
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
        case_export = case_export.apply(lambda column: column.map(csv_safe_text))
        export.download_button(
            "Export cases",
            data=case_export.to_csv(index=False),
            file_name="teqoai_cases.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption(
            "The case export includes analyst notes. Store it securely; text is "
            "escaped to reduce spreadsheet formula risks."
        )

    if queue.empty:
        st.markdown(
            empty_state(
                "No alerts match this view",
                "Change the case stage, severity, or search filters in the sidebar "
                "to restore the review queue.",
            ),
            unsafe_allow_html=True,
        )
        return

    display = queue.head(25).copy()
    columns = [
        "case_status", "risk_level", "risk_score", "time", "user", "device",
        "target_device", "case_owner", "risk_reasons",
    ]
    column_config = {
        "case_status": st.column_config.TextColumn("Case"),
        "risk_level": st.column_config.TextColumn("Severity"),
        "risk_score": st.column_config.ProgressColumn(
            "Risk score", min_value=0, max_value=100, format="%.1f"
        ),
        "time": st.column_config.NumberColumn("Elapsed seconds", format="%d"),
        "case_owner": st.column_config.TextColumn("Analyst"),
        "risk_reasons": st.column_config.TextColumn("Detection reason", width="large"),
    }
    selection = st.dataframe(
        display[columns],
        width="stretch",
        hide_index=True,
        height=420,
        key="overview-priority-table",
        on_select="rerun",
        selection_mode="single-row",
        column_config=column_config,
    )
    selected_rows = getattr(getattr(selection, "selection", None), "rows", [])
    if selected_rows:
        selected = display.iloc[selected_rows[0]]
        st.markdown(
            callout(
                "Selected alert",
                f"{selected['risk_level']} priority · {selected['user']} · "
                f"{selected['device']} → {selected['target_device']} · "
                f"score {float(selected['risk_score']):.1f}",
            ),
            unsafe_allow_html=True,
        )
        if st.button(
            "Open selected alert in Alert Investigation",
            type="primary",
            key="overview-open-selected",
        ):
            st.session_state["investigation-event"] = str(selected["_event_id"])
            st.switch_page(investigation_page)
    else:
        st.caption("Select a table row to open that alert in Alert Investigation.")

    remaining = queue.iloc[25:300]
    if not remaining.empty:
        with st.expander(f"Show {len(remaining):,} more filtered alerts"):
            st.dataframe(
                remaining[columns],
                width="stretch",
                hide_index=True,
                height=420,
                column_config=column_config,
            )
    if len(queue) > 300:
        st.caption(
            "The Security Overview shows the first 300 alerts. Use Alert Investigation "
            "to browse the full filtered queue. Event times are elapsed seconds, not "
            "local clock times."
        )


def render_investigation(queue, data, graph, cases, clean_events):
    if queue.empty:
        st.markdown(
            empty_state(
                "No alerts in this view",
                "Change the case stage, severity, or search filters in the sidebar "
                "to choose an alert for investigation.",
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<p class="teqo-toolbar-note">Choose an alert from the filtered queue. '
        "The selection stays in place while you review evidence and update the case.</p>",
        unsafe_allow_html=True,
    )
    event_ids = queue["_event_id"].tolist()[:500]
    # Keep the investigated alert selected even if saving a case moves it in
    # the priority queue. If a filter hides the alert entirely, reset the choice.
    pinned_event_id = st.session_state.get("investigation-event")
    if pinned_event_id is not None:
        queued_event_ids = set(queue["_event_id"])
        if pinned_event_id in queued_event_ids:
            if pinned_event_id not in event_ids:
                event_ids.insert(0, pinned_event_id)
        else:
            st.session_state.pop("investigation-event", None)
    selected_event_id = st.selectbox(
        "Active alert",
        options=event_ids,
        format_func=lambda event_id: event_label(
            queue.loc[queue["_event_id"] == event_id].iloc[0]
        ),
        key="investigation-event",
    )
    row = queue.loc[queue["_event_id"] == selected_event_id].iloc[0]
    event_id = str(row["_event_id"])
    case = cases.get(event_id, {})
    status_default = case.get("status", "New")
    severity_tone = {"High": "critical", "Medium": "warning", "Low": "success"}.get(
        str(row["risk_level"]), "info"
    )
    status_tone = {
        "New": "info",
        "Investigating": "warning",
        "Escalated": "critical",
        "Resolved": "success",
        "Benign": "neutral",
    }.get(status_default, "neutral")

    st.markdown(
        metric_grid(
            [
                {
                    "label": "Severity",
                    "value": str(row["risk_level"]),
                    "hint": "Risk level from the transparent risk layer",
                    "tone": severity_tone,
                    "size": "compact",
                },
                {
                    "label": "Priority score",
                    "value": f"{row['risk_score']:.1f} / 100",
                    "hint": "Relative review priority, not a probability",
                    "tone": "info",
                },
                {
                    "label": "Case status",
                    "value": status_default,
                    "hint": "Analyst-owned investigation state",
                    "tone": status_tone,
                    "size": "compact",
                },
                {
                    "label": "Assigned analyst",
                    "value": case.get("owner") or "Unassigned",
                    "hint": "Recorded in the local case store",
                    "tone": "blue",
                    "size": "compact",
                },
            ]
        ),
        unsafe_allow_html=True,
    )

    context, evidence = st.columns([1, 1], gap="medium")
    with context:
        st.markdown(
            '<div class="teqo-panel"><div class="teqo-panel-head">'
            '<h3 class="teqo-panel-title">Alert context</h3>'
            + status_badge(status_default)
            + "</div>"
            + kv_card(
                [
                    ("Account", f'<span class="teqo-mono">{html.escape(str(row["user"]))}</span>'),
                    ("Source device", f'<span class="teqo-mono">{html.escape(str(row["device"]))}</span>'),
                    ("Destination", f'<span class="teqo-mono">{html.escape(str(row["target_device"]))}</span>'),
                    ("Authentication", html.escape(f"{row['result']} · {row['action']}")),
                    ("Elapsed time", f"{int(row['time']):,} s"),
                    ("Risk level", html.escape(str(row["risk_level"]))),
                ]
            )
            + '<p class="teqo-panel-note">Event times are elapsed seconds in the loaded '
            "sample, not local clock times.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            callout(
                "Detection reason",
                str(row.get("risk_reasons", "No reason recorded")),
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            callout("Suggested next step", recommend_action(row)),
            unsafe_allow_html=True,
        )

    with evidence:
        factor_labels = {
            "factor_new_device": "First observed user-to-destination pair",
            "factor_sensitive_asset": "Destination is configured as sensitive",
            "factor_failed_login": "Authentication failed",
            "factor_burst_activity": "User activity exceeded the burst threshold",
        }
        factors = [
            (label, bool(row.get(key, False)), "")
            for key, label in factor_labels.items()
        ]
        triggered_count = sum(active for _, active, _ in factors)
        st.markdown(
            '<div class="teqo-panel"><div class="teqo-panel-head">'
            '<h3 class="teqo-panel-title">Detection evidence</h3>'
            + badge(
                f"{triggered_count} of 4 signals",
                "warning" if triggered_count else "neutral",
            )
            + "</div>"
            + factor_list(factors)
            + '<p class="teqo-panel-note">Rule factors are independently checkable; '
            "the anomaly ranking can also contribute when no factor fires. Signals "
            "require analyst validation and are not proof of compromise.</p></div>",
            unsafe_allow_html=True,
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

    timeline, twin = st.columns([1.2, 1], gap="medium")
    with timeline:
        st.markdown(
            section_header(
                "ACTIVITY CONTEXT",
                "Related authentication activity",
                "Matching account or source device within ±300 seconds.",
            ),
            unsafe_allow_html=True,
        )
        time_value = row["time"]
        related = data[
            ((data["user"] == row["user"]) | (data["device"] == row["device"]))
            & (data["time"].between(time_value - 300, time_value + 300))
        ].sort_values("time")
        if related.empty:
            st.markdown(
                empty_state(
                    "No related events",
                    "No matching activity appears in the five-minute context window.",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(
                related[["time", "user", "device", "target_device", "result", "action"]]
                .head(100),
                width="stretch",
                hide_index=True,
                height=250,
            )
        st.caption(
            "Related events use a ±300-second window and matching account or source device."
        )
    with twin:
        st.markdown(
            section_header(
                "OBSERVED PATHS",
                "Assets along observed routes",
                "Bounded links from the successful-authentication sample.",
            ),
            unsafe_allow_html=True,
        )
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
            st.markdown(
                empty_state(
                    "No onward paths",
                    "No observed outbound authentication paths were recorded for this "
                    "source device.",
                ),
                unsafe_allow_html=True,
            )
        st.caption(
            "Observed event links are not verified network reachability or permissions."
        )

    st.markdown(
        section_header(
            "CASE WORKFLOW",
            "Analyst action plan",
            "Record status, ownership, notes, and checklist progress. TeqoAI does not "
            "execute containment actions.",
        ),
        unsafe_allow_html=True,
    )
    saved_flag = f"case-saved-{event_id}"
    if st.session_state.pop(saved_flag, False):
        st.success("Case saved locally and added to the audit trail.")
    with st.container(key=f"teqo-case-form-{event_id}"):
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
            submitted = st.form_submit_button(
                "Save case and audit entry", type="primary"
            )
            if submitted:
                try:
                    save_case(DATASET, event_id, status, owner, notes, checklist)
                except (OSError, sqlite3.Error, ValueError):
                    st.error(
                        "This case could not be saved. Your latest changes were not "
                        "recorded."
                    )
                else:
                    st.session_state[saved_flag] = True
                    st.rerun()

    history = get_case_history(DATASET, event_id)
    if history:
        st.markdown(
            section_header(
                "AUDIT TRAIL",
                "Case activity",
                "Changes recorded in the local case store.",
            ),
            unsafe_allow_html=True,
        )
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
        'html,body{margin:0;padding:0;overflow:hidden;background:#07111F}'
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
        st.markdown(
            empty_state(
                "No connections to explore",
                "The loaded sample does not contain any successful authentication "
                "observations to link.",
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        callout(
            "What this view shows",
            "Directed links are built only from successful authentication observations. "
            "They describe the loaded sample, not verified network reachability, "
            "permissions, or a confirmed attack path.",
        ),
        unsafe_allow_html=True,
    )

    source_options = sorted(graph.nodes())
    default_source = (
        str(queue.iloc[0]["device"])
        if not queue.empty and str(queue.iloc[0]["device"]) in graph
        else source_options[0]
    )
    control, context = st.columns([1, 2], gap="medium")
    with control:
        selected_source = st.selectbox(
            "Starting device",
            options=source_options,
            index=source_options.index(default_source),
        )
    with context:
        st.markdown(
            '<p class="teqo-toolbar-note">Choose any observed device to highlight the '
            "bounded routes recorded from it. The priority source defaults to the "
            "highest-ranked alert in the current queue.</p>",
            unsafe_allow_html=True,
        )

    paths = get_attack_paths(graph, selected_source)
    highlighted_nodes = {selected_source}
    highlighted_edges = set()
    for path in paths[:5]:
        highlighted_nodes.update(path["path"])
        highlighted_edges.update(zip(path["path"][:-1], path["path"][1:]))

    st.markdown(
        metric_grid(
            [
                {
                    "label": "Devices and systems",
                    "value": f"{graph.number_of_nodes():,}",
                    "hint": "Observed in the loaded sample",
                    "tone": "info",
                },
                {
                    "label": "Recorded connections",
                    "value": f"{graph.number_of_edges():,}",
                    "hint": "Successful authentication observations",
                    "tone": "blue",
                },
                {
                    "label": "Highlighted routes",
                    "value": f"{len(paths):,}",
                    "hint": "Bounded onward paths from this device",
                    "tone": "warning" if paths else "default",
                },
            ]
        ),
        unsafe_allow_html=True,
    )

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
    st.markdown(
        legend(
            [
                ("Selected starting device", "info", False),
                ("Priority route", "warning", False),
                ("Other observed connection", "muted", False),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        section_header(
            "ONWARD CONNECTIONS",
            "Where activity was observed next",
            "Routes are ranked by asset criticality and limited to four hops.",
        ),
        unsafe_allow_html=True,
    )
    if paths:
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
        st.markdown(
            empty_state(
                "No onward connections",
                f"No onward connection was recorded from {selected_source}.",
            ),
            unsafe_allow_html=True,
        )
    st.caption(
        "Repeated successful observations increase edge weight. Failed attempts never "
        "create graph edges."
    )


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


def render_process_diagram(title, description, steps, kicker="PROCESS"):
    st.markdown(
        section_header(kicker, title, description),
        unsafe_allow_html=True,
    )
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
    record_count = len(data)
    user_count = int(data["user"].nunique()) if "user" in data.columns else 0
    asset_count = (
        int(pd.concat([data["device"], data["target_device"]]).nunique())
        if {"device", "target_device"}.issubset(data.columns)
        else 0
    )
    st.markdown(
        metric_grid(
            [
                {
                    "label": "Activity records",
                    "value": f"{record_count:,}",
                    "hint": "Events in the loaded sample",
                    "tone": "info",
                },
                {
                    "label": "Accounts",
                    "value": f"{user_count:,}",
                    "hint": "Distinct identities",
                    "tone": "blue",
                },
                {
                    "label": "Devices and systems",
                    "value": f"{asset_count:,}",
                    "hint": "Sources and destinations",
                    "tone": "default",
                },
            ]
        ),
        unsafe_allow_html=True,
    )

    if provenance_issues:
        st.markdown(
            callout(
                "Provenance mismatch",
                "Evaluation results are unavailable because the current activity and "
                "evaluation report do not match. Rerun the local demo pipeline before "
                "interpreting these results.",
                tone="warning",
            ),
            unsafe_allow_html=True,
        )
        return
    if report is None:
        st.markdown(
            callout(
                "Evaluation evidence unavailable",
                "Run the local demo pipeline to generate an evaluation report, then "
                "return to this page.",
            ),
            unsafe_allow_html=True,
        )
        return

    split = report.get("split", {})
    evaluation = report.get("evaluation", {})
    known_positives = int(split.get("test_known_positive_labels", 0))
    st.markdown(
        section_header(
            "CHRONOLOGICAL HOLDOUT",
            "Train on earlier events, evaluate on later events",
            f"{int(split.get('training_events', 0)):,} earlier events were used for "
            f"training and {int(split.get('test_events', 0)):,} later events were held "
            f"out. The held-out period contains {known_positives} known positive labels.",
        ),
        unsafe_allow_html=True,
    )

    model_metrics = evaluation.get("models", {})
    ranking_rows = [
        {
            "Approach": label,
            "Status": model_metrics.get(model_key, {})
            .get("status", "not available")
            .replace("_", " "),
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
        st.markdown(
            section_header(
                "RANKING",
                "Ranking metrics",
                "Average precision and ROC-AUC compare raw ranking scores with known "
                "labels in the held-out period.",
            ),
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(ranking_rows), width="stretch", hide_index=True)

    for budget_number in (30, 100):
        findings = _format_model_results(
            model_metrics, known_positives, budget_number
        )
        if findings:
            st.markdown(
                section_header(
                    "REVIEW BUDGET",
                    f"Top-{budget_number} review budget",
                    "Agreement with known labels when analysts can review only the "
                    "highest-ranked events.",
                ),
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(findings), width="stretch", hide_index=True)
    st.caption(
        "False-positive rates treat events without an exact label match as negative "
        "for calculation; some may be malicious but unlabeled."
    )

    if known_positives < 20:
        st.markdown(
            callout(
                "Small label sample",
                f"The holdout contains only {known_positives} known positive labels. "
                "These results are illustrative, not a stable measure of detection "
                "performance. Events without a ground-truth match are not confirmed benign.",
                tone="warning",
            ),
            unsafe_allow_html=True,
        )
    elif report.get("dataset_profile") == "synthetic":
        st.markdown(
            callout(
                "Synthetic demonstration",
                "This is generated demonstration activity; its metrics do not establish "
                "detection quality on LANL data or real organization networks.",
                tone="warning",
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        callout(
            "Score semantics",
            "The report evaluates raw ranking scores against known labels, not the "
            "rule-adjusted risk score. A high score is a review priority—not a "
            "probability of compromise.",
        ),
        unsafe_allow_html=True,
    )

    walk_forward = report.get("walk_forward_validation", {})
    folds = walk_forward.get("folds", []) if isinstance(walk_forward, dict) else []
    if folds:
        st.markdown(
            section_header(
                "VALIDATION",
                "Earlier chronological validation folds",
                "Expanding-window folds compare the same approaches without training "
                "on future events.",
            ),
            unsafe_allow_html=True,
        )
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
                "Each fold uses the same chronological windows across the baseline and "
                "available models. Folds without known positives are not evaluable for "
                "detection quality."
            )

    if "risk_level" in data.columns and not data.empty:
        st.markdown(
            section_header(
                "DISTRIBUTION",
                "Activity overview",
                "How the loaded events fall across the priority review bands.",
            ),
            unsafe_allow_html=True,
        )
        distribution = data["risk_level"].value_counts().reindex(
            ["High", "Medium", "Low"], fill_value=0
        )
        distribution.index = ["Priority review", "Review", "Routine"]
        st.bar_chart(distribution, color=BRAND, height=240)
    else:
        st.caption(
            "The activity distribution will appear after the demo pipeline has produced "
            "risk results."
        )


def render_model_intelligence():
    st.markdown(
        callout(
            "What a score means",
            "An anomaly score is not a probability that an account is compromised. "
            "Novelty can be legitimate, and suspicious activity can look familiar. "
            "Every ranking is a review priority for a human analyst.",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        section_header(
            "EVENT SIGNALS",
            "What the detector reads",
            "The current feature set is built from authentication outcomes, novelty, "
            "prior-hour activity, and encoded action categories.",
        )
        + feature_grid(
            [
                {
                    "icon": "alert",
                    "title": "Authentication outcome",
                    "body": "Success or failure is a model feature and risk factor. "
                    "Failures never create observed device-graph edges.",
                },
                {
                    "icon": "detect",
                    "title": "First-seen destination",
                    "body": "Whether this account-to-destination pairing appears for the "
                    "first time in the available chronology.",
                },
                {
                    "icon": "clock",
                    "title": "Prior-hour activity",
                    "body": "The account's rolling authentication count over the preceding "
                    "hour. Events at the same timestamp do not inflate one another.",
                },
                {
                    "icon": "layers",
                    "title": "Authentication method",
                    "body": "One-hot encoded categories derived from the event action. "
                    "The model uses elapsed time, not invented calendar-hour features.",
                },
            ]
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        section_header(
            "RANKING APPROACHES",
            "Three ways to prioritize",
            "The prototype keeps a transparent baseline alongside the unsupervised "
            "models so their value can be tested.",
        ),
        unsafe_allow_html=True,
    )
    model_cards = [
        (
            "Baseline",
            "accent",
            "First-seen destination",
            "A simple behavior baseline that prioritizes first-time account-to-destination "
            "pairs. It exists to make more complex model behavior testable.",
        ),
        (
            "Primary",
            "info",
            "Isolation Forest",
            "The primary unsupervised model. It is fitted on training-period event "
            "features and ranks events by isolation-based anomaly score.",
        ),
        (
            "Optional",
            "neutral",
            "Autoencoder",
            "A small reconstruction model trained in bounded batches. Higher "
            "reconstruction error contributes to the ranking; scores are not calibrated "
            "probabilities.",
        ),
    ]
    columns = st.columns(3, gap="medium")
    for column, (tag, tone, title, body) in zip(columns, model_cards):
        with column:
            st.markdown(
                '<div class="teqo-contribution"><div class="teqo-badge-row">'
                + badge(tag, tone)
                + f"</div><strong>{html.escape(title)}</strong>"
                + f"<p>{html.escape(body)}</p></div>",
                unsafe_allow_html=True,
            )

    render_process_diagram(
        "From event to human review",
        "Each stage adds context; none of them makes an automated access or "
        "containment decision.",
        [
            ("Authentication event", "Common fields and observed outcome."),
            ("Event features", "Novelty, prior-hour volume, method and outcome."),
            ("Model ranking", "Isolation Forest and optional reconstruction error."),
            ("Risk context", "Percentile plus transparent, named factors."),
            ("Analyst review", "Event context, observed paths and case decision."),
        ],
        kicker="REVIEW PIPELINE",
    )

    st.markdown(
        callout(
            "Explainability",
            "The analyst view exposes named risk factors and exact interventional "
            "Shapley contributions for the Isolation Forest anomaly score. Attributions "
            "describe this model output relative to a training background; they are not "
            "causal effects or probabilities.",
        ),
        unsafe_allow_html=True,
    )
    st.page_link(
        results_page,
        label="Review measured model evaluation",
        icon=":material/analytics:",
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
    st.markdown(
        '<div class="teqo-panel"><div class="teqo-section-kicker">RESEARCH QUESTION</div>'
        '<p class="teqo-question">Can unsupervised authentication-anomaly rankings help '
        "prioritize known red-team events in a later chronological period, compared with "
        "a first-seen-destination baseline?</p></div>",
        unsafe_allow_html=True,
    )

    render_process_diagram(
        "Train and evaluate without time leakage",
        "Labels are evaluation evidence only; they are not inputs to unsupervised model "
        "fitting.",
        [
            ("Earlier events", "Fit feature transforms and unsupervised model(s)."),
            ("Later holdout", "Keep a chronological period separate from training."),
            ("Rank events", "Score holdout events using models fitted on earlier data."),
            ("Compare", "Measure agreement with known labels at fixed review budgets."),
        ],
        kicker="EVALUATION DESIGN",
    )

    if report is None or provenance_issues:
        render_data_quality(pd.DataFrame(), data, None, provenance_issues)
    else:
        split = report.get("split", {})
        input_metadata = report.get("input", {})
        input_hash = (
            str(input_metadata.get("sha256", "unavailable"))
            if isinstance(input_metadata, dict)
            else "unavailable"
        )
        reproducibility = report.get("reproducibility", {})
        packages = (
            reproducibility.get("packages", {})
            if isinstance(reproducibility, dict)
            else {}
        )
        package_text = ", ".join(
            f"{name} {version}" for name, version in packages.items()
        ) or "unavailable"
        walk_forward = report.get("walk_forward_validation", {})
        completed_folds = (
            f"{int(walk_forward.get('completed_folds', 0))} / "
            f"{int(walk_forward.get('requested_folds', 0))}"
            if isinstance(walk_forward, dict)
            else "unavailable"
        )

        st.markdown(
            metric_grid(
                [
                    {
                        "label": "Dataset profile",
                        "value": str(report.get("dataset_profile", "unknown")),
                        "hint": "Input profile used by the pipeline",
                        "tone": "info",
                        "size": "compact",
                    },
                    {
                        "label": "Training events",
                        "value": f"{int(split.get('training_events', 0)):,}",
                        "hint": "Earlier chronological portion",
                        "tone": "blue",
                    },
                    {
                        "label": "Holdout events",
                        "value": f"{int(split.get('test_events', 0)):,}",
                        "hint": "Later held-out period",
                        "tone": "default",
                    },
                    {
                        "label": "Known positive labels",
                        "value": f"{int(split.get('test_known_positive_labels', 0)):,}",
                        "hint": "Exact ground-truth matches",
                        "tone": "warning",
                    },
                    {
                        "label": "Walk-forward folds",
                        "value": completed_folds,
                        "hint": "Completed / requested",
                        "tone": "default",
                        "size": "compact",
                    },
                ]
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            callout(
                "Provenance verified",
                "The evaluation report matches the loaded activity sample and the current "
                "pipeline schema.",
                tone="success",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            provenance_block(
                [
                    ("Run ID", report.get("run_id", "unavailable")),
                    ("Input SHA-256", input_hash[:20] + ("…" if len(input_hash) > 20 else "")),
                    ("Schema version", report.get("schema_version", "unavailable")),
                    ("Packages", package_text),
                ]
            ),
            unsafe_allow_html=True,
        )
        render_data_quality(pd.DataFrame(), data, report, provenance_issues)

    st.markdown(
        section_header(
            "INTERPRETATION",
            "Limitations and responsible interpretation",
            "These constraints apply to every metric and ranking on this page.",
        )
        + limitations(
            [
                "Ground truth is exact-match red-team labeling; an event without a label "
                "match may still be malicious.",
                "A small number of known positives cannot support stable performance claims.",
                "Synthetic demonstrations validate the pipeline and evaluation code, not "
                "real-world detection quality.",
                "Risk scores are prioritization values, not calibrated probabilities; "
                "analysts remain responsible for case decisions.",
                "The current ingestion profiles are synthetic and LANL; arbitrary "
                "organization exports need an approved, validated adapter.",
            ]
        ),
        unsafe_allow_html=True,
    )


def render_architecture():
    st.markdown(
        page_header(
            "SYSTEM · ARCHITECTURE",
            "System architecture",
            "How the local research prototype moves from authentication activity to "
            "analyst-owned review.",
            DATASET,
            "Local implementation",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        feature_grid(
            [
                {
                    "icon": "database",
                    "title": "Ingestion and preparation",
                    "body": "Supported synthetic and LANL profiles map to one event schema; "
                    "compressed logs are streamed with a bounded row count.",
                },
                {
                    "icon": "brain",
                    "title": "Detection and risk",
                    "body": "Unsupervised ranking, a first-seen baseline, and transparent "
                    "named risk factors produce a review priority.",
                },
                {
                    "icon": "investigate",
                    "title": "Analyst workspace",
                    "body": "Evidence, exact Shapley explanations, observed paths, and "
                    "local case records support a documented human decision.",
                },
            ]
        ),
        unsafe_allow_html=True,
    )

    render_process_diagram(
        "Implemented data and review flow",
        "Artifacts and processing follow the current pipeline modules and dashboard "
        "workflow.",
        [
            ("Profile input", "Synthetic demo or an authorized LANL sample."),
            ("Data preparation", "Stream, normalize and label known matches."),
            ("Chronological split", "Earlier training period; later held-out period."),
            ("Model and risk", "Unsupervised ranking with transparent risk factors."),
            ("Dashboard", "Filter, investigate, explore observed paths and manage cases."),
        ],
        kicker="DATA FLOW",
    )
    render_process_diagram(
        "Observed-path data flow",
        "Only successful authentication events create device-to-device observations.",
        [
            ("Clean events", "Read device, destination and outcome."),
            ("Successful only", "Failed attempts are excluded from graph edges."),
            ("Directed graph", "Repeated successful observations increase edge weight."),
            ("Bounded path query", "Shortest paths are limited to four hops and ranked by "
             "asset criticality."),
            ("Analyst context", "Observed sample paths—not proof of reachability or "
             "authorization."),
        ],
        kicker="OBSERVED PATHS",
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
        kicker="CASE WORKFLOW",
    )

    st.markdown(
        section_header(
            "ENGINEERING",
            "Research and engineering contributions",
            "These are implementation contributions of the prototype, not claims of "
            "validated operational effectiveness.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        contribution_grid(RESEARCH_CONTRIBUTIONS),
        unsafe_allow_html=True,
    )

    st.markdown(
        callout(
            "Local prototype",
            "Case notes are stored unencrypted on this computer. Do not enter sensitive "
            "production information. The project is not a production security product "
            "and does not provide live monitoring or automated response.",
            tone="warning",
        ),
        unsafe_allow_html=True,
    )


PAGE_TITLES = {
    "Overview": "Security overview",
    "Investigate": "Alert investigation",
    "Observed Paths": "Cyber digital twin",
    "AI Intelligence": "AI intelligence",
    "Research & Evaluation": "Model evaluation",
}

PAGE_DESCRIPTIONS = {
    "Overview": "Alert volume, severity mix, and the review queue for the loaded authentication sample.",
    "Investigate": "Inspect evidence, explanations, and record an analyst-owned case decision.",
    "Observed Paths": "Explore successful authentication relationships and where activity was observed next.",
    "AI Intelligence": "How the implemented detectors work, what signals they use, and how to read a score.",
    "Research & Evaluation": "Chronological validation, measured ranking evidence, and responsible limitations.",
}

PAGE_KICKERS = {
    "Overview": "WORKSPACE · SECURITY OVERVIEW",
    "Investigate": "WORKSPACE · ALERT INVESTIGATION",
    "Observed Paths": "WORKSPACE · CYBER DIGITAL TWIN",
    "AI Intelligence": "INTELLIGENCE · AI INTELLIGENCE",
    "Research & Evaluation": "INTELLIGENCE · MODEL EVALUATION",
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

    with st.container(key="teqo-demo-setup"):
        st.markdown(
            '<div class="teqo-demo-panel">'
            '<div class="teqo-section-kicker">DEMO DATA</div>'
            "<h3>Run the reproducible evaluation</h3>"
            "<p>Generate a small deterministic synthetic authentication dataset with "
            "known red-team labels, then run the same chronological pipeline used by "
            "the project. Nothing needs to be uploaded.</p></div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "This is a software demonstration, not evidence of performance on real "
            "organizations. Generated files and case records on Community Cloud are "
            "temporary."
        )
        if st.button(
            "Generate demo data and run evaluation",
            key=button_key,
            type="primary",
        ):
            try:
                with st.status(
                    "Preparing the synthetic evaluation...", expanded=True
                ) as status:
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
                "Demo data and evaluation results are ready. Open Model Evaluation to "
                "inspect the metrics, then use Security Overview and Alert Investigation "
                "to explore events."
            )
            st.rerun()


def render_homepage():
    with open(LOGO_FILE, "rb") as logo_file:
        logo_data = base64.b64encode(logo_file.read()).decode("ascii")
    logo_markup = f'<img src="data:image/png;base64,{logo_data}" alt="TeqoAI emblem">'
    dataset_label = (
        "Generated demonstration activity"
        if DATASET == "synthetic"
        else "Authorized LANL sample"
    )
    creator_contacts = contact_list(
        [
            {
                "href": "mailto:hobbieyano@gmail.com",
                "icon": "mail",
                "label": "hobbieyano@gmail.com",
            },
            {
                "href": "https://t.me/hobbie111",
                "icon": "telegram",
                "label": "Telegram · @hobbie111",
                "external": True,
            },
            {
                "href": "https://github.com/Juna-111",
                "icon": "github",
                "label": "GitHub · Juna-111",
                "external": True,
            },
            {
                "href": "tel:+251962703558",
                "icon": "phone",
                "label": "+251 96 270 3558",
            },
        ]
    )

    with st.container(key="teqo-home"):
        st.markdown(
            '<section class="teqo-home-hero">'
            '<div>'
            '<div class="teqo-home-meta">'
            '<span class="teqo-eyebrow">INSA SUMMER CAMP · GRADUATION PROJECT</span>'
            + environment_badge(DATASET)
            + "</div>"
            '<h1 class="teqo-home-title">Explainable <span>authentication triage</span></h1>'
            '<p class="teqo-home-lead">TeqoAI uses explainable AI to identify unusual '
            "authentication behaviour, prioritize suspicious events, and support "
            "evidence-based security investigations.</p>"
            '<p class="teqo-home-sub">A local research prototype with the analyst in '
            "control: every ranking is a review priority, never an automated decision.</p>"
            '<div class="teqo-home-tags">'
            + badge("Explainable AI", "info")
            + badge("Analyst-led", "neutral")
            + badge("Local prototype", "neutral")
            + "</div></div>"
            f'<div class="teqo-home-emblem"><div class="teqo-emblem">{logo_markup}</div></div>'
            "</section>",
            unsafe_allow_html=True,
        )

        with st.container(horizontal=True, key="teqo-home-ctas", gap="small"):
            st.page_link(
                overview_page,
                label="Open security workspace",
                icon=":material/shield:",
            )
            st.page_link(
                intelligence_page,
                label="Explore AI intelligence",
                icon=":material/psychology:",
            )

        render_demo_setup("teqo-demo-build-home")

        st.markdown(
            '<section class="teqo-home-section">'
            + section_header(
                "CAPABILITIES",
                "Detect. Investigate. Connect.",
                "Three operational surfaces built on the same authentication evidence.",
            )
            + feature_grid(
                [
                    {
                        "icon": "detect",
                        "title": "Detect",
                        "body": "Unsupervised ranking surfaces unusual authentication "
                        "events without training on malicious-activity labels.",
                    },
                    {
                        "icon": "investigate",
                        "title": "Investigate",
                        "body": "Alert context, transparent risk factors, and exact feature "
                        "attributions support a documented case decision.",
                    },
                    {
                        "icon": "connect",
                        "title": "Connect",
                        "body": "Observed device-to-device authentication links provide "
                        "investigation context—not proof of reachability.",
                    },
                ]
            )
            + "</section>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<section class="teqo-home-section">'
            + section_header(
                "APPROACH",
                "How the prototype works",
                "The pipeline keeps labels out of model fitting and leaves response "
                "decisions with the analyst.",
            )
            + '<ol class="teqo-process">'
            "<li><strong>Read activity</strong><span>Normalize authentication events from "
            "the loaded sample.</span></li>"
            "<li><strong>Prioritize</strong><span>Rank unusual behavior and add named, "
            "checkable risk factors.</span></li>"
            "<li><strong>Investigate</strong><span>Review context, explanations, observed "
            "paths, and record a case decision.</span></li>"
            "<li><strong>Evaluate</strong><span>Compare rankings with known labels at "
            "fixed review budgets.</span></li>"
            "</ol>"
            '<div class="teqo-home-facts">'
            f"<div><span>Dataset</span> · {html.escape(dataset_label)}</div>"
            "<div><span>Approach</span> · Unsupervised anomaly ranking</div>"
            "<div><span>Decision-maker</span> · Human analyst</div>"
            "</div></section>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="teqo-about-strip">'
            "<div>"
            '<div class="teqo-section-kicker">ABOUT THE PROJECT</div>'
            "<h2>Built by Hobby Yano</h2>"
            "<p>TeqoAI was developed as an INSA Summer Camp graduation project exploring "
            "explainable AI for authentication security. The prototype combines "
            "unsupervised anomaly ranking, transparent risk factors, analyst-owned case "
            "records, and evidence-bounded relationship views.</p>"
            + creator_contacts
            + "</div>"
            '<div class="teqo-about-note">'
            "<strong>Research prototype, not a production security product.</strong>"
            "<p>"
            + (
                "This build uses generated demonstration activity; its results do not "
                "establish accuracy on real networks. TeqoAI supports analyst review and "
                "does not perform automated response."
                if DATASET == "synthetic"
                else "Results depend on the selected dataset and available confirmed "
                "labels. Scores are review priorities, not probabilities, and TeqoAI "
                "does not perform automated response."
            )
            + "</p></div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="teqo-home-footer">'
            "<span>TEQOAI · Explainable authentication triage</span>"
            "<span>INSA Summer Camp · Created by Hobby Yano</span>"
            "</div>",
            unsafe_allow_html=True,
        )


def render_analyst_page(selected_page):
    if not os.path.exists(CLEAN_FILE) or not os.path.exists(RISK_FILE):
        st.markdown(
            page_header(
                PAGE_KICKERS[selected_page],
                PAGE_TITLES[selected_page],
                PAGE_DESCRIPTIONS[selected_page],
                DATASET,
            ),
            unsafe_allow_html=True,
        )
        render_demo_setup(f"teqo-demo-build-{selected_page.lower().replace(' ', '-')}")
        if selected_page == "AI Intelligence":
            render_model_intelligence()
        elif selected_page == "Research & Evaluation":
            render_research_and_evaluation(None, pd.DataFrame(), [])
        else:
            st.markdown(
                empty_state(
                    "No processed events yet",
                    "Run the local demo pipeline from the About page or the panel above, "
                    "then return to this page.",
                ),
                unsafe_allow_html=True,
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

    st.markdown(
        page_header(
            PAGE_KICKERS[selected_page],
            PAGE_TITLES[selected_page],
            PAGE_DESCRIPTIONS[selected_page],
            DATASET,
        ),
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
        '<div class="teqo-foot">TEQOAI · Local research prototype · '
        "Analyst review only · No automated response</div>",
        unsafe_allow_html=True,
    )


home_page = st.Page(
    render_homepage,
    title="About TeqoAI",
    icon=":material/info:",
    default=True,
)
overview_page = st.Page(
    lambda: render_analyst_page("Overview"),
    title="Security Overview",
    icon=":material/space_dashboard:",
    url_path="overview",
)
investigation_page = st.Page(
    lambda: render_analyst_page("Investigate"),
    title="Alert Investigation",
    icon=":material/search:",
    url_path="investigate",
)
paths_page = st.Page(
    lambda: render_analyst_page("Observed Paths"),
    title="Cyber Digital Twin",
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
    title="Model Evaluation",
    icon=":material/analytics:",
    url_path="research",
)
architecture_page = st.Page(
    render_architecture,
    title="Architecture",
    icon=":material/schema:",
    url_path="architecture",
)

current_page = st.navigation(
    {
        "Workspace": [overview_page, investigation_page, paths_page],
        "Intelligence": [intelligence_page, results_page],
        "System": [architecture_page, home_page],
    },
    position="sidebar",
)
current_page.run()
with st.sidebar:
    st.markdown('<div class="teqo-sidebar-rule"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="teqo-sidebar-block">'
        '<div class="teqo-brand">TEQO<span>AI</span></div>'
        '<div class="teqo-sidebar-sub">Explainable authentication triage</div>'
        f'<div class="teqo-sidebar-status">{environment_badge(DATASET)}</div>'
        '<div class="teqo-sidebar-note">Local research prototype · Analyst review only · '
        "No automated response</div></div>",
        unsafe_allow_html=True,
    )
