"""Security Overview page."""

import altair as alt
import pandas as pd
import streamlit as st

from components import (
    callout,
    empty_state,
    kv_rows,
    metric_grid,
    risk_distribution,
    section_header,
)
from src.case_store import csv_safe_text, export_cases
from src.dataset_paths import DATASET

SUCCESS_COLOR = "#32B783"
FAILURE_COLOR = "#F04452"
MEDIUM_COLOR = "#F2A93B"
LOW_COLOR = "#32B783"


def _timeline_bins(data, bin_count=28):
    times = pd.to_numeric(data.get("time"), errors="coerce")
    if times is None:
        return None, None
    valid = data.assign(_elapsed=times).dropna(subset=["_elapsed"])
    if valid.empty or valid["_elapsed"].nunique() < 2:
        return None, None
    start = float(valid["_elapsed"].min())
    end = float(valid["_elapsed"].max())
    if end <= start:
        return None, None
    span = end - start
    valid["_bin"] = (
        ((valid["_elapsed"] - start) / span * bin_count)
        .clip(upper=bin_count - 1)
        .astype(int)
    )
    result = (
        valid.get("result", pd.Series("", index=valid.index))
        .astype("string")
        .str.strip()
        .str.casefold()
    )
    valid = valid.assign(
        _success=result.eq("success").fillna(False),
        _failure=result.eq("failure").fillna(False),
        _high=valid.get("risk_level", pd.Series("", index=valid.index)).eq("High"),
        _medium=valid.get("risk_level", pd.Series("", index=valid.index)).eq("Medium"),
        _low=valid.get("risk_level", pd.Series("", index=valid.index)).eq("Low"),
    )
    grouped = valid.groupby("_bin")
    rows = []
    for index in range(bin_count):
        subset = grouped.get_group(index) if index in grouped.groups else None
        rows.append(
            {
                "start": start + span * index / bin_count,
                "end": start + span * (index + 1) / bin_count,
                "Events": 0 if subset is None else int(len(subset)),
                "Successful": 0 if subset is None else int(subset["_success"].sum()),
                "Failed": 0 if subset is None else int(subset["_failure"].sum()),
                "High": 0 if subset is None else int(subset["_high"].sum()),
                "Medium": 0 if subset is None else int(subset["_medium"].sum()),
                "Low": 0 if subset is None else int(subset["_low"].sum()),
            }
        )
    return pd.DataFrame(rows), valid


def _extract_time_window(state):
    selection = getattr(state, "selection", None)
    if selection is None and isinstance(state, dict):
        selection = state.get("selection")
    if not isinstance(selection, dict):
        return None
    window = selection.get("window", selection)
    if not isinstance(window, dict):
        return None
    if {"start", "end"}.issubset(window):
        return float(window["start"]), float(window["end"])
    for value in window.values():
        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(isinstance(item, (int, float)) for item in value)
        ):
            return float(min(value)), float(max(value))
    return None


def _render_activity_timeline(data):
    st.markdown(
        section_header(
            "AUTHENTICATION TIMELINE",
            "Activity across the loaded period",
            "Event timestamps are elapsed seconds relative to the dataset, not local "
            "clock times. Drag across the chart to inspect a time window.",
        ),
        unsafe_allow_html=True,
    )
    bins, valid = _timeline_bins(data)
    if bins is None:
        st.markdown(
            empty_state(
                "Timeline unavailable",
                "The loaded sample does not contain enough distinct elapsed-second "
                "values to aggregate authentication activity over time.",
            ),
            unsafe_allow_html=True,
        )
        return
    if bins["Events"].sum() == 0:
        st.markdown(
            empty_state(
                "No activity to plot",
                "No loaded events have usable elapsed-second timestamps.",
            ),
            unsafe_allow_html=True,
        )
        return

    mode = st.segmented_control(
        "Timeline measure",
        options=["Event volume", "Alert volume"],
        default="Event volume",
        key="overview-timeline-mode",
        label_visibility="collapsed",
    )
    if mode == "Alert volume":
        value_columns = ["High", "Medium", "Low"]
        colors = [FAILURE_COLOR, MEDIUM_COLOR, LOW_COLOR]
    else:
        value_columns = ["Successful", "Failed"]
        colors = [SUCCESS_COLOR, FAILURE_COLOR]
    long = bins.melt(
        id_vars=["start", "end"],
        value_vars=value_columns,
        var_name="Measure",
        value_name="Count",
    )
    brush = alt.selection_interval(encodings=["x"], name="window")
    chart = (
        alt.Chart(long)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X(
                "start:Q",
                title="Elapsed seconds (dataset-relative)",
                axis=alt.Axis(
                    labelExpr="format(datum.value, ',.0f')",
                    labelAngle=0,
                    tickCount=8,
                ),
            ),
            x2=alt.X2("end:Q"),
            y=alt.Y(
                "Count:Q",
                title="Events per interval",
                axis=alt.Axis(labelExpr="format(datum.value, ',.0f')"),
            ),
            color=alt.Color(
                "Measure:N",
                scale=alt.Scale(domain=value_columns, range=colors),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
            tooltip=[
                alt.Tooltip("start:Q", title="Window start (s)", format=","),
                alt.Tooltip("end:Q", title="Window end (s)", format=","),
                alt.Tooltip("Measure:N", title="Measure"),
                alt.Tooltip("Count:Q", title="Events", format=","),
            ],
        )
        .add_params(brush)
        .properties(height=250)
    )
    state = st.altair_chart(
        chart,
        width="stretch",
        key="overview-timeline-chart",
        on_select="rerun",
        selection_mode="window",
    )
    window = _extract_time_window(state)
    if window is not None:
        window_start, window_end = window
        selected = valid.loc[
            valid["_elapsed"].between(window_start, window_end, inclusive="both")
        ]
        if selected.empty:
            st.caption("The selected window contains no loaded events.")
        else:
            st.markdown(
                '<div class="teqo-panel-note">'
                f"Selected window: {len(selected):,} events · "
                f"{int(selected['_success'].sum()):,} successful · "
                f"{int(selected['_failure'].sum()):,} failed · "
                f"{int(selected['_high'].sum()):,} high-risk · "
                f"{int(selected['_medium'].sum()):,} medium-risk. "
                f"Elapsed seconds {window_start:,.0f}–{window_end:,.0f}."
                "</div>",
                unsafe_allow_html=True,
            )
            st.button(
                "Clear timeline selection",
                key="overview-timeline-clear",
                on_click=lambda: st.session_state.pop(
                    "overview-timeline-chart", None
                ),
            )
    st.caption(
        "Event volume separates successful and failed authentication results. Alert "
        "volume uses the rule-based risk levels. Both use the same elapsed-second axis."
    )


def render_overview(data, queue, cases, investigation_page):
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
                    event_id
                    for event_id, case in cases.items()
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

    _render_activity_timeline(data)

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
            if not queue.empty
            else 0
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
                    "_event_id",
                    "time",
                    "user",
                    "device",
                    "target_device",
                    "risk_level",
                    "risk_score",
                    "risk_reasons",
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
        "case_status",
        "risk_level",
        "risk_score",
        "time",
        "user",
        "device",
        "target_device",
        "case_owner",
        "risk_reasons",
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