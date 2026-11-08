"""Sidebar filters shared by the operational workspace pages."""

import streamlit as st

from src.case_store import CASE_STATUSES


def apply_filters(data, cases):
    """Apply sidebar review filters and return the current alert queue."""
    st.sidebar.markdown(
        '<div class="teqo-eyebrow">Refine alerts · optional</div>',
        unsafe_allow_html=True,
    )
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
        status_order = {
            "Escalated": 0,
            "New": 1,
            "Investigating": 2,
            "Resolved": 3,
            "Benign": 4,
        }
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