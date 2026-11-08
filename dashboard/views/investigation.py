"""Alert Investigation page and exact Shapley presentation."""

import html
import json
import sqlite3

import altair as alt
import pandas as pd
import streamlit as st

from components import (
    badge,
    callout,
    empty_state,
    factor_list,
    kv_card,
    metric_grid,
    section_header,
    status_badge,
)
from data_access import load_analysis_resources
from src.case_store import CASE_STATUSES, CHECKLIST_ITEMS, get_case_history, save_case
from src.dataset_paths import DATASET
from src.digital_twin import explain_event, get_attack_paths

POSITIVE_COLOR = "#FFB454"
NEGATIVE_COLOR = "#38BDF8"
ZERO_COLOR = "#5B7290"


def format_event_value(value):
    """Render mixed numeric/categorical event values as display text."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"{float(value):g}"


def format_background(item):
    """Describe the sampled training reference for one grouped feature."""
    background = item.get("background") or {}
    if background.get("kind") == "category":
        return (
            f"{background['value']} · "
            f"{float(background['share']):.0%} of sampled background"
        )
    if "value" in background:
        return f"mean {float(background['value']):g}"
    return "Not available"


def format_event_record(row):
    """Turn the alert row into a readable field/value table."""
    records = [{"Field": "Event ID", "Value": str(row.get("_event_id", ""))}]
    for field, value in row.items():
        if field == "_event_id":
            continue
        if isinstance(value, float) and pd.isna(value):
            text = ""
        elif isinstance(value, str):
            text = value
        else:
            text = format_event_value(value)
        records.append({"Field": field.replace("_", " ").title(), "Value": text})
    return pd.DataFrame(records)


def driver_frame(items):
    return pd.DataFrame(
        [
            {
                "Feature": item["label"],
                "Contribution": float(item["shap_value"]),
                "Event value": format_event_value(item["value"]),
                "Training reference": format_background(item),
            }
            for item in items
        ]
    )


def driver_chart(frame, direction, limit):
    """Build one signed horizontal bar chart with a shared zero reference."""
    increasing = direction == "positive"
    color = POSITIVE_COLOR if increasing else NEGATIVE_COLOR
    sort_order = "-x" if increasing else "x"
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X(
                "Contribution:Q",
                title="Shapley contribution (anomaly-score units)",
                scale=alt.Scale(domain=[-limit, limit]),
                axis=alt.Axis(format="+.3f", tickCount=5, labelAngle=0),
            ),
            y=alt.Y(
                "Feature:N",
                sort=sort_order,
                title=None,
                axis=alt.Axis(labelLimit=210, labelFontSize=11),
            ),
            color=alt.value(color),
            tooltip=[
                alt.Tooltip("Feature:N"),
                alt.Tooltip("Event value:N"),
                alt.Tooltip("Training reference:N"),
                alt.Tooltip("Contribution:Q", format="+.6f"),
            ],
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(
            align="left" if increasing else "right",
            dx=4 if increasing else -4,
            fontSize=10,
            color=color,
        )
        .encode(
            x=alt.X(
                "Contribution:Q",
                scale=alt.Scale(domain=[-limit, limit]),
            ),
            y=alt.Y("Feature:N", sort=sort_order),
            text=alt.Text("Contribution:Q", format="+.3f"),
        )
    )
    zero = (
        alt.Chart(pd.DataFrame({"Contribution": [0.0]}))
        .mark_rule(color=ZERO_COLOR, strokeDash=[3, 3])
        .encode(x="Contribution:Q")
    )
    return (bars + labels + zero).properties(height=max(170, 36 * len(frame) + 40))


def render_shap_explanation(explanation):
    st.markdown("#### Exact Shapley decomposition")
    st.caption(
        "Output: Isolation Forest anomaly score (negative decision_function); "
        "higher scores indicate more anomalous events. Every bar is a signed "
        "contribution to this event's score relative to the training background."
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

    positive = explanation["positive_drivers"][:6]
    negative = explanation["negative_drivers"][:6]
    contributions = [
        abs(float(item["shap_value"]))
        for item in explanation["positive_drivers"] + explanation["negative_drivers"]
    ]
    limit = max(contributions, default=0.0) * 1.2 or 1.0
    positive_column, negative_column = st.columns(2, gap="medium")
    with positive_column:
        st.markdown("**Positive drivers · increase the anomaly score**")
        if positive:
            st.altair_chart(
                driver_chart(driver_frame(positive), "positive", limit),
                width="stretch",
            )
        else:
            st.caption("No feature increased this event's score.")
    with negative_column:
        st.markdown("**Negative drivers · decrease the anomaly score**")
        if negative:
            st.altair_chart(
                driver_chart(driver_frame(negative), "negative", limit),
                width="stretch",
            )
        else:
            st.caption("No feature decreased this event's score.")

    interpretations = []
    if explanation["positive_drivers"]:
        top = explanation["positive_drivers"][0]
        interpretations.append(
            f"The largest upward contribution is {top['label']} "
            f"(event {format_event_value(top['value'])}; training "
            f"{format_background(top)}), adding {top['shap_value']:+.3f}."
        )
    if explanation["negative_drivers"]:
        bottom = explanation["negative_drivers"][0]
        interpretations.append(
            f"The largest downward contribution is {bottom['label']} "
            f"(event {format_event_value(bottom['value'])}; training "
            f"{format_background(bottom)}), subtracting {abs(bottom['shap_value']):.3f}."
        )
    if interpretations:
        st.markdown(
            callout(
                "Reading the strongest contributions",
                " ".join(interpretations)
                + " Bars on both sides share the same zero reference and scale.",
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Full feature values and training comparison"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature": item["label"],
                        "Event value": format_event_value(item["value"]),
                        "Training reference": format_background(item),
                        "Shapley contribution": item["shap_value"],
                    }
                    for item in explanation["features"]
                ]
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "Shapley contribution": st.column_config.NumberColumn(format="%.6f"),
            },
        )
        st.caption(
            "Numeric references are sampled training means. Categorical references "
            "show the most common sampled state and its share. Comparisons are "
            "descriptive and do not imply expected behaviour."
        )

    with st.expander("Pairwise interaction details"):
        interactions = explanation["interactions"]
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

    with st.expander("Method, background sample, and additivity check"):
        st.markdown(f"**Method.** {explanation['method']}")
        st.markdown(
            f"**Background.** {explanation['background']['reference']} "
            f"{explanation['background']['rows_used']} of "
            f"{explanation['background']['available_training_rows']} training-period "
            "events were sampled."
        )
        st.markdown(
            f"**Additivity residual.** {explanation['additivity_error']:.3g} "
            "(expected value plus feature contributions minus the event prediction)."
        )
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
        section_header(
            "ALERT SUMMARY",
            "What was observed",
            "The event under review, its review priority, and the recommended next "
            "step. Scores rank review effort; they are not probabilities.",
        ),
        unsafe_allow_html=True,
    )
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

    context, summary_side = st.columns([1.15, 1], gap="medium")
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
    with summary_side:
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

    st.markdown(
        section_header(
            "EVIDENCE",
            "Why this alert was raised",
            "Independently checkable rule signals and the raw event record. Signals "
            "require analyst validation and are not proof of compromise.",
        ),
        unsafe_allow_html=True,
    )
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
        + '<p class="teqo-panel-note">The anomaly ranking can also contribute when no '
        "rule factor fires. Use the AI explanation to see which event features drove "
        "the model score.</p></div>",
        unsafe_allow_html=True,
    )
    with st.expander("Raw event record"):
        st.dataframe(
            format_event_record(row),
            width="stretch",
            hide_index=True,
            height=320,
        )
        st.caption(
            "Internal identifiers are included for traceability within this local "
            "prototype. Event times remain elapsed seconds."
        )

    st.markdown(
        section_header(
            "AI EXPLANATION",
            "Why the anomaly model scored this event",
            "Exact Shapley contributions for the Isolation Forest score. Positive "
            "values push the score above the training background; negative values "
            "pull it below.",
        ),
        unsafe_allow_html=True,
    )
    explanation_key = f"shap-explanation-{event_id}"
    explanation = st.session_state.get(explanation_key)
    if explanation is None:
        st.caption(
            "Generate the exact decomposition for this event. The calculation uses "
            "the current model artifacts and the training-period background, and is "
            "kept while this page stays open."
        )
    button_label = (
        "Generate exact Shapley explanation"
        if explanation is None
        else "Refresh Shapley explanation"
    )
    if st.button(button_label, key=f"shap-{event_id}"):
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
        except (
            ArithmeticError,
            IndexError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            st.error(f"The Shapley explanation could not be generated: {error}")
        else:
            st.session_state[explanation_key] = explanation
            st.rerun()
    if explanation is not None:
        render_shap_explanation(explanation)

    st.markdown(
        section_header(
            "ACTIVITY CONTEXT",
            "Surrounding activity and observed routes",
            "Related authentication activity and bounded links from the successful-"
            "authentication sample.",
        ),
        unsafe_allow_html=True,
    )
    timeline, twin = st.columns([1.2, 1], gap="medium")
    with timeline:
        st.markdown("**Related authentication activity**")
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
            "Related events use a ±300-second window and matching account or source "
            "device. Times are elapsed seconds, not clock times."
        )
    with twin:
        st.markdown("**Assets along observed routes**")
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
            "CASE MANAGEMENT",
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
        with st.expander("Audit trail"):
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
            st.caption("Changes recorded in the local case store.")