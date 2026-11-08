"""AI Intelligence page."""

import html

import streamlit as st

from components import (
    badge,
    callout,
    feature_grid,
    metric_grid,
    process_diagram,
    section_header,
)
from data_access import probe_model_artifacts
from formatting import MODEL_LABELS, format_metric


def _runtime_status(
    probe_entry, score_present, active_label, inactive_label
):
    """Translate a probed artifact result into an explicit runtime state."""
    if probe_entry.get("loaded"):
        if score_present:
            return active_label, "success"
        return inactive_label, "info"
    if probe_entry.get("reason") == "dependency":
        return "Not installed", "neutral"
    if probe_entry.get("reason") == "artifact":
        return "Not available", "neutral"
    return "Unavailable", "neutral"


def _status_card(
    role,
    role_tone,
    title,
    status,
    status_tone,
    body,
    inputs,
    outputs,
    limitations,
):
    facts = [
        ("Inputs", inputs),
        ("Outputs", outputs),
        ("Limitations", limitations),
    ]
    fact_markup = "".join(
        f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>"
        for label, value in facts
    )
    return (
        '<div class="teqo-contribution teqo-model-card">'
        '<div class="teqo-badge-row">'
        + badge(role, role_tone)
        + badge(status, status_tone, dot=True)
        + "</div>"
        f"<strong>{html.escape(title)}</strong>"
        f"<p>{html.escape(body)}</p>"
        f'<dl class="teqo-model-facts">{fact_markup}</dl>'
        "</div>"
    )


def _model_status_cards(probe, risk_events):
    columns = set(risk_events.columns) if risk_events is not None else set()
    baseline_active = "anomaly_score_first_seen_baseline" in columns
    iforest_active = "anomaly_score_iforest" in columns
    autoencoder_active = (
        "anomaly_score_autoencoder" in columns
        and bool(risk_events["anomaly_score_autoencoder"].notna().any())
    )

    if baseline_active:
        baseline_status = "Available for comparison"
        baseline_tone = "info"
    elif risk_events is not None:
        baseline_status = "Score column missing"
        baseline_tone = "warning"
    else:
        baseline_status = "No results loaded"
        baseline_tone = "neutral"

    iforest_status, iforest_tone = _runtime_status(
        probe.get("isolation_forest", {}),
        iforest_active,
        "Loaded · primary ranking active",
        "Loaded · scores not present",
    )
    autoencoder_status, autoencoder_tone = _runtime_status(
        probe.get("autoencoder", {}),
        autoencoder_active,
        "Loaded · optional scores available",
        "Loaded · optional, not applied",
    )

    cards = [
        (
            "Rule baseline",
            "info",
            "First-seen destination baseline",
            baseline_status,
            baseline_tone,
            "A transparent comparator, not a machine-learning model. It ranks "
            "first-time account-to-destination pairs so the learned models can be "
            "tested against a simple rule.",
            "First-seen pairing flag.",
            "Novelty indicator used by the risk layer.",
            "Rewards novelty only; familiar malicious activity ranks low.",
        ),
        (
            "Primary ML",
            "accent",
            "Isolation Forest",
            iforest_status,
            iforest_tone,
            "The primary unsupervised anomaly model used for event ranking. It is "
            "loaded from local artifacts and scored against the current feature set.",
            "Eight encoded event features: outcome, novelty, prior-hour volume, "
            "authentication and logon categories.",
            "Continuous anomaly score (negative decision_function).",
            "Unsupervised ranking; the score is not a calibrated probability.",
        ),
        (
            "Optional ML",
            "neutral",
            "Autoencoder",
            autoencoder_status,
            autoencoder_tone,
            "An optional reconstruction model. When its artifacts are present and "
            "applied, reconstruction error contributes a second anomaly score; it is "
            "not required for the primary ranking.",
            "The same eight encoded features, standardized.",
            "Mean squared reconstruction error.",
            "Fixed-epoch training, not calibrated, and may be unavailable on a host.",
        ),
    ]
    columns_ui = st.columns(3, gap="medium")
    for column, card in zip(columns_ui, cards):
        with column:
            st.markdown(_status_card(*card), unsafe_allow_html=True)


def _render_quality_snapshot(report):
    models = {}
    if isinstance(report, dict):
        evaluation = report.get("evaluation")
        if isinstance(evaluation, dict):
            models = evaluation.get("models") or {}
    if not models:
        st.markdown(
            callout(
                "Measured quality unavailable",
                "No evaluation results are loaded, so model quality is not shown. "
                "Runtime availability above does not imply measured performance.",
                tone="warning",
            ),
            unsafe_allow_html=True,
        )
        return

    items = []
    for key, label in MODEL_LABELS.items():
        model = models.get(key) or {}
        evaluated = model.get("status") == "evaluated"
        average_precision = model.get("average_precision")
        roc_auc = model.get("roc_auc")
        if evaluated and average_precision is not None:
            hint_parts = []
            if roc_auc is not None:
                hint_parts.append(f"ROC-AUC {format_metric(roc_auc)}")
            if model.get("test_events") is not None:
                hint_parts.append(f"{int(model['test_events'])} test events")
            items.append(
                {
                    "label": label,
                    "value": format_metric(average_precision),
                    "hint": " · ".join(hint_parts) or "Average precision",
                    "tone": "info",
                    "size": "compact",
                }
            )
        else:
            items.append(
                {
                    "label": label,
                    "value": "Not available",
                    "hint": "No evaluated result in the stored report",
                    "tone": "neutral",
                    "size": "compact",
                }
            )
    st.markdown(
        metric_grid(items),
        unsafe_allow_html=True,
    )
    st.caption(
        "Average precision from the latest stored chronological test. Unavailable "
        "results are shown as not available, never as zero. Open Model Evaluation "
        "for budgets, fold results, and provenance."
    )


def render_model_intelligence(report=None, risk_events=None, results_page=None):
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
            "MODEL STATUS",
            "What is actually running",
            "Availability is checked at render time from local model artifacts and "
            "score columns in the loaded results. Runtime presence is not a quality "
            "claim.",
        ),
        unsafe_allow_html=True,
    )
    probe = probe_model_artifacts()
    _model_status_cards(probe, risk_events)
    st.caption(
        "The baseline is implemented in the risk layer and needs no model artifact. "
        "The Isolation Forest is the primary learned model. The autoencoder is "
        "optional and never required for ranking."
    )

    st.markdown(
        section_header(
            "MEASURED QUALITY",
            "Latest chronological test at a glance",
            "Average precision from the stored evaluation report for the same test "
            "period. These values compare agreement with known labels, not proof of "
            "malicious activity.",
        ),
        unsafe_allow_html=True,
    )
    _render_quality_snapshot(report)

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
    if results_page is not None:
        st.page_link(
            results_page,
            label="Review measured model evaluation",
            icon=":material/analytics:",
        )


def render_process_diagram(title, description, steps, kicker="PROCESS"):
    """Compatibility wrapper for the shared process diagram component."""
    st.markdown(
        process_diagram(kicker, title, description, steps),
        unsafe_allow_html=True,
    )