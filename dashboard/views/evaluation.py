"""Model Evaluation and research evidence page."""

import altair as alt
import pandas as pd
import streamlit as st

from components import (
    callout,
    metric_grid,
    limitations,
    process_diagram,
    provenance_block,
    section_header,
)
from formatting import MODEL_LABELS, format_metric

BRAND = "#36B9D6"
SECONDARY = "#8B7CF6"
BUDGET_COLORS = ["#36B9D6", "#F2A93B"]
FOLD_COLORS = ["#8FA6C2", "#36B9D6", "#F2A93B"]


def _evaluated_model_rows(models):
    rows = []
    for model_key, label in MODEL_LABELS.items():
        metrics = models.get(model_key, {})
        if not isinstance(metrics, dict):
            continue
        if metrics.get("status") != "evaluated":
            continue
        if metrics.get("average_precision") is None and metrics.get("roc_auc") is None:
            continue
        rows.append((model_key, label, metrics))
    return rows


def _ranking_comparison_chart(models):
    rows = []
    for _, label, metrics in _evaluated_model_rows(models):
        for metric, value in (
            ("Average precision", metrics.get("average_precision")),
            ("ROC-AUC", metrics.get("roc_auc")),
        ):
            if value is None:
                continue
            rows.append(
                {
                    "Approach": label,
                    "Metric": metric,
                    "Value": float(value),
                }
            )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("Approach:N", title=None, sort=None),
            yOffset=alt.YOffset("Metric:N", sort=["Average precision", "ROC-AUC"]),
            x=alt.X(
                "Value:Q",
                title="Score (share of maximum)",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=["Average precision", "ROC-AUC"],
                    range=[BRAND, SECONDARY],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("Approach:N"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Value:Q", format=".4f"),
            ],
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(align="left", dx=3, fontSize=10, color="#DCEAF5")
        .encode(
            y=alt.Y("Approach:N", sort=None),
            yOffset=alt.YOffset("Metric:N", sort=["Average precision", "ROC-AUC"]),
            x=alt.X("Value:Q", scale=alt.Scale(domain=[0, 1])),
            text=alt.Text("Value:Q", format=".3f"),
        )
    )
    return (bars + labels).properties(height=max(170, 58 * len(frame) + 50))


def _budget_frame(models, known_positives):
    rows = []
    for _, label, metrics in _evaluated_model_rows(models):
        for budget_number in (30, 100):
            budget = metrics.get("alert_budgets", {}).get(str(budget_number), {})
            hits = budget.get("known_positive_hits")
            if hits is None:
                continue
            rows.append(
                {
                    "Approach": label,
                    "Budget": f"Top {budget_number}",
                    "Known-positive hits": int(hits),
                    "Precision": budget.get("precision"),
                    "Recall": budget.get("recall"),
                    "Known positives": known_positives,
                }
            )
    return pd.DataFrame(rows)


def _budget_chart(frame, known_positives):
    if frame.empty:
        return None
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("Approach:N", title=None, sort=None),
            yOffset=alt.YOffset("Budget:N", sort=["Top 30", "Top 100"]),
            x=alt.X(
                "Known-positive hits:Q",
                title=f"Known-positive events reached (of {known_positives})",
                scale=alt.Scale(domain=[0, max(known_positives, 1)]),
                axis=alt.Axis(tickMinStep=1),
            ),
            color=alt.Color(
                "Budget:N",
                scale=alt.Scale(
                    domain=["Top 30", "Top 100"], range=BUDGET_COLORS
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("Approach:N"),
                alt.Tooltip("Budget:N"),
                alt.Tooltip("Known-positive hits:Q"),
                alt.Tooltip("Precision:Q", format=".3f"),
                alt.Tooltip("Recall:Q", format=".3f"),
            ],
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(align="left", dx=3, fontSize=10, color="#DCEAF5")
        .encode(
            y=alt.Y("Approach:N", sort=None),
            yOffset=alt.YOffset("Budget:N", sort=["Top 30", "Top 100"]),
            x=alt.X("Known-positive hits:Q", scale=alt.Scale(domain=[0, max(known_positives, 1)])),
            text=alt.Text("Known-positive hits:Q"),
        )
    )
    reference = (
        alt.Chart(pd.DataFrame({"Known-positive hits": [known_positives]}))
        .mark_rule(color="#5B7290", strokeDash=[4, 4])
        .encode(x="Known-positive hits:Q")
    )
    return (bars + labels + reference).properties(
        height=max(170, 54 * frame["Approach"].nunique() + 50)
    )


def _fold_frame(folds):
    rows = []
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        fold_models = fold.get("models", {})
        for model_key, label in MODEL_LABELS.items():
            metrics = fold_models.get(model_key, {})
            if not isinstance(metrics, dict):
                continue
            if metrics.get("status") != "evaluated":
                continue
            if metrics.get("average_precision") is None:
                continue
            rows.append(
                {
                    "Fold": int(fold.get("fold", 0)),
                    "Approach": label,
                    "Average precision": float(metrics["average_precision"]),
                    "ROC-AUC": metrics.get("roc_auc"),
                    "Known positives": metrics.get("known_positive_events"),
                    "Validation events": metrics.get("test_events"),
                    "Start (s)": fold.get("validation_start_elapsed_seconds"),
                    "End (s)": fold.get("validation_end_elapsed_seconds"),
                }
            )
    return pd.DataFrame(rows)


def _fold_chart(frame):
    if frame.empty:
        return None
    labels = list(dict.fromkeys(frame["Approach"].tolist()))
    chart = (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=55), strokeWidth=2.2)
        .encode(
            x=alt.X(
                "Fold:O",
                title="Chronological validation fold (later elapsed time →)",
            ),
            y=alt.Y(
                "Average precision:Q",
                title="Average precision",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "Approach:N",
                scale=alt.Scale(domain=labels, range=FOLD_COLORS[: len(labels)]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                alt.Tooltip("Approach:N"),
                alt.Tooltip("Fold:O"),
                alt.Tooltip("Average precision:Q", format=".4f"),
                alt.Tooltip("ROC-AUC:Q", format=".4f"),
                alt.Tooltip("Known positives:Q"),
                alt.Tooltip("Validation events:Q"),
                alt.Tooltip("Start (s):Q"),
                alt.Tooltip("End (s):Q"),
            ],
        )
        .properties(height=280)
    )
    return chart


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
                    if evaluated and budget.get("known_positive_hits") is not None
                    else "Not available"
                ),
                "Precision": format_metric(budget.get("precision"), percent=True),
                "Recall": format_metric(budget.get("recall"), percent=True),
                "F1": format_metric(budget.get("f1_score"), percent=True),
                "False-positive rate": format_metric(
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
            "Average precision": format_metric(
                model_metrics.get(model_key, {}).get("average_precision"),
                percent=True,
            ),
            "ROC-AUC": format_metric(
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
        ranking_chart = _ranking_comparison_chart(model_metrics)
        if ranking_chart is not None:
            st.altair_chart(ranking_chart, width="stretch")
            evaluated_labels = [
                label for _, label, _ in _evaluated_model_rows(model_metrics)
            ]
            unavailable_labels = [
                label
                for model_key, label in MODEL_LABELS.items()
                if label not in evaluated_labels
                and isinstance(model_metrics.get(model_key, {}), dict)
            ]
            caption = (
                "Only approaches with an evaluated numeric result are plotted. "
                "An unavailable result is omitted rather than treated as zero."
            )
            if unavailable_labels:
                caption += " Not evaluated: " + ", ".join(unavailable_labels) + "."
            st.caption(caption)
        st.dataframe(pd.DataFrame(ranking_rows), width="stretch", hide_index=True)

    budget_frame = _budget_frame(model_metrics, known_positives)
    if budget_frame.empty:
        st.markdown(
            callout(
                "Review-budget results unavailable",
                "No evaluated approach in the stored report has known-positive hits at "
                "the review budgets. No zero values are substituted.",
                tone="warning",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            section_header(
                "REVIEW BUDGET",
                "Known positives reached at fixed review budgets",
                "Agreement with known labels when analysts can review only the "
                "highest-ranked events. A larger budget reviews more events, so hit "
                "counts are not directly comparable across budgets.",
            ),
            unsafe_allow_html=True,
        )
        budget_chart = _budget_chart(budget_frame, known_positives)
        if budget_chart is not None:
            st.altair_chart(budget_chart, width="stretch")
            st.caption(
                f"The dashed line marks all {known_positives} known positive labels in "
                "the holdout. Precision and recall are available in the tables below."
            )
        for budget_number in (30, 100):
            findings = _format_model_results(
                model_metrics, known_positives, budget_number
            )
            if not findings:
                continue
            with st.expander(f"Top-{budget_number} review-budget details"):
                st.dataframe(
                    pd.DataFrame(findings), width="stretch", hide_index=True
                )
                st.caption(
                    "Known matches count reviewed events that exactly match a known "
                    "positive label. Missing results are shown as not available."
                )
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
        completed = walk_forward.get("completed_folds")
        requested = walk_forward.get("requested_folds")
        st.markdown(
            section_header(
                "VALIDATION",
                "Earlier chronological validation folds",
                "Expanding-window folds compare the same approaches without training "
                "on future events. Only completed, evaluated folds are plotted.",
            ),
            unsafe_allow_html=True,
        )
        fold_frame = _fold_frame(folds)
        fold_chart = _fold_chart(fold_frame)
        if fold_chart is not None:
            st.altair_chart(fold_chart, width="stretch")
            st.caption(
                "Average precision by validation fold. Hover for the fold window, "
                "known-positive count, validation size, and ROC-AUC. Validation "
                "windows are elapsed seconds, not calendar dates."
            )
        else:
            st.markdown(
                callout(
                    "No evaluable fold results",
                    "The stored report contains folds, but none has an evaluated "
                    "average-precision result to plot.",
                    tone="warning",
                ),
                unsafe_allow_html=True,
            )
        fold_rows = []
        for fold in folds:
            fold_models = fold.get("models", {})
            start = fold.get("validation_start_elapsed_seconds")
            end = fold.get("validation_end_elapsed_seconds")
            window = (
                f"{int(start):,}–{int(end):,} s"
                if start is not None and end is not None
                else "unavailable"
            )
            for model_key, label in MODEL_LABELS.items():
                metrics = fold_models.get(model_key, {})
                if not isinstance(metrics, dict):
                    continue
                budget = metrics.get("alert_budgets", {}).get("30", {})
                hits = budget.get("known_positive_hits")
                fold_rows.append(
                    {
                        "Fold": fold.get("fold"),
                        "Validation window": window,
                        "Approach": label,
                        "Validation events": metrics.get("test_events"),
                        "Known positive labels": metrics.get("known_positive_events"),
                        "Status": metrics.get("status", "not available").replace("_", " "),
                        "Average precision": format_metric(
                            metrics.get("average_precision"), percent=True
                        ),
                        "Known matches in top 30": hits if hits is not None else "—",
                    }
                )
        if fold_rows:
            st.dataframe(pd.DataFrame(fold_rows), width="stretch", hide_index=True)
            count_text = (
                f"{int(completed)} of {int(requested)} requested folds completed"
                if completed is not None and requested is not None
                else "Completed fold count unavailable"
            )
            st.caption(
                f"{count_text}. Each fold uses the same chronological windows across "
                "the baseline and available models. Folds without known positives are "
                "not evaluable for detection quality. Missing metrics are not shown as "
                "zero."
            )
    elif isinstance(walk_forward, dict):
        st.markdown(
            callout(
                "Walk-forward validation unavailable",
                "The stored report does not contain completed chronologically validated "
                "folds. No fold results are shown.",
                tone="warning",
            ),
            unsafe_allow_html=True,
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


def render_research_and_evaluation(report, data, provenance_issues):
    st.markdown(
        '<div class="teqo-panel"><div class="teqo-section-kicker">RESEARCH QUESTION</div>'
        '<p class="teqo-question">Can unsupervised authentication-anomaly rankings help '
        "prioritize known red-team events in a later chronological period, compared with "
        "a first-seen-destination baseline?</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        process_diagram(
            "EVALUATION DESIGN",
            "Train and evaluate without time leakage",
            "Labels are evaluation evidence only; they are not inputs to unsupervised model "
            "fitting.",
            [
                ("Earlier events", "Fit feature transforms and unsupervised model(s)."),
                ("Later holdout", "Keep a chronological period separate from training."),
                ("Rank events", "Score holdout events using models fitted on earlier data."),
                ("Compare", "Measure agreement with known labels at fixed review budgets."),
            ],
        ),
        unsafe_allow_html=True,
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
                "Provenance checks passed",
                "The report's pipeline schema, input SHA-256, and event run identifiers "
                "match the loaded activity results.",
                tone="info",
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