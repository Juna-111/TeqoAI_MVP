"""About TeqoAI page and demo preparation."""

import base64
import html
import os

import streamlit as st

from components import (
    badge,
    callout,
    contact_list,
    contribution_grid,
    environment_badge,
    feature_grid,
    kv_rows,
    limitations,
    section_header,
)
from src.dataset_paths import BASE_DIR, CLEAN_FILE, DATASET, RISK_FILE
from src.demo_pipeline import run_synthetic_demo

LOGO_FILE = os.path.join(BASE_DIR, "assets", "teqoai-logo.png")


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


def render_about(overview_page, intelligence_page):
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
            "<div>"
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
                "OVERVIEW",
                "What TeqoAI is",
                "An explainable triage workspace for authentication events.",
            )
            + "<p>TeqoAI reads authentication activity, ranks events that differ from "
            "an account's earlier behaviour, and presents the evidence behind each "
            "ranking. An analyst reviews the alert, inspects exact feature "
            "attributions and observed account/device links, and records a case "
            "decision. The prototype is local: it reads exported logs and writes "
            "case records, with no live collection or automated response.</p>"
            '<div class="teqo-home-facts">'
            f"<div><span>Dataset</span> · {html.escape(dataset_label)}</div>"
            "<div><span>Ranking</span> · Unsupervised anomaly score</div>"
            "<div><span>Decision-maker</span> · Human analyst</div>"
            "</div></section>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<section class="teqo-home-section">'
            + section_header(
                "THE PROBLEM",
                "Why authentication triage needs explainability",
                "Volume is only part of the difficulty; context and evidence matter too.",
            )
            + feature_grid(
                [
                    {
                        "icon": "database",
                        "title": "Too many events to read",
                        "body": "Authentication logs are high-volume and mostly routine, "
                        "so analysts need a defensible way to choose what to review first.",
                    },
                    {
                        "icon": "investigate",
                        "title": "Context is buried",
                        "body": "A single event says little on its own. Investigators need "
                        "the account's earlier behaviour, peer activity, and observed "
                        "connections in one place.",
                    },
                    {
                        "icon": "brain",
                        "title": "Opaque scores are hard to defend",
                        "body": "An anomaly score without visible reasoning cannot support "
                        "a documented decision, review by another analyst, or an honest "
                        "statement of model limitations.",
                    },
                ]
            )
            + "</section>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<section class="teqo-home-section">'
            + section_header(
                "CONTRIBUTIONS",
                "What the prototype contributes",
                "Each contribution maps to code in the repository rather than a "
                "hypothetical capability.",
            )
            + contribution_grid(
                [
                    (
                        "Leakage-resistant evaluation",
                        "A chronological holdout compares the Isolation Forest ranking "
                        "with a first-seen baseline and an optional autoencoder using "
                        "fixed review budgets. Unavailable results are reported as "
                        "unavailable, never as zero performance.",
                        "src/evaluation.py",
                    ),
                    (
                        "Exact feature attribution",
                        "Interventional Shapley values over grouped event features and "
                        "pairwise interactions explain each anomaly score, with "
                        "background comparisons and an additivity check.",
                        "src/digital_twin.py · dashboard/views/investigation.py",
                    ),
                    (
                        "Transparent risk factors",
                        "Named, independently checkable rules add context to the model "
                        "ranking. Every adjustment is visible, and the result remains a "
                        "review priority rather than a probability.",
                        "src/risk_engine.py",
                    ),
                    (
                        "Evidence-bounded relationships",
                        "Observed account and device authentication links give "
                        "investigators context without asserting reachability, "
                        "authorization, or compromise.",
                        "src/digital_twin.py · dashboard/views/digital_twin.py",
                    ),
                    (
                        "Analyst-owned case records",
                        "Case status, notes, and follow-up decisions are stored locally "
                        "with timestamps, so the investigation trail stays with the "
                        "analyst instead of an automated action.",
                        "src/case_store.py · dashboard/views/investigation.py",
                    ),
                ]
            )
            + "</section>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<section class="teqo-home-section">'
            + section_header(
                "IMPLEMENTATION SCOPE",
                "What is implemented — and what is not",
                "The boundary is part of the contribution: this is a research "
                "prototype, not a production security platform.",
            )
            + kv_rows(
                [
                    ("Data profiles", "Synthetic generator and an authorized LANL sample"),
                    ("Detection", "Isolation Forest, first-seen baseline, optional autoencoder"),
                    ("Risk", "Named rule-assisted score shown as a review priority"),
                    ("Explanation", "Exact Shapley values with background comparison"),
                    ("Investigation", "Alert context, raw event record, observed paths"),
                    ("Cases", "Local analyst decisions, notes, and status history"),
                    ("Evaluation", "Chronological holdout and fixed review budgets"),
                    ("Provenance", "Run identifier and input checksum verification"),
                ]
            )
            + '<div class="teqo-section-kicker teqo-scope-kicker">DELIBERATELY OUT OF SCOPE</div>'
            + limitations(
                [
                    "Live monitoring, streaming ingestion, or scheduled collection.",
                    "Automated containment, blocking, or any response action.",
                    "Identity-provider, SIEM, or ticketing integrations.",
                    "Production case management, encryption at rest, or multi-user access control.",
                    "Log adapters beyond the documented synthetic and LANL layouts.",
                    "Inferring reachability or compromise from observed authentication links.",
                ]
            )
            + "</section>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<section class="teqo-home-section">'
            + section_header(
                "LIMITATIONS",
                "What the results do and do not support",
                "These limits apply to the current build and are stated explicitly "
                "throughout the interface.",
            )
            + limitations(
                [
                    "Confirmed labels are sparse. Events without a confirmed label are "
                    "unknown, not known-benign, and the current pipeline expects binary labels.",
                    (
                        "This build uses generated demonstration activity; its metrics "
                        "exercise the software pipeline and are not evidence of accuracy "
                        "on real organizations."
                        if DATASET == "synthetic"
                        else "Results depend on the selected dataset's time range, "
                        "confirmed-label coverage, and the number of positive events "
                        "in the holdout."
                    ),
                    "Anomaly and risk scores are review priorities, not probabilities "
                    "of malicious activity.",
                    "Risk rules and their point values are prototype heuristics and have "
                    "not been validated against an organization's authentication policy.",
                    "Case notes are stored unencrypted in a local SQLite database without "
                    "user authentication or multi-user controls.",
                    "Observed paths show authentication links only; they do not establish "
                    "network reachability, authorization, or compromise.",
                    "Data preparation processes a bounded prefix of the input, so longer "
                    "time ranges may be truncated.",
                ]
            )
            + "</section>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<section class="teqo-home-section">'
            + section_header(
                "TECHNOLOGY & REPOSITORY",
                "How it is built",
                "Standard Python tooling, local artifacts, and a focused repository layout.",
            )
            + kv_rows(
                [
                    ("Language", "Python 3"),
                    ("Data", "pandas, NumPy, gzip-compressed CSV artifacts"),
                    ("Models", "scikit-learn Isolation Forest; optional PyTorch autoencoder"),
                    ("Graph", "NetworkX over observed authentication links"),
                    ("Interface", "Streamlit, Altair charts, custom HTML graph component"),
                    ("Storage", "CSV artifacts and a local SQLite case store"),
                    ("Pipeline", "src/ — preparation, model, risk, evaluation, provenance"),
                    ("Dashboard", "dashboard/ — Streamlit app and focused view modules"),
                    ("Artifacts", "data/ raw and processed files; models/ serialized models"),
                    ("Checks", "tests/ — data, model, graph, case, and provenance checks"),
                ]
            )
            + "</section>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<section class="teqo-home-section">'
            + section_header(
                "ACKNOWLEDGEMENTS",
                "Project context",
                "TeqoAI was developed as a graduation project for INSA Summer Camp.",
            )
            + "<p>The prototype follows the publicly documented LANL Cyber1 "
            "authentication dataset layout and was built with open-source Python, "
            "pandas, NumPy, scikit-learn, PyTorch (optional), NetworkX, Streamlit, and "
            "Altair. Thanks to the INSA Summer Camp organizers and mentors for the "
            "programme, and to the maintainers of these libraries.</p></section>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="teqo-about-strip">'
            "<div>"
            '<div class="teqo-section-kicker">CREATOR</div>'
            "<h2>Hobby Yano</h2>"
            "<p>Design and development of the TeqoAI prototype, from data preparation "
            "and model evaluation to the analyst workspace and explanations.</p>"
            + creator_contacts
            + "</div>"
            '<div class="teqo-about-note">'
            "<strong>Research prototype, not a production security product.</strong>"
            "<p>TeqoAI supports analyst review and does not perform automated response. "
            "Scores, explanations, and observed links should be interpreted within the "
            "limitations stated above.</p></div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="teqo-home-footer">'
            "<span>TEQOAI · Explainable authentication triage</span>"
            "<span>INSA Summer Camp · Created by Hobby Yano</span>"
            "</div>",
            unsafe_allow_html=True,
        )