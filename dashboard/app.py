"""TeqoAI analyst dashboard: application setup, navigation and page orchestration.

Page rendering lives in ``dashboard/views``. The directory is intentionally not
named ``pages``: Streamlit reserves that name beside the entrypoint for its
automatic multipage navigation, which would shadow these routes on a hard
page load. Shared data loading is in ``dashboard/data_access.py`` and sidebar
filtering in ``dashboard/filters.py``.
"""

import json
import os
import sqlite3
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
for project_path in (BASE_DIR, DASHBOARD_DIR):
    if project_path not in sys.path:
        sys.path.insert(0, project_path)

from components import empty_state, environment_badge, page_header  # noqa: E402
from data_access import (  # noqa: E402
    cached_file_sha256,
    load_data,
    load_graph,
)
from filters import apply_filters  # noqa: E402
from views.about import render_demo_setup, render_homepage  # noqa: E402
from views.architecture import render_architecture  # noqa: E402
from views.digital_twin import render_twin  # noqa: E402
from views.evaluation import (  # noqa: E402
    render_data_quality,
    render_research_and_evaluation,
)
from views.intelligence import render_model_intelligence  # noqa: E402
from views.investigation import render_investigation  # noqa: E402
from views.overview import render_overview  # noqa: E402
from src.case_store import get_cases  # noqa: E402
from src.dataset_paths import (  # noqa: E402
    CLEAN_FILE,
    DATASET,
    EVALUATION_FILE,
    RISK_FILE,
)
from src.provenance import validate_pipeline_provenance  # noqa: E402
from theme import load_styles  # noqa: E402

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


PAGE_TITLES = {
    "Overview": "Security overview",
    "Investigate": "Alert investigation",
    "Observed Paths": "Cyber digital twin",
    "AI Intelligence": "AI intelligence",
    "Research & Evaluation": "Model evaluation",
}

PAGE_DESCRIPTIONS = {
    "Overview": (
        "Alert volume, severity mix, and the review queue for the loaded "
        "authentication sample."
    ),
    "Investigate": (
        "Inspect evidence, explanations, and record an analyst-owned case decision."
    ),
    "Observed Paths": (
        "Explore successful authentication relationships and where activity was "
        "observed next."
    ),
    "AI Intelligence": (
        "How the implemented detectors work, what signals they use, and how to read "
        "a score."
    ),
    "Research & Evaluation": (
        "Chronological validation, measured ranking evidence, and responsible "
        "limitations."
    ),
}

PAGE_KICKERS = {
    "Overview": "WORKSPACE · SECURITY OVERVIEW",
    "Investigate": "WORKSPACE · ALERT INVESTIGATION",
    "Observed Paths": "WORKSPACE · CYBER DIGITAL TWIN",
    "AI Intelligence": "INTELLIGENCE · AI INTELLIGENCE",
    "Research & Evaluation": "INTELLIGENCE · MODEL EVALUATION",
}


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
        render_demo_setup(
            f"teqo-demo-build-{selected_page.lower().replace(' ', '-')}"
        )
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
        "risk_level",
        "risk_score",
        "time",
        "user",
        "device",
        "target_device",
        "result",
        "action",
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
            if "pipeline_run_id" in risk_events
            else []
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
        render_overview(risk_events, filtered_events, case_records, investigation_page)
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
            render_twin(graph, filtered_events, clean_events)
        except (KeyError, ValueError, RuntimeError):
            st.error(
                "The observed connections could not be prepared. Refresh the demo "
                "or try again later."
            )
    elif selected_page == "AI Intelligence":
        render_model_intelligence(report, risk_events, results_page)
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
    lambda: render_homepage(overview_page, intelligence_page),
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