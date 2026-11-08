"""Architecture page."""

import html

import streamlit as st

from components import (
    callout,
    contribution_grid,
    feature_grid,
    page_header,
    process_diagram,
    section_header,
)
from src.dataset_paths import DATASET

MODULE_STAGES = [
    {
        "index": "01",
        "kicker": "INPUTS & PREPARATION",
        "title": "Raw activity to clean events",
        "modules": [
            (
                "src/generate_synthetic_data.py",
                "Writes deterministic synthetic authentication and label files.",
            ),
            (
                "src/data_prep.py",
                "Streams raw logs, maps the common event schema, and writes clean events.",
            ),
            (
                "src/dataset_paths.py",
                "Resolves dataset-specific raw, processed, and model paths.",
            ),
        ],
        "artifacts": ["data/raw/*.txt.gz", "auth_events_clean.csv"],
    },
    {
        "index": "02",
        "kicker": "SCORING & EVALUATION",
        "title": "Features, models, metrics",
        "modules": [
            (
                "src/model.py",
                "Builds features, fits the chronological split, scores events, and "
                "trains the optional autoencoder.",
            ),
            (
                "src/evaluation.py",
                "Calculates ranking and fixed-budget review metrics.",
            ),
        ],
        "artifacts": [
            "models/*.pkl",
            "models/autoencoder.pt",
            "auth_events_scored.csv",
            "evaluation_report.json",
        ],
    },
    {
        "index": "03",
        "kicker": "RISK & PROVENANCE",
        "title": "Review priority with checks",
        "modules": [
            (
                "src/risk_engine.py",
                "Combines the anomaly percentile with named, independently checkable "
                "risk factors.",
            ),
            (
                "src/provenance.py",
                "Creates and validates run identifiers, input hashes, and report links.",
            ),
        ],
        "artifacts": ["auth_events_risk.csv", "pipeline_run_id · SHA-256"],
    },
    {
        "index": "04",
        "kicker": "RUNTIME & REVIEW",
        "title": "Dashboard and analyst state",
        "modules": [
            (
                "dashboard/app.py",
                "Loads results, applies shared filters, and dispatches page views.",
            ),
            (
                "dashboard/views/*.py",
                "Renders Overview, Investigation, Twin, Intelligence, Evaluation, and "
                "Architecture pages.",
            ),
            (
                "src/digital_twin.py",
                "Builds the successful-authentication graph, bounded paths, and exact "
                "Shapley explanations.",
            ),
            (
                "src/case_store.py",
                "Stores analyst case state and audit history in local SQLite.",
            ),
        ],
        "artifacts": ["dashboard/graph_view/index.html", "data/state/*.sqlite3"],
    },
]


def _module_stage(stage):
    modules = "".join(
        "<li><code>{}</code><span>{}</span></li>".format(
            html.escape(path), html.escape(role)
        )
        for path, role in stage["modules"]
    )
    artifacts = "".join(
        f"<code>{html.escape(artifact)}</code>" for artifact in stage["artifacts"]
    )
    return (
        '<section class="teqo-arch-stage" role="listitem">'
        '<div class="teqo-arch-stage-head">'
        f'<span class="teqo-arch-index">{html.escape(stage["index"])}</span>'
        f'<span>{html.escape(stage["kicker"])}</span></div>'
        f"<h3>{html.escape(stage['title'])}</h3>"
        f'<ul class="teqo-arch-modules">{modules}</ul>'
        '<div class="teqo-arch-artifacts"><span>Artifacts</span>'
        f"{artifacts}</div></section>"
    )


def render_module_architecture():
    stages = '<div class="teqo-arch-arrow" aria-hidden="true">→</div>'.join(
        _module_stage(stage) for stage in MODULE_STAGES
    )
    return (
        '<div class="teqo-arch-flow" role="list" '
        'aria-label="Module-level repository data flow">'
        f"{stages}</div>"
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

    st.markdown(
        section_header(
            "MODULE MAP",
            "How the repository is connected",
            "Each box is an actual module or generated artifact in this repository. "
            "Arrows show the direction of data and control; no external service, API, "
            "or database is implied.",
        )
        + render_module_architecture(),
        unsafe_allow_html=True,
    )
    st.caption(
        "The dashboard reads generated artifacts and does not retrain models. Only "
        "case records are written at runtime, to a local SQLite file."
    )

    st.markdown(
        process_diagram(
            "DATA FLOW",
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
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        process_diagram(
            "OBSERVED PATHS",
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
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        process_diagram(
            "CASE WORKFLOW",
            "Local analyst case workflow",
            "The analyst owns status, notes and decisions; TeqoAI does not execute containment.",
            [
                ("Select an event", "Review its authentication and risk context."),
                ("Record decision", "Set status, owner, notes and checklist progress."),
                ("Persist locally", "SQLite stores current case records on this computer."),
                ("Audit changes", "Changed case state is recorded in a local audit history."),
            ],
        ),
        unsafe_allow_html=True,
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