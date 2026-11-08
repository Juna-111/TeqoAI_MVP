"""Immersive TeqoAI homepage: cinematic landing, explorer, and demo story.

The homepage uses its own deterministic synthetic graph and event examples.
Nothing in this module reads processed event files, model results, evaluation
outputs, or case records used by the analyst workspace.
"""

import base64
import os

import streamlit as st

from components import badge, section_header
from src.dataset_paths import BASE_DIR
from views.digital_twin import teqo_graph

LOGO_FILE = os.path.join(BASE_DIR, "assets", "teqoai-logo.png")

EXPLORER_NODES = [
    {"id": "a.mensah", "type": "account", "type_label": "Account"},
    {"id": "j.okafor", "type": "account", "type_label": "Account"},
    {"id": "r.tan", "type": "account", "type_label": "Account"},
    {"id": "l.moreau", "type": "account", "type_label": "Account"},
    {"id": "WS-021", "type": "device", "type_label": "Device"},
    {"id": "WS-044", "type": "device", "type_label": "Device"},
    {"id": "LT-118", "type": "device", "type_label": "Device"},
    {"id": "DEV-UNK-7", "type": "device", "type_label": "Device", "unusual": True},
    {"id": "SSO-GATEWAY", "type": "service", "type_label": "Service"},
    {"id": "MAIL-01", "type": "service", "type_label": "Service"},
    {"id": "VPN-HUB", "type": "service", "type_label": "Service"},
    {"id": "HR-PORTAL", "type": "service", "type_label": "Service"},
    {"id": "DB-FIN", "type": "service", "type_label": "Service"},
]

EXPLORER_EDGES = [
    {"source": "a.mensah", "target": "WS-021", "weight": 12},
    {"source": "a.mensah", "target": "SSO-GATEWAY", "weight": 9},
    {"source": "WS-021", "target": "SSO-GATEWAY", "weight": 7},
    {"source": "WS-021", "target": "MAIL-01", "weight": 5},
    {"source": "j.okafor", "target": "WS-044", "weight": 10},
    {"source": "j.okafor", "target": "VPN-HUB", "weight": 4},
    {"source": "WS-044", "target": "HR-PORTAL", "weight": 6},
    {"source": "WS-044", "target": "SSO-GATEWAY", "weight": 5},
    {"source": "r.tan", "target": "LT-118", "weight": 8},
    {"source": "r.tan", "target": "VPN-HUB", "weight": 5},
    {"source": "LT-118", "target": "SSO-GATEWAY", "weight": 6},
    {"source": "LT-118", "target": "MAIL-01", "weight": 4},
    {"source": "l.moreau", "target": "LT-118", "weight": 3},
    {"source": "a.mensah", "target": "DEV-UNK-7", "weight": 2, "unusual": True},
    {"source": "DEV-UNK-7", "target": "SSO-GATEWAY", "weight": 4, "unusual": True},
    {"source": "DEV-UNK-7", "target": "HR-PORTAL", "weight": 2, "unusual": True},
    {"source": "VPN-HUB", "target": "SSO-GATEWAY", "weight": 6},
    {"source": "SSO-GATEWAY", "target": "HR-PORTAL", "weight": 8},
    {"source": "HR-PORTAL", "target": "DB-FIN", "weight": 3},
]

EXPLORER_POSITIONS = {
    "a.mensah": [178, 168],
    "j.okafor": [156, 470],
    "r.tan": [820, 300],
    "l.moreau": [884, 448],
    "WS-021": [334, 318],
    "WS-044": [338, 520],
    "LT-118": [704, 508],
    "DEV-UNK-7": [356, 118],
    "SSO-GATEWAY": [522, 296],
    "MAIL-01": [722, 152],
    "VPN-HUB": [520, 118],
    "HR-PORTAL": [724, 402],
    "DB-FIN": [600, 556],
}

EVENTS = {
    "unusual": {
        "label": "Unusual example",
        "account": "a.mensah",
        "device": "DEV-UNK-7",
        "service": "SSO-GATEWAY",
        "context": (
            "A first-seen device reaches the identity service outside the pattern "
            "used in the typical example. This sequence is hand-built for the demo."
        ),
        "indicator": "Anomaly indicator · illustrative",
        "tone": "warn",
        "emphasis": 4,
        "unusual": True,
        "signals": [
            {
                "label": "New device",
                "edge": "account-device",
                "detail": (
                    "DEV-UNK-7 has no earlier link to a.mensah in this generated "
                    "demonstration window."
                ),
            },
            {
                "label": "Unusual hour",
                "edge": "account-device",
                "detail": (
                    "The synthetic event is timestamped 03:14, outside the daytime "
                    "pattern used in the typical example."
                ),
            },
            {
                "label": "Repeated service attempts",
                "edge": "device-service",
                "detail": (
                    "The generated sequence shows several SSO attempts after the new "
                    "device appears."
                ),
            },
        ],
    },
    "typical": {
        "label": "Typical example",
        "account": "a.mensah",
        "device": "WS-021",
        "service": "SSO-GATEWAY",
        "context": (
            "A known device reaches a service the account has used before in this "
            "demonstration. Nothing here asserts that the activity is benign."
        ),
        "indicator": "No anomaly highlighted · illustrative",
        "tone": "calm",
        "emphasis": 1,
        "unusual": False,
        "signals": [
            {
                "label": "Known device",
                "edge": "account-device",
                "detail": (
                    "WS-021 already has illustrative links to a.mensah elsewhere in "
                    "this synthetic window."
                ),
            },
            {
                "label": "Consistent hour",
                "edge": "account-device",
                "detail": (
                    "The synthetic event is timestamped 10:42, inside the window used "
                    "for the typical example."
                ),
            },
            {
                "label": "Familiar service",
                "edge": "device-service",
                "detail": (
                    "SSO-GATEWAY appears earlier in the illustrative activity for this "
                    "account and device."
                ),
            },
        ],
    },
}


def _logo_data_uri():
    try:
        with open(LOGO_FILE, "rb") as logo_file:
            encoded = base64.b64encode(logo_file.read()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/png;base64,{encoded}"


def _explorer_payload():
    node_details = {}
    for node in EXPLORER_NODES:
        outgoing = sum(
            1 for edge in EXPLORER_EDGES if edge["source"] == node["id"]
        )
        incoming = sum(
            1 for edge in EXPLORER_EDGES if edge["target"] == node["id"]
        )
        observations = sum(
            edge["weight"]
            for edge in EXPLORER_EDGES
            if node["id"] in (edge["source"], edge["target"])
        )
        node_details[node["id"]] = {
            "kicker": f"SYNTHETIC DEMO · {node['type_label'].upper()}",
            "title": node["id"],
            "subtitle": f"Illustrative {node['type_label'].lower()} in the homepage network.",
            "rows": [
                ["Entity type", node["type_label"]],
                ["Link direction", f"{outgoing} outgoing · {incoming} incoming"],
                ["Illustrative observations", observations],
            ],
            "note": (
                "Generated for this demonstration only. Not loaded activity, live "
                "monitoring, or evidence of compromise."
            ),
        }

    edge_details = {}
    for edge in EXPLORER_EDGES:
        edge_details[f"{edge['source']} -> {edge['target']}"] = {
            "kicker": "SYNTHETIC DEMO · RELATIONSHIP",
            "title": f"{edge['source']} → {edge['target']}",
            "subtitle": "Illustrative authentication direction",
            "rows": [
                ["Direction", f"{edge['source']} → {edge['target']}"],
                ["Observations", edge["weight"]],
                ["Context", "Observed link only"],
            ],
            "note": (
                "Generated demonstration data. An observed link does not establish "
                "network reachability, authorization, or compromise."
            ),
        }

    return node_details, edge_details


def _hero_markup(logo_uri):
    brand_image = f'<img src="{logo_uri}" alt="" />' if logo_uri else ""
    logo_image = (
        f'<image href="{logo_uri}" x="166" y="166" width="108" height="108" '
        'opacity="0.94" />'
        if logo_uri
        else '<circle cx="220" cy="220" r="46" fill="rgba(54,185,214,0.16)" />'
    )
    return (
        '<section class="teqo-landing-hero" aria-labelledby="teqo-landing-title">'
        '<nav class="teqo-landing-nav" aria-label="Homepage sections">'
        '<div class="teqo-landing-brand">'
        f"{brand_image}<span>TEQO<span>AI</span></span>"
        "</div>"
        '<div class="teqo-landing-navlinks">'
        '<a href="#teqo-explorer">Network</a>'
        '<a href="#teqo-signals">Signals</a>'
        '<a href="#teqo-discover">Explore</a>'
        "</div></nav>"
        '<div class="teqo-landing-copy">'
        '<div class="teqo-home-meta">'
        '<span class="teqo-eyebrow">EXPLAINABLE AUTHENTICATION TRIAGE</span>'
        + badge("Illustrative synthetic demo", "warning", dot=True)
        + "</div>"
        '<h1 class="teqo-landing-title" id="teqo-landing-title">'
        'See beyond the <span>connection.</span></h1>'
        '<p class="teqo-landing-lead">Explore the intelligence behind every '
        "authentication signal.</p>"
        '<p class="teqo-landing-sub">A local research prototype that ranks unusual '
        "authentication activity, explains the evidence behind each ranking, and "
        "leaves every decision with an analyst.</p>"
        '<div class="teqo-landing-ctas">'
        '<a class="teqo-landing-cta primary" href="#teqo-explorer">'
        "Enter the Experience</a>"
        '<a class="teqo-landing-cta secondary" href="#teqo-network-canvas">'
        "Explore the Network</a>"
        "</div>"
        '<div class="teqo-landing-indicators">'
        '<span><i class="teqo-landing-dot" aria-hidden="true"></i>'
        "Account → device → service</span>"
        "<span>Human review only</span>"
        "<span>No automated response</span>"
        "</div></div>"
        '<div class="teqo-landing-visual">'
        '<svg viewBox="0 0 440 440" role="img" aria-label="Illustrative network of '
        'accounts, devices, and services with animated signal paths">'
        "<defs>"
        '<radialGradient id="teqo-hero-glow" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%" stop-color="#36b9d6" stop-opacity="0.22"/>'
        '<stop offset="100%" stop-color="#36b9d6" stop-opacity="0"/>'
        "</radialGradient>"
        "</defs>"
        '<circle cx="220" cy="220" r="176" fill="none" '
        'stroke="rgba(148,163,184,0.12)"/>'
        '<circle cx="220" cy="220" r="128" fill="none" '
        'stroke="rgba(148,163,184,0.1)" stroke-dasharray="2 7"/>'
        '<circle cx="220" cy="220" r="96" fill="url(#teqo-hero-glow)"/>'
        '<path class="teqo-hero-edge" d="M 86 122 L 220 56"/>'
        '<path class="teqo-hero-edge strong" d="M 220 56 L 358 150"/>'
        '<path class="teqo-hero-edge" d="M 358 150 L 318 330"/>'
        '<path class="teqo-hero-edge" d="M 318 330 L 220 382"/>'
        '<path class="teqo-hero-edge" d="M 220 382 L 92 310"/>'
        '<path class="teqo-hero-edge" d="M 92 310 L 86 122"/>'
        '<path class="teqo-hero-edge strong" d="M 92 310 L 358 150"/>'
        '<circle class="teqo-hero-particle" cx="150" cy="86" r="2"/>'
        '<circle class="teqo-hero-particle" cx="292" cy="98" r="1.6" '
        'style="animation-delay:-2.2s"/>'
        '<circle class="teqo-hero-particle" cx="340" cy="238" r="1.8" '
        'style="animation-delay:-4.1s"/>'
        '<circle class="teqo-hero-particle" cx="268" cy="356" r="1.5" '
        'style="animation-delay:-1.3s"/>'
        '<circle class="teqo-hero-particle" cx="150" cy="346" r="1.8" '
        'style="animation-delay:-3.4s"/>'
        f"{logo_image}"
        '<a class="teqo-hero-node" href="#teqo-network-canvas" '
        'aria-label="Explore the a.mensah account in the network">'
        '<circle class="teqo-hero-halo" cx="86" cy="122" r="19" fill="#8b7cf6"/>'
        '<circle class="teqo-hero-ring" cx="86" cy="122" r="9" stroke="#8b7cf6"/>'
        '<circle cx="86" cy="122" r="3" fill="#8b7cf6"/>'
        '<text class="teqo-hero-label" x="86" y="148">a.mensah</text></a>'
        '<a class="teqo-hero-node" href="#teqo-network-canvas" '
        'aria-label="Explore the DEV-UNK-7 device in the network">'
        '<circle class="teqo-hero-halo" cx="220" cy="56" r="19" fill="#f2a93b"/>'
        '<circle class="teqo-hero-ring" cx="220" cy="56" r="9" stroke="#f2a93b"/>'
        '<circle cx="220" cy="56" r="3" fill="#f2a93b"/>'
        '<text class="teqo-hero-label" x="220" y="84">DEV-UNK-7</text></a>'
        '<a class="teqo-hero-node" href="#teqo-network-canvas" '
        'aria-label="Explore the SSO-GATEWAY service in the network">'
        '<circle class="teqo-hero-halo" cx="358" cy="150" r="19" fill="#5477f5"/>'
        '<circle class="teqo-hero-ring" cx="358" cy="150" r="9" stroke="#5477f5"/>'
        '<circle cx="358" cy="150" r="3" fill="#5477f5"/>'
        '<text class="teqo-hero-label" x="358" y="176">SSO-GATEWAY</text></a>'
        '<a class="teqo-hero-node" href="#teqo-network-canvas" '
        'aria-label="Explore the VPN-HUB service in the network">'
        '<circle class="teqo-hero-halo" cx="318" cy="330" r="18" fill="#5477f5"/>'
        '<circle class="teqo-hero-ring" cx="318" cy="330" r="8" stroke="#5477f5"/>'
        '<circle cx="318" cy="330" r="3" fill="#5477f5"/>'
        '<text class="teqo-hero-label" x="318" y="356">VPN-HUB</text></a>'
        '<a class="teqo-hero-node" href="#teqo-network-canvas" '
        'aria-label="Explore the MAIL-01 service in the network">'
        '<circle class="teqo-hero-halo" cx="220" cy="382" r="18" fill="#5477f5"/>'
        '<circle class="teqo-hero-ring" cx="220" cy="382" r="8" stroke="#5477f5"/>'
        '<circle cx="220" cy="382" r="3" fill="#5477f5"/>'
        '<text class="teqo-hero-label" x="220" y="408">MAIL-01</text></a>'
        '<a class="teqo-hero-node" href="#teqo-network-canvas" '
        'aria-label="Explore the WS-021 device in the network">'
        '<circle class="teqo-hero-halo" cx="92" cy="310" r="18" fill="#36b9d6"/>'
        '<circle class="teqo-hero-ring" cx="92" cy="310" r="8" stroke="#36b9d6"/>'
        '<circle cx="92" cy="310" r="3" fill="#36b9d6"/>'
        '<text class="teqo-hero-label" x="92" y="336">WS-021</text></a>'
        "</svg></div></section>"
    )


def _event_map(event, selected_signal):
    active_edge = selected_signal["edge"] if selected_signal else None

    def edge_class(edge_id):
        classes = ["teqo-event-edge"]
        if active_edge:
            classes.append("active" if edge_id == active_edge else "dim")
        return " ".join(classes)

    device_class = "teqo-event-node unusual" if event["unusual"] else "teqo-event-node"
    indicator_color = "#f2a93b" if event["unusual"] else "#32b783"
    return (
        f'<div class="teqo-event-map">'
        f'<svg viewBox="0 0 400 190" role="img" aria-label="Illustrative '
        f'authentication sequence from {event["account"]} through '
        f'{event["device"]} to {event["service"]}">'
        '<defs><marker id="teqo-event-arrow" markerWidth="8" markerHeight="8" '
        'refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" '
        'fill="#36b9d6"/></marker></defs>'
        f'<path class="{edge_class("account-device")}" '
        'd="M 84 88 L 176 88" marker-end="url(#teqo-event-arrow)"/>'
        f'<path class="{edge_class("device-service")}" '
        'd="M 224 88 L 316 88" marker-end="url(#teqo-event-arrow)"/>'
        '<g class="teqo-event-node" style="color:#8b7cf6">'
        '<circle class="teqo-event-halo" cx="62" cy="88" r="30"/>'
        '<circle cx="62" cy="88" r="20" stroke="#8b7cf6"/>'
        '<circle class="teqo-event-core" cx="62" cy="88" r="4"/>'
        f'<text x="62" y="136">{event["account"]}</text>'
        '<text class="teqo-event-type" x="62" y="152">ACCOUNT</text></g>'
        f'<g class="{device_class}" style="color:{indicator_color}">'
        '<circle class="teqo-event-halo" cx="200" cy="88" r="30"/>'
        f'<circle cx="200" cy="88" r="20" stroke="{indicator_color}"/>'
        '<circle class="teqo-event-core" cx="200" cy="88" r="4"/>'
        f'<text x="200" y="136">{event["device"]}</text>'
        '<text class="teqo-event-type" x="200" y="152">DEVICE</text></g>'
        '<g class="teqo-event-node" style="color:#5477f5">'
        '<circle class="teqo-event-halo" cx="338" cy="88" r="30"/>'
        '<circle cx="338" cy="88" r="20" stroke="#5477f5"/>'
        '<circle class="teqo-event-core" cx="338" cy="88" r="4"/>'
        f'<text x="338" y="136">{event["service"]}</text>'
        '<text class="teqo-event-type" x="338" y="152">SERVICE</text></g>'
        "</svg></div>"
    )


def _signal_markup(event, selected_signal):
    segments = "".join(
        '<i class="on"></i>' if index < event["emphasis"] else "<i></i>"
        for index in range(5)
    )
    tone = "calm" if event["tone"] == "calm" else "warn"
    card = (
        '<div class="teqo-anomaly-card">'
        f'<span class="teqo-anomaly-badge" data-tone="{tone}">'
        f'<i aria-hidden="true"></i>{event["indicator"]}</span>'
        f'<h3>{event["account"]} → {event["device"]} → {event["service"]}</h3>'
        f"<p>{event['context']}</p>"
        f'<div class="teqo-anomaly-scale" data-tone="{tone}" '
        f'role="img" aria-label="Illustrative emphasis {event["emphasis"]} of 5. '
        'This is not a score.">' + segments + "</div>"
        '<p class="teqo-anomaly-note">Illustrative emphasis only — not a model '
        "prediction, probability, or SHAP value.</p></div>"
    )
    if selected_signal:
        card += (
            '<div class="teqo-signal-detail">'
            '<div class="teqo-signal-kicker">ILLUSTRATIVE SIGNAL RELATIONSHIP</div>'
            f"<h4>{selected_signal['label']}</h4>"
            f"<p>{selected_signal['detail']}</p>"
            '<p class="teqo-anomaly-note">Links show authentication direction only; '
            "they do not establish reachability or compromise.</p></div>"
        )
    else:
        card += (
            '<p class="teqo-signal-prompt">Select a signal above to trace the '
            "relationship it describes in this synthetic sequence.</p>"
        )
    return card


def _preview_ai():
    return (
        '<div class="teqo-disc-preview" data-accent="violet" aria-hidden="true">'
        '<svg viewBox="0 0 200 96">'
        '<path d="M40 66 L84 34 L126 58 L162 28" fill="none" '
        'stroke="rgba(139,124,246,0.6)" stroke-width="1.4" stroke-dasharray="3 4"/>'
        '<circle cx="40" cy="66" r="5" fill="#8b7cf6"/>'
        '<circle cx="84" cy="34" r="6" fill="#36b9d6"/>'
        '<circle cx="126" cy="58" r="4" fill="#8b7cf6"/>'
        '<circle cx="162" cy="28" r="5" fill="#36b9d6"/>'
        '<rect x="30" y="78" width="40" height="3" rx="1.5" fill="rgba(148,163,184,0.3)"/>'
        '<rect x="78" y="78" width="64" height="3" rx="1.5" fill="rgba(139,124,246,0.5)"/>'
        "</svg></div>"
    )


def _preview_investigation():
    return (
        '<div class="teqo-disc-preview" data-accent="warning" aria-hidden="true">'
        '<svg viewBox="0 0 200 96">'
        '<rect x="26" y="20" width="148" height="12" rx="4" fill="rgba(148,163,184,0.12)"/>'
        '<rect x="26" y="20" width="62" height="12" rx="4" fill="rgba(54,185,214,0.5)"/>'
        '<rect x="26" y="42" width="148" height="12" rx="4" fill="rgba(148,163,184,0.12)"/>'
        '<rect x="26" y="42" width="104" height="12" rx="4" fill="rgba(242,169,59,0.55)"/>'
        '<rect x="26" y="64" width="148" height="12" rx="4" fill="rgba(148,163,184,0.12)"/>'
        '<rect x="26" y="64" width="44" height="12" rx="4" fill="rgba(148,163,184,0.3)"/>'
        '<circle cx="170" cy="48" r="6" fill="none" stroke="#f2a93b" stroke-width="1.6"/>'
        '<path d="M175 53 L182 60" stroke="#f2a93b" stroke-width="1.6"/>'
        "</svg></div>"
    )


def _preview_twin():
    return (
        '<div class="teqo-disc-preview" aria-hidden="true">'
        '<svg viewBox="0 0 200 96">'
        '<path d="M36 64 L84 30 L130 52 L168 34" fill="none" '
        'stroke="rgba(54,185,214,0.45)" stroke-width="1.3"/>'
        '<path d="M84 30 L102 74" fill="none" stroke="rgba(84,119,245,0.45)" '
        'stroke-width="1.3"/>'
        '<path d="M130 52 L102 74" fill="none" stroke="rgba(139,124,246,0.45)" '
        'stroke-width="1.3"/>'
        '<circle cx="36" cy="64" r="6" fill="#8b7cf6"/>'
        '<circle cx="84" cy="30" r="5" fill="#36b9d6"/>'
        '<circle cx="130" cy="52" r="6" fill="#5477f5"/>'
        '<circle cx="168" cy="34" r="4" fill="#5477f5"/>'
        '<circle cx="102" cy="74" r="5" fill="#32b783"/>'
        "</svg></div>"
    )


def _preview_evaluation():
    return (
        '<div class="teqo-disc-preview" aria-hidden="true">'
        '<svg viewBox="0 0 200 96">'
        '<rect x="34" y="52" width="18" height="26" rx="3" fill="rgba(54,185,214,0.28)"/>'
        '<rect x="62" y="38" width="18" height="40" rx="3" fill="rgba(54,185,214,0.44)"/>'
        '<rect x="90" y="26" width="18" height="52" rx="3" fill="rgba(54,185,214,0.6)"/>'
        '<rect x="118" y="44" width="18" height="34" rx="3" fill="rgba(139,124,246,0.5)"/>'
        '<rect x="146" y="32" width="18" height="46" rx="3" fill="rgba(84,119,245,0.55)"/>'
        '<path d="M34 30 L162 18" stroke="rgba(242,169,59,0.55)" stroke-width="1.2" '
        'stroke-dasharray="3 4"/>'
        "</svg></div>"
    )


def _preview_about(logo_uri):
    logo = f'<img src="{logo_uri}" alt="" />' if logo_uri else ""
    return f'<div class="teqo-disc-preview" aria-hidden="true">{logo}</div>'


def _render_discovery(logo_uri, intelligence_page, investigation_page, twin_page,
                      evaluation_page, about_page):
    st.markdown('<div class="teqo-anchor" id="teqo-discover"></div>',
                unsafe_allow_html=True)
    st.markdown(
        section_header(
            "DISCOVER",
            "Where the evidence leads.",
            "Open a working area. Each destination keeps its own data scope and "
            "limitations.",
        ),
        unsafe_allow_html=True,
    )
    cards = [
        (
            "ai-intelligence",
            intelligence_page,
            "AI Intelligence",
            ":material/psychology:",
            "How detectors, features, and explanations fit together.",
            _preview_ai(),
        ),
        (
            "investigation",
            investigation_page,
            "Alert Investigation",
            ":material/search:",
            "Inspect evidence and record an analyst-owned decision.",
            _preview_investigation(),
        ),
        (
            "digital-twin",
            twin_page,
            "Cyber Digital Twin",
            ":material/hub:",
            "Explore observed authentication relationships.",
            _preview_twin(),
        ),
        (
            "model-evaluation",
            evaluation_page,
            "Model Evaluation",
            ":material/analytics:",
            "Review chronological holdout evidence and limits.",
            _preview_evaluation(),
        ),
        (
            "about",
            about_page,
            "About TeqoAI",
            ":material/info:",
            "Project context, scope, and stated limitations.",
            _preview_about(logo_uri),
        ),
    ]
    columns = st.columns(5, gap="small")
    for column, (slug, page, title, icon, note, preview) in zip(columns, cards):
        with column:
            with st.container(key=f"teqo-disc-{slug}"):
                st.markdown(preview, unsafe_allow_html=True)
                st.page_link(page, label=title, icon=icon)
                st.markdown(f'<p class="teqo-disc-note">{note}</p>',
                            unsafe_allow_html=True)


def render_home(intelligence_page, investigation_page, twin_page, evaluation_page,
                about_page):
    logo_uri = _logo_data_uri()

    st.markdown(_hero_markup(logo_uri), unsafe_allow_html=True)

    st.markdown('<div class="teqo-anchor" id="teqo-explorer"></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<section class="teqo-landing-section">'
        + section_header(
            "INTERACTIVE NETWORK EXPLORER",
            "Follow the signal, not just the alert.",
            "Select an account, device, or service. Click a relationship to inspect "
            "direction and illustrative observation counts. Search, filter by type, "
            "zoom, pan, focus, and reset without leaving the page.",
        )
        + "</section>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="teqo-landing-band">'
        '<div class="teqo-landing-legend" aria-label="Entity types">'
        '<span><i style="background:#8b7cf6"></i>Account</span>'
        '<span><i style="background:#36b9d6"></i>Device</span>'
        '<span><i style="background:#5477f5"></i>Service</span>'
        "</div>"
        '<span class="teqo-landing-legend-note">'
        "Illustrative synthetic activity · not live monitoring</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="teqo-anchor" id="teqo-network-canvas"></div>',
                unsafe_allow_html=True)

    node_details, edge_details = _explorer_payload()
    args = {
        "nodes": EXPLORER_NODES,
        "edges": EXPLORER_EDGES,
        "positions": EXPLORER_POSITIONS,
        "path_nodes": [],
        "path_edges": [],
        "matches": [],
        "nonce": 1,
        "fit_token": 0,
        "mode": "presentation",
        "emit": False,
        "node_details": node_details,
        "edge_details": edge_details,
    }
    with st.container(key="teqo-landing-explorer"):
        teqo_graph(args=args, key="teqo-home-graph", default=None)
    st.markdown(
        '<p class="teqo-landing-note">Generated demonstration graph. Entity names '
        "and observation counts are invented for the homepage and are not loaded "
        "customer data. A relationship shows authentication direction only; it does "
        "not prove reachability, authorization, or compromise.</p>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="teqo-anchor" id="teqo-signals"></div>',
                unsafe_allow_html=True)
    st.markdown(
        '<section class="teqo-landing-section">'
        + section_header(
            "ILLUSTRATIVE ANOMALY EXPLORATION",
            "What changed in the pattern?",
            "Switch between two hand-built synthetic events, then select a "
            "contributing signal to trace the relationship it describes.",
        )
        + "</section>",
        unsafe_allow_html=True,
    )

    event_labels = ["Unusual example", "Typical example"]
    event_choice = st.segmented_control(
        "Illustrative event",
        event_labels,
        default="Unusual example",
        key="teqo-home-event",
    )
    event_key = "typical" if event_choice == "Typical example" else "unusual"
    event = EVENTS[event_key]

    signal_options = [signal["label"] for signal in event["signals"]]
    selected_label = st.pills(
        "Trace a contributing signal",
        signal_options,
        selection_mode="single",
        key=f"teqo-signal-{event_key}",
    )
    selected_signal = next(
        (signal for signal in event["signals"] if signal["label"] == selected_label),
        None,
    )

    map_column, panel_column = st.columns([1.1, 0.9], gap="large")
    with map_column:
        st.markdown(_event_map(event, selected_signal), unsafe_allow_html=True)
        st.markdown(
            '<p class="teqo-landing-note">Illustrative event sequence. Direction '
            "shows which entity authenticated to which; it does not establish "
            "reachability or compromise.</p>",
            unsafe_allow_html=True,
        )
    with panel_column:
        st.markdown(_signal_markup(event, selected_signal), unsafe_allow_html=True)

    st.markdown(
        '<p class="teqo-landing-note">Clearly labelled demonstration: event values, '
        "signal emphasis, and sequences are synthetic and hand-set. They are not "
        "model predictions, SHAP values, alert records, or evidence of malicious "
        "activity.</p>",
        unsafe_allow_html=True,
    )

    _render_discovery(
        logo_uri,
        intelligence_page,
        investigation_page,
        twin_page,
        evaluation_page,
        about_page,
    )

    st.markdown(
        '<div class="teqo-landing-footer">'
        "<span>TEQOAI · See beyond the connection</span>"
        "<span>Illustrative synthetic homepage · Analyst review only</span>"
        "</div>",
        unsafe_allow_html=True,
    )
