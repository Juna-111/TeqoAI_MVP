"""Cyber Digital Twin: interactive observed-authentication graph workspace."""

import html
import os

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from components import (
    badge,
    empty_state,
    kv_rows,
    section_header,
)
from src.digital_twin import get_attack_paths

GRAPH_COMPONENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph_view"
)
teqo_graph = components.declare_component(
    "teqo_observed_graph", path=GRAPH_COMPONENT_DIR
)

GRAPH_HEIGHT = 640
GRAPH_PADDING = 55

TYPE_LABELS = {
    "workstation": "Workstation",
    "server": "Server",
    "database": "Database",
    "application": "Application",
    "other": "Other asset",
}

CRITICALITY_LABELS = {
    0: "0 · user identity",
    1: "1 · workstation",
    2: "2 · server",
    3: "3 · application",
    4: "4 · database",
}


def _type_label(asset_type):
    return TYPE_LABELS.get(asset_type, str(asset_type).replace("_", " ").title())


def _successful(clean_events):
    if "result" not in clean_events:
        return clean_events.iloc[0:0]
    return clean_events.loc[
        clean_events["result"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("success")
        .fillna(False)
    ]


@st.cache_data(show_spinner=False)
def _layout(node_ids, edge_pairs, height=GRAPH_HEIGHT):
    """Deterministic spring layout normalised into the component viewport."""
    layout_graph = nx.DiGraph()
    layout_graph.add_nodes_from(node_ids)
    layout_graph.add_edges_from(edge_pairs)
    raw = nx.spring_layout(layout_graph, seed=42, k=1.05, iterations=90)
    if not raw:
        return {}
    xs = [float(point[0]) for point in raw.values()]
    ys = [float(point[1]) for point in raw.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    horizontal_span = 1000 - GRAPH_PADDING * 2
    vertical_span = max(height - GRAPH_PADDING * 2, 120)
    return {
        node_id: [
            round(
                GRAPH_PADDING + (float(point[0]) - min_x) / span_x * horizontal_span,
                2,
            ),
            round(
                GRAPH_PADDING + (float(point[1]) - min_y) / span_y * vertical_span,
                2,
            ),
        ]
        for node_id, point in raw.items()
    }


def _reset_selection():
    st.session_state["twin-selected"] = None
    st.session_state["twin-edge"] = None
    st.session_state["twin-nonce"] = st.session_state.get("twin-nonce", 0) + 1


def _clear_filters(all_types, default_source):
    st.session_state["twin-search"] = ""
    st.session_state["twin-types"] = list(all_types)
    st.session_state["twin-hops"] = 4
    st.session_state["twin-path-source"] = default_source
    _reset_selection()


def _fit_request():
    st.session_state["twin-fit"] = st.session_state.get("twin-fit", 0) + 1


def _use_selected_as_start():
    if st.session_state.get("twin-selected"):
        st.session_state["twin-path-source"] = st.session_state["twin-selected"]


def _workspace_header(
    visible_nodes,
    total_nodes,
    visible_edges,
    total_edges,
    route_count,
    selected,
):
    stats = [
        f'<span class="teqo-twin-stat"><b>{visible_nodes:,}</b> of '
        f"{total_nodes:,} entities</span>",
        f'<span class="teqo-twin-stat"><b>{visible_edges:,}</b> of '
        f"{total_edges:,} relationships</span>",
        f'<span class="teqo-twin-stat"><b>{route_count:,}</b> bounded '
        f"route{'' if route_count == 1 else 's'}</span>",
    ]
    if selected:
        stats.append(
            '<span class="teqo-twin-stat is-selected">Selected: '
            f"<b>{html.escape(str(selected))}</b></span>"
        )
    return (
        '<div class="teqo-twin-head">'
        '<div class="teqo-twin-head-copy">'
        '<div class="teqo-section-kicker">GRAPH WORKSPACE</div>'
        "<h2>Observed authentication map</h2>"
        "</div>"
        f'<div class="teqo-twin-stats">{"".join(stats)}</div>'
        "</div>"
    )


def _edge_context(edge_events):
    if edge_events.empty:
        return "No successful observations remain for this relationship."
    methods = (
        edge_events["action"].astype(str).value_counts().head(3).index.tolist()
        if "action" in edge_events
        else []
    )
    accounts = (
        edge_events["user"].astype(str).nunique() if "user" in edge_events else 0
    )
    latest = (
        pd.to_numeric(edge_events["time"], errors="coerce").max()
        if "time" in edge_events
        else float("nan")
    )
    method_text = ", ".join(methods) if methods else "unavailable"
    time_text = (
        f"{int(latest):,} elapsed seconds" if pd.notna(latest) else "unavailable"
    )
    return (
        f"{len(edge_events):,} successful observations · {accounts:,} account"
        f"{'' if accounts == 1 else 's'} · methods: {method_text} · "
        f"latest: {time_text}"
    )


def _selected_entity_panel(graph, selected, clean_success, queue):
    node = graph.nodes[selected]
    involving = clean_success.loc[
        clean_success["device"].eq(selected)
        | clean_success["target_device"].eq(selected)
    ]
    incoming_edges = [
        (source, target)
        for source, target in graph.in_edges(selected)
    ]
    outgoing_edges = [
        (source, target)
        for source, target in graph.out_edges(selected)
    ]
    outgoing_weight = sum(
        int(graph[selected][target].get("weight", 1))
        for target in graph.successors(selected)
    )
    incoming_weight = sum(
        int(graph[source][selected].get("weight", 1))
        for source in graph.predecessors(selected)
    )
    criticality = node.get("criticality")
    criticality_text = CRITICALITY_LABELS.get(
        criticality, str(criticality) if criticality is not None else "unavailable"
    )
    accounts = (
        int(involving["user"].astype(str).nunique()) if "user" in involving else 0
    )

    st.markdown(
        section_header(
            "ENTITY INTELLIGENCE",
            str(selected),
            "Values are computed from the loaded successful authentication sample.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        badge(_type_label(node.get("asset_type", "other")), "info")
        + " "
        + badge(f"{len(involving):,} observed events", "neutral"),
        unsafe_allow_html=True,
    )
    if st.session_state.get("twin-path-source") != selected:
        st.button(
            "Trace paths from this entity",
            key=f"twin-trace-{selected}",
            on_click=_use_selected_as_start,
            width="stretch",
        )
    st.markdown(
        kv_rows(
            [
                ("Entity type", _type_label(node.get("asset_type", "other"))),
                ("Configured criticality", criticality_text),
                ("Incoming relationships", f"{len(incoming_edges):,}"),
                ("Outgoing relationships", f"{len(outgoing_edges):,}"),
                (
                    "Observed successes as source",
                    f"{outgoing_weight:,}",
                ),
                (
                    "Observed successes as destination",
                    f"{incoming_weight:,}",
                ),
                ("Distinct accounts observed", f"{accounts:,}"),
            ]
        ),
        unsafe_allow_html=True,
    )

    node_risk = queue.loc[
        queue["device"].astype(str).eq(selected)
        | queue["target_device"].astype(str).eq(selected)
    ] if not queue.empty else queue
    risk_parts = []
    for level in ("High", "Medium", "Low"):
        count = int((node_risk["risk_level"] == level).sum())
        if count:
            risk_parts.append(f"{count:,} {level.lower()}")
    if len(node_risk):
        max_score = pd.to_numeric(node_risk["risk_score"], errors="coerce").max()
        st.markdown(
            '<p class="teqo-panel-note">Risk-scored events in the loaded sample: '
            f"{', '.join(risk_parts)}. Highest risk score: "
            f"{max_score:.0f}.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="teqo-panel-note">No risk-scored events in the loaded sample '
            "reference this entity. Risk data may be filtered out or unavailable."
            "</p>",
            unsafe_allow_html=True,
        )


def _entity_relationships(graph, selected):
    relationship_rows = []
    for source, target in graph.out_edges(selected):
        weight = graph[source][target].get("weight", 1)
        relationship_rows.append(
            {
                "Source": source,
                "Destination": target,
                "Direction": "Outgoing",
                "Successes": int(weight),
            }
        )
    for source, target in graph.in_edges(selected):
        weight = graph[source][target].get("weight", 1)
        relationship_rows.append(
            {
                "Source": source,
                "Destination": target,
                "Direction": "Incoming",
                "Successes": int(weight),
            }
        )
    relationship_rows.sort(
        key=lambda row: (-row["Successes"], row["Source"], row["Destination"])
    )
    st.markdown("**Observed relationships**")
    if relationship_rows:
        relationship_markup = "".join(
            '<div class="teqo-path-row">'
            f"<strong>{html.escape(str(row['Source']))} → "
            f"{html.escape(str(row['Destination']))}</strong>"
            f"<span>{row['Direction']} · {row['Successes']:,} successful "
            f"observation{'' if row['Successes'] == 1 else 's'}</span>"
            "</div>"
            for row in relationship_rows[:8]
        )
        st.markdown(
            f'<div class="teqo-path-list">{relationship_markup}</div>',
            unsafe_allow_html=True,
        )
        if len(relationship_rows) > 8:
            st.caption(
                f"Showing the 8 strongest of {len(relationship_rows):,} observed "
                "relationships. Select a line in the graph to inspect one."
            )
    else:
        st.caption("No observed relationships touch this entity.")


def _entity_activity(selected, clean_success):
    involving = clean_success.loc[
        clean_success["device"].eq(selected)
        | clean_success["target_device"].eq(selected)
    ]
    recent = involving.copy()
    if "time" in recent:
        recent = recent.sort_values("time", ascending=False)
    st.markdown("**Recent related activity**")
    if recent.empty:
        st.caption("No related authenticated activity was found for this entity.")
        return
    if "device" in recent and "target_device" in recent:
        recent["peer"] = recent.apply(
            lambda event: (
                event["target_device"]
                if str(event["device"]) == str(selected)
                else event["device"]
            ),
            axis=1,
        )
    recent_markup = []
    for _, event in recent.head(6).iterrows():
        details = []
        if "time" in event and pd.notna(event["time"]):
            details.append(f"t={int(event['time']):,}")
        if "user" in event:
            details.append(str(event["user"]))
        if "action" in event:
            details.append(str(event["action"]))
        if "result" in event:
            details.append(str(event["result"]))
        peer = event["peer"] if "peer" in event else "related event"
        recent_markup.append(
            '<div class="teqo-path-row">'
            f"<strong>{html.escape(str(peer))}</strong>"
            f"<span>{html.escape(' · '.join(details))}</span>"
            "</div>"
        )
    st.markdown(
        f'<div class="teqo-path-list">{"".join(recent_markup)}</div>',
        unsafe_allow_html=True,
    )


def _valid_visible_edge(edge, visible_set, visible_graph):
    """Return a normalised edge tuple only if it exists in the visible graph."""
    if not isinstance(edge, (list, tuple)) or len(edge) != 2:
        return None
    source, target = edge
    if not isinstance(source, str) or not isinstance(target, str):
        return None
    if source not in visible_set or target not in visible_set:
        return None
    if not visible_graph.has_edge(source, target):
        return None
    return (source, target)


def _relationship_panel(graph, selected_edge, clean_success):
    if not selected_edge:
        st.markdown(
            '<p class="teqo-panel-note">Select a relationship line in the graph to '
            "inspect its direction, observation count, and authentication context."
            "</p>",
            unsafe_allow_html=True,
        )
        return
    source, target = selected_edge
    weight = (
        graph[source][target].get("weight", 0)
        if graph.has_edge(source, target)
        else 0
    )
    st.markdown("**Selected observed relationship**")
    st.markdown(
        kv_rows(
            [
                ("Source entity", source),
                ("Destination entity", target),
                ("Direction", f"{source} → {target}"),
                ("Successful observations", f"{int(weight):,}"),
            ]
        ),
        unsafe_allow_html=True,
    )
    edge_events = clean_success.loc[
        clean_success["device"].astype(str).eq(str(source))
        & clean_success["target_device"].astype(str).eq(str(target))
    ]
    st.caption(_edge_context(edge_events))


def _paths_panel(graph, path_source, max_hops):
    if path_source not in graph:
        st.caption("Select a starting entity that exists in the observed graph.")
        return []
    paths = get_attack_paths(graph, path_source, max_hops=max_hops)
    if paths:
        path_rows = "".join(
            '<div class="teqo-path-row">'
            f"<strong>{html.escape(str(path['target']))}</strong>"
            "<span>"
            f"{html.escape(_type_label(path['target_type']))} · "
            f"{path['hops']} hop{'' if path['hops'] == 1 else 's'}"
            "</span>"
            f"<code>{html.escape(' → '.join(path['path']))}</code>"
            "</div>"
            for path in paths[:8]
        )
        st.markdown(
            f'<div class="teqo-path-list">{path_rows}</div>',
            unsafe_allow_html=True,
        )
        if len(paths) > 8:
            st.caption(f"Showing 8 of {len(paths):,} observed destinations.")
    else:
        st.markdown(
            empty_state(
                "No onward connections",
                f"No onward connection was recorded from {path_source} within "
                f"{max_hops} hop{'' if max_hops == 1 else 's'}.",
            ),
            unsafe_allow_html=True,
        )
    return paths


def render_twin(graph, queue, clean_events):
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

    clean_success = _successful(clean_events)
    all_nodes = sorted(graph.nodes(), key=str)
    all_types = sorted(
        {graph.nodes[node].get("asset_type", "other") for node in all_nodes}
    )
    node_types = {
        node: graph.nodes[node].get("asset_type", "other") for node in all_nodes
    }

    default_source = (
        str(queue.iloc[0]["device"])
        if not queue.empty and str(queue.iloc[0]["device"]) in graph
        else all_nodes[0]
    )
    for key, value in (
        ("twin-selected", None),
        ("twin-edge", None),
        ("twin-nonce", 0),
        ("twin-fit", 0),
        ("twin-search", ""),
        ("twin-types", list(all_types)),
        ("twin-path-source", default_source),
        ("twin-hops", 4),
    ):
        st.session_state.setdefault(key, value)

    if st.session_state["twin-path-source"] not in graph:
        st.session_state["twin-path-source"] = default_source
    if (
        st.session_state["twin-selected"] is not None
        and st.session_state["twin-selected"] not in graph
    ):
        st.session_state["twin-selected"] = None
        st.session_state["twin-edge"] = None

    selected_types = st.session_state["twin-types"]
    visible_nodes = [
        node
        for node in all_nodes
        if not selected_types or node_types[node] in selected_types
    ]
    visible_set = set(visible_nodes)
    visible_graph = graph.subgraph(visible_nodes).copy()
    query = st.session_state["twin-search"].strip().lower()
    matches = [node for node in visible_nodes if query and query in str(node).lower()]

    selected = st.session_state["twin-selected"]
    if selected and selected not in visible_set:
        st.session_state["twin-selected"] = None
        st.session_state["twin-edge"] = None
        selected = None
    selected_edge = st.session_state.get("twin-edge")
    if selected_edge and not visible_graph.has_edge(*selected_edge):
        st.session_state["twin-edge"] = None
        selected_edge = None

    path_source = st.session_state["twin-path-source"]
    max_hops = int(st.session_state["twin-hops"])
    paths = (
        get_attack_paths(visible_graph, path_source, max_hops=max_hops)
        if path_source in visible_graph
        else []
    )
    highlighted_nodes = {path_source} if path_source in visible_graph else set()
    highlighted_edges = set()
    for path in paths[:8]:
        highlighted_nodes.update(path["path"])
        highlighted_edges.update(zip(path["path"][:-1], path["path"][1:]))

    st.session_state.setdefault("twin-focus", False)
    focus_mode = bool(st.session_state.get("twin-focus"))

    with st.container(key="twin-workspace"):
        st.markdown(
            _workspace_header(
                len(visible_nodes),
                len(all_nodes),
                visible_graph.number_of_edges(),
                graph.number_of_edges(),
                len(paths),
                selected,
            ),
            unsafe_allow_html=True,
        )

        search_column, filters_column, actions_column = st.columns(
            [2.5, 1.0, 2.55], gap="small", vertical_alignment="bottom"
        )
        with search_column:
            st.text_input(
                "Search entities",
                key="twin-search",
                placeholder="Account, workstation, server, or service",
                help="Matching entities are highlighted; the graph is not reduced.",
            )
        with filters_column:
            with st.popover("Filters & paths", width="stretch"):
                st.markdown("**Visible entities**")
                st.multiselect(
                    "Entity types",
                    options=all_types,
                    key="twin-types",
                    format_func=_type_label,
                    help=(
                        "Filters operate on the observed graph. Clearing every type "
                        "hides all entities."
                    ),
                )
                st.markdown("**Path exploration**")
                st.selectbox(
                    "Starting entity",
                    options=all_nodes,
                    key="twin-path-source",
                    help="Bounded observed paths are calculated from this entity.",
                )
                st.slider(
                    "Maximum path length",
                    min_value=1,
                    max_value=6,
                    key="twin-hops",
                    help="Hops are counted along observed authentication relationships.",
                )
                st.button(
                    "Use selected entity as start",
                    key="twin-use-selected",
                    on_click=_use_selected_as_start,
                    width="stretch",
                    disabled=not selected,
                )
                st.button(
                    "Clear filters",
                    key="twin-clear",
                    on_click=_clear_filters,
                    args=(all_types, default_source),
                    width="stretch",
                )
        with actions_column:
            fit_column, reset_column, focus_column = st.columns(
                [1, 1, 1.35], gap="small", vertical_alignment="bottom"
            )
            with fit_column:
                st.button(
                    "Fit graph",
                    key="twin-fit-button",
                    on_click=_fit_request,
                    width="stretch",
                    help="Re-center the whole observed graph in the canvas.",
                )
            with reset_column:
                st.button(
                    "Reset",
                    key="twin-reset-selection",
                    on_click=_reset_selection,
                    width="stretch",
                    disabled=not selected and not selected_edge,
                    help="Clear the current node or relationship selection.",
                )
            with focus_column:
                st.toggle(
                    "Focus graph",
                    key="twin-focus",
                    help=(
                        "Hide the intelligence dock so the graph uses the full "
                        "workspace width."
                    ),
                )

        if focus_mode:
            graph_column = st.container()
            details_column = None
        else:
            graph_column, details_column = st.columns(
                [3.2, 1.25], gap="medium", vertical_alignment="top"
            )

        with graph_column:
            source_text = (
                html.escape(str(path_source))
                if path_source in visible_graph
                else "no entity"
            )
            st.markdown(
                f'<p class="teqo-panel-note">Path exploration starts at '
                f"<strong>{source_text}</strong> · {len(paths):,} bounded "
                f"route{'' if len(paths) == 1 else 's'} · drag to pan, scroll to "
                "zoom</p>",
                unsafe_allow_html=True,
            )
            if not visible_nodes:
                st.markdown(
                    empty_state(
                        "No entities match the current filters",
                        "Select at least one entity type or clear the filters to "
                        "restore the observed graph.",
                    ),
                    unsafe_allow_html=True,
                )
                event = None
            else:
                node_ids = tuple(visible_nodes)
                edge_pairs = tuple(
                    sorted(
                        (str(source), str(target))
                        for source, target in visible_graph.edges()
                    )
                )
                positions = _layout(node_ids, edge_pairs, GRAPH_HEIGHT)
                args = {
                    "nodes": [
                        {
                            "id": node,
                            "type": node_types[node],
                            "criticality": graph.nodes[node].get("criticality"),
                        }
                        for node in visible_nodes
                    ],
                    "edges": [
                        {
                            "source": source,
                            "target": target,
                            "weight": int(
                                visible_graph[source][target].get("weight", 1)
                            ),
                        }
                        for source, target in visible_graph.edges()
                    ],
                    "positions": positions,
                    "selected": selected,
                    "selected_edge": list(selected_edge) if selected_edge else None,
                    "path_nodes": sorted(highlighted_nodes, key=str),
                    "path_edges": [list(edge) for edge in sorted(highlighted_edges)],
                    "matches": matches,
                    "nonce": st.session_state["twin-nonce"],
                    "fit_token": st.session_state["twin-fit"],
                    "height": GRAPH_HEIGHT,
                }
                event = teqo_graph(args=args, key="twin-graph", default=None)
            st.caption(
                "Built only from successful authentication observations; a link is not "
                "proof of reachability, authorization, or a confirmed attack path. "
                "Highlighted routes follow observed successes and are investigation "
                "context, not confirmed attack paths."
            )
            if focus_mode:
                st.caption(
                    "Focus mode hides the intelligence dock. Turn off Focus graph "
                    "to inspect the selected entity, relationship, and routes."
                )

        if (
            isinstance(event, dict)
            and event.get("nonce") == st.session_state["twin-nonce"]
        ):
            event_node = event.get("node")
            node_is_valid = (
                isinstance(event_node, str) and event_node in visible_set
            )
            event_edge = _valid_visible_edge(
                event.get("edge"), visible_set, visible_graph
            )
            if node_is_valid or event_edge is not None:
                current_node = st.session_state["twin-selected"]
                current_edge = st.session_state.get("twin-edge")
                if event_edge is not None:
                    # A relationship can be inspected without a selected node.
                    # When a node is selected, keep it alongside the new edge.
                    new_node = (
                        current_node
                        if current_node in visible_set
                        else event_node if node_is_valid else None
                    )
                    new_edge = event_edge
                else:
                    new_node = event_node
                    new_edge = None
                if new_node != current_node or new_edge != current_edge:
                    st.session_state["twin-selected"] = new_node
                    st.session_state["twin-edge"] = new_edge
                    st.rerun()

        if details_column is not None:
            with details_column:
                with st.container(key="twin-dock"):
                    selected = st.session_state["twin-selected"]
                    selected_edge = st.session_state.get("twin-edge")
                    if selected:
                        _selected_entity_panel(
                            graph, selected, clean_success, queue
                        )
                    elif selected_edge:
                        st.markdown(
                            section_header(
                                "RELATIONSHIP INTELLIGENCE",
                                "Relationship selected",
                                "The selected connection is shown below, including "
                                "its direction and observation count. Select a node "
                                "to inspect that entity instead.",
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            section_header(
                                "INTELLIGENCE DOCK",
                                "No entity selected",
                                "Select any node in the graph to inspect its observed "
                                "relationships and related authentication activity.",
                            ),
                            unsafe_allow_html=True,
                        )
                    _relationship_panel(graph, selected_edge, clean_success)

        if selected and not focus_mode:
            with st.container(key="twin-evidence"):
                relationship_column, activity_column = st.columns(
                    [1, 1], gap="medium", vertical_alignment="top"
                )
                with relationship_column:
                    _entity_relationships(visible_graph, selected)
                with activity_column:
                    _entity_activity(selected, clean_success)

        st.markdown(
            section_header(
                "OBSERVED ROUTES",
                "Onward observed paths",
                f"Shortest recorded routes from {path_source}, bounded to "
                f"{max_hops} hop{'' if max_hops == 1 else 's'}. "
                "High-criticality destinations first.",
            ),
            unsafe_allow_html=True,
        )
        with st.container(key="twin-routes"):
            _paths_panel(visible_graph, path_source, max_hops)

    st.caption(
        "Repeated successful observations increase relationship weight. Entity type, "
        "criticality, and counts come from the loaded graph and event data; unavailable "
        "values are shown as unavailable rather than estimated."
    )