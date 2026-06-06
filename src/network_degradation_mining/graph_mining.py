from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd

from network_degradation_mining.io import ensure_directory, read_csv, write_csv
from network_degradation_mining.plotting import (
    plot_context_degradation_graph,
    plot_context_pair_evidence_matrix,
    plot_external_reference_graph,
    plot_graph_schema,
    plot_top_degradation_context_pair_lift,
)

TOP_EDGE_LIMIT = 100
TOP_NODE_LIMIT = 100
TOP_COMMUNITY_LIMIT = 50
MAX_COMMUNITY_ITERATIONS = 10

WAREHOUSE_VIEW = "warehouse_measurement"
CONTEXT_VIEW = "context_cooccurrence"
EXTERNAL_VIEW = "external_reference_evidence"


def _empty_table(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _clean_value(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"
    text = str(value).strip()
    return text if text else "<missing>"


def _node_type_map(nodes: pd.DataFrame) -> dict[str, str]:
    if nodes.empty or {"node_id", "node_type"}.difference(nodes.columns):
        return {}
    return dict(zip(nodes["node_id"].astype(str), nodes["node_type"].astype(str)))


def _node_label_map(nodes: pd.DataFrame) -> dict[str, str]:
    if nodes.empty or "node_id" not in nodes.columns:
        return {}
    label_column = "node_label" if "node_label" in nodes.columns else "node_id"
    return dict(zip(nodes["node_id"].astype(str), nodes[label_column].astype(str)))


def _view_summary(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    views = sorted(
        set(nodes.get("graph_view", pd.Series(dtype=str)).dropna().astype(str))
        | set(edges.get("graph_view", pd.Series(dtype=str)).dropna().astype(str))
    )
    for view in views:
        node_view = nodes.get("graph_view", pd.Series()).astype(str)
        edge_view = edges.get("graph_view", pd.Series()).astype(str)
        node_count = int(node_view.eq(view).sum())
        edge_count = int(edge_view.eq(view).sum())
        rows.append(
            {
                "graph_view": view,
                "node_count": node_count,
                "edge_count": edge_count,
            }
        )
    return pd.DataFrame(rows)


def _node_type_summary(nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty:
        return _empty_table(["graph_view", "node_type", "node_count"])
    return (
        nodes.groupby(["graph_view", "node_type"], dropna=False)
        .size()
        .reset_index(name="node_count")
        .sort_values(["graph_view", "node_count"], ascending=[True, False])
    )


def _edge_type_summary(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return _empty_table(["graph_view", "edge_type", "edge_count"])
    return (
        edges.groupby(["graph_view", "edge_type"], dropna=False)
        .size()
        .reset_index(name="edge_count")
        .sort_values(["graph_view", "edge_count"], ascending=[True, False])
    )


def _graph_mining_scope_audit(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policies = [
        (
            WAREHOUSE_VIEW,
            "warehouse_schema_and_transition_analysis",
            "prediction_eligible",
        ),
        (
            CONTEXT_VIEW,
            "context_cooccurrence_degradation_signature_mining",
            "prediction_feature_support",
        ),
        (
            EXTERNAL_VIEW,
            "external_reference_evidence_organization",
            "reference_only",
        ),
    ]
    node_views = nodes.get("graph_view", pd.Series(dtype=str)).astype(str)
    edge_views = edges.get("graph_view", pd.Series(dtype=str)).astype(str)
    for graph_view, mining_role, prediction_role in policies:
        rows.append(
            {
                "graph_view": graph_view,
                "mining_role": mining_role,
                "prediction_role": prediction_role,
                "node_count": int(node_views.eq(graph_view).sum()),
                "edge_count": int(edge_views.eq(graph_view).sum()),
                "status": "ok" if node_views.eq(graph_view).any() else "review",
            }
        )
    return pd.DataFrame(rows)


def _build_adjacency(
    edges: pd.DataFrame,
    weight_column: str = "edge_weight",
) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    if edges.empty:
        return adjacency
    for row in edges.itertuples(index=False):
        data = row._asdict()
        source = str(data.get("source_node_id"))
        target = str(data.get("target_node_id"))
        weight = pd.to_numeric(data.get(weight_column, 1.0), errors="coerce")
        weight_value = float(weight) if pd.notna(weight) else 1.0
        adjacency[source][target] = adjacency[source].get(target, 0.0) + weight_value
        adjacency[target][source] = adjacency[target].get(source, 0.0) + weight_value
    return adjacency


def _connected_components(adjacency: dict[str, dict[str, float]]) -> dict[str, int]:
    component_ids: dict[str, int] = {}
    component_index = 0
    for node_id in adjacency:
        if node_id in component_ids:
            continue
        queue: deque[str] = deque([node_id])
        component_ids[node_id] = component_index
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, {}):
                if neighbor in component_ids:
                    continue
                component_ids[neighbor] = component_index
                queue.append(neighbor)
        component_index += 1
    return component_ids


def _weighted_label_propagation(
    adjacency: dict[str, dict[str, float]],
) -> dict[str, str]:
    labels = {node_id: node_id for node_id in adjacency}
    ordered_nodes = sorted(
        adjacency,
        key=lambda node_id: (-sum(adjacency[node_id].values()), node_id),
    )
    for _ in range(MAX_COMMUNITY_ITERATIONS):
        changed = 0
        for node_id in ordered_nodes:
            neighbors = adjacency.get(node_id, {})
            if not neighbors:
                continue
            scores: defaultdict[str, float] = defaultdict(float)
            for neighbor, weight in neighbors.items():
                scores[labels[neighbor]] += weight
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            best_label = ranked[0][0]
            if labels[node_id] != best_label:
                labels[node_id] = best_label
                changed += 1
        if changed == 0:
            break
    compact_ids = {
        label: f"community_{index + 1:04d}"
        for index, label in enumerate(sorted(set(labels.values())))
    }
    return {node_id: compact_ids[label] for node_id, label in labels.items()}


def _degree_summary(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty:
        return _empty_table(
            [
                "node_id",
                "graph_view",
                "node_type",
                "degree_total",
                "weighted_degree",
            ]
        )
    node_rows = nodes[["node_id", "graph_view", "node_type"]].copy()
    edge_values = edges.copy()
    if edge_values.empty:
        node_rows["degree_total"] = 0
        node_rows["weighted_degree"] = 0.0
        return node_rows

    edge_values["edge_weight_numeric"] = pd.to_numeric(
        edge_values.get("edge_weight", 1.0), errors="coerce"
    ).fillna(1.0)
    degree_count = pd.concat(
        [
            edge_values["source_node_id"].astype(str),
            edge_values["target_node_id"].astype(str),
        ],
        ignore_index=True,
    ).value_counts()
    weighted: Counter[str] = Counter()
    for row in edge_values.itertuples(index=False):
        data = row._asdict()
        weight = float(data.get("edge_weight_numeric", 1.0))
        weighted[str(data["source_node_id"])] += weight
        weighted[str(data["target_node_id"])] += weight

    node_rows["degree_total"] = (
        node_rows["node_id"].astype(str).map(degree_count).fillna(0).astype(int)
    )
    node_rows["weighted_degree"] = (
        node_rows["node_id"].astype(str).map(weighted).fillna(0.0).astype(float)
    )
    return node_rows.sort_values("weighted_degree", ascending=False)



def _top_nodes_by_degree_by_view(
    degree_summary: pd.DataFrame,
    top_per_view: int = 25,
) -> pd.DataFrame:
    if degree_summary.empty:
        return _empty_table(
            [
                "graph_view",
                "node_id",
                "node_type",
                "degree_total",
                "weighted_degree",
                "view_rank",
            ]
        )
    output = degree_summary.copy()
    output["weighted_degree"] = pd.to_numeric(
        output.get("weighted_degree", 0.0),
        errors="coerce",
    ).fillna(0.0)
    output["degree_total"] = pd.to_numeric(
        output.get("degree_total", 0),
        errors="coerce",
    ).fillna(0).astype(int)
    ranked = []
    for _graph_view, group in output.groupby("graph_view", dropna=False):
        view_rows = group.sort_values(
            ["weighted_degree", "degree_total"],
            ascending=False,
        ).head(top_per_view).copy()
        view_rows.insert(0, "view_rank", range(1, len(view_rows) + 1))
        ranked.append(view_rows)
    if not ranked:
        return pd.DataFrame()
    return pd.concat(ranked, ignore_index=True, sort=False)


def _graph_view_report_table(view_summary: pd.DataFrame) -> pd.DataFrame:
    roles = {
        WAREHOUSE_VIEW: (
            "Warehouse record, session, context, and transition structure",
            "Eligible for warehouse-only prediction and transition analysis",
            "Main text table",
        ),
        CONTEXT_VIEW: (
            "Compact operating-context co-occurrence graph for degradation signatures",
            "Supports graph mining and graph-derived feature construction",
            "Main text table and primary graph-mining evidence",
        ),
        EXTERNAL_VIEW: (
            "Reference-source, metric, context, and comparison-evidence graph",
            "Reference-only; excluded from supervised prediction",
            "Main text table or supplement depending on page budget",
        ),
    }
    if view_summary.empty:
        return _empty_table(
            [
                "graph_view",
                "node_count",
                "edge_count",
                "scientific_role",
                "prediction_role",
                "recommended_reporting_use",
            ]
        )
    rows = []
    for row in view_summary.itertuples(index=False):
        data = row._asdict()
        graph_view = str(data.get("graph_view"))
        scientific_role, prediction_role, reporting_use = roles.get(
            graph_view,
            ("Unspecified graph view", "Review before prediction use", "Supplement"),
        )
        rows.append(
            {
                "graph_view": graph_view,
                "node_count": int(data.get("node_count", 0)),
                "edge_count": int(data.get("edge_count", 0)),
                "scientific_role": scientific_role,
                "prediction_role": prediction_role,
                "recommended_reporting_use": reporting_use,
            }
        )
    return pd.DataFrame(rows)


def _compact_context_label(label: Any) -> str:
    text = _clean_value(label)
    replacements = {
        "network_type=": "Network: ",
        "radio_band=": "Band: ",
        "application=": "App: ",
        "mobility=": "Mobility: ",
        "weather=": "Weather: ",
        "obstruction=": "Obstruction: ",
        "congestion=": "Congestion: ",
        "tower_load=": "Tower load: ",
        "time_bin=": "Time: ",
        "area_type=": "Area: ",
        "infrastructure=": "Infrastructure: ",
    }
    for source, target in replacements.items():
        if text.startswith(source):
            return target + text.removeprefix(source).replace("_", " ")
    return text.replace("_", " ")


def _context_pair_evidence_matrix(
    high_context_edges: pd.DataFrame,
    top_n: int = 12,
) -> pd.DataFrame:
    columns = [
        "main_text_rank",
        "context_pair_label",
        "source_label",
        "target_label",
        "row_count",
        "support_fraction",
        "degraded_count",
        "degradation_rate",
        "baseline_degradation_rate",
        "degradation_lift",
    ]
    if high_context_edges.empty:
        return _empty_table(columns)
    required = {"source_label", "target_label", "row_count", "degradation_lift"}
    if not required.issubset(high_context_edges.columns):
        return _empty_table(columns)

    output = high_context_edges.copy()
    for column in [
        "row_count",
        "support_fraction",
        "degraded_count",
        "degradation_rate",
        "baseline_degradation_rate",
        "degradation_lift",
    ]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.sort_values(
        ["degradation_lift", "degradation_rate", "row_count"],
        ascending=False,
    ).head(top_n)
    output["source_label"] = output["source_label"].map(_compact_context_label)
    output["target_label"] = output["target_label"].map(_compact_context_label)
    output["context_pair_label"] = (
        output["source_label"].astype(str) + " + " + output["target_label"].astype(str)
    )
    output.insert(0, "main_text_rank", range(1, len(output) + 1))
    available_columns = [
        column for column in columns if column in output.columns
    ]
    return output[available_columns].reset_index(drop=True)



def _context_edges(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty or "graph_view" not in edges.columns:
        return pd.DataFrame()
    return edges.loc[edges["graph_view"].astype(str).eq(CONTEXT_VIEW)].copy()


def _context_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty or "graph_view" not in nodes.columns:
        return pd.DataFrame()
    return nodes.loc[nodes["graph_view"].astype(str).eq(CONTEXT_VIEW)].copy()


def _context_edge_summary(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    top_limit: int | None = None,
) -> pd.DataFrame:
    context_edges = _context_edges(edges)
    if context_edges.empty:
        return _empty_table(
            [
                "source_label",
                "target_label",
                "row_count",
                "degradation_rate",
                "degradation_lift",
            ]
        )
    labels = _node_label_map(nodes)
    output = context_edges.copy()
    output["source_label"] = output["source_node_id"].astype(str).map(labels)
    output["target_label"] = output["target_node_id"].astype(str).map(labels)
    numeric_columns = [
        "row_count",
        "support_fraction",
        "degraded_count",
        "degradation_rate",
        "baseline_degradation_rate",
        "degradation_lift",
    ]
    for column in numeric_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    sort_columns = ["degradation_lift", "degradation_rate", "row_count"]
    available = [column for column in sort_columns if column in output.columns]
    ranked = output.sort_values(available, ascending=False)
    if top_limit is not None:
        return ranked.head(top_limit)
    return ranked


def _context_node_summary(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    context_nodes = _context_nodes(nodes)
    context_edges = _context_edges(edges)
    if context_nodes.empty:
        return _empty_table(
            [
                "node_id",
                "node_label",
                "context_family",
                "row_count",
                "incident_edge_count",
                "incident_weight",
                "max_edge_degradation_lift",
            ]
        )

    incident_count: Counter[str] = Counter()
    incident_weight: Counter[str] = Counter()
    max_lift: defaultdict[str, float] = defaultdict(float)
    for row in context_edges.itertuples(index=False):
        data = row._asdict()
        weight = pd.to_numeric(data.get("row_count", 1), errors="coerce")
        lift = pd.to_numeric(data.get("degradation_lift", 0), errors="coerce")
        weight_value = float(weight) if pd.notna(weight) else 1.0
        lift_value = float(lift) if pd.notna(lift) else 0.0
        for column in ("source_node_id", "target_node_id"):
            node_id = str(data[column])
            incident_count[node_id] += 1
            incident_weight[node_id] += weight_value
            max_lift[node_id] = max(max_lift[node_id], lift_value)

    output = context_nodes.copy()
    output["incident_edge_count"] = (
        output["node_id"].astype(str).map(incident_count).fillna(0).astype(int)
    )
    output["incident_weight"] = (
        output["node_id"].astype(str).map(incident_weight).fillna(0.0).astype(float)
    )
    output["max_edge_degradation_lift"] = (
        output["node_id"].astype(str).map(max_lift).fillna(0.0).astype(float)
    )
    return output.sort_values(
        ["max_edge_degradation_lift", "incident_weight"],
        ascending=False,
    ).head(TOP_NODE_LIMIT)


def _context_communities(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    context_nodes = _context_nodes(nodes)
    context_edges = _context_edges(edges)
    if context_nodes.empty:
        return _empty_table(
            [
                "community_id",
                "node_count",
                "edge_count",
                "mean_degradation_rate",
                "max_degradation_lift",
                "top_context_values",
            ]
        )
    adjacency = _build_adjacency(context_edges, weight_column="row_count")
    for node_id in context_nodes["node_id"].astype(str):
        adjacency.setdefault(node_id, {})
    communities = _weighted_label_propagation(adjacency)
    node_frame = context_nodes[["node_id", "node_label"]].copy()
    node_frame["community_id"] = node_frame["node_id"].astype(str).map(communities)

    edge_frame = context_edges.copy()
    edge_frame["source_community"] = edge_frame["source_node_id"].astype(str).map(
        communities
    )
    edge_frame["target_community"] = edge_frame["target_node_id"].astype(str).map(
        communities
    )
    same_community = edge_frame["source_community"].eq(
        edge_frame["target_community"]
    )
    intra_edges = edge_frame.loc[same_community]

    rows = []
    for community_id, group in node_frame.groupby("community_id", dropna=False):
        community_edges = intra_edges.loc[
            intra_edges["source_community"].eq(community_id)
        ]
        rates = pd.to_numeric(
            community_edges.get("degradation_rate", pd.Series(dtype=float)),
            errors="coerce",
        )
        lifts = pd.to_numeric(
            community_edges.get("degradation_lift", pd.Series(dtype=float)),
            errors="coerce",
        )
        node_labels = sorted(group["node_label"].astype(str).tolist())[:8]
        rows.append(
            {
                "community_id": community_id,
                "node_count": int(len(group)),
                "edge_count": int(len(community_edges)),
                "mean_degradation_rate": (
                    float(rates.mean()) if not rates.empty else 0.0
                ),
                "max_degradation_lift": float(lifts.max()) if not lifts.empty else 0.0,
                "top_context_values": "; ".join(node_labels),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["max_degradation_lift", "edge_count"], ascending=False)
        .head(TOP_COMMUNITY_LIMIT)
    )


def _high_degradation_context_edges(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    output = summary.copy()
    if "row_count" in output.columns:
        minimum_count = max(25, int(output["row_count"].quantile(0.25)))
        output = output.loc[output["row_count"] >= minimum_count]
    if "degradation_lift" in output.columns:
        output = output.loc[output["degradation_lift"] >= 1.0]
    return output.head(TOP_EDGE_LIMIT)


def _warehouse_transition_summary(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return _empty_table(
            [
                "source_service_degraded",
                "target_service_degraded",
                "transition_count",
                "transition_fraction",
            ]
        )
    transition_edges = edges.loc[
        edges.get("edge_type", pd.Series(dtype=str)).astype(str).eq("next_measurement")
    ].copy()
    if transition_edges.empty:
        return _empty_table(
            [
                "source_service_degraded",
                "target_service_degraded",
                "transition_count",
                "transition_fraction",
            ]
        )
    grouped = (
        transition_edges.groupby(
            ["source_service_degraded", "target_service_degraded"],
            dropna=False,
        )
        .size()
        .reset_index(name="transition_count")
    )
    total = grouped["transition_count"].sum()
    grouped["transition_fraction"] = grouped["transition_count"] / total if total else 0
    return grouped.sort_values("transition_count", ascending=False)


def _external_edges(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty or "graph_view" not in edges.columns:
        return pd.DataFrame()
    return edges.loc[edges["graph_view"].astype(str).eq(EXTERNAL_VIEW)].copy()


def _external_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty or "graph_view" not in nodes.columns:
        return pd.DataFrame()
    return nodes.loc[nodes["graph_view"].astype(str).eq(EXTERNAL_VIEW)].copy()


def _external_evidence_summary(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
) -> pd.DataFrame:
    external_nodes = _external_nodes(nodes)
    external_edges = _external_edges(edges)
    rows = [
        {"metric": "external_node_count", "value": int(len(external_nodes))},
        {"metric": "external_edge_count", "value": int(len(external_edges))},
        {
            "metric": "source_node_count",
            "value": int(external_nodes["node_type"].astype(str).eq("source").sum())
            if not external_nodes.empty
            else 0,
        },
        {
            "metric": "metric_node_count",
            "value": int(external_nodes["node_type"].astype(str).eq("metric").sum())
            if not external_nodes.empty
            else 0,
        },
        {
            "metric": "comparison_node_count",
            "value": int(
                external_nodes["node_type"].astype(str).eq("comparison").sum()
            )
            if not external_nodes.empty
            else 0,
        },
    ]
    return pd.DataFrame(rows)


def _external_metric_support(edges: pd.DataFrame) -> pd.DataFrame:
    external_edges = _external_edges(edges)
    metric_edges = external_edges.loc[
        external_edges.get("edge_type", pd.Series(dtype=str))
        .astype(str)
        .eq("source_provides_metric")
    ].copy()
    if metric_edges.empty:
        return _empty_table(
            [
                "metric",
                "source_count",
                "total_non_missing_count",
                "main_text_candidate_count",
            ]
        )
    metric_edges["non_missing_count"] = pd.to_numeric(
        metric_edges["non_missing_count"], errors="coerce"
    ).fillna(0)
    metric_edges["is_main_text_candidate"] = metric_edges.get(
        "recommended_use",
        pd.Series(dtype=str),
    ).astype(str).eq("main_text_candidate")
    return (
        metric_edges.groupby("metric", dropna=False)
        .agg(
            source_count=("source_dataset", "nunique"),
            total_non_missing_count=("non_missing_count", "sum"),
            main_text_candidate_count=("is_main_text_candidate", "sum"),
        )
        .reset_index()
        .sort_values(["main_text_candidate_count", "source_count"], ascending=False)
    )


def _external_comparison_edges(edges: pd.DataFrame) -> pd.DataFrame:
    external_edges = _external_edges(edges)
    comparison_edges = external_edges.loc[
        external_edges.get("edge_type", pd.Series(dtype=str))
        .astype(str)
        .isin(
            {
                "synnetqos_has_comparison",
                "reference_has_comparison",
                "comparison_uses_metric",
            }
        )
    ].copy()
    if comparison_edges.empty:
        return comparison_edges
    for column in ("median_difference", "wasserstein_distance", "ks_statistic"):
        if column in comparison_edges.columns:
            comparison_edges[column] = pd.to_numeric(
                comparison_edges[column], errors="coerce"
            )
    return comparison_edges.sort_values(
        ["recommended_use", "metric", "reference_dataset"],
        ascending=[True, True, True],
    )



def run_graph_mining(
    graph_nodes_path: str | Path,
    graph_edges_path: str | Path,
    results_dir: str | Path,
    figures_dir: str | Path,
) -> dict[str, Path]:
    results_path = ensure_directory(results_dir)
    figures_path = ensure_directory(figures_dir)

    nodes = read_csv(graph_nodes_path, low_memory=False)
    edges = read_csv(graph_edges_path, low_memory=False)

    view_summary = _view_summary(nodes, edges)
    node_type_summary = _node_type_summary(nodes)
    edge_type_summary = _edge_type_summary(edges)
    graph_mining_scope_audit = _graph_mining_scope_audit(nodes, edges)
    degree_summary = _degree_summary(nodes, edges)
    top_nodes_by_degree_by_view = _top_nodes_by_degree_by_view(degree_summary)
    graph_view_report_table = _graph_view_report_table(view_summary)

    context_edge_summary = _context_edge_summary(nodes, edges)
    context_edge_summary_top = context_edge_summary.head(TOP_EDGE_LIMIT)
    context_node_summary = _context_node_summary(nodes, edges)
    context_communities = _context_communities(nodes, edges)
    high_context_edges = _high_degradation_context_edges(context_edge_summary)
    context_pair_evidence_matrix = _context_pair_evidence_matrix(high_context_edges)
    transition_summary = _warehouse_transition_summary(edges)

    external_summary = _external_evidence_summary(nodes, edges)
    external_metric_support = _external_metric_support(edges)
    external_comparison_edges = _external_comparison_edges(edges)

    output_paths: dict[str, Path] = {
        "graph_view_summary": write_csv(
            view_summary,
            results_path / "graph_view_summary.csv",
        ),
        "node_type_summary": write_csv(
            node_type_summary,
            results_path / "node_type_summary.csv",
        ),
        "edge_type_summary": write_csv(
            edge_type_summary,
            results_path / "edge_type_summary.csv",
        ),
        "degree_summary": write_csv(
            degree_summary,
            results_path / "degree_summary.csv",
        ),
        "graph_mining_scope_audit": write_csv(
            graph_mining_scope_audit,
            results_path / "graph_mining_scope_audit.csv",
        ),
        "top_nodes_by_degree": write_csv(
            degree_summary.head(TOP_NODE_LIMIT),
            results_path / "top_nodes_by_degree.csv",
        ),
        "top_nodes_by_degree_by_view": write_csv(
            top_nodes_by_degree_by_view,
            results_path / "top_nodes_by_degree_by_view.csv",
        ),
        "graph_view_report_table": write_csv(
            graph_view_report_table,
            results_path / "graph_view_report_table.csv",
        ),
        "context_edge_summary": write_csv(
            context_edge_summary,
            results_path / "context_edge_summary.csv",
        ),
        "context_node_summary": write_csv(
            context_node_summary,
            results_path / "context_node_summary.csv",
        ),
        "context_community_profiles": write_csv(
            context_communities,
            results_path / "context_community_profiles.csv",
        ),
        "high_degradation_context_edges": write_csv(
            high_context_edges,
            results_path / "high_degradation_context_edges.csv",
        ),
        "context_pair_evidence_matrix": write_csv(
            context_pair_evidence_matrix,
            results_path / "context_pair_evidence_matrix.csv",
        ),
        "warehouse_transition_summary": write_csv(
            transition_summary,
            results_path / "warehouse_transition_summary.csv",
        ),
        "external_evidence_summary": write_csv(
            external_summary,
            results_path / "external_evidence_summary.csv",
        ),
        "external_metric_support_summary": write_csv(
            external_metric_support,
            results_path / "external_metric_support_summary.csv",
        ),
        "external_comparison_edge_summary": write_csv(
            external_comparison_edges,
            results_path / "external_comparison_edge_summary.csv",
        ),
    }

    schema_figure = plot_graph_schema(figures_path / "graph_schema.pdf")
    output_paths["graph_schema_figure"] = schema_figure

    context_figure = plot_context_degradation_graph(
        _context_nodes(nodes),
        context_edge_summary_top,
        figures_path / "context_degradation_graph.pdf",
    )
    if context_figure is not None:
        output_paths["context_degradation_graph"] = context_figure

    context_pair_evidence_figure = plot_context_pair_evidence_matrix(
        context_pair_evidence_matrix,
        figures_path / "context_pair_evidence_matrix.pdf",
    )
    if context_pair_evidence_figure is not None:
        output_paths["context_pair_evidence_matrix_figure"] = (
            context_pair_evidence_figure
        )

    top_context_lift_figure = plot_top_degradation_context_pair_lift(
        context_pair_evidence_matrix,
        figures_path / "top_degradation_context_pair_lift.pdf",
    )
    if top_context_lift_figure is not None:
        output_paths["top_degradation_context_pair_lift_figure"] = (
            top_context_lift_figure
        )

    external_figure = plot_external_reference_graph(
        nodes,
        edges,
        figures_path / "external_reference_graph.pdf",
    )
    if external_figure is not None:
        output_paths["external_reference_graph"] = external_figure

    return output_paths
