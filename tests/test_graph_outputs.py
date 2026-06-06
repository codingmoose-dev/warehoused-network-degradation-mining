from __future__ import annotations

from pathlib import Path

import pandas as pd

from network_degradation_mining.gnn import run_graph_neural_benchmark
from network_degradation_mining.graph_mining import run_graph_mining
from network_degradation_mining.io import write_csv


def _write_small_graph(root: Path) -> tuple[Path, Path, Path]:
    nodes_dir = root / "nodes"
    edges_dir = root / "edges"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    edges_dir.mkdir(parents=True, exist_ok=True)

    nodes: list[dict[str, object]] = []
    for session_index in range(4):
        for step_index in range(6):
            measurement_id = f"m{session_index}_{step_index}"
            nodes.append(
                {
                    "node_id": f"measurement:{measurement_id}",
                    "node_type": "measurement",
                    "graph_view": "warehouse_measurement",
                    "node_label": measurement_id,
                    "measurement_id": measurement_id,
                    "session_id": f"session_{session_index}",
                    "service_degraded": int((session_index + step_index) % 2 == 0),
                }
            )

    for value in ["4G", "5G"]:
        nodes.append(
            {
                "node_id": f"context:network:{value}",
                "node_type": "network_type",
                "graph_view": "warehouse_measurement",
                "node_label": f"network_type={value}",
            }
        )
    for value in ["High", "Low"]:
        nodes.append(
            {
                "node_id": f"context:congestion:{value}",
                "node_type": "congestion",
                "graph_view": "warehouse_measurement",
                "node_label": f"congestion={value}",
            }
        )
    for index, value in enumerate(
        ["network_type=4G", "congestion=High", "obstruction=High"]
    ):
        nodes.append(
            {
                "node_id": f"context_cooccurrence:{index}",
                "node_type": "context_value",
                "graph_view": "context_cooccurrence",
                "node_label": value,
                "context_family": value.split("=")[0],
            }
        )
    nodes.extend(
        [
            {
                "node_id": "external:source:synnetqos",
                "node_type": "source",
                "graph_view": "external_reference_evidence",
                "node_label": "SynNetQoS",
            },
            {
                "node_id": "external:metric:latency",
                "node_type": "metric",
                "graph_view": "external_reference_evidence",
                "node_label": "Latency",
            },
        ]
    )

    edges: list[dict[str, object]] = []
    for session_index in range(4):
        for step_index in range(6):
            measurement_id = f"m{session_index}_{step_index}"
            measurement_node = f"measurement:{measurement_id}"
            network_value = "4G" if step_index % 2 == 0 else "5G"
            congestion_value = (
                "High" if (session_index + step_index) % 2 == 0 else "Low"
            )
            degraded = int((session_index + step_index) % 2 == 0)
            next_degraded = int((session_index + step_index + 1) % 2 == 0)
            edges.extend(
                [
                    {
                        "edge_id": f"edge:network:{measurement_id}",
                        "source_node_id": measurement_node,
                        "target_node_id": f"context:network:{network_value}",
                        "edge_type": "uses_network_type",
                        "graph_view": "warehouse_measurement",
                        "edge_weight": 1,
                    },
                    {
                        "edge_id": f"edge:congestion:{measurement_id}",
                        "source_node_id": measurement_node,
                        "target_node_id": f"context:congestion:{congestion_value}",
                        "edge_type": "has_congestion_level",
                        "graph_view": "warehouse_measurement",
                        "edge_weight": 1,
                    },
                ]
            )
            if step_index < 5:
                edges.append(
                    {
                        "edge_id": f"edge:transition:{measurement_id}",
                        "source_node_id": measurement_node,
                        "target_node_id": (
                            f"measurement:m{session_index}_{step_index + 1}"
                        ),
                        "edge_type": "next_measurement",
                        "graph_view": "warehouse_measurement",
                        "edge_weight": 1,
                        "source_service_degraded": degraded,
                        "target_service_degraded": next_degraded,
                    }
                )
    edges.extend(
        [
            {
                "edge_id": "context_pair_1",
                "source_node_id": "context_cooccurrence:0",
                "target_node_id": "context_cooccurrence:1",
                "edge_type": "context_cooccurs",
                "graph_view": "context_cooccurrence",
                "edge_weight": 100,
                "row_count": 100,
                "degraded_count": 80,
                "degradation_rate": 0.80,
                "baseline_degradation_rate": 0.50,
                "degradation_lift": 1.60,
                "support_fraction": 0.20,
            },
            {
                "edge_id": "context_pair_2",
                "source_node_id": "context_cooccurrence:0",
                "target_node_id": "context_cooccurrence:2",
                "edge_type": "context_cooccurs",
                "graph_view": "context_cooccurrence",
                "edge_weight": 90,
                "row_count": 90,
                "degraded_count": 70,
                "degradation_rate": 0.78,
                "baseline_degradation_rate": 0.50,
                "degradation_lift": 1.56,
                "support_fraction": 0.18,
            },
            {
                "edge_id": "external_metric_1",
                "source_node_id": "external:source:synnetqos",
                "target_node_id": "external:metric:latency",
                "edge_type": "source_provides_metric",
                "graph_view": "external_reference_evidence",
                "edge_weight": 1,
                "source_dataset": "synnetqos",
                "metric": "latency_ms",
                "non_missing_count": 24,
                "recommended_use": "main_text_candidate",
            },
        ]
    )

    nodes_path = write_csv(pd.DataFrame(nodes), nodes_dir / "all_nodes.csv")
    edges_path = write_csv(pd.DataFrame(edges), edges_dir / "all_edges.csv")
    supervised_path = write_csv(
        pd.DataFrame(
            [
                {
                    "model_name": "supervised_baseline",
                    "accuracy": 0.70,
                    "balanced_accuracy": 0.70,
                    "precision": 0.70,
                    "recall": 0.70,
                    "f1": 0.70,
                    "roc_auc": 0.70,
                }
            ]
        ),
        root / "supervised_model_comparison.csv",
    )
    return nodes_path, edges_path, supervised_path


def test_graph_outputs_are_written(tmp_path: Path) -> None:
    nodes_path, edges_path, _ = _write_small_graph(tmp_path)

    output_paths = run_graph_mining(
        graph_nodes_path=nodes_path,
        graph_edges_path=edges_path,
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
    )

    assert "graph_view_report_table" in output_paths
    assert "top_nodes_by_degree_by_view" in output_paths
    assert "context_pair_evidence_matrix" in output_paths
    assert "context_pair_evidence_matrix_figure" in output_paths
    assert output_paths["context_pair_evidence_matrix"].exists()


def test_true_gnn_benchmark_is_audited(tmp_path: Path) -> None:
    nodes_path, edges_path, supervised_path = _write_small_graph(tmp_path)

    output_paths = run_graph_neural_benchmark(
        graph_nodes_path=nodes_path,
        graph_edges_path=edges_path,
        results_dir=tmp_path / "results",
        figures_dir=tmp_path / "figures",
        supervised_model_comparison_path=supervised_path,
    )

    assert "gnn_model_comparison" in output_paths
    assert "true_gnn_audit" in output_paths
    audit = pd.read_csv(output_paths["true_gnn_audit"])
    assert "true_gnn_external_reference_exclusion" in set(audit["check_name"])
