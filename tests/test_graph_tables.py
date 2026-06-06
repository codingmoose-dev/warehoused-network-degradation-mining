from __future__ import annotations

from pathlib import Path

import pandas as pd

from network_degradation_mining import graph_builder


def _warehouse_frame() -> pd.DataFrame:
    row_count = 24
    index_values = list(range(row_count))
    return pd.DataFrame(
        {
            "measurement_id": [
                f"measurement_{index:03d}" for index in index_values
            ],
            "session_id": [f"session_{index // 6:02d}" for index in index_values],
            "session_step_index": [index % 6 for index in index_values],
            "time_of_day_bin": [
                "Evening" if index % 3 == 0 else "Morning"
                for index in index_values
            ],
            "area_type": [
                "Urban" if index % 2 == 0 else "Rural"
                for index in index_values
            ],
            "network_type": [
                "4G" if index % 4 == 0 else "5G" for index in index_values
            ],
            "infrastructure_profile": [
                "macro" if index % 2 == 0 else "small_cell"
                for index in index_values
            ],
            "band": [
                "LTE Anchor" if index % 4 == 0 else "n78"
                for index in index_values
            ],
            "app_type": [
                "Streaming" if index % 5 == 0 else "Browsing"
                for index in index_values
            ],
            "movement_speed": [
                "Driving" if index % 3 == 0 else "Static"
                for index in index_values
            ],
            "weather": [
                "Rain" if index % 7 == 0 else "Clear" for index in index_values
            ],
            "obstruction_level": [
                "High" if index % 6 == 0 else "Low" for index in index_values
            ],
            "congestion_level": [
                "High" if index % 5 == 0 else "Low" for index in index_values
            ],
            "tower_load": [
                "High" if index % 5 == 0 else "Low" for index in index_values
            ],
            "service_degraded": [
                1 if index % 5 == 0 or index % 7 == 0 else 0
                for index in index_values
            ],
            "high_latency": [1 if index % 5 == 0 else 0 for index in index_values],
            "high_jitter": [1 if index % 7 == 0 else 0 for index in index_values],
        }
    )


def _write_external_reference_inputs(
    external_reference_dir: Path,
    external_results_dir: Path,
) -> None:
    external_reference_dir.mkdir(parents=True, exist_ok=True)
    external_results_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "source_dataset": ["vienna_4g5g", "vienna_4g5g"],
            "technology": ["5g", "5g"],
            "measurement_source": ["phone", "phone"],
            "measurement_context": ["drive_test", "drive_test"],
            "application_context": ["Download", "Download"],
            "mobility_context": ["mobile_measurement", "mobile_measurement"],
            "signal_dbm": [-84.0, -88.0],
            "download_throughput_mbps": [110.0, 120.0],
        }
    ).to_csv(external_reference_dir / "vienna_reference_clean.csv", index=False)

    pd.DataFrame(
        {
            "source_dataset": ["campus_qos", "campus_qos"],
            "technology": ["5g", "5g"],
            "measurement_source": ["controlled_testbed", "controlled_testbed"],
            "measurement_context": ["controlled_testbed", "controlled_testbed"],
            "application_context": ["Download", "Download"],
            "mobility_context": ["Static", "Static"],
            "jitter_ms": [1.0, 2.0],
            "packet_loss_fraction": [0.0, 0.01],
            "offered_downlink_mbps": [50.0, 60.0],
        }
    ).to_csv(external_reference_dir / "campus_qos_reference_clean.csv", index=False)

    pd.DataFrame(
        {
            "source_dataset": ["synnetqos", "vienna_4g5g", "campus_qos"],
            "row_count": [24, 2, 2],
            "signal_dbm_non_missing_count": [24, 2, 0],
            "signal_dbm_non_missing_fraction": [1.0, 1.0, 0.0],
            "download_throughput_mbps_non_missing_count": [24, 2, 0],
            "download_throughput_mbps_non_missing_fraction": [1.0, 1.0, 0.0],
            "jitter_ms_non_missing_count": [24, 0, 2],
            "jitter_ms_non_missing_fraction": [1.0, 0.0, 1.0],
            "packet_loss_fraction_non_missing_count": [24, 0, 2],
            "packet_loss_fraction_non_missing_fraction": [1.0, 0.0, 1.0],
            "offered_downlink_mbps_non_missing_count": [24, 0, 2],
            "offered_downlink_mbps_non_missing_fraction": [1.0, 0.0, 1.0],
        }
    ).to_csv(external_results_dir / "external_reference_coverage.csv", index=False)

    pd.DataFrame(
        {
            "source_dataset": ["vienna_4g5g", "campus_qos"],
            "metric": ["signal_dbm", "jitter_ms"],
            "comparison_status": ["available", "available"],
            "recommended_use": ["main_text_candidate", "main_text_candidate"],
            "claim_scope": [
                "selected-variable plausibility",
                "selected-variable plausibility",
            ],
        }
    ).to_csv(external_results_dir / "external_metric_readiness.csv", index=False)

    pd.DataFrame(
        {
            "metric": ["signal_dbm", "jitter_ms"],
            "reference_dataset": ["vienna_4g5g", "campus_qos"],
            "comparison_context": ["unfiltered", "unfiltered"],
            "recommended_use": ["main_text_candidate", "main_text_candidate"],
            "comparison_scope": ["radio", "qos"],
            "median_difference": [1.2, 0.4],
            "wasserstein_distance": [2.1, 0.5],
            "ks_statistic": [0.2, 0.1],
        }
    ).to_csv(
        external_results_dir / "external_pairwise_distance_summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "measurement_context": ["drive_test", "controlled_testbed"],
            "application_context": ["Download", "Download"],
            "mobility_context": ["mobile_measurement", "Static"],
        }
    ).to_csv(external_results_dir / "external_context_coverage.csv", index=False)

    pd.DataFrame({"summary_row": ["available"]}).to_csv(
        external_results_dir / "external_summary.csv",
        index=False,
    )


def test_external_context_edges_use_source_dataset_key() -> None:
    clean_tables = {
        "vienna_4g5g": pd.DataFrame(
            {
                "measurement_context": ["drive_test", "drive_test"],
                "mobility_context": ["mobile_measurement", "mobile_measurement"],
            }
        )
    }

    edges = graph_builder._build_external_context_edges(clean_tables)

    assert not edges.empty
    assert set(edges["source_dataset"]) == {"vienna_4g5g"}
    assert edges["graph_view"].eq("external_reference_evidence").all()


def test_graph_tables_build_three_views_without_missing_endpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        graph_builder,
        "build_joined_warehouse_frame",
        lambda warehouse_dir: _warehouse_frame(),
    )
    monkeypatch.setattr(
        graph_builder,
        "add_degradation_labels",
        lambda dataframe: dataframe,
    )

    external_reference_dir = tmp_path / "external_reference"
    external_results_dir = tmp_path / "external_results"
    _write_external_reference_inputs(external_reference_dir, external_results_dir)

    output_paths = graph_builder.build_graph_tables(
        warehouse_dir=tmp_path / "warehouse",
        graph_nodes_dir=tmp_path / "graph" / "nodes",
        graph_edges_dir=tmp_path / "graph" / "edges",
        results_dir=tmp_path / "results" / "graph",
        external_reference_dir=external_reference_dir,
        external_results_dir=external_results_dir,
    )

    nodes = pd.read_csv(output_paths["all_nodes"])
    edges = pd.read_csv(output_paths["all_edges"])
    audit = pd.read_csv(output_paths["graph_build_audit"])

    assert set(nodes["graph_view"]) == {
        "warehouse_measurement",
        "context_cooccurrence",
        "external_reference_evidence",
    }
    assert set(edges["graph_view"]) == {
        "warehouse_measurement",
        "context_cooccurrence",
        "external_reference_evidence",
    }
    node_ids = set(nodes["node_id"].astype(str))
    assert set(edges["source_node_id"].astype(str)).issubset(node_ids)
    assert set(edges["target_node_id"].astype(str)).issubset(node_ids)
    assert audit["status"].ne("failed").all()
