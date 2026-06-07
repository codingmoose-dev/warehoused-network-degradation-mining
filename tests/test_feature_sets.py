from __future__ import annotations

from network_degradation_mining.features import (
    build_association_transaction_table,
    build_clustering_feature_table,
    build_joined_warehouse_frame,
    build_supervised_degradation_table,
)


def test_mining_feature_tables_have_expected_primary_columns(
    sample_warehouse_dir,
) -> None:
    joined = build_joined_warehouse_frame(sample_warehouse_dir)

    supervised = build_supervised_degradation_table(joined)
    clustering = build_clustering_feature_table(joined)
    transactions = build_association_transaction_table(joined)

    assert "service_degraded" in supervised.columns
    assert "service_degraded" in clustering.columns
    assert "items" in transactions.columns
    assert transactions["items"].str.contains("service_degraded=").all()
