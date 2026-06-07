from __future__ import annotations

from network_degradation_mining.features import load_warehouse_tables
from network_degradation_mining.schema import DIMENSIONS


def test_warehouse_fact_and_dimensions_have_stable_keys(sample_warehouse_dir) -> None:
    tables = load_warehouse_tables(sample_warehouse_dir)
    fact = tables["fact"]

    assert len(fact) == 6
    assert fact["measurement_id"].is_unique

    for spec in DIMENSIONS:
        dimension = tables[spec.table_name.removeprefix("dim_")]
        assert spec.id_column in dimension.columns
        assert dimension[spec.id_column].is_unique
        assert spec.id_column in fact.columns
        assert fact[spec.id_column].notna().all()
