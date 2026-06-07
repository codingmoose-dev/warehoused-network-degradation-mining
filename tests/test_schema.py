from __future__ import annotations

from network_degradation_mining import schema


def test_dimension_specs_have_unique_table_and_key_names() -> None:
    table_names = [spec.table_name for spec in schema.DIMENSIONS]
    id_columns = [spec.id_column for spec in schema.DIMENSIONS]

    assert len(table_names) == len(set(table_names))
    assert len(id_columns) == len(set(id_columns))
    assert "fact_network_measurement" in schema.WAREHOUSE_TABLE_ORDER


def test_fact_column_groups_do_not_overlap() -> None:
    identifiers = set(schema.FACT_IDENTIFIER_COLUMNS)
    measures = set(schema.FACT_MEASURE_COLUMNS)

    assert identifiers.isdisjoint(measures)
    assert "measurement_id" in identifiers
    assert "signal_strength_dbm" in measures
