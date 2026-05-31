from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from network_degradation_mining.io import ensure_directory, read_csv, write_csv
from network_degradation_mining.schema import (
    DIMENSIONS,
    FACT_IDENTIFIER_COLUMNS,
    FACT_MEASURE_COLUMNS,
    DimensionSpec,
)

MISSING_TOKEN = "<missing>"


def _available_columns(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in dataframe.columns]


def _normalize_key_value(value: Any) -> str:
    if pd.isna(value):
        return MISSING_TOKEN
    return str(value).strip()


def _hash_key(prefix: str, values: tuple[Any, ...], length: int = 12) -> str:
    payload = "||".join(_normalize_key_value(value) for value in values)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def _dimension_prefix(id_column: str) -> str:
    return id_column.removesuffix("_id")


def _time_of_day_bin(hour: Any) -> str:
    if pd.isna(hour):
        return MISSING_TOKEN

    try:
        hour_int = int(hour)
    except (TypeError, ValueError):
        return MISSING_TOKEN

    if 0 <= hour_int <= 5:
        return "night"
    if 6 <= hour_int <= 11:
        return "morning"
    if 12 <= hour_int <= 17:
        return "afternoon"
    if 18 <= hour_int <= 23:
        return "evening"

    return MISSING_TOKEN


def prepare_warehouse_source(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()

    if "timestamp" in output.columns:
        output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")

    if "hour" not in output.columns and "timestamp" in output.columns:
        output["hour"] = output["timestamp"].dt.hour

    if "hour" in output.columns:
        output["hour"] = pd.to_numeric(output["hour"], errors="coerce").astype("Int64")
        output["time_of_day_bin"] = output["hour"].map(_time_of_day_bin)
    else:
        output["time_of_day_bin"] = MISSING_TOKEN

    return output


def build_dimension(dataframe: pd.DataFrame, spec: DimensionSpec) -> pd.DataFrame:
    source_columns = _available_columns(dataframe, spec.source_columns)

    if not source_columns:
        return pd.DataFrame(columns=[spec.id_column])

    dimension = dataframe[source_columns].drop_duplicates().copy()
    dimension = dimension.sort_values(source_columns, kind="mergesort").reset_index(
        drop=True
    )

    prefix = _dimension_prefix(spec.id_column)
    dimension.insert(
        0,
        spec.id_column,
        [
            _hash_key(
                prefix=prefix, values=tuple(row[column] for column in source_columns)
            )
            for _, row in dimension.iterrows()
        ],
    )

    return dimension


def _build_dimension_lookup(
    dimension: pd.DataFrame, spec: DimensionSpec
) -> dict[tuple[Any, ...], str]:
    source_columns = [
        column for column in spec.source_columns if column in dimension.columns
    ]

    if not source_columns:
        return {}

    lookup: dict[tuple[Any, ...], str] = {}

    for _, row in dimension.iterrows():
        key = tuple(_normalize_key_value(row[column]) for column in source_columns)
        lookup[key] = str(row[spec.id_column])

    return lookup


def _attach_dimension_id(
    fact: pd.DataFrame,
    source: pd.DataFrame,
    dimension: pd.DataFrame,
    spec: DimensionSpec,
) -> pd.DataFrame:
    source_columns = _available_columns(source, spec.source_columns)

    if not source_columns:
        fact[spec.id_column] = pd.NA
        return fact

    lookup = _build_dimension_lookup(dimension, spec)

    keys = source[source_columns].apply(
        lambda row: tuple(
            _normalize_key_value(row[column]) for column in source_columns
        ),
        axis=1,
    )

    fact[spec.id_column] = keys.map(lookup).astype("string")
    return fact


def build_fact_table(
    dataframe: pd.DataFrame, dimensions: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    identifier_columns = _available_columns(dataframe, FACT_IDENTIFIER_COLUMNS)
    measure_columns = _available_columns(dataframe, FACT_MEASURE_COLUMNS)

    fact = dataframe[identifier_columns + measure_columns].copy()

    for spec in DIMENSIONS:
        fact = _attach_dimension_id(
            fact=fact,
            source=dataframe,
            dimension=dimensions[spec.table_name],
            spec=spec,
        )

    id_columns = [spec.id_column for spec in DIMENSIONS]
    leading_columns = _available_columns(fact, FACT_IDENTIFIER_COLUMNS) + id_columns
    remaining_columns = [
        column for column in fact.columns if column not in leading_columns
    ]

    return fact[leading_columns + remaining_columns]


def build_warehouse_integrity_summary(
    source: pd.DataFrame,
    fact: pd.DataFrame,
    dimensions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "check_name": "source_row_count",
            "table_name": "synnetqos_core",
            "check_value": int(len(source)),
            "status": "ok",
        },
        {
            "check_name": "fact_row_count",
            "table_name": "fact_network_measurement",
            "check_value": int(len(fact)),
            "status": "ok",
        },
        {
            "check_name": "fact_row_count_matches_source",
            "table_name": "fact_network_measurement",
            "check_value": bool(len(fact) == len(source)),
            "status": "ok" if len(fact) == len(source) else "failed",
        },
    ]

    if "measurement_id" in fact.columns:
        duplicate_count = int(fact["measurement_id"].duplicated().sum())
        rows.append(
            {
                "check_name": "duplicate_measurement_id_count",
                "table_name": "fact_network_measurement",
                "check_value": duplicate_count,
                "status": "ok" if duplicate_count == 0 else "failed",
            }
        )

    for spec in DIMENSIONS:
        dimension = dimensions[spec.table_name]

        rows.append(
            {
                "check_name": "dimension_row_count",
                "table_name": spec.table_name,
                "check_value": int(len(dimension)),
                "status": "ok",
            }
        )

        if spec.id_column in fact.columns:
            missing_fk_count = int(fact[spec.id_column].isna().sum())
            rows.append(
                {
                    "check_name": "missing_foreign_key_count",
                    "table_name": spec.table_name,
                    "check_value": missing_fk_count,
                    "status": "ok" if missing_fk_count == 0 else "review",
                }
            )

        if spec.id_column in dimension.columns:
            duplicate_dimension_id_count = int(
                dimension[spec.id_column].duplicated().sum()
            )
            rows.append(
                {
                    "check_name": "duplicate_dimension_id_count",
                    "table_name": spec.table_name,
                    "check_value": duplicate_dimension_id_count,
                    "status": "ok" if duplicate_dimension_id_count == 0 else "failed",
                }
            )

        if spec.table_name == "dim_time" and len(dimension) >= len(source):
            rows.append(
                {
                    "check_name": "time_dimension_smaller_than_source",
                    "table_name": spec.table_name,
                    "check_value": False,
                    "status": "failed",
                }
            )
        elif spec.table_name == "dim_time":
            rows.append(
                {
                    "check_name": "time_dimension_smaller_than_source",
                    "table_name": spec.table_name,
                    "check_value": True,
                    "status": "ok",
                }
            )

    return pd.DataFrame(rows)


def build_warehouse_column_coverage(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for spec in DIMENSIONS:
        for column in spec.source_columns:
            rows.append(
                {
                    "table_name": spec.table_name,
                    "column_name": column,
                    "column_role": "dimension_attribute",
                    "available": column in source.columns,
                }
            )

    for column in FACT_IDENTIFIER_COLUMNS:
        rows.append(
            {
                "table_name": "fact_network_measurement",
                "column_name": column,
                "column_role": "identifier",
                "available": column in source.columns,
            }
        )

    for column in FACT_MEASURE_COLUMNS:
        rows.append(
            {
                "table_name": "fact_network_measurement",
                "column_name": column,
                "column_role": "measure",
                "available": column in source.columns,
            }
        )

    return pd.DataFrame(rows)


def build_warehouse_tables(
    input_path: str | Path,
    warehouse_dir: str | Path,
    results_dir: str | Path,
) -> dict[str, Path]:
    input_path = Path(input_path)
    warehouse_dir = ensure_directory(warehouse_dir)
    results_dir = ensure_directory(results_dir)

    source = prepare_warehouse_source(read_csv(input_path, low_memory=False))

    dimensions = {spec.table_name: build_dimension(source, spec) for spec in DIMENSIONS}

    fact = build_fact_table(source, dimensions)

    integrity_summary = build_warehouse_integrity_summary(
        source=source,
        fact=fact,
        dimensions=dimensions,
    )
    column_coverage = build_warehouse_column_coverage(source)

    failed_checks = integrity_summary.loc[integrity_summary["status"] == "failed"]
    if not failed_checks.empty:
        failed_names = ", ".join(
            f"{row.table_name}:{row.check_name}"
            for row in failed_checks.itertuples(index=False)
        )
        raise ValueError(f"Warehouse build failed checks: {failed_names}")

    output_paths: dict[str, Path] = {
        "fact_network_measurement": warehouse_dir / "fact_network_measurement.csv",
        "warehouse_integrity_summary": results_dir / "warehouse_integrity_summary.csv",
        "warehouse_column_coverage": results_dir / "warehouse_column_coverage.csv",
    }

    write_csv(fact, output_paths["fact_network_measurement"])

    for spec in DIMENSIONS:
        path = warehouse_dir / f"{spec.table_name}.csv"
        write_csv(dimensions[spec.table_name], path)
        output_paths[spec.table_name] = path

    write_csv(integrity_summary, output_paths["warehouse_integrity_summary"])
    write_csv(column_coverage, output_paths["warehouse_column_coverage"])

    return output_paths
