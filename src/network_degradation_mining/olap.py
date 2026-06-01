from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from network_degradation_mining.features import (
    add_degradation_labels,
    build_joined_warehouse_frame,
)
from network_degradation_mining.io import ensure_directory, write_csv
from network_degradation_mining.plotting import (
    add_labels_to_rates,
    build_warehouse_interaction_matrix_table,
    plot_warehouse_degradation_signature,
)

FREQUENCY_COLUMNS = [
    "network_type",
    "band",
    "congestion_level",
    "tower_load",
    "app_type",
    "movement_speed",
    "weather",
    "obstruction_level",
    "time_of_day_bin",
    "area_type",
    "video_quality_label",
    "service_degraded",
]

DEGRADATION_GROUPS = {
    "network_type": "degradation_by_network_type.csv",
    "band": "degradation_by_band.csv",
    "congestion_level": "degradation_by_congestion.csv",
    "app_type": "degradation_by_app.csv",
    "movement_speed": "degradation_by_mobility.csv",
    "time_of_day_bin": "degradation_by_time_of_day.csv",
    "area_type": "degradation_by_area_type.csv",
}

CROSSTAB_PAIRS = [
    ("app_type", "congestion_level"),
    ("band", "congestion_level"),
    ("network_type", "movement_speed"),
]


def _available_columns(dataframe: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in dataframe.columns]


def _clean_group_value(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"
    text = str(value).strip()
    return text if text else "<missing>"


def build_frequency_tables(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    row_count = len(dataframe)
    for column in _available_columns(dataframe, FREQUENCY_COLUMNS):
        counts = dataframe[column].map(_clean_group_value).value_counts(dropna=False)
        for value, count in counts.items():
            rows.append(
                {
                    "column_name": column,
                    "value": value,
                    "count": int(count),
                    "fraction": float(count / row_count) if row_count else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_degradation_rate_table(
    dataframe: pd.DataFrame, group_column: str
) -> pd.DataFrame:
    if group_column not in dataframe.columns:
        return pd.DataFrame(
            columns=[group_column, "row_count", "degraded_count", "degradation_rate"]
        )

    working = dataframe[[group_column, "service_degraded"]].copy()
    working[group_column] = working[group_column].map(_clean_group_value)
    grouped = (
        working.groupby(group_column, dropna=False)["service_degraded"]
        .agg(row_count="size", degraded_count="sum", degradation_rate="mean")
        .reset_index()
        .sort_values(["degradation_rate", "row_count"], ascending=[False, False])
    )
    grouped["degraded_count"] = grouped["degraded_count"].astype(int)
    grouped["row_count"] = grouped["row_count"].astype(int)
    return grouped


def build_degradation_crosstabs(dataframe: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row_column, column_column in CROSSTAB_PAIRS:
        if (
            row_column not in dataframe.columns
            or column_column not in dataframe.columns
        ):
            continue
        working = dataframe[[row_column, column_column, "service_degraded"]].copy()
        working[row_column] = working[row_column].map(_clean_group_value)
        working[column_column] = working[column_column].map(_clean_group_value)
        grouped = (
            working.groupby([row_column, column_column], dropna=False)[
                "service_degraded"
            ]
            .agg(row_count="size", degraded_count="sum", degradation_rate="mean")
            .reset_index()
        )
        grouped.insert(0, "row_group", row_column)
        grouped.insert(1, "column_group", column_column)
        grouped = grouped.rename(
            columns={row_column: "row_value", column_column: "column_value"}
        )
        frames.append(grouped)
    if not frames:
        return pd.DataFrame(
            columns=[
                "row_group",
                "column_group",
                "row_value",
                "column_value",
                "row_count",
                "degraded_count",
                "degradation_rate",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def build_olap_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric": "row_count",
            "value": int(len(dataframe)),
        },
        {
            "metric": "service_degraded_count",
            "value": int(dataframe["service_degraded"].sum()),
        },
        {
            "metric": "service_degraded_fraction",
            "value": float(dataframe["service_degraded"].mean())
            if len(dataframe)
            else 0.0,
        },
    ]
    if "session_id" in dataframe.columns:
        rows.append(
            {
                "metric": "session_count",
                "value": int(dataframe["session_id"].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def _summary_metric(summary: pd.DataFrame, metric: str) -> float | None:
    if summary.empty or {"metric", "value"}.difference(summary.columns):
        return None
    values = summary.loc[summary["metric"].astype(str).eq(metric), "value"]
    if values.empty:
        return None
    return float(values.iloc[0])


def run_descriptive_warehouse_mining(
    warehouse_dir: str | Path,
    results_dir: str | Path,
    figures_dir: str | Path,
) -> dict[str, Path]:
    output_dir = ensure_directory(results_dir)
    figure_dir = ensure_directory(figures_dir)

    joined = add_degradation_labels(build_joined_warehouse_frame(warehouse_dir))
    frequency = build_frequency_tables(joined)
    crosstabs = build_degradation_crosstabs(joined)
    summary = build_olap_summary(joined)

    output_paths: dict[str, Path] = {
        "frequency_tables": output_dir / "frequency_tables.csv",
        "degradation_crosstabs": output_dir / "degradation_crosstabs.csv",
        "olap_summary": output_dir / "olap_summary.csv",
    }

    write_csv(frequency, output_paths["frequency_tables"])
    write_csv(crosstabs, output_paths["degradation_crosstabs"])
    write_csv(summary, output_paths["olap_summary"])

    rate_tables: dict[str, pd.DataFrame] = {}
    for group_column, filename in DEGRADATION_GROUPS.items():
        table = build_degradation_rate_table(joined, group_column)
        rate_tables[group_column] = table
        path = output_dir / filename
        write_csv(table, path)
        output_paths[filename.removesuffix(".csv")] = path

    signature_table = add_labels_to_rates(rate_tables=rate_tables)
    if not signature_table.empty:
        signature_table_path = output_dir / "warehouse_degradation_signature_table.csv"
        write_csv(signature_table, signature_table_path)
        output_paths["warehouse_degradation_signature_table"] = signature_table_path

        signature_figure_path = (
            figure_dir / "warehouse_degradation_signature_forest_plot.pdf"
        )
        plot_warehouse_degradation_signature(
            rate_tables=rate_tables,
            figure_path=signature_figure_path,
            overall_rate=_summary_metric(summary, "service_degraded_fraction"),
        )
        output_paths["warehouse_degradation_signature_forest_plot"] = (
            signature_figure_path
        )

    interaction_table = build_warehouse_interaction_matrix_table(crosstabs)
    if not interaction_table.empty:
        interaction_table_path = (
            output_dir / "warehouse_degradation_interaction_matrix_table.csv"
        )
        write_csv(interaction_table, interaction_table_path)
        output_paths["warehouse_degradation_interaction_matrix_table"] = (
            interaction_table_path
        )

    return output_paths
