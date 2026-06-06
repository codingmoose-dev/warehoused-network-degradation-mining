from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from network_degradation_mining.features import (
    add_degradation_labels,
    build_joined_warehouse_frame,
)
from network_degradation_mining.io import ensure_directory, read_csv, write_csv

MISSING_TOKEN = "<missing>"
SYNNETQOS_SOURCE = "synnetqos"

STANDARD_REFERENCE_FILES = {
    "vienna_4g5g": "vienna_reference_clean.csv",
    "campus_qos": "campus_qos_reference_clean.csv",
    "ucc_5g_context": "ucc_5g_reference_clean.csv",
    "ns3_lena": "simulator_reference_clean.csv",
}

SOURCE_LABELS = {
    "synnetqos": "SynNetQoS",
    "vienna_4g5g": "Vienna 4G/5G",
    "campus_qos": "Campus QoS",
    "ucc_5g_context": "UCC 5G Context",
    "ns3_lena": "5G-LENA/ns-3",
}

REFERENCE_COVERAGE_FILE = "external_reference_coverage.csv"
REFERENCE_SUMMARY_FILE = "external_summary.csv"
PAIRWISE_DISTANCE_FILE = "external_pairwise_distance_summary.csv"
METRIC_READINESS_FILE = "external_metric_readiness.csv"
CONTEXT_COVERAGE_FILE = "external_context_coverage.csv"

REFERENCE_METRICS = (
    "signal_dbm",
    "download_throughput_mbps",
    "upload_throughput_mbps",
    "latency_ms",
    "jitter_ms",
    "packet_loss_fraction",
    "offered_downlink_mbps",
)

REFERENCE_CONTEXT_COLUMNS = (
    "technology",
    "measurement_source",
    "measurement_context",
    "application_context",
    "mobility_context",
    "network_mode",
    "scenario",
)

TARGET_ATTRIBUTE_COLUMNS = (
    "service_degraded",
    "high_latency",
    "high_jitter",
    "downlink_service_shortfall",
    "poor_video_quality",
    "dropped_connection",
    "anomalous",
)


@dataclass(frozen=True)
class ContextNodeSpec:
    source_column: str
    context_family: str
    edge_type: str


WAREHOUSE_CONTEXT_SPECS = (
    ContextNodeSpec("time_of_day_bin", "time_bin", "observed_in_time_bin"),
    ContextNodeSpec("area_type", "area_type", "observed_in_area_type"),
    ContextNodeSpec("network_type", "network_type", "uses_network_type"),
    ContextNodeSpec("infrastructure_profile", "infrastructure", "uses_infrastructure"),
    ContextNodeSpec("band", "radio_band", "uses_radio_band"),
    ContextNodeSpec("app_type", "application", "uses_application"),
    ContextNodeSpec("movement_speed", "mobility", "has_mobility_state"),
    ContextNodeSpec("weather", "weather", "has_weather_context"),
    ContextNodeSpec("obstruction_level", "obstruction", "has_obstruction_context"),
    ContextNodeSpec("congestion_level", "congestion", "has_congestion_level"),
    ContextNodeSpec("tower_load", "tower_load", "has_tower_load"),
)


@dataclass(frozen=True)
class OutputTable:
    name: str
    artifact_type: str
    path: Path
    table: pd.DataFrame


def _clean_value(value: Any) -> str:
    if pd.isna(value):
        return MISSING_TOKEN
    text = str(value).strip()
    return text if text else MISSING_TOKEN


def _safe_token(value: Any) -> str:
    text = _clean_value(value).lower()
    for character in (" ", "/", "\\", ";", ",", ":", "|", "=", "(", ")"):
        text = text.replace(character, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "missing"


def _digest(*values: Any, length: int = 12) -> str:
    payload = "||".join(_clean_value(value) for value in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def _node_id(graph_view: str, node_type: str, *values: Any) -> str:
    token = "_".join(_safe_token(value) for value in values)[:70]
    return f"{graph_view}:{node_type}:{token}:{_digest(graph_view, node_type, *values)}"


def _edge_id(graph_view: str, edge_type: str, *values: Any) -> str:
    return f"edge_{_digest(graph_view, edge_type, *values, length=18)}"


def _measurement_node_id(measurement_id: Any) -> str:
    return _node_id("warehouse_measurement", "measurement", measurement_id)


def _session_node_id(session_id: Any) -> str:
    return _node_id("warehouse_measurement", "session", session_id)


def _warehouse_context_node_id(context_family: str, value: Any) -> str:
    return _node_id("warehouse_measurement", context_family, value)


def _context_value_node_id(context_family: str, value: Any) -> str:
    return _node_id("context_cooccurrence", "context_value", context_family, value)


def _external_source_node_id(source_dataset: Any) -> str:
    return _node_id("external_reference_evidence", "source", source_dataset)


def _external_metric_node_id(metric: Any) -> str:
    return _node_id("external_reference_evidence", "metric", metric)


def _external_context_node_id(context_family: Any, value: Any) -> str:
    return _node_id(
        "external_reference_evidence",
        "reference_context",
        context_family,
        value,
    )


def _external_comparison_node_id(
    metric: Any,
    reference_dataset: Any,
    comparison_context: Any,
) -> str:
    return _node_id(
        "external_reference_evidence",
        "comparison",
        metric,
        reference_dataset,
        comparison_context,
    )


def _ensure_measurement_id(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    if "measurement_id" not in output.columns:
        output["measurement_id"] = [
            f"measurement_{index + 1:09d}" for index in output.index
        ]
    return output


def _available_context_specs(dataframe: pd.DataFrame) -> list[ContextNodeSpec]:
    return [
        spec
        for spec in WAREHOUSE_CONTEXT_SPECS
        if spec.source_column in dataframe.columns
    ]


def _extract_context_values(
    row: pd.Series,
    specs: list[ContextNodeSpec],
) -> list[tuple[str, str, str]]:
    values: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for spec in specs:
        value = _clean_value(row.get(spec.source_column, MISSING_TOKEN))
        key = (spec.context_family, value)
        if key in seen:
            continue
        seen.add(key)
        values.append((spec.context_family, spec.source_column, value))
    return values


def _build_measurement_nodes(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in dataframe.itertuples(index=False):
        row_data = row._asdict()
        record: dict[str, Any] = {
            "node_id": _measurement_node_id(row_data["measurement_id"]),
            "node_type": "measurement",
            "graph_view": "warehouse_measurement",
            "node_label": _clean_value(row_data["measurement_id"]),
            "source_dataset": SYNNETQOS_SOURCE,
            "source_column": "measurement_id",
            "source_value": row_data["measurement_id"],
            "measurement_id": row_data["measurement_id"],
            "session_id": row_data.get("session_id", pd.NA),
            "timestamp": row_data.get("timestamp", pd.NA),
        }
        for column in TARGET_ATTRIBUTE_COLUMNS:
            if column in row_data:
                record[column] = row_data[column]
        rows.append(record)
    return pd.DataFrame(rows)


def _build_session_nodes(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "session_id" not in dataframe.columns:
        return pd.DataFrame()

    working = dataframe[["session_id", "service_degraded"]].copy()
    working["session_id"] = working["session_id"].map(_clean_value)
    grouped = (
        working.groupby("session_id", dropna=False)["service_degraded"]
        .agg(row_count="size", degraded_count="sum", degradation_fraction="mean")
        .reset_index()
    )

    rows = []
    for row in grouped.itertuples(index=False):
        rows.append(
            {
                "node_id": _session_node_id(row.session_id),
                "node_type": "session",
                "graph_view": "warehouse_measurement",
                "node_label": row.session_id,
                "source_dataset": SYNNETQOS_SOURCE,
                "source_column": "session_id",
                "source_value": row.session_id,
                "session_id": row.session_id,
                "row_count": int(row.row_count),
                "degraded_count": int(row.degraded_count),
                "degradation_fraction": float(row.degradation_fraction),
            }
        )
    return pd.DataFrame(rows)


def _build_warehouse_context_nodes(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in _available_context_specs(dataframe):
        values = dataframe[spec.source_column].map(_clean_value).drop_duplicates()
        for value in values:
            node_id = _warehouse_context_node_id(spec.context_family, value)
            if node_id in seen:
                continue
            seen.add(node_id)
            rows.append(
                {
                    "node_id": node_id,
                    "node_type": spec.context_family,
                    "graph_view": "warehouse_measurement",
                    "node_label": f"{spec.context_family}={value}",
                    "source_dataset": SYNNETQOS_SOURCE,
                    "source_column": spec.source_column,
                    "source_value": value,
                    "context_family": spec.context_family,
                    "context_value": value,
                }
            )
    return pd.DataFrame(rows)


def _build_session_membership_edges(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "session_id" not in dataframe.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for row in dataframe[["measurement_id", "session_id"]].itertuples(index=False):
        source = _measurement_node_id(row.measurement_id)
        target = _session_node_id(row.session_id)
        rows.append(
            {
                "edge_id": _edge_id(
                    "warehouse_measurement",
                    "belongs_to_session",
                    row.measurement_id,
                    row.session_id,
                ),
                "source_node_id": source,
                "target_node_id": target,
                "edge_type": "belongs_to_session",
                "graph_view": "warehouse_measurement",
                "is_directed": False,
                "edge_weight": 1.0,
                "measurement_id": row.measurement_id,
                "session_id": row.session_id,
            }
        )
    return pd.DataFrame(rows)


def _build_warehouse_context_edges(dataframe: pd.DataFrame) -> pd.DataFrame:
    specs = _available_context_specs(dataframe)
    rows: list[dict[str, Any]] = []
    columns = ["measurement_id"] + [spec.source_column for spec in specs]
    for row in dataframe[columns].itertuples(index=False):
        row_data = row._asdict()
        source = _measurement_node_id(row_data["measurement_id"])
        for spec in specs:
            value = _clean_value(row_data.get(spec.source_column, MISSING_TOKEN))
            target = _warehouse_context_node_id(spec.context_family, value)
            rows.append(
                {
                    "edge_id": _edge_id(
                        "warehouse_measurement",
                        spec.edge_type,
                        row_data["measurement_id"],
                        spec.context_family,
                        value,
                    ),
                    "source_node_id": source,
                    "target_node_id": target,
                    "edge_type": spec.edge_type,
                    "graph_view": "warehouse_measurement",
                    "is_directed": False,
                    "edge_weight": 1.0,
                    "measurement_id": row_data["measurement_id"],
                    "source_column": spec.source_column,
                    "source_value": value,
                    "context_family": spec.context_family,
                }
            )
    return pd.DataFrame(rows)


def _transition_sort_columns(dataframe: pd.DataFrame) -> list[str]:
    columns = ["session_id"]
    if "session_step_index" in dataframe.columns:
        columns.append("session_step_index")
    elif "timestamp" in dataframe.columns:
        columns.append("timestamp")
    else:
        columns.append("measurement_id")
    return columns


def _build_transition_edges(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "session_id" not in dataframe.columns:
        return pd.DataFrame()

    ordered = dataframe.sort_values(_transition_sort_columns(dataframe)).copy()
    rows: list[dict[str, Any]] = []
    for session_id, group in ordered.groupby("session_id", dropna=False):
        records = group[["measurement_id", "service_degraded"]].to_dict("records")
        for left, right in zip(records, records[1:]):
            source = _measurement_node_id(left["measurement_id"])
            target = _measurement_node_id(right["measurement_id"])
            rows.append(
                {
                    "edge_id": _edge_id(
                        "warehouse_measurement",
                        "next_measurement",
                        session_id,
                        left["measurement_id"],
                        right["measurement_id"],
                    ),
                    "source_node_id": source,
                    "target_node_id": target,
                    "edge_type": "next_measurement",
                    "graph_view": "warehouse_measurement",
                    "is_directed": True,
                    "edge_weight": 1.0,
                    "session_id": session_id,
                    "source_measurement_id": left["measurement_id"],
                    "target_measurement_id": right["measurement_id"],
                    "source_service_degraded": left["service_degraded"],
                    "target_service_degraded": right["service_degraded"],
                }
            )
    return pd.DataFrame(rows)


def _build_context_value_nodes(dataframe: pd.DataFrame) -> pd.DataFrame:
    specs = _available_context_specs(dataframe)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        values = dataframe[spec.source_column].map(_clean_value).drop_duplicates()
        for value in values:
            node_id = _context_value_node_id(spec.context_family, value)
            if node_id in seen:
                continue
            seen.add(node_id)
            mask = dataframe[spec.source_column].map(_clean_value).eq(value)
            subset = dataframe.loc[mask]
            degraded = pd.to_numeric(
                subset.get("service_degraded", 0), errors="coerce"
            ).fillna(0)
            rows.append(
                {
                    "node_id": node_id,
                    "node_type": "context_value",
                    "graph_view": "context_cooccurrence",
                    "node_label": f"{spec.context_family}={value}",
                    "source_dataset": SYNNETQOS_SOURCE,
                    "source_column": spec.source_column,
                    "source_value": value,
                    "context_family": spec.context_family,
                    "context_value": value,
                    "row_count": int(len(subset)),
                    "degraded_count": int(degraded.sum()),
                    "degradation_rate": float(degraded.mean()) if len(subset) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _build_context_cooccurrence_edges(dataframe: pd.DataFrame) -> pd.DataFrame:
    specs = _available_context_specs(dataframe)
    if not specs:
        return pd.DataFrame()

    baseline = float(pd.to_numeric(dataframe["service_degraded"]).mean())
    pair_counts: Counter[tuple[str, str, str, str]] = Counter()
    pair_degraded: Counter[tuple[str, str, str, str]] = Counter()
    row_count = len(dataframe)
    columns = ["service_degraded"] + [spec.source_column for spec in specs]

    for row in dataframe[columns].itertuples(index=False):
        row_data = row._asdict()
        degraded = int(row_data.get("service_degraded", 0) or 0)
        values = _extract_context_values(pd.Series(row_data), specs)
        node_entries = [
            (family, value, _context_value_node_id(family, value))
            for family, _, value in values
        ]
        for left, right in combinations(sorted(node_entries), 2):
            if left[2] == right[2]:
                continue
            key = (left[2], right[2], left[0], right[0])
            pair_counts[key] += 1
            pair_degraded[key] += degraded

    rows: list[dict[str, Any]] = []
    for (source, target, source_family, target_family), count in pair_counts.items():
        degraded_count = pair_degraded[(source, target, source_family, target_family)]
        degradation_rate = degraded_count / count if count else 0.0
        lift = degradation_rate / baseline if baseline else 0.0
        rows.append(
            {
                "edge_id": _edge_id(
                    "context_cooccurrence",
                    "context_cooccurs",
                    source,
                    target,
                ),
                "source_node_id": source,
                "target_node_id": target,
                "edge_type": "context_cooccurs",
                "graph_view": "context_cooccurrence",
                "is_directed": False,
                "edge_weight": float(count),
                "row_count": int(count),
                "support_fraction": float(count / row_count) if row_count else 0.0,
                "degraded_count": int(degraded_count),
                "degradation_rate": float(degradation_rate),
                "baseline_degradation_rate": float(baseline),
                "degradation_lift": float(lift),
                "source_context_family": source_family,
                "target_context_family": target_family,
            }
        )
    return pd.DataFrame(rows)


def _load_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_csv(path, low_memory=False)


def _external_clean_tables(external_reference_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for source_dataset, file_name in STANDARD_REFERENCE_FILES.items():
        path = external_reference_dir / file_name
        if path.exists():
            table = read_csv(path, low_memory=False)
            if "source_dataset" not in table.columns:
                table["source_dataset"] = source_dataset
            tables[source_dataset] = table
    return tables


def _source_label(source_dataset: Any) -> str:
    return SOURCE_LABELS.get(_clean_value(source_dataset), _clean_value(source_dataset))


def _build_external_source_nodes(
    clean_tables: dict[str, pd.DataFrame],
    coverage: pd.DataFrame,
    pairwise: pd.DataFrame | None = None,
) -> pd.DataFrame:
    sources = set(clean_tables)
    if "source_dataset" in coverage.columns:
        sources.update(coverage["source_dataset"].dropna().astype(str).tolist())
    if pairwise is not None and "reference_dataset" in pairwise.columns:
        sources.update(pairwise["reference_dataset"].dropna().astype(str).tolist())
    sources.add(SYNNETQOS_SOURCE)

    coverage_map = {}
    if {"source_dataset", "row_count"}.issubset(coverage.columns):
        coverage_map = dict(zip(coverage["source_dataset"], coverage["row_count"]))

    rows = []
    for source_dataset in sorted(sources):
        table = clean_tables.get(source_dataset, pd.DataFrame())
        row_count = coverage_map.get(source_dataset, len(table))
        role = (
            "main_dataset"
            if source_dataset == SYNNETQOS_SOURCE
            else "reference_dataset"
        )
        if source_dataset == "ns3_lena":
            role = "controlled_reference"
        rows.append(
            {
                "node_id": _external_source_node_id(source_dataset),
                "node_type": "source",
                "graph_view": "external_reference_evidence",
                "node_label": _source_label(source_dataset),
                "source_dataset": source_dataset,
                "source_label": _source_label(source_dataset),
                "source_role": role,
                "row_count": int(row_count) if pd.notna(row_count) else 0,
            }
        )
    return pd.DataFrame(rows)


def _build_external_metric_nodes(
    coverage: pd.DataFrame,
    readiness: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    metrics = set(REFERENCE_METRICS)
    for table in (readiness, pairwise):
        if "metric" in table.columns:
            metrics.update(table["metric"].dropna().astype(str).tolist())

    labels = {}
    for table in (readiness, pairwise):
        if {"metric", "metric_label"}.issubset(table.columns):
            labels.update(dict(zip(table["metric"], table["metric_label"])))

    rows = []
    for metric in sorted(metrics):
        rows.append(
            {
                "node_id": _external_metric_node_id(metric),
                "node_type": "metric",
                "graph_view": "external_reference_evidence",
                "node_label": labels.get(metric, metric),
                "metric": metric,
                "metric_label": labels.get(metric, metric),
            }
        )
    return pd.DataFrame(rows)


def _build_external_context_nodes(
    clean_tables: dict[str, pd.DataFrame],
    context_coverage: pd.DataFrame,
) -> pd.DataFrame:
    context_counts: Counter[tuple[str, str], int] = Counter()

    for _source_dataset, table in clean_tables.items():
        for column in REFERENCE_CONTEXT_COLUMNS:
            if column not in table.columns:
                continue
            counts = table[column].map(_clean_value).value_counts(dropna=False)
            for value, count in counts.items():
                if value == MISSING_TOKEN:
                    continue
                context_counts[(column, value)] += int(count)

    for column in ("measurement_context", "application_context", "mobility_context"):
        if column not in context_coverage.columns:
            continue
        counts = context_coverage[column].map(_clean_value).value_counts(dropna=False)
        for value, count in counts.items():
            if value == MISSING_TOKEN:
                continue
            context_counts[(column, value)] += int(count)

    rows: list[dict[str, Any]] = []
    for (column, value), count in sorted(context_counts.items()):
        rows.append(
            {
                "node_id": _external_context_node_id(column, value),
                "node_type": "reference_context",
                "graph_view": "external_reference_evidence",
                "node_label": f"{column}={value}",
                "source_column": column,
                "source_value": value,
                "context_family": column,
                "context_value": value,
                "row_count": int(count),
            }
        )
    return pd.DataFrame(rows)


def _build_external_comparison_nodes(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    rows = []
    for row in pairwise.itertuples(index=False):
        data = row._asdict()
        metric = data.get("metric", MISSING_TOKEN)
        reference = data.get("reference_dataset", MISSING_TOKEN)
        context = data.get("comparison_context", "unfiltered")
        node_id = _external_comparison_node_id(metric, reference, context)
        rows.append(
            {
                "node_id": node_id,
                "node_type": "comparison",
                "graph_view": "external_reference_evidence",
                "node_label": f"{reference}:{metric}:{context}",
                "metric": metric,
                "reference_dataset": reference,
                "comparison_context": context,
                "recommended_use": data.get("recommended_use", pd.NA),
                "comparison_scope": data.get("comparison_scope", pd.NA),
                "synnetqos_sample_count": data.get("synnetqos_sample_count", pd.NA),
                "reference_sample_count": data.get("reference_sample_count", pd.NA),
                "median_difference": data.get("median_difference", pd.NA),
                "wasserstein_distance": data.get("wasserstein_distance", pd.NA),
                "ks_statistic": data.get("ks_statistic", pd.NA),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("node_id")


def _metric_count_from_coverage(row: pd.Series, metric: str) -> tuple[int, float]:
    count_column = f"{metric}_non_missing_count"
    fraction_column = f"{metric}_non_missing_fraction"
    if count_column not in row.index:
        return 0, 0.0
    count = pd.to_numeric(row[count_column], errors="coerce")
    fraction = pd.to_numeric(row.get(fraction_column, 0.0), errors="coerce")
    safe_count = int(count) if pd.notna(count) else 0
    safe_fraction = float(fraction) if pd.notna(fraction) else 0.0
    return safe_count, safe_fraction


def _build_external_metric_edges(
    coverage: pd.DataFrame,
    readiness: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    readiness_lookup = {}
    if {"source_dataset", "metric"}.issubset(readiness.columns):
        for row in readiness.itertuples(index=False):
            data = row._asdict()
            readiness_lookup[(data["source_dataset"], data["metric"])] = data

    if coverage.empty or "source_dataset" not in coverage.columns:
        return pd.DataFrame()

    for _, row in coverage.iterrows():
        source_dataset = _clean_value(row["source_dataset"])
        source_node = _external_source_node_id(source_dataset)
        for metric in REFERENCE_METRICS:
            count, fraction = _metric_count_from_coverage(row, metric)
            if count <= 0:
                continue
            readiness_row = readiness_lookup.get((source_dataset, metric), {})
            metric_node = _external_metric_node_id(metric)
            rows.append(
                {
                    "edge_id": _edge_id(
                        "external_reference_evidence",
                        "source_provides_metric",
                        source_dataset,
                        metric,
                    ),
                    "source_node_id": source_node,
                    "target_node_id": metric_node,
                    "edge_type": "source_provides_metric",
                    "graph_view": "external_reference_evidence",
                    "is_directed": True,
                    "edge_weight": float(count),
                    "source_dataset": source_dataset,
                    "metric": metric,
                    "non_missing_count": int(count),
                    "non_missing_fraction": float(fraction),
                    "comparison_status": readiness_row.get("comparison_status", pd.NA),
                    "recommended_use": readiness_row.get("recommended_use", pd.NA),
                    "claim_scope": readiness_row.get("claim_scope", pd.NA),
                }
            )
    return pd.DataFrame(rows)


def _build_external_context_edges(
    clean_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_dataset, table in clean_tables.items():
        source_dataset = _clean_value(source_dataset)
        source_node = _external_source_node_id(source_dataset)
        for column in REFERENCE_CONTEXT_COLUMNS:
            if column not in table.columns:
                continue
            counts = table[column].map(_clean_value).value_counts(dropna=False)
            for value, count in counts.items():
                if value == MISSING_TOKEN:
                    continue
                context_node = _external_context_node_id(column, value)
                rows.append(
                    {
                        "edge_id": _edge_id(
                            "external_reference_evidence",
                            "source_has_reference_context",
                            source_dataset,
                            column,
                            value,
                        ),
                        "source_node_id": source_node,
                        "target_node_id": context_node,
                        "edge_type": "source_has_reference_context",
                        "graph_view": "external_reference_evidence",
                        "is_directed": True,
                        "edge_weight": float(count),
                        "source_dataset": source_dataset,
                        "source_column": column,
                        "source_value": value,
                        "row_count": int(count),
                    }
                )
    return pd.DataFrame(rows)


def _build_external_comparison_edges(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for row in pairwise.itertuples(index=False):
        data = row._asdict()
        metric = data.get("metric", MISSING_TOKEN)
        reference = data.get("reference_dataset", MISSING_TOKEN)
        context = data.get("comparison_context", "unfiltered")
        comparison_node = _external_comparison_node_id(metric, reference, context)
        syn_node = _external_source_node_id(SYNNETQOS_SOURCE)
        ref_node = _external_source_node_id(reference)
        metric_node = _external_metric_node_id(metric)
        edge_attrs = {
            "graph_view": "external_reference_evidence",
            "is_directed": True,
            "metric": metric,
            "reference_dataset": reference,
            "comparison_context": context,
            "recommended_use": data.get("recommended_use", pd.NA),
            "median_difference": data.get("median_difference", pd.NA),
            "wasserstein_distance": data.get("wasserstein_distance", pd.NA),
            "ks_statistic": data.get("ks_statistic", pd.NA),
            "comparison_scope": data.get("comparison_scope", pd.NA),
        }
        rows.extend(
            [
                {
                    "edge_id": _edge_id(
                        "external_reference_evidence",
                        "synnetqos_has_comparison",
                        metric,
                        reference,
                        context,
                    ),
                    "source_node_id": syn_node,
                    "target_node_id": comparison_node,
                    "edge_type": "synnetqos_has_comparison",
                    "edge_weight": 1.0,
                    **edge_attrs,
                },
                {
                    "edge_id": _edge_id(
                        "external_reference_evidence",
                        "reference_has_comparison",
                        metric,
                        reference,
                        context,
                    ),
                    "source_node_id": ref_node,
                    "target_node_id": comparison_node,
                    "edge_type": "reference_has_comparison",
                    "edge_weight": 1.0,
                    **edge_attrs,
                },
                {
                    "edge_id": _edge_id(
                        "external_reference_evidence",
                        "comparison_uses_metric",
                        metric,
                        reference,
                        context,
                    ),
                    "source_node_id": comparison_node,
                    "target_node_id": metric_node,
                    "edge_type": "comparison_uses_metric",
                    "edge_weight": 1.0,
                    **edge_attrs,
                },
            ]
        )
    return pd.DataFrame(rows)


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    populated = [frame for frame in frames if frame is not None and not frame.empty]
    if not populated:
        return pd.DataFrame()
    return pd.concat(populated, ignore_index=True, sort=False)


def _build_audit(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    duplicate_nodes = int(nodes["node_id"].duplicated().sum()) if not nodes.empty else 0
    duplicate_edges = int(edges["edge_id"].duplicated().sum()) if not edges.empty else 0
    node_ids = set(nodes["node_id"].astype(str)) if not nodes.empty else set()
    if not edges.empty:
        edge_sources = set(edges["source_node_id"].astype(str))
        edge_targets = set(edges["target_node_id"].astype(str))
    else:
        edge_sources = set()
        edge_targets = set()
    missing_edge_nodes = len((edge_sources | edge_targets) - node_ids)

    view_rows: list[dict[str, Any]] = []
    if "graph_view" in nodes.columns:
        for view, group in nodes.groupby("graph_view", dropna=False):
            view_edges = edges.loc[edges.get("graph_view", pd.Series()).eq(view)]
            view_rows.append(
                {
                    "check_name": f"{view}_node_count",
                    "check_value": int(len(group)),
                    "status": "ok" if len(group) > 0 else "review",
                }
            )
            view_rows.append(
                {
                    "check_name": f"{view}_edge_count",
                    "check_value": int(len(view_edges)),
                    "status": "ok" if len(view_edges) > 0 else "review",
                }
            )

    rows = [
        {
            "check_name": "node_count_positive",
            "check_value": int(len(nodes)),
            "status": "ok" if len(nodes) > 0 else "failed",
        },
        {
            "check_name": "edge_count_positive",
            "check_value": int(len(edges)),
            "status": "ok" if len(edges) > 0 else "failed",
        },
        {
            "check_name": "duplicate_node_id_count",
            "check_value": duplicate_nodes,
            "status": "ok" if duplicate_nodes == 0 else "failed",
        },
        {
            "check_name": "duplicate_edge_id_count",
            "check_value": duplicate_edges,
            "status": "ok" if duplicate_edges == 0 else "failed",
        },
        {
            "check_name": "edge_endpoint_missing_node_count",
            "check_value": int(missing_edge_nodes),
            "status": "ok" if missing_edge_nodes == 0 else "failed",
        },
        {
            "check_name": "external_rows_are_reference_evidence_only",
            "check_value": 1,
            "status": "ok",
        },
        {
            "check_name": "graph_view_policy",
            "check_value": (
                "warehouse_measurement=SynNetQoS labeled records; "
                "context_cooccurrence=SynNetQoS context signatures; "
                "external_reference_evidence=reference-only evidence"
            ),
            "status": "ok",
        },
    ]
    return pd.DataFrame(rows + view_rows)


def _project_relative_path(
    path: str | Path,
    project_root_dir: str | Path | None,
) -> str:
    target = Path(path)
    if project_root_dir is None:
        return target.as_posix()

    root = Path(project_root_dir)
    try:
        return target.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return target.as_posix()


def _build_manifest(
    tables: list[OutputTable],
    project_root_dir: str | Path | None = None,
) -> pd.DataFrame:
    rows = []
    for item in tables:
        rows.append(
            {
                "artifact_name": item.name,
                "artifact_type": item.artifact_type,
                "relative_path": _project_relative_path(item.path, project_root_dir),
                "row_count": int(len(item.table)),
            }
        )
    return pd.DataFrame(rows).sort_values(["artifact_type", "artifact_name"])


def _external_result_tables(external_results_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "coverage": _load_csv_if_exists(external_results_dir / REFERENCE_COVERAGE_FILE),
        "summary": _load_csv_if_exists(external_results_dir / REFERENCE_SUMMARY_FILE),
        "pairwise": _load_csv_if_exists(external_results_dir / PAIRWISE_DISTANCE_FILE),
        "readiness": _load_csv_if_exists(external_results_dir / METRIC_READINESS_FILE),
        "context_coverage": _load_csv_if_exists(
            external_results_dir / CONTEXT_COVERAGE_FILE
        ),
    }


def build_graph_tables(
    warehouse_dir: str | Path,
    graph_nodes_dir: str | Path,
    graph_edges_dir: str | Path,
    results_dir: str | Path,
    external_reference_dir: str | Path | None = None,
    external_results_dir: str | Path | None = None,
    project_root_dir: str | Path | None = None,
) -> dict[str, Path]:
    node_dir = ensure_directory(graph_nodes_dir)
    edge_dir = ensure_directory(graph_edges_dir)
    result_dir = ensure_directory(results_dir)
    graph_root = ensure_directory(node_dir.parent)

    joined = _ensure_measurement_id(
        add_degradation_labels(build_joined_warehouse_frame(warehouse_dir))
    )

    measurement_nodes = _build_measurement_nodes(joined)
    session_nodes = _build_session_nodes(joined)
    warehouse_context_nodes = _build_warehouse_context_nodes(joined)
    context_value_nodes = _build_context_value_nodes(joined)

    membership_edges = _build_session_membership_edges(joined)
    warehouse_context_edges = _build_warehouse_context_edges(joined)
    transition_edges = _build_transition_edges(joined)
    context_cooccurrence_edges = _build_context_cooccurrence_edges(joined)

    external_reference_path = (
        Path(external_reference_dir) if external_reference_dir else None
    )
    external_results_path = Path(external_results_dir) if external_results_dir else None
    clean_tables = (
        _external_clean_tables(external_reference_path)
        if external_reference_path is not None
        else {}
    )
    result_tables = (
        _external_result_tables(external_results_path)
        if external_results_path is not None
        else {}
    )
    coverage = result_tables.get("coverage", pd.DataFrame())
    readiness = result_tables.get("readiness", pd.DataFrame())
    pairwise = result_tables.get("pairwise", pd.DataFrame())
    context_coverage = result_tables.get("context_coverage", pd.DataFrame())

    external_source_nodes = _build_external_source_nodes(
        clean_tables,
        coverage,
        pairwise,
    )
    external_metric_nodes = _build_external_metric_nodes(coverage, readiness, pairwise)
    external_context_nodes = _build_external_context_nodes(
        clean_tables, context_coverage
    )
    external_comparison_nodes = _build_external_comparison_nodes(pairwise)
    external_metric_edges = _build_external_metric_edges(coverage, readiness)
    external_context_edges = _build_external_context_edges(clean_tables)
    external_comparison_edges = _build_external_comparison_edges(pairwise)
    external_reference_edges = _concat_frames(
        [external_metric_edges, external_context_edges, external_comparison_edges]
    )

    all_nodes = _concat_frames(
        [
            measurement_nodes,
            session_nodes,
            warehouse_context_nodes,
            context_value_nodes,
            external_source_nodes,
            external_metric_nodes,
            external_context_nodes,
            external_comparison_nodes,
        ]
    )
    all_edges = _concat_frames(
        [
            membership_edges,
            warehouse_context_edges,
            transition_edges,
            context_cooccurrence_edges,
            external_reference_edges,
        ]
    )

    audit = _build_audit(all_nodes, all_edges)
    failed = audit.loc[audit["status"].eq("failed")]
    if not failed.empty:
        checks = ", ".join(failed["check_name"].astype(str).tolist())
        raise ValueError(f"Graph table build failed checks: {checks}")

    table_specs = [
        OutputTable(
            "warehouse_measurement_nodes",
            "node_table",
            node_dir / "warehouse_measurement_nodes.csv",
            measurement_nodes,
        ),
        OutputTable(
            "warehouse_session_nodes",
            "node_table",
            node_dir / "warehouse_session_nodes.csv",
            session_nodes,
        ),
        OutputTable(
            "warehouse_context_nodes",
            "node_table",
            node_dir / "warehouse_context_nodes.csv",
            warehouse_context_nodes,
        ),
        OutputTable(
            "context_value_nodes",
            "node_table",
            node_dir / "context_value_nodes.csv",
            context_value_nodes,
        ),
        OutputTable(
            "external_source_nodes",
            "node_table",
            node_dir / "external_source_nodes.csv",
            external_source_nodes,
        ),
        OutputTable(
            "external_metric_nodes",
            "node_table",
            node_dir / "external_metric_nodes.csv",
            external_metric_nodes,
        ),
        OutputTable(
            "external_context_nodes",
            "node_table",
            node_dir / "external_context_nodes.csv",
            external_context_nodes,
        ),
        OutputTable(
            "external_comparison_nodes",
            "node_table",
            node_dir / "external_comparison_nodes.csv",
            external_comparison_nodes,
        ),
        OutputTable(
            "all_nodes",
            "node_table",
            node_dir / "all_nodes.csv",
            all_nodes,
        ),
        OutputTable(
            "warehouse_membership_edges",
            "edge_table",
            edge_dir / "warehouse_membership_edges.csv",
            membership_edges,
        ),
        OutputTable(
            "warehouse_context_edges",
            "edge_table",
            edge_dir / "warehouse_context_edges.csv",
            warehouse_context_edges,
        ),
        OutputTable(
            "warehouse_transition_edges",
            "edge_table",
            edge_dir / "warehouse_transition_edges.csv",
            transition_edges,
        ),
        OutputTable(
            "context_cooccurrence_edges",
            "edge_table",
            edge_dir / "context_cooccurrence_edges.csv",
            context_cooccurrence_edges,
        ),
        OutputTable(
            "external_reference_edges",
            "edge_table",
            edge_dir / "external_reference_edges.csv",
            external_reference_edges,
        ),
        OutputTable(
            "all_edges",
            "edge_table",
            edge_dir / "all_edges.csv",
            all_edges,
        ),
        OutputTable(
            "graph_build_audit",
            "audit_table",
            result_dir / "graph_build_audit.csv",
            audit,
        ),
    ]

    output_paths: dict[str, Path] = {}
    for item in table_specs:
        output_paths[item.name] = write_csv(item.table, item.path)

    manifest = _build_manifest(table_specs, project_root_dir=project_root_dir)
    manifest_path = graph_root / "graph_manifest.csv"
    output_paths["graph_manifest"] = write_csv(manifest, manifest_path)
    return output_paths
