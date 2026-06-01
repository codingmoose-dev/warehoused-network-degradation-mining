from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from network_degradation_mining.io import ensure_directory, read_csv, write_csv

WAREHOUSE_TABLES = {
    "fact": "fact_network_measurement.csv",
    "time": "dim_time.csv",
    "area": "dim_area.csv",
    "device": "dim_device.csv",
    "network": "dim_network.csv",
    "radio": "dim_radio.csv",
    "application": "dim_application.csv",
    "mobility": "dim_mobility.csv",
    "environment": "dim_environment.csv",
    "service_state": "dim_service_state.csv",
}

JOIN_SPECS = (
    ("dim_time.csv", "time_id"),
    ("dim_area.csv", "area_id"),
    ("dim_device.csv", "device_id"),
    ("dim_network.csv", "network_id"),
    ("dim_radio.csv", "radio_id"),
    ("dim_application.csv", "application_id"),
    ("dim_mobility.csv", "mobility_id"),
    ("dim_environment.csv", "environment_id"),
    ("dim_service_state.csv", "service_state_id"),
)

IDENTIFIER_COLUMNS = [
    "measurement_id",
    "source_row_number",
    "user_id",
    "session_id",
    "tower_id",
    "timestamp",
    "session_step_index",
    "session_record_count",
]

SUPERVISED_NUMERIC_FEATURES = [
    "hour",
    "is_weekend",
    "signal_strength_dbm",
    "los_probability",
    "distance_2d_m",
    "distance_3d_m",
    "breakpoint_distance_m",
    "path_loss_db",
    "deterministic_path_loss_db",
    "shadow_fading_db",
    "fast_fading_db",
    "obstruction_penalty_db",
    "weather_penalty_db",
    "mobility_penalty_db",
    "indoor_penalty_db",
    "contextual_penalty_db",
    "effective_tx_power_dbm",
    "signal_strength_unclipped_dbm",
    "rsrp_clipped",
    "link_capacity_downlink_mbps",
    "link_capacity_upload_mbps",
    "offered_downlink_mbps",
    "offered_upload_mbps",
    "battery_level_percent",
    "temperature_c",
    "connected_duration_min",
    "interval_handover_count",
    "cumulative_handover_count",
    "activity_factor",
    "data_usage_mb",
    "distance_to_tower_km",
    "carrier_frequency_ghz",
]

SUPERVISED_CATEGORICAL_FEATURES = [
    "day_of_week",
    "time_of_day_bin",
    "deployment_area",
    "area_type",
    "ue_profile",
    "ue_capability_class",
    "network_type",
    "operator_profile",
    "infrastructure_profile",
    "propagation_model",
    "propagation_scenario",
    "los_state",
    "band",
    "app_type",
    "movement_speed",
    "weather",
    "obstruction_level",
    "congestion_level",
    "tower_load",
]

CLUSTERING_NUMERIC_FEATURES = [
    "signal_strength_dbm",
    "los_probability",
    "distance_2d_m",
    "path_loss_db",
    "contextual_penalty_db",
    "link_capacity_downlink_mbps",
    "offered_downlink_mbps",
    "download_speed_mbps",
    "offered_upload_mbps",
    "upload_speed_mbps",
    "latency_ms",
    "jitter_ms",
    "ping_ms",
    "interval_handover_count",
    "activity_factor",
    "data_usage_mb",
    "distance_to_tower_km",
    "throughput_satisfaction_ratio",
    "downlink_shortfall_fraction",
]

CLUSTERING_CATEGORICAL_FEATURES = [
    "network_type",
    "infrastructure_profile",
    "band",
    "app_type",
    "movement_speed",
    "weather",
    "obstruction_level",
    "congestion_level",
    "tower_load",
    "time_of_day_bin",
]

TARGET_COMPONENT_COLUMNS = [
    "download_speed_mbps",
    "upload_speed_mbps",
    "latency_ms",
    "jitter_ms",
    "ping_ms",
    "video_quality",
    "video_quality_label",
    "dropped_connection",
    "anomalous",
    "throughput_satisfaction_ratio",
    "downlink_shortfall_fraction",
    "high_latency",
    "high_jitter",
    "downlink_service_shortfall",
    "poor_video_quality",
]

LABEL_THRESHOLDS = {
    "high_latency_ms": 150.0,
    "high_jitter_ms": 1.0,
    "low_throughput_ratio": 0.35,
    "poor_video_quality_max": 2.0,
}

BIN_SPECS = {
    "signal_strength_dbm": [-float("inf"), -115, -100, -85, float("inf")],
    "latency_ms": [-float("inf"), 50, 150, 300, float("inf")],
    "jitter_ms": [-float("inf"), 0.25, 1.0, 2.5, float("inf")],
    "downlink_shortfall_fraction": [-float("inf"), 0.2, 0.5, 0.8, float("inf")],
    "offered_downlink_mbps": [-float("inf"), 10, 50, 200, float("inf")],
}

BIN_LABELS = {
    "signal_strength_dbm": [
        "signal_very_weak",
        "signal_weak",
        "signal_moderate",
        "signal_strong",
    ],
    "latency_ms": [
        "latency_low",
        "latency_moderate",
        "latency_high",
        "latency_extreme",
    ],
    "jitter_ms": ["jitter_low", "jitter_moderate", "jitter_high", "jitter_extreme"],
    "downlink_shortfall_fraction": [
        "shortfall_low",
        "shortfall_moderate",
        "shortfall_high",
        "shortfall_extreme",
    ],
    "offered_downlink_mbps": [
        "offered_low",
        "offered_moderate",
        "offered_high",
        "offered_very_high",
    ],
}


def _available_columns(dataframe: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in dataframe.columns]


def _to_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.astype("string").str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


def _numeric(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(pd.NA, index=dataframe.index, dtype="Float64")
    return pd.to_numeric(dataframe[column], errors="coerce")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_numeric = pd.to_numeric(numerator, errors="coerce")
    denominator_numeric = pd.to_numeric(denominator, errors="coerce")
    result = numerator_numeric / denominator_numeric.where(denominator_numeric > 0)
    return result.replace([float("inf"), -float("inf")], pd.NA)


def load_warehouse_tables(warehouse_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(warehouse_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in WAREHOUSE_TABLES.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing warehouse table: {path}")
        tables[name] = read_csv(path, low_memory=False)
    return tables


def build_joined_warehouse_frame(warehouse_dir: str | Path) -> pd.DataFrame:
    root = Path(warehouse_dir)
    fact_path = root / WAREHOUSE_TABLES["fact"]
    if not fact_path.exists():
        raise FileNotFoundError(f"Missing fact table: {fact_path}")

    output = read_csv(fact_path, low_memory=False)
    for filename, key in JOIN_SPECS:
        dimension_path = root / filename
        if not dimension_path.exists() or key not in output.columns:
            continue
        dimension = read_csv(dimension_path, low_memory=False)
        if key not in dimension.columns:
            continue
        output = output.merge(dimension, on=key, how="left", validate="many_to_one")

    return output


def add_degradation_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()

    output["throughput_satisfaction_ratio"] = _safe_divide(
        _numeric(output, "download_speed_mbps"),
        _numeric(output, "offered_downlink_mbps"),
    ).clip(lower=0, upper=2)
    output["downlink_shortfall_fraction"] = (
        1 - output["throughput_satisfaction_ratio"]
    ).clip(lower=0, upper=1)

    output["high_latency"] = (
        _numeric(output, "latency_ms") >= LABEL_THRESHOLDS["high_latency_ms"]
    )
    output["high_jitter"] = (
        _numeric(output, "jitter_ms") >= LABEL_THRESHOLDS["high_jitter_ms"]
    )
    output["downlink_service_shortfall"] = (
        output["throughput_satisfaction_ratio"]
        <= LABEL_THRESHOLDS["low_throughput_ratio"]
    ) & _numeric(output, "offered_downlink_mbps").notna()

    if "dropped_connection" in output.columns:
        output["dropped_connection"] = _to_bool(output["dropped_connection"])
    else:
        output["dropped_connection"] = False

    if "anomalous" in output.columns:
        output["anomalous"] = _to_bool(output["anomalous"])
    else:
        output["anomalous"] = False

    app_type = output.get("app_type", pd.Series("", index=output.index)).astype(
        "string"
    )
    video_quality = _numeric(output, "video_quality")
    output["poor_video_quality"] = (
        app_type.str.lower().eq("streaming")
        & video_quality.notna()
        & (video_quality <= LABEL_THRESHOLDS["poor_video_quality_max"])
    )

    output["service_degraded"] = (
        output["high_latency"].fillna(False)
        | output["high_jitter"].fillna(False)
        | output["downlink_service_shortfall"].fillna(False)
        | output["dropped_connection"].fillna(False)
        | output["poor_video_quality"].fillna(False)
        | output["anomalous"].fillna(False)
    ).astype(int)

    return output


def build_supervised_degradation_table(joined: pd.DataFrame) -> pd.DataFrame:
    labeled = add_degradation_labels(joined)

    id_columns = _available_columns(
        labeled, ["measurement_id", "session_id", "timestamp"]
    )
    numeric_features = _available_columns(labeled, SUPERVISED_NUMERIC_FEATURES)
    categorical_features = _available_columns(labeled, SUPERVISED_CATEGORICAL_FEATURES)
    target_columns = [
        "service_degraded",
        "high_latency",
        "high_jitter",
        "downlink_service_shortfall",
        "poor_video_quality",
    ]

    selected = id_columns + target_columns + numeric_features + categorical_features
    output = labeled[selected].copy()

    for column in target_columns:
        if column in output.columns:
            output[column] = output[column].astype(int)

    return output


def build_clustering_feature_table(joined: pd.DataFrame) -> pd.DataFrame:
    labeled = add_degradation_labels(joined)
    selected = _available_columns(
        labeled,
        ["measurement_id", "session_id", "timestamp", "service_degraded"]
        + CLUSTERING_NUMERIC_FEATURES
        + CLUSTERING_CATEGORICAL_FEATURES,
    )
    return labeled[selected].copy()


def _bin_column(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(pd.NA, index=dataframe.index, dtype="string")
    return pd.cut(
        pd.to_numeric(dataframe[column], errors="coerce"),
        bins=BIN_SPECS[column],
        labels=BIN_LABELS[column],
        include_lowest=True,
    ).astype("string")


def _item_value(prefix: str, value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    safe = text.replace(" ", "_").replace(";", "_").replace(",", "_")
    return f"{prefix}={safe}"


def build_association_transaction_table(joined: pd.DataFrame) -> pd.DataFrame:
    labeled = add_degradation_labels(joined)
    output = pd.DataFrame(index=labeled.index)
    output["measurement_id"] = labeled.get(
        "measurement_id", pd.Series(pd.NA, index=labeled.index)
    )
    output["session_id"] = labeled.get(
        "session_id", pd.Series(pd.NA, index=labeled.index)
    )
    output["service_degraded"] = labeled["service_degraded"].astype(int)

    for column in BIN_SPECS:
        labeled[f"{column}_bin"] = _bin_column(labeled, column)

    item_columns = [
        "network_type",
        "infrastructure_profile",
        "band",
        "app_type",
        "movement_speed",
        "weather",
        "obstruction_level",
        "congestion_level",
        "tower_load",
        "time_of_day_bin",
        "service_degraded",
        "signal_strength_dbm_bin",
        "latency_ms_bin",
        "jitter_ms_bin",
        "downlink_shortfall_fraction_bin",
        "offered_downlink_mbps_bin",
    ]

    transaction_items: list[str] = []
    for _, row in labeled.iterrows():
        items: list[str] = []
        for column in item_columns:
            if column not in labeled.columns:
                continue
            prefix = column.removesuffix("_bin")
            item = _item_value(prefix, row[column])
            if item is not None:
                items.append(item)
        transaction_items.append(";".join(sorted(set(items))))

    output["items"] = transaction_items
    return output


def build_label_audit(supervised: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    row_count = len(supervised)
    for column in [
        "service_degraded",
        "high_latency",
        "high_jitter",
        "downlink_service_shortfall",
        "poor_video_quality",
        "dropped_connection",
        "anomalous",
    ]:
        if column not in supervised.columns:
            continue
        values = (
            pd.to_numeric(supervised[column], errors="coerce").fillna(0).astype(int)
        )
        positive_count = int(values.sum())
        rows.append(
            {
                "label_name": column,
                "row_count": int(row_count),
                "positive_count": positive_count,
                "positive_fraction": float(positive_count / row_count)
                if row_count
                else 0.0,
                "status": "ok" if 0 < positive_count < row_count else "review",
            }
        )
    return pd.DataFrame(rows)


def build_mining_feature_tables(
    warehouse_dir: str | Path,
    mining_tables_dir: str | Path,
    results_dir: str | Path,
) -> dict[str, Path]:
    output_dir = ensure_directory(mining_tables_dir)
    audit_dir = ensure_directory(results_dir)

    joined = build_joined_warehouse_frame(warehouse_dir)
    supervised = build_supervised_degradation_table(joined)
    clustering = build_clustering_feature_table(joined)
    transactions = build_association_transaction_table(joined)
    label_audit = build_label_audit(add_degradation_labels(joined))

    paths = {
        "supervised_degradation_table": output_dir / "supervised_degradation_table.csv",
        "clustering_feature_table": output_dir / "clustering_feature_table.csv",
        "association_transaction_table": output_dir
        / "association_transaction_table.csv",
        "label_audit": audit_dir / "mining_label_audit.csv",
    }

    write_csv(supervised, paths["supervised_degradation_table"])
    write_csv(clustering, paths["clustering_feature_table"])
    write_csv(transactions, paths["association_transaction_table"])
    write_csv(label_audit, paths["label_audit"])

    return paths
