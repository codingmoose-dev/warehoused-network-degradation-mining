from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionSpec:
    table_name: str
    id_column: str
    source_columns: tuple[str, ...]


TIME_COLUMNS = (
    "hour",
    "day_of_week",
    "is_weekend",
    "time_of_day_bin",
)

AREA_COLUMNS = (
    "deployment_area",
    "area_type",
)

DEVICE_COLUMNS = (
    "ue_profile",
    "ue_capability_class",
)

NETWORK_COLUMNS = (
    "network_type",
    "operator_profile",
    "infrastructure_profile",
)

RADIO_COLUMNS = (
    "propagation_model",
    "propagation_scenario",
    "los_state",
    "carrier_frequency_ghz",
    "band",
)

APPLICATION_COLUMNS = (
    "app_type",
)

MOBILITY_COLUMNS = (
    "movement_speed",
)

ENVIRONMENT_COLUMNS = (
    "weather",
    "obstruction_level",
    "is_indoor",
)

SERVICE_STATE_COLUMNS = (
    "congestion_level",
    "tower_load",
    "video_quality_label",
    "vonr_enabled",
    "anomalous",
    "dropped_connection",
)


DIMENSIONS = (
    DimensionSpec(
        table_name="dim_time",
        id_column="time_id",
        source_columns=TIME_COLUMNS,
    ),
    DimensionSpec(
        table_name="dim_area",
        id_column="area_id",
        source_columns=AREA_COLUMNS,
    ),
    DimensionSpec(
        table_name="dim_device",
        id_column="device_id",
        source_columns=DEVICE_COLUMNS,
    ),
    DimensionSpec(
        table_name="dim_network",
        id_column="network_id",
        source_columns=NETWORK_COLUMNS,
    ),
    DimensionSpec(
        table_name="dim_radio",
        id_column="radio_id",
        source_columns=RADIO_COLUMNS,
    ),
    DimensionSpec(
        table_name="dim_application",
        id_column="application_id",
        source_columns=APPLICATION_COLUMNS,
    ),
    DimensionSpec(
        table_name="dim_mobility",
        id_column="mobility_id",
        source_columns=MOBILITY_COLUMNS,
    ),
    DimensionSpec(
        table_name="dim_environment",
        id_column="environment_id",
        source_columns=ENVIRONMENT_COLUMNS,
    ),
    DimensionSpec(
        table_name="dim_service_state",
        id_column="service_state_id",
        source_columns=SERVICE_STATE_COLUMNS,
    ),
)


FACT_IDENTIFIER_COLUMNS = (
    "measurement_id",
    "source_row_number",
    "user_id",
    "session_id",
    "tower_id",
    "timestamp",
    "session_step_index",
    "session_record_count",
)

FACT_MEASURE_COLUMNS = (
    "synthetic_latitude",
    "synthetic_longitude",
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
    "offered_downlink_mbps",
    "download_speed_mbps",
    "link_capacity_upload_mbps",
    "offered_upload_mbps",
    "upload_speed_mbps",
    "latency_ms",
    "jitter_ms",
    "ping_ms",
    "battery_level_percent",
    "temperature_c",
    "connected_duration_min",
    "interval_handover_count",
    "cumulative_handover_count",
    "activity_factor",
    "data_usage_mb",
    "distance_to_tower_km",
    "video_quality",
    "timestamp_parse_failed",
)


WAREHOUSE_TABLE_ORDER = (
    "fact_network_measurement",
    "dim_time",
    "dim_area",
    "dim_device",
    "dim_network",
    "dim_radio",
    "dim_application",
    "dim_mobility",
    "dim_environment",
    "dim_service_state",
)