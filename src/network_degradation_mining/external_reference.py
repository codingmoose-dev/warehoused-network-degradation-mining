from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from network_degradation_mining.io import (
    ensure_directory,
    read_csv,
    read_parquet,
    write_csv,
)

STANDARD_COLUMNS = [
    "reference_record_id",
    "source_dataset",
    "source_file",
    "technology",
    "measurement_source",
    "measurement_context",
    "application_context",
    "mobility_context",
    "timestamp",
    "latitude",
    "longitude",
    "operator",
    "network_mode",
    "frequency_ghz",
    "rsrp_dbm",
    "rsrq_db",
    "rssi_dbm",
    "sinr_db",
    "cqi",
    "download_throughput_mbps",
    "upload_throughput_mbps",
    "download_bitrate_raw",
    "upload_bitrate_raw",
    "raw_bitrate_unit",
    "latency_ms",
    "jitter_ms",
    "packet_loss_fraction",
    "offered_downlink_mbps",
    "offered_upload_mbps",
    "speed_raw",
    "scenario",
    "notes",
]


SUMMARY_NUMERIC_COLUMNS = [
    "frequency_ghz",
    "rsrp_dbm",
    "rsrq_db",
    "rssi_dbm",
    "sinr_db",
    "cqi",
    "download_throughput_mbps",
    "upload_throughput_mbps",
    "download_bitrate_raw",
    "upload_bitrate_raw",
    "latency_ms",
    "jitter_ms",
    "packet_loss_fraction",
    "offered_downlink_mbps",
    "offered_upload_mbps",
    "speed_raw",
]


REQUIRED_NON_EMPTY_REFERENCES = {
    "vienna_4g5g",
    "campus_qos",
    "ucc_5g_context",
}


VIENNA_FILE_SPECS = [
    ("phone_data_5g.parquet", "5g", "phone"),
    ("phone_data_lte.parquet", "lte", "phone"),
    ("scanner_data_5g.parquet", "5g", "scanner"),
    ("scanner_data_lte.parquet", "lte", "scanner"),
]


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return Path(path).name


def _missing_series(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index)


def _text_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column not in dataframe.columns:
        return _missing_series(dataframe.index)
    return dataframe[column].astype("string").str.strip()


def _numeric_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(pd.NA, index=dataframe.index, dtype="Float64")
    return pd.to_numeric(dataframe[column], errors="coerce")


def _empty_reference_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["source_row_number", *STANDARD_COLUMNS])


def _finalize_reference_frame(dataframe: pd.DataFrame, prefix: str) -> pd.DataFrame:
    output = dataframe.copy()

    for column in STANDARD_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA

    output = output[STANDARD_COLUMNS].copy()
    output.insert(0, "source_row_number", range(1, len(output) + 1))

    output["reference_record_id"] = [
        f"{prefix}_{row_number:09d}" for row_number in output["source_row_number"]
    ]

    return output


def _parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema.names)


def _read_parquet_selected(path: Path, selected_columns: list[str]) -> pd.DataFrame:
    available = _parquet_columns(path)
    columns = [column for column in selected_columns if column in available]

    if not columns:
        return pd.DataFrame()

    return read_parquet(path, columns=columns)


def _first_existing_file_by_name(root: Path, filename: str) -> Path | None:
    direct_matches = sorted(root.glob(f"*/{filename}"))
    recursive_matches = sorted(root.rglob(filename))
    matches = direct_matches + [
        path for path in recursive_matches if path not in direct_matches
    ]
    return matches[0] if matches else None


def _parse_timestamp(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")

    text = series.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    formats = (
        "%Y.%m.%d_%H.%M.%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    )

    for timestamp_format in formats:
        missing = parsed.isna() & text.notna() & (text != "")
        if not missing.any():
            break

        parsed_format = pd.to_datetime(
            text.loc[missing],
            format=timestamp_format,
            errors="coerce",
        )
        parsed.loc[missing] = parsed_format

    fallback_missing = parsed.isna() & text.notna() & (text != "")
    if fallback_missing.any():
        parsed.loc[fallback_missing] = pd.to_datetime(
            text.loc[fallback_missing],
            errors="coerce",
        )

    return parsed


def _prepare_vienna_file(
    path: Path,
    root: Path,
    technology: str,
    measurement_source: str,
) -> pd.DataFrame:
    selected_columns = [
        "time",
        "latitude",
        "longitude",
        "rsrp_dbm",
        "rsrq_db",
        "rssi_dbm",
        "sinr_db",
        "cinr_db",
        "dl_throughput_mbps",
        "ul_throughput_mbps",
        "operator",
        "frequency_khz",
    ]

    raw = _read_parquet_selected(path, selected_columns)
    if raw.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    output = pd.DataFrame(index=raw.index)
    output["source_dataset"] = "vienna_4g5g"
    output["source_file"] = _safe_relative_path(path, root)
    output["technology"] = technology
    output["measurement_source"] = measurement_source
    output["measurement_context"] = "drive_test"
    output["application_context"] = pd.NA
    output["mobility_context"] = "mobile_measurement"
    output["timestamp"] = pd.to_datetime(raw.get("time"), errors="coerce")
    output["latitude"] = _numeric_series(raw, "latitude")
    output["longitude"] = _numeric_series(raw, "longitude")
    output["operator"] = _text_series(raw, "operator")
    output["network_mode"] = technology
    output["frequency_ghz"] = _numeric_series(raw, "frequency_khz") / 1_000_000
    output["rsrp_dbm"] = _numeric_series(raw, "rsrp_dbm")
    output["rsrq_db"] = _numeric_series(raw, "rsrq_db")
    output["rssi_dbm"] = _numeric_series(raw, "rssi_dbm")

    if "sinr_db" in raw.columns:
        output["sinr_db"] = _numeric_series(raw, "sinr_db")
    elif "cinr_db" in raw.columns:
        output["sinr_db"] = _numeric_series(raw, "cinr_db")
    else:
        output["sinr_db"] = pd.NA

    output["cqi"] = pd.NA
    output["download_throughput_mbps"] = _numeric_series(raw, "dl_throughput_mbps")
    output["upload_throughput_mbps"] = _numeric_series(raw, "ul_throughput_mbps")
    output["download_bitrate_raw"] = pd.NA
    output["upload_bitrate_raw"] = pd.NA
    output["raw_bitrate_unit"] = pd.NA
    output["latency_ms"] = pd.NA
    output["jitter_ms"] = pd.NA
    output["packet_loss_fraction"] = pd.NA
    output["offered_downlink_mbps"] = pd.NA
    output["offered_upload_mbps"] = pd.NA
    output["speed_raw"] = pd.NA
    output["scenario"] = pd.NA
    output["notes"] = "selected comparable radio and throughput fields"

    return output


def prepare_vienna_reference(vienna_path: str | Path, root: Path) -> pd.DataFrame:
    dataset_root = Path(vienna_path)
    frames: list[pd.DataFrame] = []

    for filename, technology, measurement_source in VIENNA_FILE_SPECS:
        path = _first_existing_file_by_name(dataset_root, filename)
        if path is None:
            continue

        frames.append(
            _prepare_vienna_file(
                path=path,
                root=root,
                technology=technology,
                measurement_source=measurement_source,
            )
        )

    frames = [frame for frame in frames if not frame.empty]

    if not frames:
        return _empty_reference_frame()

    return _finalize_reference_frame(pd.concat(frames, ignore_index=True), "vienna")


def _prepare_campus_file(path: Path, root: Path, site: str) -> pd.DataFrame:
    raw = read_csv(path, low_memory=False)

    output = pd.DataFrame(index=raw.index)
    output["source_dataset"] = "campus_qos"
    output["source_file"] = _safe_relative_path(path, root)
    output["technology"] = "5g"
    output["measurement_source"] = site
    output["measurement_context"] = "controlled_testbed"
    output["application_context"] = pd.NA
    output["mobility_context"] = "controlled"
    output["timestamp"] = pd.NA
    output["latitude"] = pd.NA
    output["longitude"] = pd.NA
    output["operator"] = pd.NA
    output["network_mode"] = "5g"
    output["frequency_ghz"] = pd.NA
    output["rsrp_dbm"] = pd.NA
    output["rsrq_db"] = pd.NA
    output["rssi_dbm"] = pd.NA
    output["sinr_db"] = pd.NA
    output["cqi"] = pd.NA
    output["download_throughput_mbps"] = _numeric_series(raw, "mbpsactual_downlink")
    output["upload_throughput_mbps"] = _numeric_series(raw, "mbpsactual_uplink")
    output["download_bitrate_raw"] = pd.NA
    output["upload_bitrate_raw"] = pd.NA
    output["raw_bitrate_unit"] = pd.NA
    output["latency_ms"] = pd.NA
    output["jitter_ms"] = _numeric_series(raw, "meanjitterms_downlink")
    output["packet_loss_fraction"] = _numeric_series(raw, "meanloss_downlink")
    output["offered_downlink_mbps"] = _numeric_series(raw, "mbpsoffered_downlink")
    output["offered_upload_mbps"] = _numeric_series(raw, "mbpsoffered_uplink")
    output["speed_raw"] = pd.NA
    output["scenario"] = _text_series(raw, "scenario")
    output["notes"] = "controlled throughput file; downlink jitter and loss retained"

    return output


def prepare_campus_reference(campus_path: str | Path, root: Path) -> pd.DataFrame:
    campus_root = Path(campus_path)

    file_specs = [
        (campus_root / "ntnu_tput_all_Throughput.csv", "ntnu"),
        (campus_root / "wue_tput_all_Throughput.csv", "wue"),
    ]

    frames = [
        _prepare_campus_file(path, root, site)
        for path, site in file_specs
        if path.exists()
    ]

    if not frames:
        return _empty_reference_frame()

    return _finalize_reference_frame(pd.concat(frames, ignore_index=True), "campus")


def _ucc_context_from_path(path: Path, ucc_root: Path) -> dict[str, str]:
    parts = list(path.relative_to(ucc_root).parts)

    if parts and parts[0].lower() in {
        "5g-production-dataset",
        "beyond-throughput",
        "beyond_throughput",
    }:
        parts = parts[1:]

    return {
        "application_context": parts[0] if len(parts) >= 1 else "",
        "mobility_context": parts[1] if len(parts) >= 2 else "",
        "scenario": "/".join(parts[2:-1]) if len(parts) > 3 else "",
    }


def _technology_from_network_mode(network_mode: pd.Series) -> pd.Series:
    normalized = network_mode.astype("string").str.strip().str.upper()
    technology = pd.Series("unknown", index=network_mode.index, dtype="string")

    technology.loc[normalized.str.contains("5G", na=False)] = "5g"
    technology.loc[normalized.str.contains("LTE", na=False)] = "lte"
    technology.loc[normalized.str.contains("HSPA|UMTS|WCDMA", na=False, regex=True)] = (
        "3g"
    )
    technology.loc[normalized.str.contains("GSM|EDGE|GPRS", na=False, regex=True)] = (
        "2g"
    )

    return technology


def _fraction_from_percent_like(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return values

    if values.dropna().max() > 1:
        return values / 100

    return values


def _prepare_ucc_file(path: Path, ucc_root: Path, root: Path) -> pd.DataFrame:
    raw = read_csv(path, low_memory=False)
    context = _ucc_context_from_path(path, ucc_root)
    network_mode = _text_series(raw, "NetworkMode")

    output = pd.DataFrame(index=raw.index)
    output["source_dataset"] = "ucc_5g_context"
    output["source_file"] = _safe_relative_path(path, root)
    output["technology"] = _technology_from_network_mode(network_mode)
    output["measurement_source"] = "phone"
    output["measurement_context"] = "production_trace"
    output["application_context"] = context["application_context"]
    output["mobility_context"] = context["mobility_context"]
    output["timestamp"] = _parse_timestamp(raw.get("Timestamp"))
    output["latitude"] = _numeric_series(raw, "Latitude")
    output["longitude"] = _numeric_series(raw, "Longitude")
    output["operator"] = _text_series(raw, "Operatorname")
    output["network_mode"] = network_mode
    output["frequency_ghz"] = pd.NA
    output["rsrp_dbm"] = _numeric_series(raw, "RSRP")
    output["rsrq_db"] = _numeric_series(raw, "RSRQ")
    output["rssi_dbm"] = _numeric_series(raw, "RSSI")
    output["sinr_db"] = _numeric_series(raw, "SNR")
    output["cqi"] = _numeric_series(raw, "CQI")
    download_bitrate_kbps = _numeric_series(raw, "DL_bitrate")
    upload_bitrate_kbps = _numeric_series(raw, "UL_bitrate")

    output["download_throughput_mbps"] = download_bitrate_kbps / 1000
    output["upload_throughput_mbps"] = upload_bitrate_kbps / 1000
    output["download_bitrate_raw"] = download_bitrate_kbps
    output["upload_bitrate_raw"] = upload_bitrate_kbps
    output["raw_bitrate_unit"] = "kbps"
    output["latency_ms"] = _numeric_series(raw, "PINGAVG")
    output["jitter_ms"] = _numeric_series(raw, "PINGSTDEV")
    output["packet_loss_fraction"] = _fraction_from_percent_like(
        raw.get("PINGLOSS", pd.Series(pd.NA, index=raw.index))
    )
    output["offered_downlink_mbps"] = pd.NA
    output["offered_upload_mbps"] = pd.NA
    output["speed_raw"] = _numeric_series(raw, "Speed")
    output["scenario"] = context["scenario"]
    output["notes"] = (
        "production trace; source bitrate unit confirmed as kbps in source documentation; throughput_mbps derived by dividing source bitrate by 1000; ping loss converted to fraction when source values are percent-like"
    )

    return output


def prepare_ucc_reference(ucc_path: str | Path, root: Path) -> pd.DataFrame:
    ucc_root = Path(ucc_path)
    files = sorted(ucc_root.rglob("*.csv"))

    frames = [_prepare_ucc_file(path, ucc_root, root) for path in files]

    if not frames:
        return _empty_reference_frame()

    return _finalize_reference_frame(pd.concat(frames, ignore_index=True), "ucc")


SIMULATOR_EXPECTED_TRAFFIC_CONDITIONS = ("medium_load", "saturation_load")
SIMULATOR_EXPECTED_SCHEDULER_MODES = ("tdma", "ofdma")
SIMULATOR_EXPECTED_SEEDS = (1, 2, 3)
SIMULATOR_DEFAULT_EXAMPLE = "cttc-nr-simple-qos-sched"

SIMULATOR_RUN_COLUMNS = [
    "run_id",
    "traffic_condition",
    "scheduler_mode",
    "seed",
    "ns3_example",
    "ns3_version",
    "tx_packets",
    "rx_packets",
    "tx_bytes",
    "rx_bytes",
    "offered_traffic_mbps",
    "mean_throughput_mbps",
    "mean_delay_ms",
    "mean_jitter_ms",
    "packet_delivery_ratio",
    "packet_loss_fraction",
    "output_status",
    "parser_status",
    "notes",
]

SIMULATOR_ALIASES = {
    "run_id": ["run_id", "scenario_id", "simulation_id", "name"],
    "traffic_condition": [
        "traffic_condition",
        "load_condition",
        "traffic_load",
        "load_level",
    ],
    "scheduler_mode": ["scheduler_mode", "scheduler", "access_mode", "scheduler_type"],
    "seed": ["seed", "run_seed", "rng_seed", "random_seed"],
    "ns3_example": ["ns3_example", "example", "example_name", "script"],
    "ns3_version": ["ns3_version", "version", "build_version"],
    "timestamp": ["timestamp", "run_timestamp", "created_at"],
    "tx_packets": ["tx_packets", "transmitted_packets", "packets_tx", "tx_pkts"],
    "rx_packets": ["rx_packets", "received_packets", "packets_rx", "rx_pkts"],
    "tx_bytes": ["tx_bytes", "transmitted_bytes", "bytes_tx"],
    "rx_bytes": ["rx_bytes", "received_bytes", "bytes_rx"],
    "offered_traffic_mbps": [
        "offered_traffic_mbps",
        "offered_downlink_mbps",
        "offered_load_mbps",
        "offered_mbps",
        "application_rate_mbps",
    ],
    "mean_throughput_mbps": [
        "mean_throughput_mbps",
        "throughput_mbps",
        "rx_throughput_mbps",
        "download_throughput_mbps",
        "dl_throughput_mbps",
        "mean_rx_throughput_mbps",
    ],
    "mean_delay_ms": ["mean_delay_ms", "delay_ms", "latency_ms", "mean_latency_ms"],
    "mean_jitter_ms": ["mean_jitter_ms", "jitter_ms", "mean_jitter"],
    "packet_delivery_ratio": [
        "packet_delivery_ratio",
        "delivery_ratio",
        "pdr",
        "rx_tx_ratio",
    ],
    "packet_loss_fraction": [
        "packet_loss_fraction",
        "loss_fraction",
        "packet_loss_ratio",
        "loss_ratio",
    ],
    "scenario": ["scenario", "scenario_name", "run_id"],
    "output_status": ["output_status", "simulation_status", "status"],
    "parser_status": ["parser_status", "parse_status"],
    "notes": ["notes", "note", "description"],
}

SIMULATOR_EXCLUDED_CSV_NAMES = {
    "external_reference_summary.csv",
    "external_reference_preparation_audit.csv",
    "simulator_reference_clean.csv",
    "simulator_reference_readiness_audit.csv",
    "ucc_reference_quality_audit.csv",
    "ns3_lena_run_audit.csv",
    "ns3_lena_summary.csv",
}


def _normalized_column_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _find_column(dataframe: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = {_normalized_column_name(column): column for column in dataframe.columns}
    for alias in aliases:
        column = lookup.get(_normalized_column_name(alias))
        if column is not None:
            return column
    return None


def _series_from_alias(dataframe: pd.DataFrame, field: str) -> pd.Series:
    column = _find_column(dataframe, SIMULATOR_ALIASES[field])
    if column is None:
        return _missing_series(dataframe.index)
    return dataframe[column]


def _numeric_from_alias(dataframe: pd.DataFrame, field: str) -> pd.Series:
    return pd.to_numeric(_series_from_alias(dataframe, field), errors="coerce")


def _text_from_alias(dataframe: pd.DataFrame, field: str) -> pd.Series:
    return _series_from_alias(dataframe, field).astype("string").str.strip()


def _fraction_from_ratio_or_percent(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    non_missing = values.dropna()
    if non_missing.empty:
        return values
    if non_missing.max() > 1:
        return values / 100
    return values


def _has_simulator_metric_columns(dataframe: pd.DataFrame) -> bool:
    metric_fields = [
        "mean_throughput_mbps",
        "mean_delay_ms",
        "mean_jitter_ms",
        "packet_delivery_ratio",
        "packet_loss_fraction",
        "offered_traffic_mbps",
        "tx_packets",
        "rx_packets",
        "tx_bytes",
        "rx_bytes",
    ]
    return any(
        _find_column(dataframe, SIMULATOR_ALIASES[field]) is not None
        for field in metric_fields
    )


def _candidate_simulator_csv_files(simulator_reference_dir: str | Path) -> list[Path]:
    simulator_dir = Path(simulator_reference_dir)
    if not simulator_dir.exists():
        return []

    return [
        path
        for path in sorted(simulator_dir.rglob("*.csv"))
        if path.name not in SIMULATOR_EXCLUDED_CSV_NAMES
    ]


def _expected_ns3_lena_run_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for traffic_condition in SIMULATOR_EXPECTED_TRAFFIC_CONDITIONS:
        for scheduler_mode in SIMULATOR_EXPECTED_SCHEDULER_MODES:
            for seed in SIMULATOR_EXPECTED_SEEDS:
                run_id = f"{traffic_condition}_{scheduler_mode}_seed_{seed}"
                rows.append(
                    {
                        "run_id": run_id,
                        "traffic_condition": traffic_condition,
                        "scheduler_mode": scheduler_mode,
                        "seed": seed,
                        "ns3_example": SIMULATOR_DEFAULT_EXAMPLE,
                        "output_dir_label": f"ns3_lena_simple_qos/{run_id}",
                        "output_status": "expected",
                        "notes": "Expected controlled 5G-LENA reference run.",
                    }
                )
    return rows


def _scenario_from_simulator_columns(raw: pd.DataFrame, source_file: Path) -> pd.Series:
    scenario = _text_from_alias(raw, "scenario")
    missing = scenario.isna() | (scenario == "")

    traffic_condition = _text_from_alias(raw, "traffic_condition").fillna("")
    scheduler_mode = _text_from_alias(raw, "scheduler_mode").fillna("")
    seed = _text_from_alias(raw, "seed").fillna("")

    constructed = (
        traffic_condition.astype(str)
        + "_"
        + scheduler_mode.astype(str)
        + "_seed_"
        + seed.astype(str)
    ).str.strip("_")

    constructed = constructed.replace({"_seed_": "", "seed_": ""})
    fallback = source_file.parent.name if source_file.parent.name else source_file.stem
    scenario.loc[missing] = constructed.loc[missing]
    scenario = scenario.replace("", pd.NA)
    scenario = scenario.fillna(fallback)
    return scenario


def _prepare_simulator_summary_file(
    path: Path,
    simulator_reference_dir: Path,
    root: Path,
) -> pd.DataFrame:
    raw = read_csv(path, low_memory=False)
    if raw.empty or not _has_simulator_metric_columns(raw):
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    delivery_ratio = _fraction_from_ratio_or_percent(
        _numeric_from_alias(raw, "packet_delivery_ratio")
    )
    packet_loss_fraction = _fraction_from_ratio_or_percent(
        _numeric_from_alias(raw, "packet_loss_fraction")
    )
    loss_missing = packet_loss_fraction.isna() & delivery_ratio.notna()
    packet_loss_fraction.loc[loss_missing] = 1 - delivery_ratio.loc[loss_missing]

    output = pd.DataFrame(index=raw.index)
    output["source_dataset"] = "ns3_lena"
    output["source_file"] = _safe_relative_path(path, root)
    output["technology"] = "5g"
    output["measurement_source"] = "simulator"
    output["measurement_context"] = "controlled_simulation"
    output["application_context"] = "qos_reference"
    output["mobility_context"] = "controlled"
    output["timestamp"] = _parse_timestamp(_series_from_alias(raw, "timestamp"))
    output["latitude"] = pd.NA
    output["longitude"] = pd.NA
    output["operator"] = pd.NA
    output["network_mode"] = "5g_nr"
    output["frequency_ghz"] = pd.NA
    output["rsrp_dbm"] = pd.NA
    output["rsrq_db"] = pd.NA
    output["rssi_dbm"] = pd.NA
    output["sinr_db"] = pd.NA
    output["cqi"] = pd.NA
    output["download_throughput_mbps"] = _numeric_from_alias(
        raw, "mean_throughput_mbps"
    )
    output["upload_throughput_mbps"] = pd.NA
    output["download_bitrate_raw"] = pd.NA
    output["upload_bitrate_raw"] = pd.NA
    output["raw_bitrate_unit"] = pd.NA
    output["latency_ms"] = _numeric_from_alias(raw, "mean_delay_ms")
    output["jitter_ms"] = _numeric_from_alias(raw, "mean_jitter_ms")
    output["packet_loss_fraction"] = packet_loss_fraction
    output["offered_downlink_mbps"] = _numeric_from_alias(raw, "offered_traffic_mbps")
    output["offered_upload_mbps"] = pd.NA
    output["speed_raw"] = pd.NA
    output["scenario"] = _scenario_from_simulator_columns(raw, path)
    output["notes"] = (
        "controlled 5G-LENA reference run; summary metrics parsed from curated simulator output"
    )

    comparable_metrics = [
        "download_throughput_mbps",
        "latency_ms",
        "jitter_ms",
        "packet_loss_fraction",
        "offered_downlink_mbps",
    ]
    has_metric = output[comparable_metrics].notna().any(axis=1)
    return output.loc[has_metric].copy()


def prepare_simulator_reference(
    simulator_reference_dir: str | Path, root: Path
) -> pd.DataFrame:
    simulator_dir = Path(simulator_reference_dir)
    frames: list[pd.DataFrame] = []

    for path in _candidate_simulator_csv_files(simulator_dir):
        try:
            frame = _prepare_simulator_summary_file(path, simulator_dir, root)
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return _empty_reference_frame()

    return _finalize_reference_frame(pd.concat(frames, ignore_index=True), "ns3")


def _safe_manifest_value(
    dataframe: pd.DataFrame, field: str, default: Any = pd.NA
) -> pd.Series:
    if field in {
        "run_id",
        "traffic_condition",
        "scheduler_mode",
        "ns3_example",
        "ns3_version",
        "output_status",
        "parser_status",
        "notes",
    }:
        series = _text_from_alias(
            dataframe, field if field in SIMULATOR_ALIASES else "notes"
        )
        return series.fillna(default)
    return _numeric_from_alias(dataframe, field).fillna(default)


def _read_manifest_file(path: Path) -> pd.DataFrame:
    raw = read_csv(path, low_memory=False)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "traffic_condition",
                "scheduler_mode",
                "seed",
                "ns3_example",
                "output_dir_label",
                "output_status",
                "notes",
            ]
        )

    output = pd.DataFrame(index=raw.index)
    output["run_id"] = _text_from_alias(raw, "run_id")
    output["traffic_condition"] = _text_from_alias(raw, "traffic_condition")
    output["scheduler_mode"] = _text_from_alias(raw, "scheduler_mode")
    output["seed"] = _numeric_from_alias(raw, "seed").astype("Int64")
    output["ns3_example"] = _text_from_alias(raw, "ns3_example").fillna(
        SIMULATOR_DEFAULT_EXAMPLE
    )
    output["output_dir_label"] = (
        output["run_id"]
        .fillna("unlabeled_run")
        .map(lambda value: f"ns3_lena_simple_qos/{value}")
    )
    output["output_status"] = _text_from_alias(raw, "output_status").fillna("listed")
    output["notes"] = "Manifest row loaded from curated simulator reference file."
    return output


def build_ns3_lena_run_manifest(simulator_reference_dir: str | Path) -> pd.DataFrame:
    simulator_dir = Path(simulator_reference_dir)
    manifest_files = (
        sorted(simulator_dir.rglob("run_manifest.csv"))
        if simulator_dir.exists()
        else []
    )

    frames: list[pd.DataFrame] = []
    for path in manifest_files:
        try:
            frame = _read_manifest_file(path)
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)

    if frames:
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame(_expected_ns3_lena_run_rows())


def _parsed_simulator_run_ids(simulator: pd.DataFrame) -> set[str]:
    if "scenario" not in simulator.columns or simulator.empty:
        return set()
    return set(simulator["scenario"].dropna().astype(str).str.strip())


def build_ns3_lena_run_audit(
    simulator_reference_dir: str | Path, simulator: pd.DataFrame
) -> pd.DataFrame:
    manifest = build_ns3_lena_run_manifest(simulator_reference_dir)
    parsed_run_ids = _parsed_simulator_run_ids(simulator)
    csv_files = _candidate_simulator_csv_files(simulator_reference_dir)

    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        run_id = str(getattr(row, "run_id", "")).strip()
        parsed = run_id in parsed_run_ids
        rows.append(
            {
                "source_dataset": "ns3_lena",
                "run_id": run_id,
                "traffic_condition": getattr(row, "traffic_condition", pd.NA),
                "scheduler_mode": getattr(row, "scheduler_mode", pd.NA),
                "seed": getattr(row, "seed", pd.NA),
                "ns3_example": getattr(row, "ns3_example", SIMULATOR_DEFAULT_EXAMPLE),
                "curated_csv_file_count": int(len(csv_files)),
                "parsed_reference_row_count": int(
                    (simulator["scenario"].astype(str) == run_id).sum()
                )
                if parsed and "scenario" in simulator.columns
                else 0,
                "output_status": "parsed" if parsed else "waiting_for_curated_summary",
                "parser_status": "ok" if parsed else "review",
                "notes": "Parsed simulator summary is available."
                if parsed
                else "No parsed simulator summary row found for this expected run.",
            }
        )

    if not rows:
        rows.append(
            {
                "source_dataset": "ns3_lena",
                "run_id": "",
                "traffic_condition": "",
                "scheduler_mode": "",
                "seed": pd.NA,
                "ns3_example": SIMULATOR_DEFAULT_EXAMPLE,
                "curated_csv_file_count": int(len(csv_files)),
                "parsed_reference_row_count": int(len(simulator)),
                "output_status": "waiting_for_controlled_runs",
                "parser_status": "review",
                "notes": "No simulator manifest or curated summary CSV files were found.",
            }
        )

    return pd.DataFrame(rows)


def build_ns3_lena_summary(simulator: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "download_throughput_mbps",
        "latency_ms",
        "jitter_ms",
        "packet_loss_fraction",
        "offered_downlink_mbps",
    ]

    if simulator.empty:
        return pd.DataFrame(
            columns=[
                "source_dataset",
                "group_name",
                "group_value",
                "metric",
                "column_name",
                "value",
            ]
        )

    groups = [("all", "all", simulator)]
    if "scenario" in simulator.columns:
        groups.extend(
            ("scenario", str(name), frame)
            for name, frame in simulator.groupby("scenario", dropna=False)
        )

    for group_name, group_value, frame in groups:
        rows.append(
            {
                "source_dataset": "ns3_lena",
                "group_name": group_name,
                "group_value": group_value,
                "metric": "row_count",
                "column_name": "",
                "value": int(len(frame)),
            }
        )
        for column in metrics:
            series = (
                pd.to_numeric(frame[column], errors="coerce")
                if column in frame.columns
                else pd.Series(dtype="float64")
            )
            non_missing = series.dropna()
            rows.append(
                {
                    "source_dataset": "ns3_lena",
                    "group_name": group_name,
                    "group_value": group_value,
                    "metric": "non_missing_count",
                    "column_name": column,
                    "value": int(non_missing.shape[0]),
                }
            )
            if non_missing.empty:
                continue
            for metric_name, value in {
                "mean": non_missing.mean(),
                "median": non_missing.median(),
                "min": non_missing.min(),
                "max": non_missing.max(),
            }.items():
                rows.append(
                    {
                        "source_dataset": "ns3_lena",
                        "group_name": group_name,
                        "group_value": group_value,
                        "metric": metric_name,
                        "column_name": column,
                        "value": float(value),
                    }
                )

    return pd.DataFrame(rows)


def _ns3_local_readiness_rows(ns3_lena_root: str | Path | None) -> list[dict[str, Any]]:
    configured = ns3_lena_root is not None and str(ns3_lena_root).strip() != ""
    ns3_root = Path(ns3_lena_root).expanduser() if configured else None
    root_exists = bool(ns3_root and ns3_root.exists())

    if root_exists and ns3_root is not None:
        launcher_exists = any((ns3_root / name).exists() for name in ["ns3", "waf"])
        nr_module_present = any(
            (ns3_root / parent / "nr").exists() for parent in ["src", "contrib"]
        )
        candidate_examples = sorted(ns3_root.rglob("cttc-nr-simple-qos-sched.cc"))
        nr_example_count = len(list(ns3_root.rglob("cttc-nr*.cc")))
    else:
        launcher_exists = False
        nr_module_present = False
        candidate_examples = []
        nr_example_count = 0

    return [
        {
            "source_dataset": "ns3_lena",
            "check_name": "local_ns3_root_configured",
            "check_value": bool(configured),
            "status": "ok" if configured else "review",
        },
        {
            "source_dataset": "ns3_lena",
            "check_name": "local_ns3_root_exists",
            "check_value": bool(root_exists),
            "status": "ok" if root_exists else "review",
        },
        {
            "source_dataset": "ns3_lena",
            "check_name": "local_ns3_launcher_present",
            "check_value": bool(launcher_exists),
            "status": "ok" if launcher_exists else "review",
        },
        {
            "source_dataset": "ns3_lena",
            "check_name": "local_nr_module_present",
            "check_value": bool(nr_module_present),
            "status": "ok" if nr_module_present else "review",
        },
        {
            "source_dataset": "ns3_lena",
            "check_name": "candidate_qos_example_present",
            "check_value": bool(candidate_examples),
            "status": "ok" if candidate_examples else "review",
        },
        {
            "source_dataset": "ns3_lena",
            "check_name": "local_nr_example_file_count",
            "check_value": int(nr_example_count),
            "status": "ok" if nr_example_count else "review",
        },
    ]


def build_external_reference_summary(
    references: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for source_dataset, dataframe in references.items():
        rows.append(
            {
                "source_dataset": source_dataset,
                "metric": "row_count",
                "column_name": "",
                "value": int(len(dataframe)),
            }
        )

        for column in SUMMARY_NUMERIC_COLUMNS:
            if column not in dataframe.columns:
                continue

            series = pd.to_numeric(dataframe[column], errors="coerce")
            non_missing = series.dropna()
            missing_fraction = float(series.isna().mean()) if len(series) else pd.NA

            rows.append(
                {
                    "source_dataset": source_dataset,
                    "metric": "non_missing_count",
                    "column_name": column,
                    "value": int(non_missing.shape[0]),
                }
            )

            rows.append(
                {
                    "source_dataset": source_dataset,
                    "metric": "missing_fraction",
                    "column_name": column,
                    "value": missing_fraction,
                }
            )

            if non_missing.empty:
                continue

            rows.extend(
                [
                    {
                        "source_dataset": source_dataset,
                        "metric": "mean",
                        "column_name": column,
                        "value": float(non_missing.mean()),
                    },
                    {
                        "source_dataset": source_dataset,
                        "metric": "std",
                        "column_name": column,
                        "value": float(non_missing.std()),
                    },
                    {
                        "source_dataset": source_dataset,
                        "metric": "min",
                        "column_name": column,
                        "value": float(non_missing.min()),
                    },
                    {
                        "source_dataset": source_dataset,
                        "metric": "q25",
                        "column_name": column,
                        "value": float(non_missing.quantile(0.25)),
                    },
                    {
                        "source_dataset": source_dataset,
                        "metric": "median",
                        "column_name": column,
                        "value": float(non_missing.median()),
                    },
                    {
                        "source_dataset": source_dataset,
                        "metric": "q75",
                        "column_name": column,
                        "value": float(non_missing.quantile(0.75)),
                    },
                    {
                        "source_dataset": source_dataset,
                        "metric": "max",
                        "column_name": column,
                        "value": float(non_missing.max()),
                    },
                ]
            )

    return pd.DataFrame(rows)


def build_external_reference_audit(references: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for source_dataset, dataframe in references.items():
        row_count = int(len(dataframe))

        if source_dataset in REQUIRED_NON_EMPTY_REFERENCES and row_count == 0:
            row_status = "failed"
        elif source_dataset == "ns3_lena" and row_count == 0:
            row_status = "review"
        else:
            row_status = "ok"

        rows.append(
            {
                "source_dataset": source_dataset,
                "check_name": "row_count",
                "check_value": row_count,
                "status": row_status,
            }
        )

        rows.append(
            {
                "source_dataset": source_dataset,
                "check_name": "column_count",
                "check_value": int(len(dataframe.columns)),
                "status": "ok" if len(dataframe.columns) > 0 else "failed",
            }
        )

        if "reference_record_id" in dataframe.columns:
            duplicate_count = int(dataframe["reference_record_id"].duplicated().sum())
            rows.append(
                {
                    "source_dataset": source_dataset,
                    "check_name": "duplicate_reference_record_id_count",
                    "check_value": duplicate_count,
                    "status": "ok" if duplicate_count == 0 else "failed",
                }
            )

    return pd.DataFrame(rows)


UCC_RANGE_RULES = {
    "rsrp_dbm": (-156.0, -20.0),
    "rsrq_db": (-50.0, 0.0),
    "rssi_dbm": (-120.0, -20.0),
    "sinr_db": (-30.0, 50.0),
    "cqi": (0.0, 15.0),
    "latency_ms": (0.0, 2000.0),
    "jitter_ms": (0.0, 1000.0),
    "packet_loss_fraction": (0.0, 1.0),
    "speed_raw": (0.0, 250.0),
    "download_bitrate_raw": (0.0, None),
    "upload_bitrate_raw": (0.0, None),
    "download_throughput_mbps": (0.0, None),
    "upload_throughput_mbps": (0.0, None),
}


def _value_or_missing(value: Any) -> str:
    if pd.isna(value):
        return "<missing>"
    text = str(value).strip()
    return text if text else "<missing>"


def _float_or_na(value: Any) -> float | pd.NA:
    if pd.isna(value):
        return pd.NA
    return float(value)


def _distribution_audit_rows(
    dataframe: pd.DataFrame,
    column: str,
    max_values: int = 50,
) -> list[dict[str, Any]]:
    if column not in dataframe.columns:
        return [
            {
                "source_dataset": "ucc_5g_context",
                "audit_section": "distribution",
                "check_name": "column_missing",
                "column_name": column,
                "group_name": "",
                "group_value": "",
                "row_count": int(len(dataframe)),
                "non_missing_count": 0,
                "missing_fraction": pd.NA,
                "min": pd.NA,
                "q25": pd.NA,
                "median": pd.NA,
                "q75": pd.NA,
                "max": pd.NA,
                "affected_count": int(len(dataframe)),
                "affected_fraction": 1.0 if len(dataframe) else 0.0,
                "status": "failed",
                "notes": "Expected UCC context column is absent.",
            }
        ]

    values = dataframe[column].map(_value_or_missing)
    counts = values.value_counts(dropna=False).head(max_values)

    rows: list[dict[str, Any]] = []
    for value, count in counts.items():
        rows.append(
            {
                "source_dataset": "ucc_5g_context",
                "audit_section": "distribution",
                "check_name": "category_count",
                "column_name": column,
                "group_name": column,
                "group_value": value,
                "row_count": int(len(dataframe)),
                "non_missing_count": int(dataframe[column].notna().sum()),
                "missing_fraction": float(dataframe[column].isna().mean())
                if len(dataframe)
                else pd.NA,
                "min": pd.NA,
                "q25": pd.NA,
                "median": pd.NA,
                "q75": pd.NA,
                "max": pd.NA,
                "affected_count": int(count),
                "affected_fraction": float(count / len(dataframe))
                if len(dataframe)
                else 0.0,
                "status": "ok",
                "notes": "Observed UCC context distribution.",
            }
        )

    return rows


def _numeric_quality_row(dataframe: pd.DataFrame, column: str) -> dict[str, Any]:
    row_count = int(len(dataframe))

    if column not in dataframe.columns:
        return {
            "source_dataset": "ucc_5g_context",
            "audit_section": "numeric_quality",
            "check_name": "column_missing",
            "column_name": column,
            "group_name": "",
            "group_value": "",
            "row_count": row_count,
            "non_missing_count": 0,
            "missing_fraction": pd.NA,
            "min": pd.NA,
            "q25": pd.NA,
            "median": pd.NA,
            "q75": pd.NA,
            "max": pd.NA,
            "affected_count": row_count,
            "affected_fraction": 1.0 if row_count else 0.0,
            "status": "failed",
            "notes": "Expected UCC numeric column is absent.",
        }

    series = pd.to_numeric(dataframe[column], errors="coerce")
    non_missing = series.dropna()
    lower, upper = UCC_RANGE_RULES[column]

    if non_missing.empty:
        status = "review"
        affected_count = row_count
        affected_fraction = 1.0 if row_count else 0.0
        notes = "Column has no numeric values in the prepared UCC reference table."
        return {
            "source_dataset": "ucc_5g_context",
            "audit_section": "numeric_quality",
            "check_name": "numeric_range",
            "column_name": column,
            "group_name": "",
            "group_value": "",
            "row_count": row_count,
            "non_missing_count": 0,
            "missing_fraction": float(series.isna().mean()) if row_count else pd.NA,
            "min": pd.NA,
            "q25": pd.NA,
            "median": pd.NA,
            "q75": pd.NA,
            "max": pd.NA,
            "affected_count": affected_count,
            "affected_fraction": affected_fraction,
            "status": status,
            "notes": notes,
        }

    out_of_range = pd.Series(False, index=series.index)
    if lower is not None:
        out_of_range = out_of_range | (series < lower)
    if upper is not None:
        out_of_range = out_of_range | (series > upper)

    out_of_range = out_of_range & series.notna()
    affected_count = int(out_of_range.sum())
    affected_fraction = (
        float(affected_count / len(non_missing)) if len(non_missing) else 0.0
    )

    if column == "packet_loss_fraction" and affected_count > 0:
        status = "failed"
    elif affected_count > 0:
        status = "review"
    elif series.isna().mean() > 0.9:
        status = "review"
    else:
        status = "ok"

    notes_by_column = {
        "download_bitrate_raw": "Source bitrate is retained in kbps as reported by the UCC source documentation.",
        "upload_bitrate_raw": "Source bitrate is retained in kbps as reported by the UCC source documentation.",
        "download_throughput_mbps": "Derived from UCC DL_bitrate by converting kbps to Mbps.",
        "upload_throughput_mbps": "Derived from UCC UL_bitrate by converting kbps to Mbps.",
        "packet_loss_fraction": "Prepared value is stored as a fraction; percent-like source values are converted during UCC preparation.",
        "latency_ms": "Ping average is context-limited and is mostly populated for download traces.",
        "jitter_ms": "Ping standard deviation is context-limited and is mostly populated for download traces.",
    }

    if affected_count > 0:
        notes = (
            f"Values outside the configured review range: lower={lower}, upper={upper}."
        )
    else:
        notes = notes_by_column.get(column, "Numeric range check completed.")

    return {
        "source_dataset": "ucc_5g_context",
        "audit_section": "numeric_quality",
        "check_name": "numeric_range",
        "column_name": column,
        "group_name": "",
        "group_value": "",
        "row_count": row_count,
        "non_missing_count": int(non_missing.shape[0]),
        "missing_fraction": float(series.isna().mean()) if row_count else pd.NA,
        "min": _float_or_na(non_missing.min()),
        "q25": _float_or_na(non_missing.quantile(0.25)),
        "median": _float_or_na(non_missing.median()),
        "q75": _float_or_na(non_missing.quantile(0.75)),
        "max": _float_or_na(non_missing.max()),
        "affected_count": affected_count,
        "affected_fraction": affected_fraction,
        "status": status,
        "notes": notes,
    }


def build_ucc_reference_quality_audit(ucc: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    rows.append(
        {
            "source_dataset": "ucc_5g_context",
            "audit_section": "dataset",
            "check_name": "row_count",
            "column_name": "",
            "group_name": "",
            "group_value": "",
            "row_count": int(len(ucc)),
            "non_missing_count": int(len(ucc)),
            "missing_fraction": 0.0 if len(ucc) else pd.NA,
            "min": pd.NA,
            "q25": pd.NA,
            "median": pd.NA,
            "q75": pd.NA,
            "max": pd.NA,
            "affected_count": int(len(ucc)),
            "affected_fraction": 1.0 if len(ucc) else 0.0,
            "status": "ok" if len(ucc) else "failed",
            "notes": "Prepared UCC reference row count.",
        }
    )

    rows.extend(
        [
            {
                "source_dataset": "ucc_5g_context",
                "audit_section": "unit_handling",
                "check_name": "bitrate_unit_policy",
                "column_name": "download_bitrate_raw,upload_bitrate_raw,download_throughput_mbps,upload_throughput_mbps",
                "group_name": "",
                "group_value": "",
                "row_count": int(len(ucc)),
                "non_missing_count": int(
                    ucc[["download_bitrate_raw", "upload_bitrate_raw"]]
                    .notna()
                    .any(axis=1)
                    .sum()
                )
                if {"download_bitrate_raw", "upload_bitrate_raw"}.issubset(ucc.columns)
                else 0,
                "missing_fraction": pd.NA,
                "min": pd.NA,
                "q25": pd.NA,
                "median": pd.NA,
                "q75": pd.NA,
                "max": pd.NA,
                "affected_count": int(len(ucc)),
                "affected_fraction": 1.0 if len(ucc) else 0.0,
                "status": "ok",
                "notes": "UCC DL_bitrate and UL_bitrate are documented as kbps; raw bitrate values are retained and throughput_mbps fields are derived by dividing by 1000.",
            },
            {
                "source_dataset": "ucc_5g_context",
                "audit_section": "unit_handling",
                "check_name": "packet_loss_unit_policy",
                "column_name": "packet_loss_fraction",
                "group_name": "",
                "group_value": "",
                "row_count": int(len(ucc)),
                "non_missing_count": int(ucc["packet_loss_fraction"].notna().sum())
                if "packet_loss_fraction" in ucc.columns
                else 0,
                "missing_fraction": float(ucc["packet_loss_fraction"].isna().mean())
                if "packet_loss_fraction" in ucc.columns and len(ucc)
                else pd.NA,
                "min": pd.NA,
                "q25": pd.NA,
                "median": pd.NA,
                "q75": pd.NA,
                "max": pd.NA,
                "affected_count": 0,
                "affected_fraction": 0.0,
                "status": "ok",
                "notes": "PINGLOSS is converted to a fraction when source values are percent-like.",
            },
        ]
    )

    for column in [
        "application_context",
        "mobility_context",
        "technology",
        "network_mode",
        "scenario",
    ]:
        rows.extend(_distribution_audit_rows(ucc, column))

    for column in UCC_RANGE_RULES:
        rows.append(_numeric_quality_row(ucc, column))

    return pd.DataFrame(rows)


def build_simulator_readiness_audit(
    simulator_reference_dir: str | Path,
    ns3_lena_root: str | Path | None = None,
) -> pd.DataFrame:
    simulator_dir = Path(simulator_reference_dir)
    text_files = sorted(simulator_dir.rglob("*.txt")) if simulator_dir.exists() else []
    csv_files = _candidate_simulator_csv_files(simulator_dir)

    if csv_files:
        status = "ready_for_parser"
    elif simulator_dir.exists():
        status = "waiting_for_controlled_runs"
    else:
        status = "raw_directory_missing"

    rows = [
        {
            "source_dataset": "ns3_lena",
            "check_name": "simulator_reference_dir_exists",
            "check_value": bool(simulator_dir.exists()),
            "status": "ok" if simulator_dir.exists() else "review",
        },
        {
            "source_dataset": "ns3_lena",
            "check_name": "raw_text_file_count",
            "check_value": int(len(text_files)),
            "status": "ok" if text_files else "review",
        },
        {
            "source_dataset": "ns3_lena",
            "check_name": "curated_csv_file_count",
            "check_value": int(len(csv_files)),
            "status": "ok" if csv_files else "review",
        },
        {
            "source_dataset": "ns3_lena",
            "check_name": "readiness_status",
            "check_value": status,
            "status": "ok" if status == "ready_for_parser" else "review",
        },
    ]
    rows.extend(_ns3_local_readiness_rows(ns3_lena_root))
    return pd.DataFrame(rows)


def prepare_external_references(
    vienna_path: str | Path,
    campus_path: str | Path,
    ucc_path: str | Path,
    simulator_reference_dir: str | Path,
    output_dir: str | Path,
    mining_tables_dir: str | Path,
    results_dir: str | Path,
    root: Path,
    ns3_lena_root: str | Path | None = None,
) -> dict[str, Path]:
    output_dir = ensure_directory(output_dir)
    mining_tables_dir = ensure_directory(mining_tables_dir)
    results_dir = ensure_directory(results_dir)

    vienna = prepare_vienna_reference(vienna_path=vienna_path, root=root)
    campus = prepare_campus_reference(campus_path=campus_path, root=root)
    ucc = prepare_ucc_reference(ucc_path=ucc_path, root=root)
    simulator = prepare_simulator_reference(
        simulator_reference_dir=simulator_reference_dir,
        root=root,
    )

    references = {
        "vienna_4g5g": vienna,
        "campus_qos": campus,
        "ucc_5g_context": ucc,
        "ns3_lena": simulator,
    }

    summary = build_external_reference_summary(references)
    audit = build_external_reference_audit(references)
    simulator_readiness = build_simulator_readiness_audit(
        simulator_reference_dir=simulator_reference_dir,
        ns3_lena_root=ns3_lena_root,
    )
    ns3_run_manifest = build_ns3_lena_run_manifest(simulator_reference_dir)
    ns3_run_audit = build_ns3_lena_run_audit(simulator_reference_dir, simulator)
    ns3_summary = build_ns3_lena_summary(simulator)
    ucc_quality_audit = build_ucc_reference_quality_audit(ucc)

    output_paths = {
        "vienna_reference_clean": output_dir / "vienna_reference_clean.csv",
        "campus_qos_reference_clean": output_dir / "campus_qos_reference_clean.csv",
        "ucc_5g_reference_clean": output_dir / "ucc_5g_reference_clean.csv",
        "simulator_reference_clean": output_dir / "simulator_reference_clean.csv",
        "external_reference_summary": mining_tables_dir
        / "external_reference_summary.csv",
        "external_reference_preparation_audit": results_dir
        / "external_reference_preparation_audit.csv",
        "simulator_reference_readiness_audit": results_dir
        / "simulator_reference_readiness_audit.csv",
        "ns3_lena_run_manifest": results_dir / "ns3_lena_run_manifest.csv",
        "ns3_lena_run_audit": results_dir / "ns3_lena_run_audit.csv",
        "ns3_lena_summary": results_dir / "ns3_lena_summary.csv",
        "ucc_reference_quality_audit": results_dir / "ucc_reference_quality_audit.csv",
    }

    write_csv(vienna, output_paths["vienna_reference_clean"])
    write_csv(campus, output_paths["campus_qos_reference_clean"])
    write_csv(ucc, output_paths["ucc_5g_reference_clean"])
    write_csv(simulator, output_paths["simulator_reference_clean"])
    write_csv(summary, output_paths["external_reference_summary"])
    write_csv(audit, output_paths["external_reference_preparation_audit"])
    write_csv(simulator_readiness, output_paths["simulator_reference_readiness_audit"])
    write_csv(ns3_run_manifest, output_paths["ns3_lena_run_manifest"])
    write_csv(ns3_run_audit, output_paths["ns3_lena_run_audit"])
    write_csv(ns3_summary, output_paths["ns3_lena_summary"])
    write_csv(ucc_quality_audit, output_paths["ucc_reference_quality_audit"])

    failed_checks = audit.loc[audit["status"] == "failed"]
    if not failed_checks.empty:
        failed_names = ", ".join(
            f"{row.source_dataset}:{row.check_name}"
            for row in failed_checks.itertuples(index=False)
        )
        raise ValueError(
            f"External reference preparation failed checks: {failed_names}"
        )

    return output_paths
