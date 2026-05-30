from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from network_degradation_mining.io import read_csv, write_csv


BOOLEAN_TRUE_VALUES = {"true", "1", "yes", "y"}
BOOLEAN_FALSE_VALUES = {"false", "0", "no", "n"}
BOOLEAN_VALUES = BOOLEAN_TRUE_VALUES | BOOLEAN_FALSE_VALUES


def normalize_column_name(name: str) -> str:
    value = str(name).strip()

    replacements = {
        "dBm": "dbm",
        "dB": "db",
        "GHz": "ghz",
        "Mbps": "mbps",
        "MB": "mb",
        "TX": "tx",
        "ID": "id",
        "UE": "ue",
        "LOS": "los",
        "RSRP": "rsrp",
        "VoNR": "vonr",
        "2D": "2d",
        "3D": "3d",
    }

    for source, target in replacements.items():
        value = value.replace(source, target)

    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    return value


def build_column_mapping(columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for source_column in columns:
        base_column = normalize_column_name(source_column)
        counts[base_column] = counts.get(base_column, 0) + 1

        if counts[base_column] == 1:
            standard_column = base_column
        else:
            standard_column = f"{base_column}_{counts[base_column]}"

        rows.append(
            {
                "source_column": source_column,
                "standard_column": standard_column,
                "base_standard_column": base_column,
                "duplicate_base_name": counts[base_column] > 1,
            }
        )

    return pd.DataFrame(rows)


def _strip_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()

    for column in output.columns:
        if pd.api.types.is_object_dtype(output[column]) or pd.api.types.is_string_dtype(output[column]):
            output[column] = output[column].astype("string").str.strip()

    return output


def _coerce_boolean_like_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()

    for column in output.columns:
        series = output[column]

        if pd.api.types.is_bool_dtype(series):
            output[column] = series.astype("boolean")
            continue

        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        normalized = series.dropna().astype("string").str.strip().str.lower()
        unique_values = set(normalized.unique().tolist())

        if unique_values and unique_values.issubset(BOOLEAN_VALUES):
            mapped = series.astype("string").str.strip().str.lower().map(
                {
                    "true": True,
                    "1": True,
                    "yes": True,
                    "y": True,
                    "false": False,
                    "0": False,
                    "no": False,
                    "n": False,
                }
            )
            output[column] = mapped.astype("boolean")

    return output


def _parse_timestamp_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()

    if "timestamp" not in output.columns:
        return output

    parsed = pd.to_datetime(output["timestamp"], errors="coerce")
    output["timestamp"] = parsed
    output["timestamp_parse_failed"] = parsed.isna()

    return output


def _add_stable_fields(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()

    source_row_number = pd.Series(range(1, len(output) + 1), index=output.index)

    output.insert(0, "source_row_number", source_row_number)
    output.insert(
        0,
        "measurement_id",
        source_row_number.map(lambda value: f"synnetqos_m_{value:08d}"),
    )

    sort_columns = [
        column
        for column in ["session_id", "timestamp", "source_row_number"]
        if column in output.columns
    ]

    if sort_columns:
        output = output.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    if "session_id" in output.columns:
        output["session_step_index"] = output.groupby("session_id", sort=False).cumcount() + 1
        output["session_record_count"] = output.groupby("session_id")["session_id"].transform("size")

    return output


def _build_missingness_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    row_count = len(dataframe)

    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())
        rows.append(
            {
                "column_name": column,
                "dtype": str(dataframe[column].dtype),
                "missing_count": missing_count,
                "missing_fraction": missing_count / row_count if row_count else 0.0,
                "non_missing_count": int(dataframe[column].notna().sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["missing_fraction", "column_name"],
        ascending=[False, True],
    )


def _build_value_profile(dataframe: pd.DataFrame, max_values: int = 25) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        unique_count = int(series.nunique(dropna=True))

        if unique_count > max_values:
            continue

        value_counts = series.astype("string").fillna("<missing>").value_counts(dropna=False).head(max_values)

        for value, count in value_counts.items():
            rows.append(
                {
                    "column_name": column,
                    "value": value,
                    "count": int(count),
                    "fraction": float(count / len(dataframe)) if len(dataframe) else 0.0,
                    "unique_count": unique_count,
                }
            )

    return pd.DataFrame(rows)


def _build_cleaning_audit(
    raw_dataframe: pd.DataFrame,
    clean_dataframe: pd.DataFrame,
    column_mapping: pd.DataFrame,
) -> pd.DataFrame:
    duplicate_measurement_ids = int(clean_dataframe["measurement_id"].duplicated().sum())

    timestamp_parse_failures: int | None = None
    if "timestamp_parse_failed" in clean_dataframe.columns:
        timestamp_parse_failures = int(clean_dataframe["timestamp_parse_failed"].sum())

    duplicate_session_timestamps: int | None = None
    if {"session_id", "timestamp"}.issubset(clean_dataframe.columns):
        duplicate_session_timestamps = int(clean_dataframe.duplicated(["session_id", "timestamp"]).sum())

    duplicate_base_names = int(column_mapping["duplicate_base_name"].sum())

    rows = [
        {
            "check_name": "raw_row_count",
            "check_value": int(len(raw_dataframe)),
            "status": "ok",
        },
        {
            "check_name": "output_row_count",
            "check_value": int(len(clean_dataframe)),
            "status": "ok",
        },
        {
            "check_name": "row_count_preserved",
            "check_value": bool(len(raw_dataframe) == len(clean_dataframe)),
            "status": "ok" if len(raw_dataframe) == len(clean_dataframe) else "failed",
        },
        {
            "check_name": "raw_column_count",
            "check_value": int(len(raw_dataframe.columns)),
            "status": "ok",
        },
        {
            "check_name": "output_column_count",
            "check_value": int(len(clean_dataframe.columns)),
            "status": "ok",
        },
        {
            "check_name": "duplicate_standard_column_base_count",
            "check_value": duplicate_base_names,
            "status": "review" if duplicate_base_names else "ok",
        },
        {
            "check_name": "timestamp_parse_failure_count",
            "check_value": timestamp_parse_failures,
            "status": "review" if timestamp_parse_failures else "ok",
        },
        {
            "check_name": "duplicate_measurement_id_count",
            "check_value": duplicate_measurement_ids,
            "status": "ok" if duplicate_measurement_ids == 0 else "failed",
        },
        {
            "check_name": "duplicate_session_timestamp_count",
            "check_value": duplicate_session_timestamps,
            "status": "review" if duplicate_session_timestamps else "ok",
        },
    ]

    return pd.DataFrame(rows)


def _build_dataset_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric": "row_count",
            "value": int(len(dataframe)),
        },
        {
            "metric": "column_count",
            "value": int(len(dataframe.columns)),
        },
    ]

    if "session_id" in dataframe.columns:
        rows.append(
            {
                "metric": "unique_session_count",
                "value": int(dataframe["session_id"].nunique(dropna=True)),
            }
        )

    if "user_id" in dataframe.columns:
        rows.append(
            {
                "metric": "unique_user_count",
                "value": int(dataframe["user_id"].nunique(dropna=True)),
            }
        )

    if "timestamp" in dataframe.columns:
        timestamp_min = dataframe["timestamp"].min()
        timestamp_max = dataframe["timestamp"].max()

        rows.extend(
            [
                {
                    "metric": "timestamp_min",
                    "value": "" if pd.isna(timestamp_min) else str(timestamp_min),
                },
                {
                    "metric": "timestamp_max",
                    "value": "" if pd.isna(timestamp_max) else str(timestamp_max),
                },
            ]
        )

    return pd.DataFrame(rows)


def prepare_synnetqos_core(
    input_path: str | Path,
    output_path: str | Path,
    audit_dir: str | Path,
) -> dict[str, Path]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    audit_dir = Path(audit_dir)

    raw = read_csv(input_path, low_memory=False)

    column_mapping = build_column_mapping(list(raw.columns))
    rename_map = dict(zip(column_mapping["source_column"], column_mapping["standard_column"]))

    clean = raw.rename(columns=rename_map)
    clean = _strip_text_columns(clean)
    clean = _coerce_boolean_like_columns(clean)
    clean = _parse_timestamp_column(clean)
    clean = clean.convert_dtypes()
    clean = _add_stable_fields(clean)

    cleaning_audit = _build_cleaning_audit(
        raw_dataframe=raw,
        clean_dataframe=clean,
        column_mapping=column_mapping,
    )

    failed_checks = cleaning_audit.loc[cleaning_audit["status"] == "failed"]
    if not failed_checks.empty:
        failed_names = ", ".join(failed_checks["check_name"].astype(str).tolist())
        raise ValueError(f"SynNetQoS preparation failed checks: {failed_names}")

    missingness_summary = _build_missingness_summary(clean)
    value_profile = _build_value_profile(clean)
    dataset_summary = _build_dataset_summary(clean)

    output_paths = {
        "synnetqos_core": output_path,
        "synnetqos_cleaning_audit": audit_dir / "synnetqos_cleaning_audit.csv",
        "synnetqos_column_mapping": audit_dir / "synnetqos_column_mapping.csv",
        "synnetqos_missingness_summary": audit_dir / "synnetqos_missingness_summary.csv",
        "synnetqos_value_profile": audit_dir / "synnetqos_value_profile.csv",
        "synnetqos_dataset_summary": audit_dir / "synnetqos_dataset_summary.csv",
    }

    write_csv(clean, output_paths["synnetqos_core"])
    write_csv(cleaning_audit, output_paths["synnetqos_cleaning_audit"])
    write_csv(column_mapping, output_paths["synnetqos_column_mapping"])
    write_csv(missingness_summary, output_paths["synnetqos_missingness_summary"])
    write_csv(value_profile, output_paths["synnetqos_value_profile"])
    write_csv(dataset_summary, output_paths["synnetqos_dataset_summary"])

    return output_paths