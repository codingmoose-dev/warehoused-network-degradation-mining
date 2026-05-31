from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from network_degradation_mining.config import get_dataset_items, resolve_project_path
from network_degradation_mining.io import ensure_directory, write_csv


SUPPORTED_TABLE_EXTENSIONS = {".csv", ".parquet"}


def _list_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted([item for item in path.rglob("*") if item.is_file()])
    return []


def _is_supported_table(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_TABLE_EXTENSIONS


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _count_csv_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
            reader = csv.reader(file)
            row_count = sum(1 for _ in reader)
        return max(row_count - 1, 0)
    except Exception:
        return None


def _read_csv_sample(path: Path, sample_rows: int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=sample_rows, low_memory=False)


def _inspect_csv(
    path: Path, sample_rows: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        sample = _read_csv_sample(path, sample_rows)
        row_count = _count_csv_rows(path)

        file_summary = {
            "row_count": row_count,
            "column_count": int(len(sample.columns)),
            "read_status": "ok",
            "error_message": "",
        }

        column_rows = _build_column_rows(sample=sample, sample_rows=sample_rows)
        return file_summary, column_rows

    except Exception as exc:
        return (
            {
                "row_count": None,
                "column_count": None,
                "read_status": "failed",
                "error_message": str(exc),
            },
            [],
        )


def _inspect_parquet(
    path: Path, sample_rows: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        row_count: int | None = None

        try:
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(path)
            row_count = int(parquet_file.metadata.num_rows)
        except Exception:
            row_count = None

        sample = pd.read_parquet(path)
        if len(sample) > sample_rows:
            sample = sample.head(sample_rows)

        file_summary = {
            "row_count": row_count if row_count is not None else int(len(sample)),
            "column_count": int(len(sample.columns)),
            "read_status": "ok",
            "error_message": "",
        }

        column_rows = _build_column_rows(sample=sample, sample_rows=sample_rows)
        return file_summary, column_rows

    except Exception as exc:
        return (
            {
                "row_count": None,
                "column_count": None,
                "read_status": "failed",
                "error_message": str(exc),
            },
            [],
        )


def _build_column_rows(sample: pd.DataFrame, sample_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    observed_rows = int(len(sample))

    for column in sample.columns:
        series = sample[column]
        examples = series.dropna().astype(str).head(3).tolist()

        rows.append(
            {
                "column_name": str(column),
                "dtype_in_sample": str(series.dtype),
                "sample_size": observed_rows,
                "sample_missing_count": int(series.isna().sum()),
                "sample_missing_fraction": float(series.isna().mean())
                if observed_rows > 0
                else 0.0,
                "example_values": json.dumps(examples, ensure_ascii=False),
                "sample_row_limit": int(sample_rows),
            }
        )

    return rows


def _inspect_table(
    path: Path, sample_rows: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _inspect_csv(path, sample_rows)

    if suffix == ".parquet":
        return _inspect_parquet(path, sample_rows)

    return (
        {
            "row_count": None,
            "column_count": None,
            "read_status": "unsupported",
            "error_message": f"Unsupported extension: {suffix}",
        },
        [],
    )


def audit_raw_inputs(
    registry: dict[str, Any],
    root: Path,
    output_dir: str | Path = "results/audits",
    sample_rows: int = 1000,
) -> dict[str, Path]:
    audit_dir = ensure_directory(resolve_project_path(output_dir, root))

    dataset_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []

    for dataset_id, metadata in get_dataset_items(registry):
        raw_path_value = metadata.get("raw_path")

        if raw_path_value is None:
            dataset_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": metadata.get("name", ""),
                    "role": metadata.get("role", ""),
                    "priority": metadata.get("priority", ""),
                    "raw_path": "",
                    "exists": False,
                    "is_file": False,
                    "is_dir": False,
                    "file_count": 0,
                    "supported_table_count": 0,
                    "total_size_bytes": 0,
                    "audit_status": "missing_raw_path_field",
                    "error_message": "Missing raw_path field.",
                }
            )
            continue

        raw_path = resolve_project_path(raw_path_value, root)
        files = _list_files(raw_path)
        supported_files = [path for path in files if _is_supported_table(path)]

        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": metadata.get("name", ""),
                "role": metadata.get("role", ""),
                "priority": metadata.get("priority", ""),
                "raw_path": _safe_relative_path(raw_path, root),
                "exists": raw_path.exists(),
                "is_file": raw_path.is_file(),
                "is_dir": raw_path.is_dir(),
                "file_count": len(files),
                "supported_table_count": len(supported_files),
                "total_size_bytes": sum(
                    path.stat().st_size for path in files if path.exists()
                ),
                "audit_status": "ok" if raw_path.exists() else "missing",
                "error_message": ""
                if raw_path.exists()
                else "Raw path does not exist.",
            }
        )

        for file_path in files:
            is_supported = _is_supported_table(file_path)
            file_base = {
                "dataset_id": dataset_id,
                "dataset_name": metadata.get("name", ""),
                "role": metadata.get("role", ""),
                "priority": metadata.get("priority", ""),
                "file_path": _safe_relative_path(file_path, root),
                "file_name": file_path.name,
                "extension": file_path.suffix.lower(),
                "size_bytes": file_path.stat().st_size,
                "is_supported_table": is_supported,
            }

            if not is_supported:
                file_rows.append(
                    {
                        **file_base,
                        "row_count": None,
                        "column_count": None,
                        "read_status": "not_table",
                        "error_message": "",
                    }
                )
                continue

            file_summary, file_column_rows = _inspect_table(
                file_path, sample_rows=sample_rows
            )

            file_rows.append({**file_base, **file_summary})

            for column_row in file_column_rows:
                column_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset_name": metadata.get("name", ""),
                        "file_path": _safe_relative_path(file_path, root),
                        **column_row,
                    }
                )

    dataset_manifest = pd.DataFrame(dataset_rows)
    file_inventory = pd.DataFrame(file_rows)
    column_inventory = pd.DataFrame(column_rows)

    output_paths = {
        "raw_input_manifest": audit_dir / "raw_input_manifest.csv",
        "raw_file_inventory": audit_dir / "raw_file_inventory.csv",
        "raw_column_inventory": audit_dir / "raw_column_inventory.csv",
    }

    write_csv(dataset_manifest, output_paths["raw_input_manifest"])
    write_csv(file_inventory, output_paths["raw_file_inventory"])
    write_csv(column_inventory, output_paths["raw_column_inventory"])

    return output_paths
