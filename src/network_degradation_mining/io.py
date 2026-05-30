from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_directory(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing YAML file: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {yaml_path}")

    return data


def write_yaml(data: dict[str, Any], path: str | Path) -> Path:
    output_path = ensure_parent_directory(path)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    return output_path


def read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(Path(path), **kwargs)


def write_csv(dataframe: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    output_path = ensure_parent_directory(path)
    dataframe.to_csv(output_path, index=index)
    return output_path


def read_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_parquet(Path(path), **kwargs)


def write_json(data: dict[str, Any] | list[dict[str, Any]], path: str | Path) -> Path:
    output_path = ensure_parent_directory(path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    return output_path


def file_size_bytes(path: str | Path) -> int | None:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    return target.stat().st_size