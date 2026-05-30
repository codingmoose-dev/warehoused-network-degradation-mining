from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path, root: Path | None = None) -> Path:
    base = root if root is not None else project_root()
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base / path


def load_yaml(path: str | Path) -> dict[str, Any]:
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


def load_dataset_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else project_root() / "config" / "dataset_registry.yaml"
    registry = load_yaml(registry_path)

    datasets = registry.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("dataset_registry.yaml must contain a 'datasets' mapping.")

    return registry


def load_paths(path: str | Path | None = None) -> dict[str, Any]:
    paths_path = Path(path) if path is not None else project_root() / "config" / "paths.yaml"
    return load_yaml(paths_path)


def get_dataset_items(registry: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    datasets = registry.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("Registry field 'datasets' must be a mapping.")

    items: list[tuple[str, dict[str, Any]]] = []
    for dataset_id, metadata in datasets.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"Dataset metadata must be a mapping: {dataset_id}")
        items.append((str(dataset_id), metadata))

    return items