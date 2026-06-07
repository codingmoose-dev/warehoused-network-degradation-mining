from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REQUIRED_MODEL_COLUMNS = {
    "model_name",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_supervised_model_comparison_output_schema() -> None:
    path = _project_root() / "results" / "supervised" / "model_comparison.csv"
    if not path.exists():
        pytest.skip(
            "Run scripts/06_supervised_classification.py before this output check."
        )

    table = pd.read_csv(path)
    assert REQUIRED_MODEL_COLUMNS.issubset(table.columns)
    assert not table.empty
    assert table["f1"].dropna().between(0, 1).all()


def test_graph_neural_model_comparison_output_schema() -> None:
    path = _project_root() / "results" / "graph" / "gnn_model_comparison.csv"
    if not path.exists():
        pytest.skip("Run scripts/13_gnn.py before this output check.")

    table = pd.read_csv(path)
    assert REQUIRED_MODEL_COLUMNS.issubset(table.columns)
    assert not table.empty
    assert table["f1"].dropna().between(0, 1).all()
