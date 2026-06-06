from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from network_degradation_mining.io import ensure_directory, read_csv, write_csv
from network_degradation_mining.plotting import (
    plot_gnn_vs_classical_model_comparison,
)

TARGET_COLUMN = "service_degraded"
MODEL_RANDOM_STATE = 42
NUMERIC_CLIP_LIMIT = 8.0
WAREHOUSE_VIEW = "warehouse_measurement"
EXTERNAL_VIEW = "external_reference_evidence"

GRAPH_CONTEXT_EDGE_TYPES = {
    "observed_in_time_bin",
    "observed_in_area_type",
    "uses_network_type",
    "uses_infrastructure",
    "uses_radio_band",
    "uses_application",
    "has_mobility_state",
    "has_weather_context",
    "has_obstruction_context",
    "has_congestion_level",
    "has_tower_load",
}

IDENTIFIER_COLUMNS = {
    "source_node_id",
    "node_id",
    "measurement_id",
    "session_id",
    TARGET_COLUMN,
}


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _prediction_scores(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        if proba.shape[1] > 1:
            return np.asarray(proba[:, 1], dtype=float)
        return np.asarray(proba[:, 0], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    return np.asarray(model.predict(features), dtype=float)


def _safe_numeric_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    for column in output.columns:
        if pd.api.types.is_numeric_dtype(output[column]):
            output[column] = pd.to_numeric(output[column], errors="coerce")
            output[column] = output[column].replace([np.inf, -np.inf], np.nan)
    return output


def _clip_numeric_array(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -NUMERIC_CLIP_LIMIT, NUMERIC_CLIP_LIMIT)


def _metric_row(
    model_name: str,
    y_true: pd.Series,
    predictions: np.ndarray,
    scores: np.ndarray,
    train_count: int,
    test_count: int,
) -> dict[str, Any]:
    y_array = np.asarray(y_true, dtype=int)
    pred_array = np.asarray(predictions, dtype=int)
    row: dict[str, Any] = {
        "model_name": model_name,
        "train_row_count": int(train_count),
        "test_row_count": int(test_count),
        "accuracy": float(accuracy_score(y_array, pred_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, pred_array)),
        "precision": float(precision_score(y_array, pred_array, zero_division=0)),
        "recall": float(recall_score(y_array, pred_array, zero_division=0)),
        "f1": float(f1_score(y_array, pred_array, zero_division=0)),
    }
    try:
        row["roc_auc"] = float(roc_auc_score(y_array, scores))
    except ValueError:
        row["roc_auc"] = np.nan
    return row


def _confusion_rows(
    model_name: str,
    y_true: pd.Series,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    y_array = np.asarray(y_true, dtype=int)
    pred_array = np.asarray(predictions, dtype=int)
    rows: list[dict[str, Any]] = []
    for actual in (0, 1):
        for predicted in (0, 1):
            count = int(((y_array == actual) & (pred_array == predicted)).sum())
            rows.append(
                {
                    "model_name": model_name,
                    "actual_class": actual,
                    "predicted_class": predicted,
                    "count": count,
                }
            )
    return rows


def _preprocessor(dataframe: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = [
        column
        for column in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    categorical_columns = [
        column for column in dataframe.columns if column not in numeric_columns
    ]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "clipper",
                            FunctionTransformer(_clip_numeric_array, validate=False),
                        ),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _make_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _warehouse_measurement_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    if nodes.empty:
        return pd.DataFrame()
    return nodes.loc[
        nodes["graph_view"].astype(str).eq(WAREHOUSE_VIEW)
        & nodes["node_type"].astype(str).eq("measurement")
    ].copy()


def _warehouse_edges(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    return edges.loc[edges["graph_view"].astype(str).eq(WAREHOUSE_VIEW)].copy()


def _context_edge_frame(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    measurement = _warehouse_measurement_nodes(nodes)
    measurement_ids = set(measurement["node_id"].astype(str))
    warehouse_edges = _warehouse_edges(edges)
    context_edges = warehouse_edges.loc[
        warehouse_edges["source_node_id"].astype(str).isin(measurement_ids)
        & warehouse_edges["edge_type"].astype(str).isin(GRAPH_CONTEXT_EDGE_TYPES)
    ].copy()
    if context_edges.empty:
        return pd.DataFrame(columns=["source_node_id"])
    context_edges["edge_type"] = context_edges["edge_type"].astype(str)
    context_edges["target_node_id"] = context_edges["target_node_id"].astype(str)
    return (
        context_edges.groupby(["source_node_id", "edge_type"])["target_node_id"]
        .first()
        .unstack("edge_type")
        .reset_index()
    )


def _degree_features(
    edges: pd.DataFrame,
    measurement_node_ids: set[str],
) -> pd.DataFrame:
    rows = pd.DataFrame({"source_node_id": sorted(measurement_node_ids)})
    rows["graph_degree"] = 0
    rows["transition_out_degree"] = 0
    rows["transition_in_degree"] = 0
    if edges.empty:
        return rows

    warehouse_edges = _warehouse_edges(edges)
    edge_source = warehouse_edges["source_node_id"].astype(str)
    edge_target = warehouse_edges["target_node_id"].astype(str)
    all_degree = pd.concat([edge_source, edge_target], ignore_index=True).value_counts()
    transition_edges = warehouse_edges.loc[
        warehouse_edges["edge_type"].astype(str).eq("next_measurement")
    ]
    transition_out = transition_edges["source_node_id"].astype(str).value_counts()
    transition_in = transition_edges["target_node_id"].astype(str).value_counts()
    rows["graph_degree"] = rows["source_node_id"].map(all_degree).fillna(0).astype(int)
    rows["transition_out_degree"] = (
        rows["source_node_id"].map(transition_out).fillna(0).astype(int)
    )
    rows["transition_in_degree"] = (
        rows["source_node_id"].map(transition_in).fillna(0).astype(int)
    )
    return rows


def _base_dataset(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    measurement = _warehouse_measurement_nodes(nodes)
    if measurement.empty:
        raise ValueError("Graph benchmark requires warehouse measurement nodes.")
    if TARGET_COLUMN not in measurement.columns:
        raise ValueError(
            f"Graph benchmark requires {TARGET_COLUMN} on measurement nodes."
        )

    measurement = measurement.copy()
    measurement["source_node_id"] = measurement["node_id"].astype(str)
    measurement[TARGET_COLUMN] = (
        pd.to_numeric(measurement[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    )
    measurement_node_ids = set(measurement["source_node_id"].astype(str))
    context_features = _context_edge_frame(nodes, edges)
    degree_features = _degree_features(edges, measurement_node_ids)

    base_columns = ["source_node_id", "measurement_id", "session_id", TARGET_COLUMN]
    base_columns = [column for column in base_columns if column in measurement.columns]
    output = measurement[base_columns].copy()
    output = output.merge(context_features, on="source_node_id", how="left")
    output = output.merge(degree_features, on="source_node_id", how="left")
    return _safe_numeric_frame(output)


def _split_dataset(dataframe: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    y = dataframe[TARGET_COLUMN].astype(int)
    has_groups = "session_id" in dataframe.columns
    has_groups = has_groups and dataframe["session_id"].nunique(dropna=True) > 1
    if has_groups:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=MODEL_RANDOM_STATE,
        )
        train_index, test_index = next(
            splitter.split(dataframe, y, groups=dataframe["session_id"].astype(str))
        )
        return dataframe.index[train_index], dataframe.index[test_index]

    train_index, test_index = train_test_split(
        dataframe.index,
        test_size=0.2,
        random_state=MODEL_RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )
    return pd.Index(train_index), pd.Index(test_index)


def _add_train_context_rates(
    dataframe: pd.DataFrame,
    train_index: pd.Index,
) -> pd.DataFrame:
    output = dataframe.copy()
    global_rate = float(output.loc[train_index, TARGET_COLUMN].mean())
    context_columns = [
        column for column in GRAPH_CONTEXT_EDGE_TYPES if column in output.columns
    ]
    for column in context_columns:
        rate_column = f"train_context_rate_{column}"
        train_rates = (
            output.loc[train_index]
            .groupby(column, dropna=False)[TARGET_COLUMN]
            .mean()
            .to_dict()
        )
        output[rate_column] = output[column].map(train_rates).fillna(global_rate)
    return _safe_numeric_frame(output)


def _feature_frame(
    dataframe: pd.DataFrame,
    include_context_rates: bool,
) -> pd.DataFrame:
    excluded = set(IDENTIFIER_COLUMNS)
    if not include_context_rates:
        excluded.update(
            column
            for column in dataframe.columns
            if column.startswith("train_context_rate_")
        )
    feature_columns = [column for column in dataframe.columns if column not in excluded]
    return dataframe[feature_columns].copy()


def _run_model(
    model_name: str,
    estimator: Any,
    dataframe: pd.DataFrame,
    train_index: pd.Index,
    test_index: pd.Index,
    include_context_rates: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    features = _feature_frame(dataframe, include_context_rates=include_context_rates)
    target = dataframe[TARGET_COLUMN].astype(int)
    x_train = features.loc[train_index]
    x_test = features.loc[test_index]
    y_train = target.loc[train_index]
    y_test = target.loc[test_index]

    model = Pipeline(
        steps=[
            ("preprocess", _preprocessor(features)),
            ("model", estimator),
        ]
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
        predictions = np.asarray(model.predict(x_test), dtype=int)
        scores = _prediction_scores(model, x_test)
    row = _metric_row(
        model_name=model_name,
        y_true=y_test,
        predictions=predictions,
        scores=scores,
        train_count=len(y_train),
        test_count=len(y_test),
    )
    relevant_warnings = [
        warning
        for warning in caught_warnings
        if issubclass(warning.category, (RuntimeWarning, ConvergenceWarning))
    ]
    row["model_warning_count"] = int(len(relevant_warnings))
    row["model_warning_types"] = "; ".join(
        sorted({warning.category.__name__ for warning in relevant_warnings})
    )
    return row, _confusion_rows(model_name, y_test, predictions)



def _transition_edge_index(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    dataset: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    node_to_dataset_index = dict(
        zip(dataset["source_node_id"].astype(str), range(len(dataset)))
    )
    warehouse_edges = _warehouse_edges(edges)
    transition_edges = warehouse_edges.loc[
        warehouse_edges.get("edge_type", pd.Series(dtype=str))
        .astype(str)
        .eq("next_measurement")
    ].copy()
    sources: list[int] = []
    targets: list[int] = []
    for row in transition_edges.itertuples(index=False):
        data = row._asdict()
        source = node_to_dataset_index.get(str(data.get("source_node_id")))
        target = node_to_dataset_index.get(str(data.get("target_node_id")))
        if source is None or target is None:
            continue
        sources.extend([source, target])
        targets.extend([target, source])
    if not sources:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def _transition_split_audit(
    dataset: pd.DataFrame,
    edges: pd.DataFrame,
    train_index: pd.Index,
    test_index: pd.Index,
) -> pd.DataFrame:
    train_nodes = set(dataset.loc[train_index, "source_node_id"].astype(str))
    test_nodes = set(dataset.loc[test_index, "source_node_id"].astype(str))
    train_sessions = set(dataset.loc[train_index, "session_id"].astype(str))
    test_sessions = set(dataset.loc[test_index, "session_id"].astype(str))

    warehouse_edges = _warehouse_edges(edges)
    transition_edges = warehouse_edges.loc[
        warehouse_edges.get("edge_type", pd.Series(dtype=str))
        .astype(str)
        .eq("next_measurement")
    ].copy()

    source = transition_edges.get("source_node_id", pd.Series(dtype=str)).astype(str)
    target = transition_edges.get("target_node_id", pd.Series(dtype=str)).astype(str)
    train_edge_count = int((source.isin(train_nodes) & target.isin(train_nodes)).sum())
    test_edge_count = int((source.isin(test_nodes) & target.isin(test_nodes)).sum())
    cross_split_edge_count = int(
        (
            (source.isin(train_nodes) & target.isin(test_nodes))
            | (source.isin(test_nodes) & target.isin(train_nodes))
        ).sum()
    )

    return pd.DataFrame(
        [
            {
                "check_name": "session_group_overlap_count",
                "check_value": int(len(train_sessions & test_sessions)),
                "status": "ok" if not (train_sessions & test_sessions) else "failed",
            },
            {
                "check_name": "transition_edge_cross_split_count",
                "check_value": cross_split_edge_count,
                "status": "ok" if cross_split_edge_count == 0 else "failed",
            },
            {
                "check_name": "transition_train_edge_count",
                "check_value": train_edge_count,
                "status": "ok" if train_edge_count > 0 else "review",
            },
            {
                "check_name": "transition_test_edge_count",
                "check_value": test_edge_count,
                "status": "ok" if test_edge_count > 0 else "review",
            },
            {
                "check_name": "gnn_inference_scope",
                "check_value": "offline_observed_within_session_transition_benchmark",
                "status": "ok",
            },
        ]
    )


def _validation_split(
    dataframe: pd.DataFrame,
    train_index: pd.Index,
) -> tuple[pd.Index, pd.Index]:
    train_frame = dataframe.loc[train_index]
    y_train = train_frame[TARGET_COLUMN].astype(int)
    has_groups = "session_id" in train_frame.columns
    has_groups = has_groups and train_frame["session_id"].nunique(dropna=True) > 1
    if has_groups:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=MODEL_RANDOM_STATE + 7,
        )
        inner_train, validation = next(
            splitter.split(
                train_frame,
                y_train,
                groups=train_frame["session_id"].astype(str),
            )
        )
        return train_frame.index[inner_train], train_frame.index[validation]

    if len(train_frame) < 5:
        return train_index, train_index
    inner_train, validation = train_test_split(
        train_frame.index,
        test_size=0.2,
        random_state=MODEL_RANDOM_STATE + 7,
        stratify=y_train if y_train.nunique() > 1 else None,
    )
    return pd.Index(inner_train), pd.Index(validation)


def _dense_feature_matrix(
    dataframe: pd.DataFrame,
    train_index: pd.Index,
) -> tuple[np.ndarray, list[str]]:
    features = _feature_frame(dataframe, include_context_rates=False)
    transformer = _preprocessor(features)
    transformer.fit(features.loc[train_index])
    matrix = transformer.transform(features)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=NUMERIC_CLIP_LIMIT,
        neginf=-NUMERIC_CLIP_LIMIT,
    )
    matrix = np.clip(matrix, -NUMERIC_CLIP_LIMIT, NUMERIC_CLIP_LIMIT)
    return matrix, list(features.columns)


def _torch_device() -> Any:
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalised_gcn_edges(
    edge_source: Any,
    edge_target: Any,
    node_count: int,
) -> tuple[Any, Any, Any]:
    import torch

    device = edge_source.device
    self_loops = torch.arange(node_count, dtype=torch.long, device=device)
    row = torch.cat([edge_source, self_loops])
    col = torch.cat([edge_target, self_loops])
    degree = torch.zeros(node_count, dtype=torch.float32, device=device)
    degree.index_add_(0, col, torch.ones_like(col, dtype=torch.float32))
    degree = degree.clamp(min=1.0)
    norm = degree[row].pow(-0.5) * degree[col].pow(-0.5)
    return row, col, norm


def _mean_aggregation_edges(
    edge_source: Any,
    edge_target: Any,
    node_count: int,
) -> tuple[Any, Any, Any]:
    import torch

    device = edge_source.device
    self_loops = torch.arange(node_count, dtype=torch.long, device=device)
    row = torch.cat([edge_source, self_loops])
    col = torch.cat([edge_target, self_loops])
    degree = torch.zeros(node_count, dtype=torch.float32, device=device)
    degree.index_add_(0, col, torch.ones_like(col, dtype=torch.float32))
    degree = degree.clamp(min=1.0)
    return row, col, degree


def _run_true_gnn_models(
    dataset: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    train_index: pd.Index,
    test_index: pd.Index,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    audit_rows: list[dict[str, Any]] = []
    try:
        import torch
        import torch.nn.functional as functional
        from torch import nn
    except ImportError as exc:
        return [], [], pd.DataFrame(
            [
                {
                    "check_name": "true_gnn_dependency",
                    "check_value": f"torch_unavailable: {exc}",
                    "status": "skipped",
                }
            ]
        )

    torch.manual_seed(MODEL_RANDOM_STATE)
    try:
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    except RuntimeError:
        pass

    sources, targets = _transition_edge_index(nodes, edges, dataset)
    if len(sources) == 0:
        return [], [], pd.DataFrame(
            [
                {
                    "check_name": "true_gnn_transition_edges",
                    "check_value": 0,
                    "status": "skipped",
                }
            ]
        )

    feature_matrix, feature_columns = _dense_feature_matrix(dataset, train_index)
    y = dataset[TARGET_COLUMN].astype(int).to_numpy(dtype=np.int64)
    inner_train_index, validation_index = _validation_split(dataset, train_index)
    position_by_index = {
        index: position for position, index in enumerate(dataset.index)
    }
    train_positions = np.asarray(
        [position_by_index[index] for index in inner_train_index],
        dtype=np.int64,
    )
    validation_positions = np.asarray(
        [position_by_index[index] for index in validation_index],
        dtype=np.int64,
    )
    full_train_positions = np.asarray(
        [position_by_index[index] for index in train_index],
        dtype=np.int64,
    )
    test_positions = np.asarray(
        [position_by_index[index] for index in test_index],
        dtype=np.int64,
    )

    device = _torch_device()
    x_tensor = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    edge_source = torch.tensor(sources, dtype=torch.long, device=device)
    edge_target = torch.tensor(targets, dtype=torch.long, device=device)
    train_tensor = torch.tensor(train_positions, dtype=torch.long, device=device)
    validation_tensor = torch.tensor(
        validation_positions,
        dtype=torch.long,
        device=device,
    )
    test_tensor = torch.tensor(test_positions, dtype=torch.long, device=device)

    class_counts = np.bincount(y[full_train_positions], minlength=2).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    class_weights = class_counts.sum() / (2.0 * class_counts)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    class GCNLayer(nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, features: Any, row: Any, col: Any, norm: Any) -> Any:
            transformed = self.linear(features)
            messages = transformed[row] * norm.unsqueeze(1)
            output = torch.zeros_like(transformed)
            output.index_add_(0, col, messages)
            return output

    class GraphSAGELayer(nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.self_linear = nn.Linear(input_dim, output_dim)
            self.neighbor_linear = nn.Linear(input_dim, output_dim)

        def forward(self, features: Any, row: Any, col: Any, degree: Any) -> Any:
            aggregated = torch.zeros_like(features)
            aggregated.index_add_(0, col, features[row])
            aggregated = aggregated / degree.unsqueeze(1)
            return self.self_linear(features) + self.neighbor_linear(aggregated)

    class GCNModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.layer1 = GCNLayer(input_dim, hidden_dim)
            self.layer2 = GCNLayer(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.15)

        def forward(self, features: Any, edge_source_: Any, edge_target_: Any) -> Any:
            row, col, norm = _normalised_gcn_edges(
                edge_source_,
                edge_target_,
                features.shape[0],
            )
            hidden = self.layer1(features, row, col, norm)
            hidden = functional.relu(hidden)
            hidden = self.dropout(hidden)
            return self.layer2(hidden, row, col, norm)

    class GraphSAGEModel(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
            super().__init__()
            self.layer1 = GraphSAGELayer(input_dim, hidden_dim)
            self.layer2 = GraphSAGELayer(hidden_dim, output_dim)
            self.dropout = nn.Dropout(0.15)

        def forward(self, features: Any, edge_source_: Any, edge_target_: Any) -> Any:
            row, col, degree = _mean_aggregation_edges(
                edge_source_,
                edge_target_,
                features.shape[0],
            )
            hidden = self.layer1(features, row, col, degree)
            hidden = functional.relu(hidden)
            hidden = self.dropout(hidden)
            return self.layer2(hidden, row, col, degree)

    def train_model(
        model_name: str,
        model: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        best_state: dict[str, Any] | None = None
        best_validation_f1 = -1.0
        best_epoch = 0
        patience = 18
        stale_epochs = 0
        max_epochs = 120
        for epoch in range(1, max_epochs + 1):
            model.train()
            optimiser.zero_grad()
            logits = model(x_tensor, edge_source, edge_target)
            loss = functional.cross_entropy(
                logits[train_tensor],
                y_tensor[train_tensor],
                weight=weight_tensor,
            )
            loss.backward()
            optimiser.step()

            model.eval()
            with torch.no_grad():
                validation_logits = model(x_tensor, edge_source, edge_target)[
                    validation_tensor
                ]
                validation_scores = torch.softmax(validation_logits, dim=1)[:, 1]
                validation_pred = (validation_scores >= 0.5).long().cpu().numpy()
                validation_true = y_tensor[validation_tensor].cpu().numpy()
            validation_f1 = float(
                f1_score(validation_true, validation_pred, zero_division=0)
            )
            if validation_f1 > best_validation_f1:
                best_validation_f1 = validation_f1
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            logits = model(x_tensor, edge_source, edge_target)[test_tensor]
            scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        predictions = (scores >= 0.5).astype(int)
        y_test = pd.Series(y[test_positions])
        row = _metric_row(
            model_name=model_name,
            y_true=y_test,
            predictions=predictions,
            scores=scores,
            train_count=len(full_train_positions),
            test_count=len(test_positions),
        )
        row["model_family"] = "true_gnn_optional_benchmark"
        row["feature_source"] = "warehouse_context_features_and_transition_edges"
        row["best_validation_f1"] = float(best_validation_f1)
        row["best_epoch"] = int(best_epoch)
        row["model_warning_count"] = 0
        row["model_warning_types"] = ""
        model_audit = {
            "check_name": f"{model_name}_training_scope",
            "check_value": (
                "synnetqos_warehouse_measurement_nodes_transition_edges_only"
            ),
            "status": "ok",
        }
        return row, _confusion_rows(model_name, y_test, predictions), model_audit

    input_dim = int(feature_matrix.shape[1])
    hidden_dim = 32 if input_dim >= 32 else max(8, input_dim)
    models = [
        ("graphsage_transition_gnn", GraphSAGEModel(input_dim, hidden_dim, 2)),
        ("gcn_transition_gnn", GCNModel(input_dim, hidden_dim, 2)),
    ]
    rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for model_name, model in models:
        row, matrix_rows, model_audit = train_model(model_name, model)
        rows.append(row)
        confusion_rows.extend(matrix_rows)
        audit_rows.append(model_audit)

    audit_rows.extend(
        [
            {
                "check_name": "true_gnn_dependency",
                "check_value": f"torch_available_on_{device}",
                "status": "ok",
            },
            {
                "check_name": "true_gnn_edge_policy",
                "check_value": int(len(sources)),
                "status": "ok",
            },
            {
                "check_name": "true_gnn_feature_policy",
                "check_value": "; ".join(feature_columns),
                "status": "ok",
            },
            {
                "check_name": "true_gnn_external_reference_exclusion",
                "check_value": (
                    "external_reference_evidence_nodes_not_used_for_prediction"
                ),
                "status": "ok",
            },
        ]
    )
    return rows, confusion_rows, pd.DataFrame(audit_rows)

def _load_supervised_reference(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    table = read_csv(target)
    if table.empty or "f1" not in table.columns:
        return pd.DataFrame()
    table = table.copy()
    table["model_family"] = "supervised_reference"
    return table



def _feature_set_table(dataset: pd.DataFrame) -> pd.DataFrame:
    feature_columns = _feature_frame(dataset, include_context_rates=False).columns
    rows = []
    for column in feature_columns:
        role = "graph_context"
        if column.startswith("train_context_rate_"):
            role = "training_context_rate"
        if column in {"graph_degree", "transition_out_degree", "transition_in_degree"}:
            role = "graph_degree_feature"
        rows.append({"feature_name": column, "feature_role": role})
    return pd.DataFrame(rows)


def _training_audit(
    dataset: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    feature_set: pd.DataFrame,
) -> pd.DataFrame:
    node_views = nodes.get("graph_view", pd.Series(dtype=str)).astype(str)
    edge_views = edges.get("graph_view", pd.Series(dtype=str)).astype(str)
    measurement_node_count = int(
        (
            node_views.eq(WAREHOUSE_VIEW)
            & nodes.get("node_type", pd.Series(dtype=str)).astype(str).eq("measurement")
        ).sum()
    )
    feature_names = set(feature_set["feature_name"].astype(str))
    rows = [
        {
            "check_name": "benchmark_role",
            "check_value": (
                "graph_predictive_benchmark_not_primary_claim_unless_results_justify"
            ),
            "status": "ok",
        },
        {
            "check_name": "prediction_rows_from_warehouse_measurements_only",
            "check_value": int(len(dataset)),
            "status": "ok" if len(dataset) == measurement_node_count else "review",
        },
        {
            "check_name": "target_column_removed_from_features",
            "check_value": int(TARGET_COLUMN not in feature_names),
            "status": "ok" if TARGET_COLUMN not in feature_names else "failed",
        },
        {
            "check_name": "training_context_rates_excluded_from_default_benchmark",
            "check_value": 1,
            "status": "ok",
        },
        {
            "check_name": "session_grouped_split_used",
            "check_value": int("session_id" in dataset.columns),
            "status": "ok" if "session_id" in dataset.columns else "review",
        },
        {
            "check_name": "warehouse_measurement_view_prediction_eligible",
            "check_value": int(node_views.eq(WAREHOUSE_VIEW).sum()),
            "status": "ok" if node_views.eq(WAREHOUSE_VIEW).any() else "failed",
        },
        {
            "check_name": "context_cooccurrence_view_feature_support",
            "check_value": int(edge_views.eq("context_cooccurrence").sum()),
            "status": "ok" if edge_views.eq("context_cooccurrence").any() else "review",
        },
        {
            "check_name": "external_reference_evidence_view_reference_only",
            "check_value": int(node_views.eq(EXTERNAL_VIEW).sum()),
            "status": "ok",
        },
    ]
    return pd.DataFrame(rows)


def run_graph_neural_benchmark(
    graph_nodes_path: str | Path,
    graph_edges_path: str | Path,
    results_dir: str | Path,
    figures_dir: str | Path,
    supervised_model_comparison_path: str | Path | None = None,
) -> dict[str, Path]:
    results_path = ensure_directory(results_dir)
    figures_path = ensure_directory(figures_dir)

    nodes = read_csv(graph_nodes_path, low_memory=False)
    edges = read_csv(graph_edges_path, low_memory=False)
    dataset = _base_dataset(nodes, edges)
    train_index, test_index = _split_dataset(dataset)

    experiments = [
        (
            "graph_context_logistic_regression",
            LogisticRegression(
                max_iter=600,
                solver="liblinear",
                class_weight="balanced",
                random_state=MODEL_RANDOM_STATE,
            ),
            False,
        ),
    ]

    model_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    for model_name, estimator, include_context_rates in experiments:
        row, rows = _run_model(
            model_name=model_name,
            estimator=estimator,
            dataframe=dataset,
            train_index=train_index,
            test_index=test_index,
            include_context_rates=include_context_rates,
        )
        row["model_family"] = "graph_derived_benchmark"
        row["feature_source"] = "warehouse_context_edges"
        model_rows.append(row)
        confusion_rows.extend(rows)

    true_gnn_rows, true_gnn_confusion_rows, true_gnn_audit = _run_true_gnn_models(
        dataset=dataset,
        nodes=nodes,
        edges=edges,
        train_index=train_index,
        test_index=test_index,
    )
    model_rows.extend(true_gnn_rows)
    confusion_rows.extend(true_gnn_confusion_rows)

    graph_model_comparison = pd.DataFrame(model_rows).sort_values(
        "f1",
        ascending=False,
    )
    supervised_reference = _load_supervised_reference(supervised_model_comparison_path)
    combined_model_comparison = graph_model_comparison.copy()
    if not supervised_reference.empty:
        comparable_columns = [
            column
            for column in combined_model_comparison.columns
            if column in supervised_reference.columns
        ]
        reference_subset = supervised_reference[comparable_columns].copy()
        combined_model_comparison = pd.concat(
            [combined_model_comparison, reference_subset],
            ignore_index=True,
            sort=False,
        )
        combined_model_comparison = combined_model_comparison.sort_values(
            "f1",
            ascending=False,
        )

    confusion_matrices = pd.DataFrame(confusion_rows)
    feature_set = _feature_set_table(dataset)
    split_audit = _transition_split_audit(dataset, edges, train_index, test_index)
    audit = pd.concat(
        [
            _training_audit(dataset, nodes, edges, feature_set),
            split_audit,
            true_gnn_audit,
        ],
        ignore_index=True,
        sort=False,
    )

    output_paths: dict[str, Path] = {
        "gnn_model_comparison": write_csv(
            graph_model_comparison,
            results_path / "gnn_model_comparison.csv",
        ),
        "gnn_vs_supervised_comparison": write_csv(
            combined_model_comparison,
            results_path / "gnn_vs_supervised_comparison.csv",
        ),
        "gnn_confusion_matrices": write_csv(
            confusion_matrices,
            results_path / "gnn_confusion_matrices.csv",
        ),
        "gnn_feature_set": write_csv(
            feature_set,
            results_path / "gnn_feature_set.csv",
        ),
        "gnn_training_audit": write_csv(
            audit,
            results_path / "gnn_training_audit.csv",
        ),
        "true_gnn_audit": write_csv(
            true_gnn_audit,
            results_path / "true_gnn_audit.csv",
        ),
    }

    figure_path = plot_gnn_vs_classical_model_comparison(
        combined_model_comparison,
        figures_path / "gnn_vs_classical.pdf",
    )
    if figure_path is not None:
        output_paths["gnn_vs_classical_figure"] = figure_path
    return output_paths
