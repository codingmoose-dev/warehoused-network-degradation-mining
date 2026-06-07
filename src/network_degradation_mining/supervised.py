from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from network_degradation_mining.io import ensure_directory, read_csv, write_csv
from network_degradation_mining.plotting import (
    display_model_name,
    plot_ranked_model_metric,
)

TARGET_COLUMN = "service_degraded"
EXCLUDED_COLUMNS = {
    "measurement_id",
    "session_id",
    "timestamp",
    "service_degraded",
    "high_latency",
    "high_jitter",
    "downlink_service_shortfall",
    "poor_video_quality",
    "dropped_connection",
    "anomalous",
    "download_speed_mbps",
    "upload_speed_mbps",
    "latency_ms",
    "jitter_ms",
    "ping_ms",
    "video_quality",
    "video_quality_label",
    "throughput_satisfaction_ratio",
    "downlink_shortfall_fraction",
}

SENSITIVITY_EXCLUDED_FEATURES = {
    "link_capacity_downlink_mbps",
    "link_capacity_upload_mbps",
    "offered_downlink_mbps",
    "offered_upload_mbps",
    "deployment_area",
    "area_type",
}

MODEL_RANDOM_STATE = 42
KNN_TRAIN_ROW_LIMIT = 15_000
RBF_SVM_TRAIN_ROW_LIMIT = 8_000
MODEL_TRAIN_ROW_LIMITS = {
    "knn": KNN_TRAIN_ROW_LIMIT,
    "rbf_svm": RBF_SVM_TRAIN_ROW_LIMIT,
}
PRIMARY_SPLIT_DESCRIPTION = "Session-grouped 80/20 holdout"
PRIMARY_PREPROCESSING_DESCRIPTION = (
    "Median/mode imputation; numeric standard scaling; categorical one-hot encoding"
)
SENSITIVITY_NOTES = (
    "Reduced context feature set excluding capacity, offered-traffic, "
    "and synthetic deployment-area variables."
)

REPEATED_SPLIT_RANDOM_STATES = (11, 23, 37, 51, 73)
TEMPORAL_TEST_FRACTION = 0.20
CALIBRATION_BIN_COUNT = 10

LABEL_COMPONENT_COLUMNS = [
    "high_latency",
    "high_jitter",
    "downlink_service_shortfall",
    "poor_video_quality",
    "dropped_connection",
    "anomalous",
]

IDENTIFIER_COLUMNS = {"measurement_id", "session_id", "timestamp"}
DIRECT_OUTCOME_COLUMNS = {
    "download_speed_mbps",
    "upload_speed_mbps",
    "latency_ms",
    "jitter_ms",
    "ping_ms",
    "video_quality",
    "video_quality_label",
    "throughput_satisfaction_ratio",
    "downlink_shortfall_fraction",
}
TARGET_COMPONENT_COLUMNS = {
    "high_latency",
    "high_jitter",
    "downlink_service_shortfall",
    "poor_video_quality",
    "dropped_connection",
    "anomalous",
}



@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any | None
    status: str = "ok"
    notes: str = ""


MODEL_FAMILIES = {
    "knn": "Instance-based",
    "naive_bayes": "Probabilistic",
    "decision_tree_gini": "Tree",
    "decision_tree_entropy": "Tree",
    "logistic_regression": "Linear",
    "linear_svm": "Linear SVM",
    "rbf_svm": "Kernel SVM",
    "random_forest": "Ensemble tree",
    "adaboost": "Boosting",
    "hist_gradient_boosting": "Boosting",
    "xgboost": "Boosting",
}


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _feature_columns(
    dataframe: pd.DataFrame,
    excluded_features: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    excluded = set(EXCLUDED_COLUMNS)
    if excluded_features:
        excluded.update(excluded_features)
    candidate_columns = []
    for column in dataframe.columns:
        if column in excluded:
            continue
        if dataframe[column].nunique(dropna=True) <= 1:
            continue
        candidate_columns.append(column)
    numeric_columns = [
        column
        for column in candidate_columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    categorical_columns = [
        column for column in candidate_columns if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns


def _clean_feature_frame(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    output = dataframe[numeric_columns + categorical_columns].copy()
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    for column in categorical_columns:
        output[column] = output[column].astype("string")
    return output


def _build_preprocessor(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
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


def _as_finite_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def xgboost_model() -> ModelSpec:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        return ModelSpec(
            name="xgboost",
            estimator=None,
            status="skipped_missing_dependency",
            notes=f"xgboost is not available: {exc}",
        )

    return ModelSpec(
        name="xgboost",
        estimator=XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=MODEL_RANDOM_STATE,
            n_jobs=1,
        ),
    )


def _model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "knn", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)
        ),
        ModelSpec("naive_bayes", GaussianNB()),
        ModelSpec(
            "decision_tree_gini",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=8,
                min_samples_leaf=100,
                class_weight="balanced",
                random_state=MODEL_RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "decision_tree_entropy",
            DecisionTreeClassifier(
                criterion="entropy",
                max_depth=8,
                min_samples_leaf=100,
                class_weight="balanced",
                random_state=MODEL_RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "logistic_regression",
            LogisticRegression(
                max_iter=500,
                solver="liblinear",
                class_weight="balanced",
                random_state=MODEL_RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "linear_svm",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual=False,
                max_iter=10_000,
                random_state=MODEL_RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "rbf_svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=False,
                cache_size=1024,
                random_state=MODEL_RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "random_forest",
            RandomForestClassifier(
                n_estimators=80,
                max_depth=14,
                min_samples_leaf=25,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=MODEL_RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "adaboost",
            AdaBoostClassifier(
                n_estimators=40,
                learning_rate=0.08,
                random_state=MODEL_RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=80,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                random_state=MODEL_RANDOM_STATE,
            ),
        ),
        xgboost_model(),
    ]


def _split_data(
    dataframe: pd.DataFrame,
    random_state: int = MODEL_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    X = dataframe.drop(columns=[TARGET_COLUMN])

    if (
        "session_id" in dataframe.columns
        and dataframe["session_id"].nunique(dropna=True) > 1
    ):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=random_state
        )
        groups = dataframe["session_id"].astype("string").fillna("missing_session")
        train_index, test_index = next(splitter.split(X, y, groups=groups))
        return (
            X.iloc[train_index],
            X.iloc[test_index],
            y.iloc[train_index],
            y.iloc[test_index],
        )

    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y if y.nunique() > 1 else None,
        random_state=random_state,
    )


def _split_data_temporal(
    dataframe: pd.DataFrame,
    test_fraction: float = TEMPORAL_TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if "timestamp" not in dataframe.columns:
        return _split_data(dataframe)

    working = dataframe.copy()
    working["__parsed_timestamp"] = pd.to_datetime(
        working["timestamp"], errors="coerce"
    )
    if working["__parsed_timestamp"].notna().sum() == 0:
        return _split_data(dataframe)

    y = pd.to_numeric(working[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    X = working.drop(columns=[TARGET_COLUMN, "__parsed_timestamp"])

    if (
        "session_id" in working.columns
        and working["session_id"].nunique(dropna=True) > 1
    ):
        session_order = (
            working.assign(
                __session_id=working["session_id"]
                .astype("string")
                .fillna("missing_session")
            )
            .groupby("__session_id", dropna=False)["__parsed_timestamp"]
            .min()
            .sort_values(kind="mergesort")
        )
        ordered_sessions = session_order.index.to_list()
        test_session_count = max(
            1, int(round(len(ordered_sessions) * test_fraction))
        )
        test_sessions = set(ordered_sessions[-test_session_count:])
        session_values = (
            working["session_id"].astype("string").fillna("missing_session")
        )
        test_mask = session_values.isin(test_sessions).to_numpy()
    else:
        ordered_index = working.sort_values(
            "__parsed_timestamp", kind="mergesort"
        ).index
        test_count = max(1, int(round(len(ordered_index) * test_fraction)))
        test_index = set(ordered_index[-test_count:])
        test_mask = working.index.isin(test_index)

    train_mask = ~test_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return _split_data(dataframe)

    return (
        X.loc[train_mask],
        X.loc[test_mask],
        y.loc[train_mask],
        y.loc[test_mask],
    )


def _positive_scores(
    model: Any,
    X_test: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return np.ravel(probabilities)
    if hasattr(model, "decision_function"):
        return np.ravel(model.decision_function(X_test))
    return predictions.astype(float)


def _score_predictions(
    model: Any,
    model_name: str,
    X_test: np.ndarray,
    y_test: pd.Series,
    train_row_count: int,
    predictions: np.ndarray,
) -> dict[str, Any]:
    scores = _positive_scores(model, X_test, predictions)

    row: dict[str, Any] = {
        "model_name": model_name,
        "training_status": "ok",
        "train_row_count": int(train_row_count),
        "test_row_count": int(len(y_test)),
        "positive_fraction": float(y_test.mean()) if len(y_test) else 0.0,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "notes": "",
    }

    if y_test.nunique() > 1:
        row["roc_auc"] = float(roc_auc_score(y_test, scores))
        row["pr_auc"] = float(average_precision_score(y_test, scores))
    else:
        row["roc_auc"] = pd.NA
        row["pr_auc"] = pd.NA

    return row


def _skipped_model_row(
    spec: ModelSpec,
    train_row_count: int,
    test_row_count: int,
    positive_fraction: float,
) -> dict[str, Any]:
    return {
        "model_name": spec.name,
        "training_status": spec.status,
        "train_row_count": int(train_row_count),
        "test_row_count": int(test_row_count),
        "positive_fraction": float(positive_fraction),
        "accuracy": pd.NA,
        "balanced_accuracy": pd.NA,
        "precision": pd.NA,
        "recall": pd.NA,
        "f1": pd.NA,
        "roc_auc": pd.NA,
        "pr_auc": pd.NA,
        "notes": spec.notes,
    }


def _failed_model_row(
    model_name: str,
    exc: Exception,
    train_row_count: int,
    test_row_count: int,
    positive_fraction: float,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "training_status": "failed",
        "train_row_count": int(train_row_count),
        "test_row_count": int(test_row_count),
        "positive_fraction": float(positive_fraction),
        "accuracy": pd.NA,
        "balanced_accuracy": pd.NA,
        "precision": pd.NA,
        "recall": pd.NA,
        "f1": pd.NA,
        "roc_auc": pd.NA,
        "pr_auc": pd.NA,
        "notes": str(exc),
    }


def _confusion_rows_from_predictions(
    model_name: str, y_test: pd.Series, predictions: np.ndarray
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    labels = ["not_degraded", "degraded"]
    rows: list[dict[str, Any]] = []
    for actual_index, actual_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            rows.append(
                {
                    "model_name": model_name,
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "count": int(matrix[actual_index, predicted_index]),
                }
            )
    return rows


def _feature_names(
    preprocessor: ColumnTransformer,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return numeric_columns + categorical_columns


def _importance_rows(
    model: Any,
    model_name: str,
    feature_names: list[str],
    top_n: int = 40,
) -> list[dict[str, Any]]:
    values: Any | None = None
    importance_type = ""
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(model, "coef_"):
        values = abs(model.coef_[0])
        importance_type = "absolute_coefficient"

    if values is None:
        return []

    rows = []
    for feature_name, value in zip(feature_names, values):
        rows.append(
            {
                "model_name": model_name,
                "feature_name": str(feature_name),
                "importance_type": importance_type,
                "importance_value": float(value),
            }
        )
    return sorted(rows, key=lambda item: item["importance_value"], reverse=True)[:top_n]


def _fit_training_subset(
    model_name: str,
    X_train: np.ndarray,
    y_train: pd.Series,
) -> tuple[np.ndarray, pd.Series]:
    row_limit = MODEL_TRAIN_ROW_LIMITS.get(model_name)
    if row_limit is None or len(y_train) <= row_limit:
        return X_train, y_train

    rng = np.random.default_rng(MODEL_RANDOM_STATE)
    indices = np.arange(len(y_train))
    class_values = y_train.to_numpy()
    selected: list[int] = []
    for label in sorted(set(class_values.tolist())):
        label_indices = indices[class_values == label]
        label_quota = max(1, int(row_limit * len(label_indices) / len(y_train)))
        label_quota = min(label_quota, len(label_indices))
        selected.extend(
            rng.choice(label_indices, size=label_quota, replace=False).tolist()
        )
    if len(selected) > row_limit:
        selected = rng.choice(selected, size=row_limit, replace=False).tolist()
    selected = sorted(selected)
    return X_train[selected], y_train.iloc[selected]


def _training_subset_note(
    model_name: str,
    original_train_row_count: int,
    used_train_row_count: int,
) -> str:
    if used_train_row_count >= original_train_row_count:
        return ""
    if model_name == "knn":
        return (
            "Stratified training subset used to keep the instance-based "
            "training runtime bounded."
        )
    if model_name == "rbf_svm":
        return (
            "Stratified training subset used because exact RBF SVM training "
            "scales poorly with dense one-hot features."
        )
    return "Stratified training subset used for runtime control."


def _feature_set_frame(
    numeric_columns: list[str],
    categorical_columns: list[str],
    experiment_name: str,
) -> pd.DataFrame:
    rows = [
        {
            "experiment_name": experiment_name,
            "feature_name": column,
            "feature_type": "numeric",
        }
        for column in numeric_columns
    ]
    rows.extend(
        {
            "experiment_name": experiment_name,
            "feature_name": column,
            "feature_type": "categorical",
        }
        for column in categorical_columns
    )
    return pd.DataFrame(rows)


def _build_ranked_model_table(
    model_comparison: pd.DataFrame,
    experiment_name: str,
    experiment_label: str,
    split_design: str = PRIMARY_SPLIT_DESCRIPTION,
) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    columns = ["model_name", "training_status", *metric_columns]
    available_columns = [
        column for column in columns if column in model_comparison.columns
    ]
    frame = model_comparison.loc[
        model_comparison["training_status"].eq("ok"), available_columns
    ].copy()
    if frame.empty:
        return pd.DataFrame()

    for column in metric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.sort_values(
        ["f1", "balanced_accuracy", "pr_auc"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    frame.insert(0, "overall_rank", range(1, len(frame) + 1))
    frame.insert(1, "experiment_name", experiment_name)
    frame.insert(2, "experiment_label", experiment_label)
    frame.insert(
        3, "model_family", frame["model_name"].map(MODEL_FAMILIES).fillna("Other")
    )
    frame.insert(4, "model", frame["model_name"].map(display_model_name))
    frame["split_design"] = split_design
    frame["preprocessing"] = PRIMARY_PREPROCESSING_DESCRIPTION
    return frame[
        [
            "overall_rank",
            "experiment_name",
            "experiment_label",
            "model_family",
            "model",
            "f1",
            "balanced_accuracy",
            "roc_auc",
            "pr_auc",
            "recall",
            "precision",
            "accuracy",
            "split_design",
            "preprocessing",
        ]
    ]


def _positive_probability_scores(model: Any, X_test: np.ndarray) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(X_test)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        return None
    return np.clip(np.asarray(probabilities[:, 1], dtype=float), 0.0, 1.0)


def _calibration_rows(
    model_name: str,
    y_test: pd.Series,
    probabilities: np.ndarray,
    experiment_name: str,
    experiment_label: str,
    bin_count: int = CALIBRATION_BIN_COUNT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_values = np.asarray(y_test, dtype=int)
    bins = np.linspace(0.0, 1.0, bin_count + 1)
    rows: list[dict[str, Any]] = []
    expected_calibration_error = 0.0

    for index in range(bin_count):
        lower = float(bins[index])
        upper = float(bins[index + 1])
        if index == bin_count - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        sample_count = int(mask.sum())
        if sample_count:
            mean_probability = float(probabilities[mask].mean())
            observed_fraction = float(y_values[mask].mean())
            expected_calibration_error += (
                sample_count / len(y_values)
            ) * abs(observed_fraction - mean_probability)
        else:
            mean_probability = pd.NA
            observed_fraction = pd.NA
        rows.append(
            {
                "experiment_name": experiment_name,
                "experiment_label": experiment_label,
                "model_name": model_name,
                "bin_index": index + 1,
                "bin_lower": lower,
                "bin_upper": upper,
                "sample_count": sample_count,
                "mean_predicted_probability": mean_probability,
                "observed_degradation_fraction": observed_fraction,
            }
        )

    summary = {
        "experiment_name": experiment_name,
        "experiment_label": experiment_label,
        "model_name": model_name,
        "calibration_status": "ok",
        "test_row_count": int(len(y_values)),
        "brier_score": float(brier_score_loss(y_values, probabilities)),
        "expected_calibration_error": float(expected_calibration_error),
        "calibration_bin_count": int(bin_count),
    }
    return rows, summary


def _build_repeated_grouped_summary(model_comparison: pd.DataFrame) -> pd.DataFrame:
    if model_comparison.empty:
        return pd.DataFrame()
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    frame = model_comparison.loc[model_comparison["training_status"].eq("ok")].copy()
    if frame.empty:
        return pd.DataFrame()
    for column in metric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    rows: list[dict[str, Any]] = []
    for model_name, group in frame.groupby("model_name", sort=False):
        row: dict[str, Any] = {
            "model_name": model_name,
            "model": display_model_name(model_name),
            "split_count": int(group["split_random_state"].nunique()),
            "mean_train_rows": float(group["train_row_count"].mean()),
            "mean_test_rows": float(group["test_row_count"].mean()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["f1_mean", "balanced_accuracy_mean"], ascending=[False, False]
    )


def _run_repeated_grouped_evaluation(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for random_state in REPEATED_SPLIT_RANDOM_STATES:
        X_train, X_test, y_train, y_test = _split_data(data, random_state=random_state)
        experiment = _run_model_experiment(
            data=data,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            experiment_name="repeated_session_grouped",
            experiment_label="Repeated session-grouped holdout stability analysis",
        )
        frame = experiment["model_comparison"].copy()
        frame.insert(0, "split_random_state", random_state)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {
        "model_comparison": combined,
        "summary": _build_repeated_grouped_summary(combined),
    }


def _build_label_component_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    available = [
        column for column in LABEL_COMPONENT_COLUMNS if column in dataframe.columns
    ]
    row_count = len(dataframe)
    if TARGET_COLUMN not in dataframe.columns:
        return pd.DataFrame()

    target = (
        pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    target_positive = target.eq(1)
    component_frame = pd.DataFrame(index=dataframe.index)
    for column in available:
        component_frame[column] = (
            pd.to_numeric(dataframe[column], errors="coerce")
            .fillna(0)
            .astype(int)
            .eq(1)
        )

    rows: list[dict[str, Any]] = []
    for column in available:
        values = component_frame[column]
        positive_count = int(values.sum())
        other_columns = [name for name in available if name != column]
        if other_columns:
            unique_mask = values & ~component_frame[other_columns].any(axis=1)
        else:
            unique_mask = values
        rows.append(
            {
                "component_name": column,
                "row_count": int(row_count),
                "positive_count": positive_count,
                "positive_fraction": (
                    float(positive_count / row_count) if row_count else 0.0
                ),
                "positive_among_degraded_fraction": (
                    float((values & target_positive).sum() / target_positive.sum())
                    if int(target_positive.sum())
                    else 0.0
                ),
                "unique_positive_count_among_available_components": int(
                    unique_mask.sum()
                ),
                "available_component_count": int(len(available)),
                "audit_note": (
                    "Unique counts are calculated only among label components present "
                    "in the supervised table."
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_predictor_exclusion_audit(
    dataframe: pd.DataFrame,
    full_feature_set: pd.DataFrame,
    reduced_feature_set: pd.DataFrame,
) -> pd.DataFrame:
    full_features = set(
        full_feature_set.get("feature_name", pd.Series(dtype="string")).astype(str)
    )
    reduced_features = set(
        reduced_feature_set.get("feature_name", pd.Series(dtype="string")).astype(str)
    )
    rows: list[dict[str, Any]] = []

    audited_columns = sorted(
        set(dataframe.columns)
        | EXCLUDED_COLUMNS
        | SENSITIVITY_EXCLUDED_FEATURES
        | IDENTIFIER_COLUMNS
        | DIRECT_OUTCOME_COLUMNS
        | TARGET_COMPONENT_COLUMNS
    )

    for column in audited_columns:
        reasons: list[str] = []
        if column in IDENTIFIER_COLUMNS:
            reasons.append("identifier_or_split_field")
        if column == TARGET_COLUMN:
            reasons.append("target_label")
        if column in TARGET_COMPONENT_COLUMNS:
            reasons.append("target_component")
        if column in DIRECT_OUTCOME_COLUMNS:
            reasons.append("direct_outcome_or_label_definition_field")
        if column in SENSITIVITY_EXCLUDED_FEATURES:
            reasons.append("reduced_context_exclusion")
        if not reasons and column in dataframe.columns:
            reasons.append("candidate_predictor")
        elif not reasons:
            reasons.append("not_present_in_supervised_table")

        rows.append(
            {
                "column_name": column,
                "present_in_supervised_table": bool(column in dataframe.columns),
                "used_in_full_context": bool(column in full_features),
                "used_in_reduced_context": bool(column in reduced_features),
                "exclusion_reason": ";".join(reasons),
            }
        )

    return pd.DataFrame(rows)


def _run_model_experiment(
    data: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    experiment_name: str,
    experiment_label: str,
    excluded_features: set[str] | None = None,
    split_design: str = PRIMARY_SPLIT_DESCRIPTION,
) -> dict[str, pd.DataFrame]:
    numeric_columns, categorical_columns = _feature_columns(
        X_train, excluded_features=excluded_features
    )
    X_train_features = _clean_feature_frame(
        X_train, numeric_columns, categorical_columns
    )
    X_test_features = _clean_feature_frame(X_test, numeric_columns, categorical_columns)

    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)
    X_train_processed = _as_finite_array(preprocessor.fit_transform(X_train_features))
    X_test_processed = _as_finite_array(preprocessor.transform(X_test_features))
    feature_names = _feature_names(preprocessor, numeric_columns, categorical_columns)

    rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    calibration_summary_rows: list[dict[str, Any]] = []
    positive_fraction = float(y_test.mean()) if len(y_test) else 0.0

    for spec in _model_specs():
        if spec.estimator is None:
            row = _skipped_model_row(
                spec,
                train_row_count=len(y_train),
                test_row_count=len(y_test),
                positive_fraction=positive_fraction,
            )
            row["experiment_name"] = experiment_name
            row["experiment_label"] = experiment_label
            rows.append(row)
            continue

        train_matrix, train_target = _fit_training_subset(
            spec.name, X_train_processed, y_train
        )
        try:
            spec.estimator.fit(train_matrix, train_target)
            predictions = spec.estimator.predict(X_test_processed)
            row = _score_predictions(
                spec.estimator,
                spec.name,
                X_test_processed,
                y_test,
                train_row_count=len(train_target),
                predictions=predictions,
            )
            row["experiment_name"] = experiment_name
            row["experiment_label"] = experiment_label
            notes = _training_subset_note(
                model_name=spec.name,
                original_train_row_count=len(y_train),
                used_train_row_count=len(train_target),
            )
            if experiment_name == "reduced_context":
                notes = "; ".join(
                    note for note in [SENSITIVITY_NOTES, notes] if note
                )
            row["notes"] = notes
            rows.append(row)

            for confusion_row in _confusion_rows_from_predictions(
                spec.name, y_test, predictions
            ):
                confusion_row["experiment_name"] = experiment_name
                confusion_row["experiment_label"] = experiment_label
                confusion_rows.append(confusion_row)

            for importance_row in _importance_rows(
                spec.estimator, spec.name, feature_names
            ):
                importance_row["experiment_name"] = experiment_name
                importance_row["experiment_label"] = experiment_label
                importance_rows.append(importance_row)

            probability_scores = _positive_probability_scores(
                spec.estimator, X_test_processed
            )
            if probability_scores is not None and y_test.nunique() > 1:
                model_calibration_rows, model_calibration_summary = _calibration_rows(
                    model_name=spec.name,
                    y_test=y_test,
                    probabilities=probability_scores,
                    experiment_name=experiment_name,
                    experiment_label=experiment_label,
                )
                calibration_rows.extend(model_calibration_rows)
                calibration_summary_rows.append(model_calibration_summary)
        except Exception as exc:
            row = _failed_model_row(
                spec.name,
                exc,
                train_row_count=len(train_target),
                test_row_count=len(y_test),
                positive_fraction=positive_fraction,
            )
            row["experiment_name"] = experiment_name
            row["experiment_label"] = experiment_label
            rows.append(row)

    model_comparison = pd.DataFrame(rows).sort_values(
        ["training_status", "f1"], ascending=[True, False], na_position="last"
    )
    ranked_table = _build_ranked_model_table(
        model_comparison=model_comparison,
        experiment_name=experiment_name,
        experiment_label=experiment_label,
        split_design=split_design,
    )

    return {
        "model_comparison": model_comparison,
        "confusion_matrices": pd.DataFrame(confusion_rows),
        "feature_importance": pd.DataFrame(importance_rows),
        "feature_set": _feature_set_frame(
            numeric_columns, categorical_columns, experiment_name
        ),
        "ranked_model_performance_table": ranked_table,
        "calibration_curve": pd.DataFrame(calibration_rows),
        "calibration_summary": pd.DataFrame(calibration_summary_rows),
    }


def run_supervised_classification(
    supervised_table_path: str | Path,
    results_dir: str | Path,
    figures_dir: str | Path,
) -> dict[str, Path]:
    output_dir = ensure_directory(results_dir)
    figure_output_dir = ensure_directory(figures_dir)

    data = read_csv(supervised_table_path, low_memory=False)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X_train, X_test, y_train, y_test = _split_data(data)

    primary = _run_model_experiment(
        data=data,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        experiment_name="full_context",
        experiment_label="Full operating-context feature set",
    )
    sensitivity = _run_model_experiment(
        data=data,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        experiment_name="reduced_context",
        experiment_label=(
            "Reduced context without capacity, offered traffic, "
            "or synthetic deployment-area variables"
        ),
        excluded_features=SENSITIVITY_EXCLUDED_FEATURES,
    )

    temporal_X_train, temporal_X_test, temporal_y_train, temporal_y_test = (
        _split_data_temporal(data)
    )
    temporal = _run_model_experiment(
        data=data,
        X_train=temporal_X_train,
        X_test=temporal_X_test,
        y_train=temporal_y_train,
        y_test=temporal_y_test,
        experiment_name="temporal_holdout",
        experiment_label="Session-preserving time-ordered holdout",
        split_design="Session-preserving time-ordered holdout",
    )
    repeated = _run_repeated_grouped_evaluation(data)

    calibration_curve = pd.concat(
        [
            primary["calibration_curve"],
            sensitivity["calibration_curve"],
            temporal["calibration_curve"],
        ],
        ignore_index=True,
    )
    calibration_summary = pd.concat(
        [
            primary["calibration_summary"],
            sensitivity["calibration_summary"],
            temporal["calibration_summary"],
        ],
        ignore_index=True,
    )
    label_component_summary = _build_label_component_summary(data)
    predictor_exclusion_audit = _build_predictor_exclusion_audit(
        dataframe=data,
        full_feature_set=primary["feature_set"],
        reduced_feature_set=sensitivity["feature_set"],
    )

    output_paths = {
        "model_comparison": output_dir / "model_comparison.csv",
        "confusion_matrices": output_dir / "confusion_matrices.csv",
        "feature_importance": output_dir / "feature_importance.csv",
        "feature_set": output_dir / "supervised_feature_set.csv",
        "ranked_model_performance_table": output_dir
        / "ranked_model_performance_table.csv",
        "sensitivity_model_comparison": output_dir / "sensitivity_model_comparison.csv",
        "sensitivity_confusion_matrices": output_dir
        / "sensitivity_confusion_matrices.csv",
        "sensitivity_feature_importance": output_dir
        / "sensitivity_feature_importance.csv",
        "sensitivity_feature_set": output_dir / "sensitivity_feature_set.csv",
        "sensitivity_ranked_model_performance_table": output_dir
        / "sensitivity_ranked_model_performance_table.csv",
        "temporal_holdout_model_comparison": output_dir
        / "temporal_holdout_model_comparison.csv",
        "temporal_holdout_confusion_matrices": output_dir
        / "temporal_holdout_confusion_matrices.csv",
        "temporal_holdout_feature_importance": output_dir
        / "temporal_holdout_feature_importance.csv",
        "temporal_holdout_feature_set": output_dir
        / "temporal_holdout_feature_set.csv",
        "temporal_holdout_ranked_model_performance_table": output_dir
        / "temporal_holdout_ranked_model_performance_table.csv",
        "calibration_curve": output_dir / "calibration_curve.csv",
        "calibration_summary": output_dir / "calibration_summary.csv",
        "label_component_summary": output_dir / "label_component_summary.csv",
        "predictor_exclusion_audit": output_dir / "predictor_exclusion_audit.csv",
        "repeated_grouped_model_comparison": output_dir
        / "repeated_grouped_model_comparison.csv",
        "repeated_grouped_model_summary": output_dir
        / "repeated_grouped_model_summary.csv",
    }

    write_csv(primary["model_comparison"], output_paths["model_comparison"])
    write_csv(primary["confusion_matrices"], output_paths["confusion_matrices"])
    write_csv(primary["feature_importance"], output_paths["feature_importance"])
    write_csv(primary["feature_set"], output_paths["feature_set"])
    write_csv(
        primary["ranked_model_performance_table"],
        output_paths["ranked_model_performance_table"],
    )
    write_csv(
        sensitivity["model_comparison"], output_paths["sensitivity_model_comparison"]
    )
    write_csv(
        sensitivity["confusion_matrices"],
        output_paths["sensitivity_confusion_matrices"],
    )
    write_csv(
        sensitivity["feature_importance"],
        output_paths["sensitivity_feature_importance"],
    )
    write_csv(sensitivity["feature_set"], output_paths["sensitivity_feature_set"])
    write_csv(
        sensitivity["ranked_model_performance_table"],
        output_paths["sensitivity_ranked_model_performance_table"],
    )
    write_csv(
        temporal["model_comparison"],
        output_paths["temporal_holdout_model_comparison"],
    )
    write_csv(
        temporal["confusion_matrices"],
        output_paths["temporal_holdout_confusion_matrices"],
    )
    write_csv(
        temporal["feature_importance"],
        output_paths["temporal_holdout_feature_importance"],
    )
    write_csv(temporal["feature_set"], output_paths["temporal_holdout_feature_set"])
    write_csv(
        temporal["ranked_model_performance_table"],
        output_paths["temporal_holdout_ranked_model_performance_table"],
    )
    write_csv(calibration_curve, output_paths["calibration_curve"])
    write_csv(calibration_summary, output_paths["calibration_summary"])
    write_csv(label_component_summary, output_paths["label_component_summary"])
    write_csv(predictor_exclusion_audit, output_paths["predictor_exclusion_audit"])
    write_csv(
        repeated["model_comparison"],
        output_paths["repeated_grouped_model_comparison"],
    )
    write_csv(
        repeated["summary"],
        output_paths["repeated_grouped_model_summary"],
    )

    ranked_metric_path = figure_output_dir / "model_comparison.pdf"
    ranked_metric = plot_ranked_model_metric(
        model_comparison=primary["model_comparison"],
        figure_path=ranked_metric_path,
        metric="f1",
    )
    if not ranked_metric.empty:
        output_paths["model_comparison_figure"] = ranked_metric_path

    return output_paths
