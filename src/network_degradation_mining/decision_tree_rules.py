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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from network_degradation_mining.io import ensure_directory, read_csv, write_csv
from network_degradation_mining.plotting import (
    display_field_name,
    display_value,
    plot_pruned_decision_tree,
    plot_representative_decision_paths,
)

TARGET_COLUMN = "service_degraded"
MODEL_RANDOM_STATE = 42
STABILITY_RANDOM_STATES = (11, 23, 37, 51, 73)

BASE_EXCLUDED_COLUMNS = {
    "measurement_id",
    "session_id",
    "timestamp",
    TARGET_COLUMN,
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

REDUCED_CONTEXT_EXCLUSIONS = {
    "link_capacity_downlink_mbps",
    "link_capacity_upload_mbps",
    "offered_downlink_mbps",
    "offered_upload_mbps",
    "deployment_area",
    "area_type",
}

TREE_SPECS = (
    ("gini", "gini"),
    ("entropy", "entropy"),
)


@dataclass(frozen=True)
class RuleTreeSpec:
    feature_context: str
    criterion: str
    excluded_features: frozenset[str]


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _feature_columns(
    dataframe: pd.DataFrame, excluded_features: set[str]
) -> tuple[list[str], list[str]]:
    candidate_columns: list[str] = []
    for column in dataframe.columns:
        if column in excluded_features:
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


def _clean_features(
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


def _feature_names(
    preprocessor: ColumnTransformer,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> list[str]:
    names: list[str] = []

    if numeric_columns:
        names.extend(numeric_columns)

    if categorical_columns:
        categorical_pipeline = preprocessor.named_transformers_.get("categorical")
        if categorical_pipeline is not None:
            encoder = categorical_pipeline.named_steps["encoder"]
            encoded = encoder.get_feature_names_out(categorical_columns).tolist()
            names.extend(encoded)

    return names


def _numeric_scale_lookup(
    preprocessor: ColumnTransformer,
    numeric_columns: list[str],
) -> dict[str, tuple[float, float]]:
    if not numeric_columns:
        return {}

    numeric_pipeline = preprocessor.named_transformers_.get("numeric")
    if numeric_pipeline is None:
        return {}

    scaler = numeric_pipeline.named_steps.get("scaler")
    if scaler is None or not hasattr(scaler, "mean_"):
        return {}

    return {
        column: (float(mean), float(scale))
        for column, mean, scale in zip(
            numeric_columns,
            scaler.mean_,
            scaler.scale_,
        )
    }


def _categorical_feature_parts(
    feature_name: str,
    categorical_columns: list[str],
) -> tuple[str, str] | None:
    for column in sorted(categorical_columns, key=len, reverse=True):
        prefix = f"{column}_"
        if feature_name.startswith(prefix):
            return column, feature_name[len(prefix) :]
    return None


def _base_feature_name(
    feature_name: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> str:
    if feature_name in numeric_columns:
        return feature_name

    categorical_parts = _categorical_feature_parts(feature_name, categorical_columns)
    if categorical_parts is not None:
        return categorical_parts[0]

    return feature_name


def _safe_metric(function: Any, y_true: pd.Series, y_pred: np.ndarray) -> float:
    try:
        return float(function(y_true, y_pred))
    except Exception:
        return float("nan")


def _score_tree(
    model: DecisionTreeClassifier,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(X_test)
    return {
        "accuracy": _safe_metric(accuracy_score, y_test, predictions),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_test, predictions),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }


def _format_standard_condition(
    feature_name: str,
    threshold: float,
    direction: str,
) -> str:
    feature = feature_name.replace("categorical__", "").replace("numeric__", "")
    comparator = "≤" if direction == "left" else ">"
    return f"{display_field_name(feature)} {comparator} {threshold:.4g}"


def _format_interpretable_condition(
    feature_name: str,
    threshold: float,
    direction: str,
    numeric_scale_lookup: dict[str, tuple[float, float]],
    categorical_columns: list[str],
) -> str:
    comparator = "≤" if direction == "left" else ">"

    if feature_name in numeric_scale_lookup:
        mean, scale = numeric_scale_lookup[feature_name]
        raw_threshold = threshold * scale + mean
        readable_feature = display_field_name(feature_name)
        return f"{readable_feature} {comparator} {raw_threshold:.4g}"

    categorical_parts = _categorical_feature_parts(feature_name, categorical_columns)
    if categorical_parts is not None and abs(threshold - 0.5) <= 1e-6:
        column, category = categorical_parts
        readable_column = display_field_name(column)
        readable_category = display_value(category)
        if direction == "left":
            return f"{readable_column} ≠ {readable_category}"
        return f"{readable_column} = {readable_category}"

    return _format_standard_condition(feature_name, threshold, direction)


def _extract_leaf_rules(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    numeric_scale_lookup: dict[str, tuple[float, float]],
    feature_context: str,
    criterion: str,
) -> pd.DataFrame:
    tree = model.tree_
    rows: list[dict[str, Any]] = []

    def recurse(
        node_id: int,
        conditions: list[str],
        interpretable_conditions: list[str],
        split_features: list[str],
        depth: int,
    ) -> None:
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if left_child == right_child:
            class_counts = tree.value[node_id][0]
            predicted_class = int(np.argmax(class_counts))
            class_total = float(class_counts.sum())
            positive_count = (
                float(class_counts[1]) if len(class_counts) > 1 else 0.0
            )
            degraded_rate = positive_count / class_total if class_total else 0.0

            rows.append(
                {
                    "feature_context": feature_context,
                    "criterion": criterion,
                    "rule_id": f"{feature_context}_{criterion}_leaf_{node_id}",
                    "tree_depth": int(depth),
                    "leaf_node": int(node_id),
                    "weighted_sample_count": float(
                        tree.weighted_n_node_samples[node_id]
                    ),
                    "node_sample_count": int(tree.n_node_samples[node_id]),
                    "predicted_class": predicted_class,
                    "degraded_rate_in_leaf": float(degraded_rate),
                    "condition_count": int(len(conditions)),
                    "rule_text": " AND ".join(conditions) if conditions else "ALL",
                    "rule_text_interpretable": (
                        " AND ".join(interpretable_conditions)
                        if interpretable_conditions
                        else "ALL"
                    ),
                    "split_feature_names": "|".join(split_features),
                    "threshold_units_note": (
                        "rule_text uses transformed model inputs; "
                        "rule_text_interpretable maps numeric thresholds back "
                        "to the training-set original scale where possible"
                    ),
                }
            )
            return

        feature_index = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = feature_names[feature_index]
        base_feature = _base_feature_name(
            feature_name, numeric_columns, categorical_columns
        )

        for child_id, child_direction in (
            (left_child, "left"),
            (right_child, "right"),
        ):
            recurse(
                child_id,
                [
                    *conditions,
                    _format_standard_condition(
                        feature_name, threshold, child_direction
                    ),
                ],
                [
                    *interpretable_conditions,
                    _format_interpretable_condition(
                        feature_name=feature_name,
                        threshold=threshold,
                        direction=child_direction,
                        numeric_scale_lookup=numeric_scale_lookup,
                        categorical_columns=categorical_columns,
                    ),
                ],
                [*split_features, base_feature],
                depth + 1,
            )

    recurse(0, [], [], [], 0)
    return pd.DataFrame(rows)


def _write_rule_text(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    summary: dict[str, Any],
    path: Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_lines = [
        f"feature_context: {summary['feature_context']}",
        f"criterion: {summary['criterion']}",
        f"train_rows: {summary['train_rows']}",
        f"test_rows: {summary['test_rows']}",
        f"numeric_feature_count: {summary['numeric_feature_count']}",
        f"categorical_feature_count: {summary['categorical_feature_count']}",
        f"transformed_feature_count: {summary['transformed_feature_count']}",
        f"tree_depth: {summary['tree_depth']}",
        f"leaf_count: {summary['leaf_count']}",
        f"test_balanced_accuracy: {summary['balanced_accuracy']:.6f}",
        f"test_f1: {summary['f1']:.6f}",
        (
            "threshold_note: tree text below uses transformed model inputs; "
            "see tree_leaf_rule_table.csv rule_text_interpretable for "
            "raw-scale/interpretable conditions."
        ),
        "",
        export_text(
            model,
            feature_names=feature_names,
            decimals=4,
            show_weights=True,
        ),
    ]

    output_path.write_text("\n".join(metric_lines), encoding="utf-8")
    return output_path


def _tree_split_feature_counts(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feature_index in model.tree_.feature:
        if feature_index < 0:
            continue
        feature_name = feature_names[int(feature_index)]
        base_name = _base_feature_name(
            feature_name, numeric_columns, categorical_columns
        )
        counts[base_name] = counts.get(base_name, 0) + 1
    return counts


def _fit_tree_for_seed(
    dataframe: pd.DataFrame,
    spec: RuleTreeSpec,
    random_state: int,
) -> tuple[dict[str, float], dict[str, int]]:
    excluded_features = set(BASE_EXCLUDED_COLUMNS) | set(spec.excluded_features)
    X_train_raw, X_test_raw, y_train, y_test = _split_data(
        dataframe, random_state=random_state
    )
    numeric_columns, categorical_columns = _feature_columns(
        dataframe, excluded_features
    )

    X_train = _clean_features(X_train_raw, numeric_columns, categorical_columns)
    X_test = _clean_features(X_test_raw, numeric_columns, categorical_columns)

    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)
    feature_names = _feature_names(preprocessor, numeric_columns, categorical_columns)

    model = DecisionTreeClassifier(
        criterion=spec.criterion,
        max_depth=4,
        min_samples_leaf=150,
        min_samples_split=300,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train_prepared, y_train)

    scores = _score_tree(model, X_test_prepared, y_test)
    split_counts = _tree_split_feature_counts(
        model=model,
        feature_names=feature_names,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    return scores, split_counts


def _stability_summary(
    dataframe: pd.DataFrame,
    spec: RuleTreeSpec,
) -> dict[str, Any]:
    score_rows: list[dict[str, float]] = []
    feature_counts: dict[str, int] = {}

    for random_state in STABILITY_RANDOM_STATES:
        scores, split_counts = _fit_tree_for_seed(
            dataframe=dataframe,
            spec=spec,
            random_state=random_state,
        )
        score_rows.append(scores)
        for feature_name, count in split_counts.items():
            feature_counts[feature_name] = feature_counts.get(feature_name, 0) + count

    scores_frame = pd.DataFrame(score_rows)
    top_features = sorted(
        feature_counts.items(), key=lambda item: (-item[1], item[0])
    )[:8]

    return {
        "stability_run_count": int(len(score_rows)),
        "stability_balanced_accuracy_mean": float(
            scores_frame["balanced_accuracy"].mean()
        ),
        "stability_balanced_accuracy_std": float(
            scores_frame["balanced_accuracy"].std(ddof=1)
        ),
        "stability_f1_mean": float(scores_frame["f1"].mean()),
        "stability_f1_std": float(scores_frame["f1"].std(ddof=1)),
        "stability_top_split_features": "|".join(
            feature_name for feature_name, _ in top_features
        ),
        "stability_audit_note": (
            "Repeated session-grouped holdout splits with fixed shallow-tree "
            "settings; used to assess rule-layer stability, not to tune "
            "hyperparameters."
        ),
    }


def _fit_rule_tree(
    dataframe: pd.DataFrame,
    spec: RuleTreeSpec,
    results_dir: Path,
    figures_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Path]]:
    excluded_features = set(BASE_EXCLUDED_COLUMNS) | set(spec.excluded_features)
    working = dataframe.copy()
    if TARGET_COLUMN not in working.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    X_train_raw, X_test_raw, y_train, y_test = _split_data(working)
    numeric_columns, categorical_columns = _feature_columns(working, excluded_features)

    if not numeric_columns and not categorical_columns:
        raise ValueError("No usable features remain after leakage exclusions.")

    X_train = _clean_features(X_train_raw, numeric_columns, categorical_columns)
    X_test = _clean_features(X_test_raw, numeric_columns, categorical_columns)

    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)
    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    feature_names = _feature_names(preprocessor, numeric_columns, categorical_columns)
    scale_lookup = _numeric_scale_lookup(preprocessor, numeric_columns)

    model = DecisionTreeClassifier(
        criterion=spec.criterion,
        max_depth=4,
        min_samples_leaf=150,
        min_samples_split=300,
        class_weight="balanced",
        random_state=MODEL_RANDOM_STATE,
    )
    model.fit(X_train_prepared, y_train)

    scores = _score_tree(model, X_test_prepared, y_test)

    summary: dict[str, Any] = {
        "feature_context": spec.feature_context,
        "criterion": spec.criterion,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "numeric_feature_count": int(len(numeric_columns)),
        "categorical_feature_count": int(len(categorical_columns)),
        "transformed_feature_count": int(len(feature_names)),
        "tree_depth": int(model.get_depth()),
        "leaf_count": int(model.get_n_leaves()),
        **scores,
        "split_design": "Session-grouped 80/20 holdout when session IDs are available",
        "preprocessing": (
            "Median/mode imputation; numeric standard scaling; "
            "categorical one-hot encoding"
        ),
        "exclusion_design": (
            "Target label, target components, direct outcome columns, and declared "
            "reduced-context fields excluded according to feature context."
        ),
    }

    summary.update(_stability_summary(working, spec))

    leaf_rules = _extract_leaf_rules(
        model=model,
        feature_names=feature_names,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_scale_lookup=scale_lookup,
        feature_context=spec.feature_context,
        criterion=spec.criterion,
    )

    stem = f"{spec.feature_context}_{spec.criterion}"
    rule_text_path = _write_rule_text(
        model=model,
        feature_names=feature_names,
        summary=summary,
        path=results_dir / f"tree_rules_{stem}.txt",
    )
    figure_path = plot_pruned_decision_tree(
        model=model,
        feature_names=feature_names,
        path=figures_dir / f"pruned_tree_{stem}.pdf",
    )

    outputs = {
        f"rules_{stem}": rule_text_path,
        f"figure_{stem}": figure_path,
    }

    return summary, leaf_rules, outputs



def _build_decision_rule_table(
    leaf_table: pd.DataFrame,
    feature_context: str = "reduced_context",
    criterion: str = "gini",
) -> pd.DataFrame:
    rules = leaf_table[
        (leaf_table["feature_context"] == feature_context)
        & (leaf_table["criterion"] == criterion)
    ].copy()

    if rules.empty:
        return pd.DataFrame(
            columns=[
                "feature_context",
                "criterion",
                "rule_id",
                "predicted_state",
                "class_confidence",
                "node_sample_count",
                "degraded_rate_in_leaf",
                "rule_text_interpretable",
                "threshold_units_note",
            ]
        )

    rules["predicted_class"] = rules["predicted_class"].astype(int)
    rules["degraded_rate_in_leaf"] = pd.to_numeric(
        rules["degraded_rate_in_leaf"], errors="coerce"
    ).fillna(0.0)
    rules["node_sample_count"] = pd.to_numeric(
        rules["node_sample_count"], errors="coerce"
    ).fillna(0.0)
    rules["predicted_state"] = np.where(
        rules["predicted_class"].eq(1),
        "degraded",
        "not_degraded",
    )
    rules["class_confidence"] = np.where(
        rules["predicted_class"].eq(1),
        rules["degraded_rate_in_leaf"],
        1.0 - rules["degraded_rate_in_leaf"],
    )
    rules["selection_score"] = rules["class_confidence"] * np.log1p(
        rules["node_sample_count"]
    )

    output_columns = [
        "feature_context",
        "criterion",
        "rule_id",
        "predicted_state",
        "class_confidence",
        "node_sample_count",
        "degraded_rate_in_leaf",
        "rule_text_interpretable",
        "threshold_units_note",
    ]

    return rules.sort_values(
        ["predicted_class", "selection_score", "class_confidence"],
        ascending=[False, False, False],
    )[output_columns].reset_index(drop=True)

def run_decision_tree_rule_extraction(
    supervised_table_path: str | Path,
    results_dir: str | Path,
    figures_dir: str | Path,
) -> dict[str, Path]:
    results_path = ensure_directory(results_dir)
    figures_path = ensure_directory(figures_dir)

    dataframe = read_csv(supervised_table_path, low_memory=False)

    specs = [
        RuleTreeSpec(
            feature_context="full_context",
            criterion=criterion,
            excluded_features=frozenset(),
        )
        for _, criterion in TREE_SPECS
    ] + [
        RuleTreeSpec(
            feature_context="reduced_context",
            criterion=criterion,
            excluded_features=frozenset(REDUCED_CONTEXT_EXCLUSIONS),
        )
        for _, criterion in TREE_SPECS
    ]

    summary_rows: list[dict[str, Any]] = []
    leaf_frames: list[pd.DataFrame] = []
    output_paths: dict[str, Path] = {}

    for spec in specs:
        summary, leaf_rules, outputs = _fit_rule_tree(
            dataframe=dataframe,
            spec=spec,
            results_dir=results_path,
            figures_dir=figures_path,
        )
        summary_rows.append(summary)
        leaf_frames.append(leaf_rules)
        output_paths.update(outputs)

    summary_table = pd.DataFrame(summary_rows).sort_values(
        ["feature_context", "criterion"]
    )
    leaf_table = pd.concat(leaf_frames, ignore_index=True).sort_values(
        ["feature_context", "criterion", "predicted_class", "degraded_rate_in_leaf"],
        ascending=[True, True, False, False],
    )

    summary_path = write_csv(summary_table, results_path / "tree_rule_summary.csv")
    leaf_path = write_csv(leaf_table, results_path / "tree_leaf_rule_table.csv")

    rule_table = _build_decision_rule_table(
        leaf_table=leaf_table,
        feature_context="reduced_context",
        criterion="gini",
    )
    rule_path = write_csv(
        rule_table,
        results_path / "decision_tree_rules.csv",
    )
    output_paths["decision_tree_rules"] = rule_path

    representative_paths_figure = plot_representative_decision_paths(
        leaf_rules=leaf_table,
        path=figures_path / "representative_decision_paths_reduced_context_gini.pdf",
        feature_context="reduced_context",
        criterion="gini",
        per_class=3,
    )
    output_paths["representative_decision_paths_figure"] = representative_paths_figure

    reduced_gini_rules = results_path / "tree_rules_reduced_context_gini.txt"
    reduced_entropy_rules = results_path / "tree_rules_reduced_context_entropy.txt"
    if reduced_gini_rules.exists():
        (results_path / "tree_rules_gini.txt").write_text(
            reduced_gini_rules.read_text(encoding="utf-8"), encoding="utf-8"
        )
        output_paths["rules_gini"] = results_path / "tree_rules_gini.txt"
    if reduced_entropy_rules.exists():
        (results_path / "tree_rules_entropy.txt").write_text(
            reduced_entropy_rules.read_text(encoding="utf-8"), encoding="utf-8"
        )
        output_paths["rules_entropy"] = results_path / "tree_rules_entropy.txt"

    reduced_gini_figure = figures_path / "pruned_tree_reduced_context_gini.pdf"
    if reduced_gini_figure.exists():
        alias_figure = figures_path / "pruned_tree.pdf"
        alias_figure.write_bytes(reduced_gini_figure.read_bytes())
        output_paths["pruned_tree_figure"] = alias_figure

    output_paths["tree_rule_summary"] = summary_path
    output_paths["tree_leaf_rule_table"] = leaf_path

    return output_paths
