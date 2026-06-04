from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from network_degradation_mining.io import ensure_directory, read_csv, write_csv

MODEL_RANDOM_STATE = 42
PRIMARY_FEATURE_BLOCK = "reduced_context"

IDENTIFIER_COLUMNS = {
    "measurement_id",
    "session_id",
    "timestamp",
    "time_id",
    "area_id",
    "device_id",
    "network_id",
    "radio_id",
    "application_id",
    "mobility_id",
    "environment_id",
    "service_state_id",
}

PROFILE_ONLY_COLUMNS = {
    "service_degraded",
    "dropped_connection",
    "video_quality_label",
    "high_latency",
    "high_jitter",
    "downlink_service_shortfall",
    "poor_video_quality",
    "download_speed_mbps",
    "upload_speed_mbps",
    "latency_ms",
    "jitter_ms",
    "ping_ms",
    "throughput_satisfaction_ratio",
    "downlink_shortfall_fraction",
}

CONTEXT_CLUSTERING_COLUMNS = [
    "signal_strength_dbm",
    "los_probability",
    "distance_2d_m",
    "path_loss_db",
    "contextual_penalty_db",
    "link_capacity_downlink_mbps",
    "offered_downlink_mbps",
    "offered_upload_mbps",
    "interval_handover_count",
    "activity_factor",
    "data_usage_mb",
    "distance_to_tower_km",
    "network_type",
    "infrastructure_profile",
    "band",
    "app_type",
    "movement_speed",
    "weather",
    "obstruction_level",
    "congestion_level",
    "tower_load",
    "time_of_day_bin",
]

FEATURE_BLOCKS = {
    "full_context": CONTEXT_CLUSTERING_COLUMNS,
    "reduced_context": [
        "signal_strength_dbm",
        "los_probability",
        "path_loss_db",
        "contextual_penalty_db",
        "link_capacity_downlink_mbps",
        "offered_downlink_mbps",
        "offered_upload_mbps",
        "interval_handover_count",
        "activity_factor",
        "data_usage_mb",
        "distance_to_tower_km",
        "network_type",
        "band",
        "app_type",
        "movement_speed",
        "weather",
        "obstruction_level",
        "congestion_level",
        "tower_load",
        "time_of_day_bin",
    ],
    "radio_capacity_context": [
        "signal_strength_dbm",
        "los_probability",
        "path_loss_db",
        "contextual_penalty_db",
        "link_capacity_downlink_mbps",
        "distance_to_tower_km",
        "network_type",
        "infrastructure_profile",
        "band",
        "weather",
        "obstruction_level",
    ],
    "load_application_context": [
        "offered_downlink_mbps",
        "offered_upload_mbps",
        "interval_handover_count",
        "activity_factor",
        "data_usage_mb",
        "app_type",
        "movement_speed",
        "congestion_level",
        "tower_load",
        "time_of_day_bin",
    ],
    "numeric_context_only": [
        "signal_strength_dbm",
        "los_probability",
        "distance_2d_m",
        "path_loss_db",
        "contextual_penalty_db",
        "link_capacity_downlink_mbps",
        "offered_downlink_mbps",
        "offered_upload_mbps",
        "interval_handover_count",
        "activity_factor",
        "data_usage_mb",
        "distance_to_tower_km",
    ],
}

PROFILE_NUMERIC_COLUMNS = [
    "service_degraded",
    "signal_strength_dbm",
    "download_speed_mbps",
    "upload_speed_mbps",
    "latency_ms",
    "jitter_ms",
    "ping_ms",
    "throughput_satisfaction_ratio",
    "downlink_shortfall_fraction",
    "offered_downlink_mbps",
    "offered_upload_mbps",
    "link_capacity_downlink_mbps",
    "distance_to_tower_km",
    "distance_2d_m",
    "path_loss_db",
    "contextual_penalty_db",
    "interval_handover_count",
    "activity_factor",
    "data_usage_mb",
]

PROFILE_CATEGORICAL_COLUMNS = [
    "network_type",
    "infrastructure_profile",
    "band",
    "app_type",
    "movement_speed",
    "weather",
    "obstruction_level",
    "congestion_level",
    "tower_load",
    "time_of_day_bin",
]

HEATMAP_PROFILE_COLUMNS = [
    "service_degraded_mean",
    "signal_strength_dbm_mean",
    "download_speed_mbps_mean",
    "upload_speed_mbps_mean",
    "latency_ms_mean",
    "jitter_ms_mean",
    "throughput_satisfaction_ratio_mean",
    "downlink_shortfall_fraction_mean",
    "offered_downlink_mbps_mean",
    "link_capacity_downlink_mbps_mean",
    "distance_to_tower_km_mean",
    "path_loss_db_mean",
    "contextual_penalty_db_mean",
    "interval_handover_count_mean",
    "activity_factor_mean",
]

CONTEXT_MAP_NUMERIC_COLUMNS = [
    "signal_strength_dbm",
    "los_probability",
    "distance_2d_m",
    "path_loss_db",
    "contextual_penalty_db",
    "link_capacity_downlink_mbps",
    "offered_downlink_mbps",
    "offered_upload_mbps",
    "interval_handover_count",
    "activity_factor",
    "data_usage_mb",
    "distance_to_tower_km",
]

EFFECT_SIZE_COLUMNS = [
    "service_degraded",
    "signal_strength_dbm",
    "download_speed_mbps",
    "upload_speed_mbps",
    "latency_ms",
    "jitter_ms",
    "throughput_satisfaction_ratio",
    "downlink_shortfall_fraction",
    "link_capacity_downlink_mbps",
    "offered_downlink_mbps",
    "path_loss_db",
    "contextual_penalty_db",
    "distance_to_tower_km",
    "interval_handover_count",
    "activity_factor",
]

K_RANGE = range(2, 9)
SILHOUETTE_SAMPLE_LIMIT = 8000
HIERARCHICAL_SAMPLE_LIMIT = 5000
DENSITY_SAMPLE_LIMIT = 4000
CONTEXT_MAP_SAMPLE_LIMIT = 1500
PROJECTION_SAMPLE_LIMIT = 4500
FEATURE_BLOCK_SAMPLE_LIMIT = 8000
STABILITY_SAMPLE_LIMIT = 8000
STABILITY_SEEDS = [7, 13, 23, 29, 31, 37, 42, 53]
BOOTSTRAP_REPLICATIONS = 300
NULL_BASELINE_REPLICATIONS = 1000
DBSCAN_MIN_SAMPLES = [10, 25, 50, 100]
DBSCAN_EPS_QUANTILES = [0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
OPTICS_MIN_SAMPLES = [10, 25, 50, 100]
MIN_VALID_CLUSTER_FRACTION = 0.01
MAX_VALID_NOISE_FRACTION = 0.60


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _available_columns(dataframe: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in dataframe.columns]


def _block_columns(dataframe: pd.DataFrame, block_name: str) -> list[str]:
    requested_columns = FEATURE_BLOCKS.get(
        block_name, FEATURE_BLOCKS[PRIMARY_FEATURE_BLOCK]
    )
    return [
        column
        for column in requested_columns
        if column in dataframe.columns
        and column not in IDENTIFIER_COLUMNS
        and column not in PROFILE_ONLY_COLUMNS
        and dataframe[column].nunique(dropna=True) > 1
    ]


def _feature_columns(
    dataframe: pd.DataFrame, block_name: str
) -> tuple[list[str], list[str]]:
    candidate_columns = _block_columns(dataframe, block_name)
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


def _as_dense_array(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        return np.asarray(values.toarray(), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _sample_array(
    x_values: np.ndarray,
    row_limit: int,
    random_state: int = MODEL_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = x_values.shape[0]
    if row_count <= row_limit:
        index = np.arange(row_count)
        return x_values, index

    rng = np.random.default_rng(random_state)
    index = np.sort(rng.choice(row_count, size=row_limit, replace=False))
    return x_values[index], index


def _sample_dataframe(
    dataframe: pd.DataFrame,
    row_limit: int,
    random_state: int = MODEL_RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    row_count = len(dataframe)
    if row_count <= row_limit:
        index = np.arange(row_count)
        return dataframe.reset_index(drop=True), index
    rng = np.random.default_rng(random_state)
    index = np.sort(rng.choice(row_count, size=row_limit, replace=False))
    return dataframe.iloc[index].reset_index(drop=True), index


def _cluster_count(labels: np.ndarray) -> int:
    unique_labels = set(np.asarray(labels).tolist())
    unique_labels.discard(-1)
    return len(unique_labels)


def _minimum_cluster_fraction(labels: np.ndarray) -> float:
    label_array = np.asarray(labels)
    valid = label_array[label_array != -1]
    if len(valid) == 0:
        return 0.0
    counts = pd.Series(valid).value_counts(normalize=True)
    return float(counts.min()) if not counts.empty else 0.0


def _is_valid_density_partition(row: dict[str, Any]) -> bool:
    return (
        int(row.get("cluster_count", 0)) >= 2
        and float(row.get("noise_fraction", 1.0)) <= MAX_VALID_NOISE_FRACTION
        and float(row.get("minimum_cluster_fraction", 0.0))
        >= MIN_VALID_CLUSTER_FRACTION
        and pd.notna(row.get("silhouette_score", np.nan))
    )


def _score_labels(
    x_values: np.ndarray,
    labels: np.ndarray,
    method: str,
    k: int | None,
    feature_block: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label_array = np.asarray(labels)
    cluster_count = _cluster_count(label_array)
    row: dict[str, Any] = {
        "feature_block": feature_block,
        "method": method,
        "k": k,
        "row_count": int(len(label_array)),
        "cluster_count": int(cluster_count),
        "noise_fraction": float(np.mean(label_array == -1))
        if len(label_array)
        else 0.0,
        "minimum_cluster_fraction": _minimum_cluster_fraction(label_array),
    }
    if extra:
        row.update(extra)

    if cluster_count >= 2:
        valid_mask = label_array != -1
        x_valid = x_values[valid_mask]
        labels_valid = label_array[valid_mask]
        unique_valid_labels = np.unique(labels_valid)
        if len(unique_valid_labels) >= 2 and len(x_valid) > len(unique_valid_labels):
            row["silhouette_score"] = float(silhouette_score(x_valid, labels_valid))
            row["davies_bouldin_score"] = float(
                davies_bouldin_score(x_valid, labels_valid)
            )
            row["calinski_harabasz_score"] = float(
                calinski_harabasz_score(x_valid, labels_valid)
            )
        else:
            row["silhouette_score"] = np.nan
            row["davies_bouldin_score"] = np.nan
            row["calinski_harabasz_score"] = np.nan
    else:
        row["silhouette_score"] = np.nan
        row["davies_bouldin_score"] = np.nan
        row["calinski_harabasz_score"] = np.nan

    return row


def _profile_clusters(
    dataframe: pd.DataFrame,
    labels: np.ndarray,
    method: str,
    feature_block: str,
) -> pd.DataFrame:
    working = dataframe.copy()
    working["cluster_label"] = labels

    rows: list[dict[str, Any]] = []
    numeric_columns = _available_columns(working, PROFILE_NUMERIC_COLUMNS)
    categorical_columns = _available_columns(working, PROFILE_CATEGORICAL_COLUMNS)

    for cluster_label, group in working.groupby("cluster_label", dropna=False):
        row: dict[str, Any] = {
            "feature_block": feature_block,
            "method": method,
            "cluster_label": int(cluster_label)
            if pd.notna(cluster_label)
            else cluster_label,
            "row_count": int(len(group)),
            "row_fraction": float(len(group) / len(working)) if len(working) else 0.0,
        }

        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = (
                float(values.mean()) if values.notna().any() else np.nan
            )
            row[f"{column}_median"] = (
                float(values.median()) if values.notna().any() else np.nan
            )

        for column in categorical_columns:
            mode = group[column].astype("string").dropna()
            if mode.empty:
                row[f"{column}_mode"] = ""
                row[f"{column}_mode_fraction"] = np.nan
            else:
                counts = mode.value_counts()
                row[f"{column}_mode"] = str(counts.index[0])
                row[f"{column}_mode_fraction"] = float(counts.iloc[0] / len(group))

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["method", "cluster_label"])


def _empty_cluster_profile_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = _available_columns(dataframe, PROFILE_NUMERIC_COLUMNS)
    categorical_columns = _available_columns(dataframe, PROFILE_CATEGORICAL_COLUMNS)

    columns = [
        "feature_block",
        "method",
        "cluster_label",
        "row_count",
        "row_fraction",
    ]

    for column in numeric_columns:
        columns.extend([f"{column}_mean", f"{column}_median"])

    for column in categorical_columns:
        columns.extend([f"{column}_mode", f"{column}_mode_fraction"])

    return pd.DataFrame(columns=columns)


def _best_k(validation: pd.DataFrame, method: str, feature_block: str) -> int:
    candidates = validation[
        (validation["method"] == method)
        & (validation["feature_block"] == feature_block)
        & validation["silhouette_score"].notna()
    ].copy()
    if candidates.empty:
        return 2
    return int(
        candidates.sort_values(
            ["silhouette_score", "calinski_harabasz_score"],
            ascending=[False, False],
        ).iloc[0]["k"]
    )


def _best_gmm_bic_k(validation: pd.DataFrame, feature_block: str) -> int | None:
    candidates = validation[
        (validation["method"] == "gaussian_mixture")
        & (validation["feature_block"] == feature_block)
        & validation["bic"].notna()
    ].copy()
    if candidates.empty:
        return None
    return int(candidates.sort_values("bic", ascending=True).iloc[0]["k"])


def _best_density_row(validation: pd.DataFrame, method: str) -> pd.Series | None:
    candidates = validation[validation["method"] == method].copy()
    if candidates.empty:
        return None
    valid = candidates[
        candidates.apply(lambda row: _is_valid_density_partition(row.to_dict()), axis=1)
    ]
    if valid.empty:
        return None
    return valid.sort_values(
        ["silhouette_score", "noise_fraction", "minimum_cluster_fraction"],
        ascending=[False, True, False],
    ).iloc[0]


def _prepare_matrix(
    dataframe: pd.DataFrame, block_name: str
) -> tuple[np.ndarray, list[str], list[str]]:
    numeric_columns, categorical_columns = _feature_columns(dataframe, block_name)
    if not numeric_columns and not categorical_columns:
        raise ValueError(f"No usable clustering features were found for {block_name}.")
    feature_frame = _clean_features(dataframe, numeric_columns, categorical_columns)
    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)
    x_values = _as_dense_array(preprocessor.fit_transform(feature_frame))
    return x_values, numeric_columns, categorical_columns


def _remap_labels_by_profile_rate(
    dataframe: pd.DataFrame,
    labels: np.ndarray,
    target_column: str = "service_degraded",
) -> np.ndarray:
    label_array = np.asarray(labels).copy()
    if target_column not in dataframe.columns:
        return label_array
    working = pd.DataFrame(
        {
            "label": label_array,
            "target": pd.to_numeric(dataframe[target_column], errors="coerce"),
        }
    )
    rates = (
        working[working["label"] != -1]
        .groupby("label")["target"]
        .mean()
        .sort_values(kind="mergesort")
    )
    mapping = {
        old_label: new_label for new_label, old_label in enumerate(rates.index.tolist())
    }
    remapped = np.array([mapping.get(label, label) for label in label_array], dtype=int)
    return remapped


def _make_kmeans_summary(kmeans_profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in kmeans_profile.sort_values("cluster_label").iterrows():
        degradation_rate = float(row.get("service_degraded_mean", np.nan))
        if pd.isna(degradation_rate):
            profile_label = "unlabeled profile"
        elif degradation_rate <= 0.25:
            profile_label = "stable-service profile"
        elif degradation_rate >= 0.60:
            profile_label = "elevated-degradation profile"
        else:
            profile_label = "mixed-service profile"

        rows.append(
            {
                "feature_block": row.get("feature_block", PRIMARY_FEATURE_BLOCK),
                "cluster_label": int(row["cluster_label"]),
                "profile_label": profile_label,
                "row_count": int(row["row_count"]),
                "row_fraction": float(row["row_fraction"]),
                "post_hoc_degradation_rate": degradation_rate,
                "signal_strength_dbm_mean": row.get("signal_strength_dbm_mean", np.nan),
                "download_speed_mbps_mean": row.get("download_speed_mbps_mean", np.nan),
                "latency_ms_mean": row.get("latency_ms_mean", np.nan),
                "downlink_shortfall_fraction_mean": row.get(
                    "downlink_shortfall_fraction_mean", np.nan
                ),
                "network_type_mode": row.get("network_type_mode", ""),
                "band_mode": row.get("band_mode", ""),
                "app_type_mode": row.get("app_type_mode", ""),
                "movement_speed_mode": row.get("movement_speed_mode", ""),
                "congestion_level_mode": row.get("congestion_level_mode", ""),
                "tower_load_mode": row.get("tower_load_mode", ""),
            }
        )
    return pd.DataFrame(rows)


def _feature_set_table(
    dataframe: pd.DataFrame,
    selected_block: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for block_name, requested_columns in FEATURE_BLOCKS.items():
        for column in requested_columns:
            if (
                column in dataframe.columns
                and dataframe[column].nunique(dropna=True) > 1
            ):
                role = (
                    "primary_clustering_input"
                    if block_name == selected_block
                    else "sensitivity_clustering_input"
                )
                feature_type = (
                    "numeric"
                    if pd.api.types.is_numeric_dtype(dataframe[column])
                    else "categorical"
                )
                rows.append(
                    {
                        "feature_block": block_name,
                        "feature_name": column,
                        "feature_type": feature_type,
                        "feature_role": role,
                    }
                )
    for column in sorted(PROFILE_ONLY_COLUMNS):
        if column in dataframe.columns:
            rows.append(
                {
                    "feature_block": "post_hoc_profile",
                    "feature_name": column,
                    "feature_type": "profile_only",
                    "feature_role": "post_hoc_profile",
                }
            )
    return pd.DataFrame(rows).drop_duplicates()


def _run_primary_validation(
    dataframe: pd.DataFrame,
    x_values: np.ndarray,
    x_eval: np.ndarray,
    eval_index: np.ndarray,
) -> tuple[
    pd.DataFrame,
    dict[int, KMeans],
    dict[int, np.ndarray],
    dict[int, GaussianMixture],
    dict[int, np.ndarray],
]:
    validation_rows: list[dict[str, Any]] = []

    kmeans_models: dict[int, KMeans] = {}
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=20, random_state=MODEL_RANDOM_STATE)
        labels = model.fit_predict(x_values)
        kmeans_models[k] = model
        validation_rows.append(
            _score_labels(
                x_eval,
                labels[eval_index],
                method="kmeans",
                k=k,
                feature_block=PRIMARY_FEATURE_BLOCK,
                extra={"inertia": float(model.inertia_), "fit_scope": "full_table"},
            )
        )

    x_hierarchical, hierarchical_index = _sample_array(
        x_values, HIERARCHICAL_SAMPLE_LIMIT, random_state=MODEL_RANDOM_STATE + 1
    )
    hierarchical_labels_by_k: dict[int, np.ndarray] = {}
    for k in K_RANGE:
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(x_hierarchical)
        hierarchical_labels_by_k[k] = labels
        validation_rows.append(
            _score_labels(
                x_hierarchical,
                labels,
                method="hierarchical",
                k=k,
                feature_block=PRIMARY_FEATURE_BLOCK,
                extra={"inertia": np.nan, "fit_scope": "bounded_sample"},
            )
        )

    gmm_models: dict[int, GaussianMixture] = {}
    gmm_labels_by_k: dict[int, np.ndarray] = {}
    for k in K_RANGE:
        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=1e-6,
            random_state=MODEL_RANDOM_STATE,
        )
        labels_eval = model.fit_predict(x_eval)
        gmm_models[k] = model
        gmm_labels_by_k[k] = model.predict(x_values)
        validation_rows.append(
            _score_labels(
                x_eval,
                labels_eval,
                method="gaussian_mixture",
                k=k,
                feature_block=PRIMARY_FEATURE_BLOCK,
                extra={
                    "inertia": np.nan,
                    "bic": float(model.bic(x_eval)),
                    "aic": float(model.aic(x_eval)),
                    "fit_scope": "bounded_sample",
                },
            )
        )

    return (
        pd.DataFrame(validation_rows),
        kmeans_models,
        hierarchical_labels_by_k,
        gmm_models,
        gmm_labels_by_k,
    )


def _dbscan_eps_candidates(x_values: np.ndarray) -> list[float]:
    eps_values: set[float] = set()
    for min_samples in DBSCAN_MIN_SAMPLES:
        n_neighbors = min(min_samples, max(2, len(x_values) - 1))
        if n_neighbors < 2:
            continue
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        neighbors.fit(x_values)
        distances, _ = neighbors.kneighbors(x_values)
        kth_distances = np.sort(distances[:, -1])
        quantile_values = np.quantile(kth_distances, DBSCAN_EPS_QUANTILES)
        for value in quantile_values:
            if np.isfinite(value) and value > 0:
                eps_values.add(round(float(value), 4))
    return sorted(eps_values)


def _run_density_clustering(
    dataframe: pd.DataFrame,
    x_values: np.ndarray,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    x_density, density_index = _sample_array(
        x_values, DENSITY_SAMPLE_LIMIT, random_state=MODEL_RANDOM_STATE + 2
    )
    density_source = dataframe.iloc[density_index].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    dbscan_labels_by_key: dict[tuple[float, int], np.ndarray] = {}

    for eps in _dbscan_eps_candidates(x_density):
        for min_samples in DBSCAN_MIN_SAMPLES:
            model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels = model.fit_predict(x_density)
            dbscan_labels_by_key[(eps, min_samples)] = labels
            rows.append(
                _score_labels(
                    x_density,
                    labels,
                    method="dbscan",
                    k=None,
                    feature_block=PRIMARY_FEATURE_BLOCK,
                    extra={
                        "inertia": np.nan,
                        "eps": eps,
                        "min_samples": min_samples,
                        "fit_scope": "bounded_sample",
                    },
                )
            )

    optics_labels_by_min_samples: dict[int, np.ndarray] = {}
    for min_samples in OPTICS_MIN_SAMPLES:
        min_cluster_size = max(25, int(round(0.02 * len(x_density))))
        model = OPTICS(
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            xi=0.05,
            n_jobs=-1,
        )
        labels = model.fit_predict(x_density)
        optics_labels_by_min_samples[min_samples] = labels
        rows.append(
            _score_labels(
                x_density,
                labels,
                method="optics",
                k=None,
                feature_block=PRIMARY_FEATURE_BLOCK,
                extra={
                    "inertia": np.nan,
                    "eps": np.nan,
                    "min_samples": min_samples,
                    "min_cluster_size": min_cluster_size,
                    "fit_scope": "bounded_sample",
                },
            )
        )

    validation = pd.DataFrame(rows)
    dbscan_sweep = validation[validation["method"] == "dbscan"].copy()
    optics_sweep = validation[validation["method"] == "optics"].copy()

    status_rows: list[dict[str, Any]] = []
    dbscan_profile = _empty_cluster_profile_table(density_source)
    optics_profile = _empty_cluster_profile_table(density_source)

    best_dbscan = _best_density_row(validation, "dbscan")
    if best_dbscan is None:
        status_rows.append(
            {
                "method": "dbscan",
                "status": "no_valid_density_partition",
                "reason": (
                    "tested settings did not produce at least two non-noise clusters "
                    "with acceptable noise and minimum cluster size"
                ),
                "selected_eps": np.nan,
                "selected_min_samples": np.nan,
                "selected_min_cluster_size": np.nan,
            }
        )
    else:
        eps = float(best_dbscan["eps"])
        min_samples = int(best_dbscan["min_samples"])
        labels = dbscan_labels_by_key[(eps, min_samples)]
        dbscan_profile = _profile_clusters(
            density_source, labels, "dbscan", PRIMARY_FEATURE_BLOCK
        )
        status_rows.append(
            {
                "method": "dbscan",
                "status": "valid_density_partition_found",
                "reason": (
                    "selected by silhouette among partitions satisfying "
                    "density-clustering validity rules"
                ),
                "selected_eps": eps,
                "selected_min_samples": min_samples,
                "selected_min_cluster_size": np.nan,
            }
        )

    best_optics = _best_density_row(validation, "optics")
    if best_optics is None:
        status_rows.append(
            {
                "method": "optics",
                "status": "no_valid_density_partition",
                "reason": (
                    "tested settings did not produce at least two non-noise clusters "
                    "with acceptable noise and minimum cluster size"
                ),
                "selected_eps": np.nan,
                "selected_min_samples": np.nan,
                "selected_min_cluster_size": np.nan,
            }
        )
    else:
        min_samples = int(best_optics["min_samples"])
        labels = optics_labels_by_min_samples[min_samples]
        optics_profile = _profile_clusters(
            density_source, labels, "optics", PRIMARY_FEATURE_BLOCK
        )
        status_rows.append(
            {
                "method": "optics",
                "status": "valid_density_partition_found",
                "reason": (
                    "selected by silhouette among partitions satisfying "
                    "density-clustering validity rules"
                ),
                "selected_eps": np.nan,
                "selected_min_samples": min_samples,
                "selected_min_cluster_size": int(
                    best_optics.get("min_cluster_size", np.nan)
                ),
            }
        )

    return (
        validation,
        dbscan_sweep,
        optics_sweep,
        pd.DataFrame(status_rows),
        dbscan_profile,
        optics_profile,
    )


def _run_feature_block_sensitivity(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled, _ = _sample_dataframe(
        dataframe, FEATURE_BLOCK_SAMPLE_LIMIT, random_state=MODEL_RANDOM_STATE + 11
    )
    validation_rows: list[dict[str, Any]] = []
    profile_rows: list[pd.DataFrame] = []

    for block_name in FEATURE_BLOCKS:
        try:
            x_values, _, _ = _prepare_matrix(sampled, block_name)
        except ValueError:
            continue
        labels_by_k: dict[int, np.ndarray] = {}
        for k in K_RANGE:
            model = KMeans(n_clusters=k, n_init=10, random_state=MODEL_RANDOM_STATE)
            labels = model.fit_predict(x_values)
            labels_by_k[k] = labels
            validation_rows.append(
                _score_labels(
                    x_values,
                    labels,
                    method="kmeans",
                    k=k,
                    feature_block=block_name,
                    extra={
                        "inertia": float(model.inertia_),
                        "fit_scope": "bounded_sample",
                    },
                )
            )
        block_validation = pd.DataFrame(
            [row for row in validation_rows if row["feature_block"] == block_name]
        )
        best_k = _best_k(block_validation, "kmeans", block_name)
        best_labels = _remap_labels_by_profile_rate(sampled, labels_by_k[best_k])
        profile_rows.append(
            _profile_clusters(sampled, best_labels, "kmeans", block_name)
        )

    profiles = (
        pd.concat(profile_rows, ignore_index=True) if profile_rows else pd.DataFrame()
    )
    return pd.DataFrame(validation_rows), profiles


def _run_kmeans_stability(
    dataframe: pd.DataFrame,
    selected_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sampled, _ = _sample_dataframe(
        dataframe, STABILITY_SAMPLE_LIMIT, random_state=MODEL_RANDOM_STATE + 17
    )
    x_values, _, _ = _prepare_matrix(sampled, PRIMARY_FEATURE_BLOCK)
    labels_by_seed: dict[int, np.ndarray] = {}
    seed_rows: list[dict[str, Any]] = []

    for seed in STABILITY_SEEDS:
        model = KMeans(n_clusters=selected_k, n_init=20, random_state=seed)
        labels = model.fit_predict(x_values)
        labels = _remap_labels_by_profile_rate(sampled, labels)
        labels_by_seed[seed] = labels
        row = _score_labels(
            x_values,
            labels,
            method="kmeans",
            k=selected_k,
            feature_block=PRIMARY_FEATURE_BLOCK,
            extra={
                "seed": seed,
                "inertia": float(model.inertia_),
                "fit_scope": "bounded_sample",
            },
        )
        profile = _profile_clusters(sampled, labels, "kmeans", PRIMARY_FEATURE_BLOCK)
        for _, profile_row in profile.iterrows():
            cluster_label = int(profile_row["cluster_label"])
            row[f"cluster_{cluster_label}_row_fraction"] = float(
                profile_row["row_fraction"]
            )
            if "service_degraded_mean" in profile_row:
                row[f"cluster_{cluster_label}_degradation_rate"] = float(
                    profile_row["service_degraded_mean"]
                )
        if "service_degraded" in sampled.columns:
            rates = profile["service_degraded_mean"].astype(float).to_numpy()
            row["degradation_rate_range"] = float(np.nanmax(rates) - np.nanmin(rates))
        seed_rows.append(row)

    pairwise_rows: list[dict[str, Any]] = []
    for seed_a, seed_b in combinations(STABILITY_SEEDS, 2):
        pairwise_rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "adjusted_rand_index": float(
                    adjusted_rand_score(labels_by_seed[seed_a], labels_by_seed[seed_b])
                ),
            }
        )

    pairwise = pd.DataFrame(pairwise_rows)
    summary = pd.DataFrame(
        [
            {
                "selected_k": selected_k,
                "seed_count": len(STABILITY_SEEDS),
                "mean_pairwise_adjusted_rand_index": float(
                    pairwise["adjusted_rand_index"].mean()
                )
                if not pairwise.empty
                else np.nan,
                "min_pairwise_adjusted_rand_index": float(
                    pairwise["adjusted_rand_index"].min()
                )
                if not pairwise.empty
                else np.nan,
                "max_pairwise_adjusted_rand_index": float(
                    pairwise["adjusted_rand_index"].max()
                )
                if not pairwise.empty
                else np.nan,
                "mean_degradation_rate_range": float(
                    pd.DataFrame(seed_rows)["degradation_rate_range"].mean()
                )
                if seed_rows
                and "degradation_rate_range" in pd.DataFrame(seed_rows).columns
                else np.nan,
            }
        ]
    )
    return pd.DataFrame(seed_rows), pairwise, summary


def _bootstrap_cluster_profile_intervals(
    dataframe: pd.DataFrame,
    labels: np.ndarray,
    metrics: list[str] | None = None,
    replications: int = BOOTSTRAP_REPLICATIONS,
) -> pd.DataFrame:
    if metrics is None:
        metrics = _available_columns(dataframe, EFFECT_SIZE_COLUMNS)
    working = dataframe.copy()
    working["cluster_label"] = labels
    rng = np.random.default_rng(MODEL_RANDOM_STATE + 23)
    rows: list[dict[str, Any]] = []

    for cluster_label, group in working.groupby("cluster_label"):
        for metric in metrics:
            values = (
                pd.to_numeric(group[metric], errors="coerce")
                .dropna()
                .to_numpy(dtype=float)
            )
            if len(values) == 0:
                continue
            point = float(np.mean(values))
            if len(values) == 1:
                lower = point
                upper = point
            else:
                estimates = np.empty(replications, dtype=float)
                for index in range(replications):
                    estimates[index] = float(
                        np.mean(rng.choice(values, size=len(values), replace=True))
                    )
                lower = float(np.percentile(estimates, 2.5))
                upper = float(np.percentile(estimates, 97.5))
            rows.append(
                {
                    "feature_block": PRIMARY_FEATURE_BLOCK,
                    "method": "kmeans",
                    "cluster_label": int(cluster_label),
                    "metric": metric,
                    "mean": point,
                    "ci_lower_95": lower,
                    "ci_upper_95": upper,
                    "row_count": int(len(values)),
                    "bootstrap_replications": replications,
                }
            )
    return pd.DataFrame(rows)


def _cluster_effect_sizes(dataframe: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    metrics = _available_columns(dataframe, EFFECT_SIZE_COLUMNS)
    working = dataframe.copy()
    working["cluster_label"] = labels
    if "service_degraded" not in working.columns:
        return pd.DataFrame()

    degradation_rates = (
        working.groupby("cluster_label")["service_degraded"]
        .mean()
        .sort_values(kind="mergesort")
    )
    if degradation_rates.empty:
        return pd.DataFrame()
    reference_cluster = int(degradation_rates.index[0])
    comparison_cluster = int(degradation_rates.index[-1])

    rows: list[dict[str, Any]] = []
    reference = working[working["cluster_label"] == reference_cluster]
    comparison = working[working["cluster_label"] == comparison_cluster]

    for metric in metrics:
        reference_values = pd.to_numeric(reference[metric], errors="coerce").dropna()
        comparison_values = pd.to_numeric(comparison[metric], errors="coerce").dropna()
        if reference_values.empty or comparison_values.empty:
            continue
        reference_mean = float(reference_values.mean())
        comparison_mean = float(comparison_values.mean())
        pooled_sd = float(
            np.sqrt(
                (
                    reference_values.var(ddof=1) * (len(reference_values) - 1)
                    + comparison_values.var(ddof=1) * (len(comparison_values) - 1)
                )
                / max(len(reference_values) + len(comparison_values) - 2, 1)
            )
        )
        if not np.isfinite(pooled_sd) or pooled_sd == 0:
            standardized_difference = np.nan
        else:
            standardized_difference = (comparison_mean - reference_mean) / pooled_sd

        row = {
            "feature_block": PRIMARY_FEATURE_BLOCK,
            "method": "kmeans",
            "reference_cluster": reference_cluster,
            "comparison_cluster": comparison_cluster,
            "metric": metric,
            "reference_mean": reference_mean,
            "comparison_mean": comparison_mean,
            "mean_difference": comparison_mean - reference_mean,
            "standardized_mean_difference": float(standardized_difference)
            if pd.notna(standardized_difference)
            else np.nan,
        }
        if metric == "service_degraded":
            if reference_mean > 0:
                row["risk_ratio"] = comparison_mean / reference_mean
            else:
                row["risk_ratio"] = np.nan
            row["risk_difference"] = comparison_mean - reference_mean
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_null_baseline(
    dataframe: pd.DataFrame,
    labels: np.ndarray,
    replications: int = NULL_BASELINE_REPLICATIONS,
) -> pd.DataFrame:
    if "service_degraded" not in dataframe.columns:
        return pd.DataFrame()
    y = (
        pd.to_numeric(dataframe["service_degraded"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=float)
    )
    label_array = np.asarray(labels)
    valid_labels = sorted(np.unique(label_array).tolist())
    if len(valid_labels) < 2:
        return pd.DataFrame()

    observed_rates = []
    cluster_sizes = []
    for label in valid_labels:
        mask = label_array == label
        cluster_sizes.append(int(mask.sum()))
        observed_rates.append(float(np.mean(y[mask])))
    observed_gap = float(max(observed_rates) - min(observed_rates))

    base_labels = np.concatenate(
        [np.repeat(index, size) for index, size in enumerate(cluster_sizes)]
    )
    rng = np.random.default_rng(MODEL_RANDOM_STATE + 31)
    random_gaps = np.empty(replications, dtype=float)
    for index in range(replications):
        shuffled = rng.permutation(base_labels)
        rates = [float(np.mean(y[shuffled == label])) for label in np.unique(shuffled)]
        random_gaps[index] = max(rates) - min(rates)

    p_value = float((np.sum(random_gaps >= observed_gap) + 1) / (replications + 1))
    return pd.DataFrame(
        [
            {
                "feature_block": PRIMARY_FEATURE_BLOCK,
                "method": "kmeans",
                "cluster_count": len(valid_labels),
                "observed_degradation_rate_gap": observed_gap,
                "random_gap_mean": float(np.mean(random_gaps)),
                "random_gap_std": float(np.std(random_gaps, ddof=1)),
                "random_gap_p95": float(np.percentile(random_gaps, 95)),
                "random_gap_p99": float(np.percentile(random_gaps, 99)),
                "empirical_p_value_random_gap_ge_observed": p_value,
                "replications": replications,
            }
        ]
    )


def _session_mode(series: pd.Series) -> Any:
    values = series.astype("string").dropna()
    if values.empty:
        return ""
    return values.value_counts().index[0]


def _make_session_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "session_id" not in dataframe.columns:
        return pd.DataFrame()
    numeric_columns = _available_columns(
        dataframe,
        list(
            dict.fromkeys(FEATURE_BLOCKS[PRIMARY_FEATURE_BLOCK] + ["service_degraded"])
        ),
    )
    numeric_columns = [
        column
        for column in numeric_columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    categorical_columns = [
        column
        for column in _available_columns(
            dataframe, FEATURE_BLOCKS[PRIMARY_FEATURE_BLOCK]
        )
        if column not in numeric_columns
    ]

    aggregations: dict[str, Any] = {column: "mean" for column in numeric_columns}
    for column in categorical_columns:
        aggregations[column] = _session_mode
    session_table = dataframe.groupby("session_id", as_index=False).agg(aggregations)
    session_table["record_count"] = dataframe.groupby("session_id").size().to_numpy()
    return session_table


def _run_session_level_clustering(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_table = _make_session_table(dataframe)
    if session_table.empty or len(session_table) < 3:
        return pd.DataFrame(), pd.DataFrame()
    x_values, _, _ = _prepare_matrix(session_table, PRIMARY_FEATURE_BLOCK)
    validation_rows: list[dict[str, Any]] = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in K_RANGE:
        if k >= len(session_table):
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=MODEL_RANDOM_STATE)
        labels = model.fit_predict(x_values)
        labels_by_k[k] = labels
        validation_rows.append(
            _score_labels(
                x_values,
                labels,
                method="kmeans",
                k=k,
                feature_block=PRIMARY_FEATURE_BLOCK,
                extra={"inertia": float(model.inertia_), "fit_scope": "session_table"},
            )
        )
    validation = pd.DataFrame(validation_rows)
    if validation.empty:
        return validation, pd.DataFrame()
    best_k = _best_k(validation, "kmeans", PRIMARY_FEATURE_BLOCK)
    labels = _remap_labels_by_profile_rate(session_table, labels_by_k[best_k])
    profiles = _profile_clusters(session_table, labels, "kmeans", PRIMARY_FEATURE_BLOCK)
    return validation, profiles


def _cluster_projection_table(
    dataframe: pd.DataFrame,
    x_values: np.ndarray,
    labels: np.ndarray,
    selected_k: int,
    feature_block: str = PRIMARY_FEATURE_BLOCK,
    row_limit: int = PROJECTION_SAMPLE_LIMIT,
) -> pd.DataFrame:
    x_sample, sample_index = _sample_array(
        x_values,
        row_limit,
        random_state=MODEL_RANDOM_STATE + 71,
    )
    source = dataframe.iloc[sample_index].reset_index(drop=True)
    label_sample = np.asarray(labels)[sample_index]

    if len(x_sample) == 0:
        return pd.DataFrame(
            columns=[
                "feature_block",
                "method",
                "selected_k",
                "source_row_index",
                "cluster_label",
                "cluster_name",
                "pc1",
                "pc2",
                "service_degraded",
                "degradation_status",
            ]
        )

    projection = PCA(n_components=2, random_state=MODEL_RANDOM_STATE).fit_transform(
        x_sample
    )

    output = pd.DataFrame(
        {
            "feature_block": feature_block,
            "method": "kmeans",
            "selected_k": int(selected_k),
            "source_row_index": sample_index.astype(int),
            "cluster_label": label_sample.astype(int),
            "cluster_name": [f"C{int(label)}" for label in label_sample],
            "pc1": projection[:, 0],
            "pc2": projection[:, 1],
        }
    )

    for identifier in ["measurement_id", "session_id", "timestamp"]:
        if identifier in source.columns:
            output[identifier] = source[identifier].to_numpy()

    if "service_degraded" in source.columns:
        degraded = pd.to_numeric(source["service_degraded"], errors="coerce")
        output["service_degraded"] = degraded.to_numpy()
        output["degradation_status"] = np.where(
            degraded.fillna(0.0).ge(0.5),
            "Degraded",
            "Not degraded",
        )
    else:
        output["service_degraded"] = np.nan
        output["degradation_status"] = "Unavailable"

    return output


def run_clustering_analysis(
    clustering_table_path: str | Path,
    results_dir: str | Path,
    figures_dir: str | Path | None = None,
) -> dict[str, Path]:
    _ = figures_dir
    results_path = ensure_directory(results_dir)

    dataframe = read_csv(clustering_table_path, low_memory=False)
    x_values, numeric_columns, categorical_columns = _prepare_matrix(
        dataframe, PRIMARY_FEATURE_BLOCK
    )
    x_eval, eval_index = _sample_array(x_values, SILHOUETTE_SAMPLE_LIMIT)

    (
        primary_validation,
        kmeans_models,
        hierarchical_labels_by_k,
        _gmm_models,
        gmm_labels_by_k,
    ) = _run_primary_validation(dataframe, x_values, x_eval, eval_index)

    (
        density_validation,
        dbscan_sweep,
        optics_sweep,
        density_status,
        dbscan_profile,
        optics_profile,
    ) = _run_density_clustering(dataframe, x_values)

    validation = pd.concat([primary_validation, density_validation], ignore_index=True)
    validation_path = write_csv(validation, results_path / "clustering_validation.csv")

    best_kmeans_k = _best_k(validation, "kmeans", PRIMARY_FEATURE_BLOCK)
    best_hierarchical_k = _best_k(validation, "hierarchical", PRIMARY_FEATURE_BLOCK)
    best_gmm_silhouette_k = _best_k(
        validation, "gaussian_mixture", PRIMARY_FEATURE_BLOCK
    )
    best_gmm_bic_k = _best_gmm_bic_k(validation, PRIMARY_FEATURE_BLOCK)

    kmeans_labels = kmeans_models[best_kmeans_k].predict(x_values)
    kmeans_labels = _remap_labels_by_profile_rate(dataframe, kmeans_labels)

    x_hierarchical, hierarchical_index = _sample_array(
        x_values, HIERARCHICAL_SAMPLE_LIMIT, random_state=MODEL_RANDOM_STATE + 1
    )
    hierarchical_profile_source = dataframe.iloc[hierarchical_index].reset_index(
        drop=True
    )
    hierarchical_labels = hierarchical_labels_by_k[best_hierarchical_k]
    hierarchical_labels = _remap_labels_by_profile_rate(
        hierarchical_profile_source, hierarchical_labels
    )

    gmm_labels = gmm_labels_by_k[best_gmm_silhouette_k]
    gmm_labels = _remap_labels_by_profile_rate(dataframe, gmm_labels)

    kmeans_profile = _profile_clusters(
        dataframe, kmeans_labels, "kmeans", PRIMARY_FEATURE_BLOCK
    )
    hierarchical_profile = _profile_clusters(
        hierarchical_profile_source,
        hierarchical_labels,
        "hierarchical",
        PRIMARY_FEATURE_BLOCK,
    )
    gmm_profile = _profile_clusters(
        dataframe, gmm_labels, "gaussian_mixture", PRIMARY_FEATURE_BLOCK
    )

    profile_frames = [kmeans_profile, hierarchical_profile, gmm_profile]
    if not dbscan_profile.empty:
        profile_frames.append(dbscan_profile)
    if not optics_profile.empty:
        profile_frames.append(optics_profile)
    all_profiles = pd.concat(profile_frames, ignore_index=True)

    feature_block_validation, feature_block_profiles = _run_feature_block_sensitivity(
        dataframe
    )
    kmeans_stability, kmeans_pairwise_stability, kmeans_stability_summary = (
        _run_kmeans_stability(dataframe, best_kmeans_k)
    )
    profile_intervals = _bootstrap_cluster_profile_intervals(dataframe, kmeans_labels)
    effect_sizes = _cluster_effect_sizes(dataframe, kmeans_labels)
    null_baseline = _cluster_null_baseline(dataframe, kmeans_labels)
    projection_table = _cluster_projection_table(
        dataframe,
        x_values,
        kmeans_labels,
        selected_k=best_kmeans_k,
        feature_block=PRIMARY_FEATURE_BLOCK,
    )
    session_validation, session_profiles = _run_session_level_clustering(dataframe)

    selection_rows: list[dict[str, Any]] = [
        {
            "feature_block": PRIMARY_FEATURE_BLOCK,
            "method": "kmeans",
            "selected_k": best_kmeans_k,
            "selected_eps": np.nan,
            "selected_min_samples": np.nan,
            "selection_metric": (
                "highest silhouette score on reduced context-only clustering inputs"
            ),
            "reporting_role": "primary",
            "status": "selected",
        },
        {
            "feature_block": PRIMARY_FEATURE_BLOCK,
            "method": "hierarchical",
            "selected_k": best_hierarchical_k,
            "selected_eps": np.nan,
            "selected_min_samples": np.nan,
            "selection_metric": "highest silhouette score on bounded sample",
            "reporting_role": "sensitivity",
            "status": "selected",
        },
        {
            "feature_block": PRIMARY_FEATURE_BLOCK,
            "method": "gaussian_mixture",
            "selected_k": best_gmm_silhouette_k,
            "selected_eps": np.nan,
            "selected_min_samples": np.nan,
            "selection_metric": "highest silhouette score on bounded sample",
            "reporting_role": "sensitivity",
            "status": "selected",
        },
        {
            "feature_block": PRIMARY_FEATURE_BLOCK,
            "method": "gaussian_mixture_bic_reference",
            "selected_k": best_gmm_bic_k,
            "selected_eps": np.nan,
            "selected_min_samples": np.nan,
            "selection_metric": "lowest BIC on bounded sample",
            "reporting_role": "diagnostic",
            "status": "reported_for_model_based_clustering_reference",
        },
    ]
    for _, row in density_status.iterrows():
        selection_rows.append(
            {
                "feature_block": PRIMARY_FEATURE_BLOCK,
                "method": row["method"],
                "selected_k": np.nan,
                "selected_eps": row["selected_eps"],
                "selected_min_samples": row["selected_min_samples"],
                "selection_metric": row["reason"],
                "reporting_role": "diagnostic",
                "status": row["status"],
            }
        )
    selection = pd.DataFrame(selection_rows)

    outputs: dict[str, Path] = {
        "clustering_validation": validation_path,
        "clustering_feature_set": write_csv(
            _feature_set_table(
                dataframe, PRIMARY_FEATURE_BLOCK, numeric_columns, categorical_columns
            ),
            results_path / "clustering_feature_set.csv",
        ),
        "clustering_model_selection": write_csv(
            selection, results_path / "clustering_model_selection.csv"
        ),
        "kmeans_cluster_profiles": write_csv(
            kmeans_profile, results_path / "kmeans_cluster_profiles.csv"
        ),
        "kmeans_cluster_summary": write_csv(
            _make_kmeans_summary(kmeans_profile),
            results_path / "kmeans_cluster_summary.csv",
        ),
        "hierarchical_cluster_profiles": write_csv(
            hierarchical_profile, results_path / "hierarchical_cluster_profiles.csv"
        ),
        "gaussian_mixture_cluster_profiles": write_csv(
            gmm_profile, results_path / "gaussian_mixture_cluster_profiles.csv"
        ),
        "dbscan_cluster_profiles": write_csv(
            dbscan_profile, results_path / "dbscan_cluster_profiles.csv"
        ),
        "optics_cluster_profiles": write_csv(
            optics_profile, results_path / "optics_cluster_profiles.csv"
        ),
        "cluster_profiles": write_csv(
            all_profiles, results_path / "cluster_profiles.csv"
        ),
        "dbscan_parameter_sweep": write_csv(
            dbscan_sweep, results_path / "dbscan_parameter_sweep.csv"
        ),
        "optics_parameter_sweep": write_csv(
            optics_sweep, results_path / "optics_parameter_sweep.csv"
        ),
        "density_clustering_status": write_csv(
            density_status, results_path / "density_clustering_status.csv"
        ),
        "feature_block_clustering_validation": write_csv(
            feature_block_validation,
            results_path / "feature_block_clustering_validation.csv",
        ),
        "feature_block_cluster_profiles": write_csv(
            feature_block_profiles, results_path / "feature_block_cluster_profiles.csv"
        ),
        "kmeans_stability": write_csv(
            kmeans_stability, results_path / "kmeans_stability.csv"
        ),
        "kmeans_pairwise_stability": write_csv(
            kmeans_pairwise_stability, results_path / "kmeans_pairwise_stability.csv"
        ),
        "kmeans_stability_summary": write_csv(
            kmeans_stability_summary, results_path / "kmeans_stability_summary.csv"
        ),
        "cluster_profile_confidence_intervals": write_csv(
            profile_intervals, results_path / "cluster_profile_confidence_intervals.csv"
        ),
        "cluster_profile_effect_sizes": write_csv(
            effect_sizes, results_path / "cluster_profile_effect_sizes.csv"
        ),
        "cluster_null_baseline": write_csv(
            null_baseline, results_path / "cluster_null_baseline.csv"
        ),
        "cluster_projection_table": write_csv(
            projection_table, results_path / "cluster_projection_table.csv"
        ),
        "session_level_clustering_validation": write_csv(
            session_validation, results_path / "session_level_clustering_validation.csv"
        ),
        "session_level_cluster_profiles": write_csv(
            session_profiles, results_path / "session_level_cluster_profiles.csv"
        ),
    }

    return outputs
