from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from network_degradation_mining.io import ensure_directory, read_csv, write_csv
from network_degradation_mining.plotting import (
    plot_external_metric_distribution,
)

REFERENCE_FILES = {
    "vienna_4g5g": "vienna_reference_clean.csv",
    "campus_qos": "campus_qos_reference_clean.csv",
    "ucc_5g_context": "ucc_5g_reference_clean.csv",
    "ns3_lena": "simulator_reference_clean.csv",
}

REQUIRED_REFERENCE_DATASETS = frozenset(
    {"vienna_4g5g", "campus_qos", "ucc_5g_context", "ns3_lena"}
)

SOURCE_DISPLAY_NAMES = {
    "synnetqos": "SynNetQoS",
    "vienna_4g5g": "Vienna 4G/5G",
    "campus_qos": "Campus QoS",
    "ucc_5g_context": "UCC 5G Context",
    "ns3_lena": "5G-LENA/ns-3",
}

SOURCE_ORDER = {
    "synnetqos": 0,
    "vienna_4g5g": 1,
    "campus_qos": 2,
    "ucc_5g_context": 3,
    "ns3_lena": 4,
}

REFERENCE_COLUMNS = [
    "source_dataset",
    "technology",
    "measurement_source",
    "measurement_context",
    "application_context",
    "mobility_context",
    "rsrp_dbm",
    "download_throughput_mbps",
    "upload_throughput_mbps",
    "latency_ms",
    "jitter_ms",
    "packet_loss_fraction",
    "offered_downlink_mbps",
    "offered_upload_mbps",
]

METRIC_SPECS = {
    "signal_dbm": {
        "metric_label": "Signal / RSRP",
        "synnetqos_column": "signal_strength_dbm",
        "reference_column": "rsrp_dbm",
        "figure_name": "signal_distribution_comparison.pdf",
        "x_label": "Signal / RSRP (dBm)",
        "metric_role": "radio_measurement",
        "claim_scope": "selected radio-strength plausibility comparison",
        "main_text_candidate": True,
    },
    "download_throughput_mbps": {
        "metric_label": "Download throughput",
        "synnetqos_column": "download_speed_mbps",
        "reference_column": "download_throughput_mbps",
        "figure_name": "throughput_distribution_comparison.pdf",
        "x_label": "Download throughput (Mbps)",
        "metric_role": "service_performance",
        "claim_scope": "selected throughput plausibility comparison",
        "main_text_candidate": True,
    },
    "upload_throughput_mbps": {
        "metric_label": "Upload throughput",
        "synnetqos_column": "upload_speed_mbps",
        "reference_column": "upload_throughput_mbps",
        "figure_name": "upload_throughput_distribution_comparison.pdf",
        "x_label": "Upload throughput (Mbps)",
        "metric_role": "service_performance",
        "claim_scope": "selected upload-throughput plausibility comparison",
        "main_text_candidate": False,
    },
    "latency_ms": {
        "metric_label": "Latency",
        "synnetqos_column": "latency_ms",
        "reference_column": "latency_ms",
        "figure_name": "latency_distribution_comparison.pdf",
        "x_label": "Latency (ms)",
        "metric_role": "service_performance",
        "claim_scope": "selected latency plausibility comparison",
        "main_text_candidate": False,
    },
    "jitter_ms": {
        "metric_label": "Jitter",
        "synnetqos_column": "jitter_ms",
        "reference_column": "jitter_ms",
        "figure_name": "jitter_distribution_comparison.pdf",
        "x_label": "Jitter (ms)",
        "metric_role": "service_performance",
        "claim_scope": "selected jitter plausibility comparison",
        "main_text_candidate": False,
    },
    "packet_loss_fraction": {
        "metric_label": "Packet loss fraction",
        "synnetqos_column": None,
        "reference_column": "packet_loss_fraction",
        "figure_name": "packet_loss_distribution_comparison.pdf",
        "x_label": "Packet loss fraction",
        "metric_role": "reference_only_quality_metric",
        "claim_scope": (
            "reference-only availability summary; SynNetQoS does not expose a "
            "direct packet-loss field in this comparison table"
        ),
        "main_text_candidate": False,
    },
    "offered_downlink_mbps": {
        "metric_label": "Offered downlink traffic",
        "synnetqos_column": "offered_downlink_mbps",
        "reference_column": "offered_downlink_mbps",
        "figure_name": "offered_downlink_distribution_comparison.pdf",
        "x_label": "Offered downlink traffic (Mbps)",
        "metric_role": "traffic_context",
        "claim_scope": "selected traffic-demand plausibility comparison",
        "main_text_candidate": False,
    },
}

UCC_DOWNLOAD_CONTEXT_FILTER_NOTE = (
    "UCC throughput comparisons use only application_context=Download. The full "
    "UCC production trace includes streaming sessions with bursty or idle bitrate "
    "periods, so the unfiltered throughput distribution is retained for coverage "
    "audit but not used as the main throughput comparison."
)

METRIC_SOURCE_FILTERS = {
    ("download_throughput_mbps", "ucc_5g_context"): {
        "application_context": {"Download"},
        "comparison_suffix": "Download context",
        "filter_note": UCC_DOWNLOAD_CONTEXT_FILTER_NOTE,
    },
    ("upload_throughput_mbps", "ucc_5g_context"): {
        "application_context": {"Download"},
        "comparison_suffix": "Download context",
        "filter_note": (
            "UCC upload throughput is shown only for Download-context rows to "
            "avoid mixing streaming idle periods; use as supplementary evidence."
        ),
    },
}

DISTRIBUTION_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
MIN_COMPARISON_RECORDS = 20
PAIRWISE_SAMPLE_LIMIT = 10000
MODEL_RANDOM_STATE = 42
PLOT_SAMPLE_LIMIT = 15000


def _source_label(source_dataset: Any) -> str:
    text = str(source_dataset)
    return SOURCE_DISPLAY_NAMES.get(text, text.replace("_", " ").title())


def _source_order(source_dataset: Any) -> int:
    return SOURCE_ORDER.get(str(source_dataset), 99)


def _empty_reference_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REFERENCE_COLUMNS)


def _safe_read_reference(path: Path, columns: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = [column for column in columns if column in header.columns]
    if not available:
        return pd.DataFrame(columns=columns)
    return read_csv(path, usecols=available, low_memory=False)


def _load_reference_frames(
    external_reference_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_dir = Path(external_reference_dir)
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []

    for source_dataset, filename in REFERENCE_FILES.items():
        path = reference_dir / filename
        if not path.exists():
            audit_rows.append(
                {
                    "source_dataset": source_dataset,
                    "file_name": filename,
                    "dataset_role": "required",
                    "status": "missing_required",
                    "row_count": 0,
                    "available_column_count": 0,
                    "missing_standard_columns": ";".join(REFERENCE_COLUMNS),
                    "notes": (
                        "clean reference file not found; required comparison "
                        "source is missing"
                    ),
                }
            )
            continue

        try:
            frame = _safe_read_reference(path, REFERENCE_COLUMNS)
        except Exception as exc:
            audit_rows.append(
                {
                    "source_dataset": source_dataset,
                    "file_name": filename,
                    "dataset_role": "required",
                    "status": "read_error",
                    "row_count": 0,
                    "available_column_count": 0,
                    "missing_standard_columns": ";".join(REFERENCE_COLUMNS),
                    "notes": f"could not read clean reference file: {exc}",
                }
            )
            continue

        if "source_dataset" not in frame.columns:
            frame["source_dataset"] = source_dataset

        available_columns = set(frame.columns)
        for column in REFERENCE_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA

        frame = frame[REFERENCE_COLUMNS].copy()
        frames.append(frame)

        missing_columns = [
            column for column in REFERENCE_COLUMNS if column not in available_columns
        ]
        audit_rows.append(
            {
                "source_dataset": source_dataset,
                "file_name": filename,
                "dataset_role": "required",
                "status": "ok",
                "row_count": int(len(frame)),
                "available_column_count": int(len(available_columns)),
                "missing_standard_columns": ";".join(missing_columns),
                "notes": "clean reference file loaded",
            }
        )

    references = (
        pd.concat([frame.astype("object") for frame in frames], ignore_index=True)
        if frames
        else _empty_reference_frame()
    )
    return references, pd.DataFrame(audit_rows)


def _first_available_series(
    dataframe: pd.DataFrame,
    columns: tuple[str, ...],
    default: Any = pd.NA,
) -> pd.Series:
    for column in columns:
        if column in dataframe.columns:
            return dataframe[column]
    return pd.Series(default, index=dataframe.index)


def _build_synnetqos_reference_frame(synnetqos_core_path: str | Path) -> pd.DataFrame:
    candidate_columns = {
        "network_type",
        "app_type",
        "application_context",
        "movement_speed",
        "mobility_context",
    }
    candidate_columns.update(
        spec["synnetqos_column"]
        for spec in METRIC_SPECS.values()
        if spec["synnetqos_column"]
    )

    header = pd.read_csv(synnetqos_core_path, nrows=0)
    available = [column for column in sorted(candidate_columns) if column in header]

    if not available:
        return _empty_reference_frame()

    raw = read_csv(synnetqos_core_path, usecols=available, low_memory=False)

    output = pd.DataFrame(index=raw.index)
    output["source_dataset"] = "synnetqos"
    output["technology"] = _first_available_series(raw, ("network_type",))
    output["measurement_source"] = "synthetic_generator"
    output["measurement_context"] = "synthetic_session_record"
    output["application_context"] = _first_available_series(
        raw, ("application_context", "app_type")
    )
    output["mobility_context"] = _first_available_series(
        raw, ("mobility_context", "movement_speed")
    )
    output["rsrp_dbm"] = pd.to_numeric(
        raw.get("signal_strength_dbm", pd.Series(pd.NA, index=raw.index)),
        errors="coerce",
    )
    output["download_throughput_mbps"] = pd.to_numeric(
        raw.get("download_speed_mbps", pd.Series(pd.NA, index=raw.index)),
        errors="coerce",
    )
    output["upload_throughput_mbps"] = pd.to_numeric(
        raw.get("upload_speed_mbps", pd.Series(pd.NA, index=raw.index)),
        errors="coerce",
    )
    output["latency_ms"] = pd.to_numeric(
        raw.get("latency_ms", pd.Series(pd.NA, index=raw.index)),
        errors="coerce",
    )
    output["jitter_ms"] = pd.to_numeric(
        raw.get("jitter_ms", pd.Series(pd.NA, index=raw.index)),
        errors="coerce",
    )
    output["packet_loss_fraction"] = pd.NA
    output["offered_downlink_mbps"] = pd.to_numeric(
        raw.get("offered_downlink_mbps", pd.Series(pd.NA, index=raw.index)),
        errors="coerce",
    )
    output["offered_upload_mbps"] = pd.NA

    return output[REFERENCE_COLUMNS].copy()


def _numeric_values(dataframe: pd.DataFrame, column: str) -> pd.Series:
    if column not in dataframe.columns:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(dataframe[column], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def _summarize_distribution(
    values: pd.Series,
    source_dataset: str,
    metric: str,
    technology: Any,
    measurement_context: Any,
    application_context: Any,
    mobility_context: Any,
) -> dict[str, Any]:
    spec = METRIC_SPECS[metric]
    row: dict[str, Any] = {
        "source_dataset": source_dataset,
        "source_label": _source_label(source_dataset),
        "metric": metric,
        "metric_label": spec["metric_label"],
        "metric_role": spec["metric_role"],
        "technology": technology,
        "measurement_context": measurement_context,
        "application_context": application_context,
        "mobility_context": mobility_context,
        "non_missing_count": int(values.notna().sum()),
        "claim_scope": spec["claim_scope"],
    }

    clean = values.dropna()
    if clean.empty:
        row.update(
            {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "p05": np.nan,
                "p25": np.nan,
                "median": np.nan,
                "p75": np.nan,
                "p95": np.nan,
                "max": np.nan,
                "iqr": np.nan,
            }
        )
        return row

    quantiles = clean.quantile(DISTRIBUTION_QUANTILES)
    row.update(
        {
            "mean": float(clean.mean()),
            "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
            "min": float(clean.min()),
            "p05": float(quantiles.loc[0.05]),
            "p25": float(quantiles.loc[0.25]),
            "median": float(quantiles.loc[0.5]),
            "p75": float(quantiles.loc[0.75]),
            "p95": float(quantiles.loc[0.95]),
            "max": float(clean.max()),
            "iqr": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
        }
    )
    return row


def _distribution_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_dataset",
        "source_label",
        "metric",
        "metric_label",
        "metric_role",
        "technology",
        "measurement_context",
        "application_context",
        "mobility_context",
        "non_missing_count",
        "mean",
        "std",
        "min",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "max",
        "iqr",
        "claim_scope",
    ]
    rows: list[dict[str, Any]] = []
    if dataframe.empty:
        return pd.DataFrame(columns=columns)

    group_columns = [
        "source_dataset",
        "technology",
        "measurement_context",
        "application_context",
        "mobility_context",
    ]
    for group_key, group in dataframe.groupby(group_columns, dropna=False):
        (
            source_dataset,
            technology,
            measurement_context,
            application_context,
            mobility_context,
        ) = group_key
        for metric, spec in METRIC_SPECS.items():
            values = _numeric_values(group, spec["reference_column"])
            rows.append(
                _summarize_distribution(
                    values=values,
                    source_dataset=str(source_dataset),
                    metric=metric,
                    technology=technology,
                    measurement_context=measurement_context,
                    application_context=application_context,
                    mobility_context=mobility_context,
                )
            )

    output = pd.DataFrame(rows, columns=columns)
    if output.empty:
        return output

    output["source_order"] = output["source_dataset"].map(SOURCE_ORDER).fillna(99)
    output = output.sort_values(
        [
            "metric",
            "source_order",
            "technology",
            "measurement_context",
            "application_context",
            "mobility_context",
        ]
    )
    return output.drop(columns=["source_order"])


def _source_role(source_dataset: str) -> str:
    if source_dataset == "synnetqos":
        return "main_dataset"
    if source_dataset in REQUIRED_REFERENCE_DATASETS:
        return "required_reference"
    return "reference"


def _apply_metric_source_filter(
    dataframe: pd.DataFrame,
    metric: str,
    source_dataset: str,
) -> tuple[pd.DataFrame, str, str, str]:
    spec = METRIC_SOURCE_FILTERS.get((metric, source_dataset))
    if spec is None:
        return dataframe, _source_label(source_dataset), "unfiltered", ""

    filtered = dataframe.copy()
    application_values = spec.get("application_context")
    if application_values is not None:
        allowed = {str(value) for value in application_values}
        source_values = filtered["application_context"].astype("string")
        filtered = filtered[source_values.isin(allowed)]

    suffix = str(spec["comparison_suffix"])
    label = f"{_source_label(source_dataset)} ({suffix})"
    return filtered, label, suffix, str(spec["filter_note"])


def _comparison_long_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "metric",
        "metric_label",
        "metric_role",
        "source_dataset",
        "source_label",
        "source_role",
        "comparison_context",
        "filter_note",
        "value",
        "claim_scope",
    ]
    rows: list[pd.DataFrame] = []

    if dataframe.empty:
        return pd.DataFrame(columns=columns)

    for metric, spec in METRIC_SPECS.items():
        value_column = spec["reference_column"]

        for source_dataset, group in dataframe.groupby("source_dataset", dropna=False):
            source_dataset = str(source_dataset)
            filtered, source_label, context, filter_note = _apply_metric_source_filter(
                dataframe=group,
                metric=metric,
                source_dataset=source_dataset,
            )
            values = _numeric_values(filtered, value_column)
            if values.empty:
                continue

            rows.append(
                pd.DataFrame(
                    {
                        "metric": metric,
                        "metric_label": spec["metric_label"],
                        "metric_role": spec["metric_role"],
                        "source_dataset": source_dataset,
                        "source_label": source_label,
                        "source_role": _source_role(source_dataset),
                        "comparison_context": context,
                        "filter_note": filter_note,
                        "value": values.to_numpy(),
                        "claim_scope": spec["claim_scope"],
                    }
                )
            )

    if not rows:
        return pd.DataFrame(columns=columns)

    output = pd.concat(rows, ignore_index=True)
    return output[columns].copy()


def _sample_values(values: pd.Series, limit: int, random_state: int) -> np.ndarray:
    clean = values.dropna().astype(float)
    if len(clean) <= limit:
        return clean.to_numpy()
    return clean.sample(n=limit, random_state=random_state).to_numpy()


def _median_ratio(syn_median: float, reference_median: float) -> float:
    if pd.isna(reference_median) or reference_median == 0:
        return np.nan
    return float(syn_median / reference_median)


def _recommended_pairwise_use(metric: str, source_dataset: str) -> str:
    spec = METRIC_SPECS[metric]
    if source_dataset == "ns3_lena":
        return "controlled_reference_only"
    if spec["main_text_candidate"]:
        return "main_text_candidate"
    return "supplementary_or_table"


def _pairwise_distances(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "metric",
        "metric_label",
        "reference_dataset",
        "reference_label",
        "comparison_context",
        "filter_note",
        "synnetqos_sample_count",
        "reference_sample_count",
        "synnetqos_median",
        "reference_median",
        "median_difference",
        "absolute_median_difference",
        "median_ratio",
        "synnetqos_iqr",
        "reference_iqr",
        "ks_statistic",
        "ks_p_value",
        "wasserstein_distance",
        "recommended_use",
        "comparison_scope",
        "interpretation_note",
    ]
    rows: list[dict[str, Any]] = []
    if comparison.empty:
        return pd.DataFrame(columns=columns)

    for metric, metric_frame in comparison.groupby("metric", dropna=False):
        syn_values = pd.to_numeric(
            metric_frame.loc[
                metric_frame["source_dataset"].eq("synnetqos"), "value"
            ],
            errors="coerce",
        ).dropna()
        if len(syn_values) < MIN_COMPARISON_RECORDS:
            continue

        syn_sample = _sample_values(
            syn_values, PAIRWISE_SAMPLE_LIMIT, MODEL_RANDOM_STATE
        )
        syn_q25, syn_q75 = np.quantile(syn_sample, [0.25, 0.75])
        syn_median = float(np.median(syn_sample))
        reference_frame = metric_frame[
            ~metric_frame["source_dataset"].eq("synnetqos")
        ].copy()

        group_columns = [
            "source_dataset",
            "source_label",
            "comparison_context",
            "filter_note",
        ]
        for group_key, group in reference_frame.groupby(group_columns, dropna=False):
            source_dataset, source_label, comparison_context, filter_note = group_key
            ref_values = pd.to_numeric(group["value"], errors="coerce").dropna()
            if len(ref_values) < MIN_COMPARISON_RECORDS:
                continue

            ref_sample = _sample_values(
                ref_values, PAIRWISE_SAMPLE_LIMIT, MODEL_RANDOM_STATE + 1
            )
            ref_q25, ref_q75 = np.quantile(ref_sample, [0.25, 0.75])
            ref_median = float(np.median(ref_sample))
            ks = ks_2samp(
                syn_sample,
                ref_sample,
                alternative="two-sided",
                mode="auto",
            )

            metric_text = str(metric)
            source_text = str(source_dataset)
            rows.append(
                {
                    "metric": metric_text,
                    "metric_label": METRIC_SPECS[metric_text]["metric_label"],
                    "reference_dataset": source_text,
                    "reference_label": source_label,
                    "comparison_context": comparison_context,
                    "filter_note": filter_note,
                    "synnetqos_sample_count": int(len(syn_sample)),
                    "reference_sample_count": int(len(ref_sample)),
                    "synnetqos_median": syn_median,
                    "reference_median": ref_median,
                    "median_difference": float(syn_median - ref_median),
                    "absolute_median_difference": float(
                        abs(syn_median - ref_median)
                    ),
                    "median_ratio": _median_ratio(syn_median, ref_median),
                    "synnetqos_iqr": float(syn_q75 - syn_q25),
                    "reference_iqr": float(ref_q75 - ref_q25),
                    "ks_statistic": float(ks.statistic),
                    "ks_p_value": float(ks.pvalue),
                    "wasserstein_distance": float(
                        wasserstein_distance(syn_sample, ref_sample)
                    ),
                    "recommended_use": _recommended_pairwise_use(
                        metric_text, source_text
                    ),
                    "comparison_scope": (
                        "selected-variable plausibility comparison; "
                        "not field validation"
                    ),
                    "interpretation_note": (
                        "KS p-values are descriptive because the datasets differ "
                        "in collection design and sample size; "
                        "emphasize medians, IQRs, and distance measures."
                    ),
                }
            )

    output = pd.DataFrame(rows, columns=columns)
    if output.empty:
        return output

    output["source_order"] = output["reference_dataset"].map(SOURCE_ORDER).fillna(99)
    output = output.sort_values(["metric", "source_order", "reference_label"])
    return output.drop(columns=["source_order"])


def _coverage_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if dataframe.empty:
        base_columns = ["source_dataset", "source_label", "row_count"]
        metric_columns = [
            item
            for metric in METRIC_SPECS
            for item in (
                f"{metric}_non_missing_count",
                f"{metric}_non_missing_fraction",
            )
        ]
        return pd.DataFrame(columns=[*base_columns, *metric_columns])

    for source_dataset, group in dataframe.groupby("source_dataset", dropna=False):
        source_dataset = str(source_dataset)
        row: dict[str, Any] = {
            "source_dataset": source_dataset,
            "source_label": _source_label(source_dataset),
            "row_count": int(len(group)),
        }
        for metric, spec in METRIC_SPECS.items():
            values = _numeric_values(group, spec["reference_column"])
            row[f"{metric}_non_missing_count"] = int(len(values))
            row[f"{metric}_non_missing_fraction"] = (
                float(len(values) / len(group)) if len(group) else 0.0
            )
        rows.append(row)

    output = pd.DataFrame(rows)
    output["source_order"] = output["source_dataset"].map(SOURCE_ORDER).fillna(99)
    return output.sort_values("source_order").drop(columns=["source_order"])


def _context_coverage_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_dataset",
        "source_label",
        "measurement_context",
        "application_context",
        "mobility_context",
        "row_count",
        "download_throughput_non_missing_count",
        "upload_throughput_non_missing_count",
        "latency_non_missing_count",
        "jitter_non_missing_count",
    ]
    if dataframe.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    group_columns = [
        "source_dataset",
        "measurement_context",
        "application_context",
        "mobility_context",
    ]
    for group_key, group in dataframe.groupby(group_columns, dropna=False):
        source_dataset, measurement_context, application_context, mobility_context = (
            group_key
        )
        rows.append(
            {
                "source_dataset": source_dataset,
                "source_label": _source_label(source_dataset),
                "measurement_context": measurement_context,
                "application_context": application_context,
                "mobility_context": mobility_context,
                "row_count": int(len(group)),
                "download_throughput_non_missing_count": int(
                    _numeric_values(group, "download_throughput_mbps").shape[0]
                ),
                "upload_throughput_non_missing_count": int(
                    _numeric_values(group, "upload_throughput_mbps").shape[0]
                ),
                "latency_non_missing_count": int(
                    _numeric_values(group, "latency_ms").shape[0]
                ),
                "jitter_non_missing_count": int(
                    _numeric_values(group, "jitter_ms").shape[0]
                ),
            }
        )

    output = pd.DataFrame(rows, columns=columns)
    output["source_order"] = output["source_dataset"].map(SOURCE_ORDER).fillna(99)
    return output.sort_values(
        [
            "source_order",
            "measurement_context",
            "application_context",
            "mobility_context",
        ]
    ).drop(columns=["source_order"])


def _metric_readiness_table(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "metric",
        "metric_label",
        "source_dataset",
        "source_label",
        "source_role",
        "comparison_context",
        "non_missing_count",
        "comparison_status",
        "recommended_use",
        "claim_scope",
        "filter_note",
    ]
    rows: list[dict[str, Any]] = []

    for metric, spec in METRIC_SPECS.items():
        metric_frame = comparison[comparison["metric"].eq(metric)]
        syn_count = int(
            metric_frame.loc[
                metric_frame["source_dataset"].eq("synnetqos"), "value"
            ].shape[0]
        )
        syn_ready = syn_count >= MIN_COMPARISON_RECORDS

        source_labels = {
            "synnetqos": "SynNetQoS",
            **SOURCE_DISPLAY_NAMES,
        }
        observed_sources = metric_frame[
            ["source_dataset", "source_label", "comparison_context", "filter_note"]
        ].drop_duplicates()

        observed_source_names = set(observed_sources["source_dataset"].astype(str))
        missing_sources = [
            source for source in source_labels if source not in observed_source_names
        ]
        if missing_sources:
            missing_source_rows = pd.DataFrame(
                [
                    {
                        "source_dataset": source,
                        "source_label": source_labels[source],
                        "comparison_context": "unavailable",
                        "filter_note": "",
                    }
                    for source in missing_sources
                ]
            )
            observed_sources = pd.concat(
                [observed_sources, missing_source_rows],
                ignore_index=True,
            )

        for record in observed_sources.to_dict("records"):
            source_dataset = str(record["source_dataset"])
            source_frame = metric_frame[
                metric_frame["source_dataset"].eq(source_dataset)
                & metric_frame["source_label"].eq(record["source_label"])
            ]
            count = int(source_frame["value"].shape[0])

            if source_dataset == "synnetqos":
                status = "available" if syn_ready else "insufficient_or_unavailable"
            elif count < MIN_COMPARISON_RECORDS:
                status = "insufficient_or_unavailable"
            elif not syn_ready:
                status = "reference_only"
            else:
                status = "pairwise_comparable"

            if status == "pairwise_comparable":
                recommended_use = _recommended_pairwise_use(metric, source_dataset)
            elif status == "reference_only":
                recommended_use = "coverage_or_limitations_only"
            else:
                recommended_use = "not_reported"

            rows.append(
                {
                    "metric": metric,
                    "metric_label": spec["metric_label"],
                    "source_dataset": source_dataset,
                    "source_label": record["source_label"],
                    "source_role": _source_role(source_dataset),
                    "comparison_context": record["comparison_context"],
                    "non_missing_count": count,
                    "comparison_status": status,
                    "recommended_use": recommended_use,
                    "claim_scope": spec["claim_scope"],
                    "filter_note": record["filter_note"],
                }
            )

    output = pd.DataFrame(rows, columns=columns)
    output["source_order"] = output["source_dataset"].map(SOURCE_ORDER).fillna(99)
    return output.sort_values(["metric", "source_order", "source_label"]).drop(
        columns=["source_order"]
    )


def _external_summary(pairwise_distances: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "metric",
        "metric_label",
        "reference_dataset",
        "reference_label",
        "comparison_context",
        "synnetqos_n",
        "reference_n",
        "synnetqos_median",
        "reference_median",
        "median_difference",
        "synnetqos_iqr",
        "reference_iqr",
        "wasserstein_distance",
        "ks_statistic",
        "recommended_use",
        "filter_note",
    ]

    if pairwise_distances.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for row in pairwise_distances.to_dict("records"):
        rows.append(
            {
                "metric": row["metric"],
                "metric_label": row["metric_label"],
                "reference_dataset": row["reference_dataset"],
                "reference_label": row["reference_label"],
                "comparison_context": row["comparison_context"],
                "synnetqos_n": row["synnetqos_sample_count"],
                "reference_n": row["reference_sample_count"],
                "synnetqos_median": row["synnetqos_median"],
                "reference_median": row["reference_median"],
                "median_difference": row["median_difference"],
                "synnetqos_iqr": row["synnetqos_iqr"],
                "reference_iqr": row["reference_iqr"],
                "wasserstein_distance": row["wasserstein_distance"],
                "ks_statistic": row["ks_statistic"],
                "recommended_use": row["recommended_use"],
                "filter_note": row["filter_note"],
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _context_filter_audit(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric": metric,
            "source_dataset": source_dataset,
            "source_label": records["source_label"].iloc[0],
            "comparison_context": records["comparison_context"].iloc[0],
            "row_count_after_filter": int(len(records)),
            "filter_note": records["filter_note"].iloc[0],
        }
        for (metric, source_dataset), records in comparison.groupby(
            ["metric", "source_dataset"], dropna=False
        )
        if str(records["filter_note"].iloc[0]).strip()
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "metric",
            "source_dataset",
            "source_label",
            "comparison_context",
            "row_count_after_filter",
            "filter_note",
        ],
    )


def _comparison_audit(
    load_audit: pd.DataFrame,
    combined: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    loaded_sources = (
        set(combined["source_dataset"].astype(str)) if not combined.empty else set()
    )
    missing_required = sorted(REQUIRED_REFERENCE_DATASETS - loaded_sources)
    ucc_download_rows = comparison[
        comparison["metric"].eq("download_throughput_mbps")
        & comparison["source_dataset"].eq("ucc_5g_context")
    ]

    rows = [
        {
            "check_name": "comparison_claim_scope",
            "status": "ok",
            "notes": (
                "External references are used for selected-variable plausibility "
                "comparison only; datasets are not merged into the main mining table."
            ),
        },
        {
            "check_name": "main_dataset_source",
            "status": "ok" if "synnetqos" in loaded_sources else "warning",
            "notes": "SynNetQoS remains the main warehouse and mining dataset.",
        },
        {
            "check_name": "required_reference_sources",
            "status": "ok" if not missing_required else "warning",
            "notes": (
                "All required reference sources were available."
                if not missing_required
                else "Missing required clean references: " + ";".join(missing_required)
            ),
        },
        {
            "check_name": "ucc_download_context_filter",
            "status": (
                "ok"
                if len(ucc_download_rows) >= MIN_COMPARISON_RECORDS
                else "warning"
            ),
            "notes": UCC_DOWNLOAD_CONTEXT_FILTER_NOTE,
        },
        {
            "check_name": "simulator_reference_source",
            "status": "ok" if "ns3_lena" in loaded_sources else "warning",
            "notes": (
                "The 5G-LENA/ns-3 simulator reference is a required controlled "
                "reference source for this comparison layer."
            ),
        },
        {
            "check_name": "statistical_test_scope",
            "status": "ok",
            "notes": (
                "KS statistics and p-values are descriptive because the datasets "
                "have different collection designs and sample sizes."
            ),
        },
        {
            "check_name": "reference_load_audit_rows",
            "status": "ok" if not load_audit.empty else "warning",
            "notes": f"Reference load audit row count: {len(load_audit)}.",
        },
    ]
    return pd.DataFrame(rows)


def run_external_reference_comparison(
    synnetqos_core_path: str | Path,
    external_reference_dir: str | Path,
    results_dir: str | Path,
    figures_dir: str | Path,
) -> dict[str, Path]:
    results_path = ensure_directory(results_dir)
    figures_path = ensure_directory(figures_dir)

    synnetqos_frame = _build_synnetqos_reference_frame(synnetqos_core_path)
    reference_frame, load_audit = _load_reference_frames(external_reference_dir)
    frames_to_combine = [
        frame.astype("object")
        for frame in (synnetqos_frame, reference_frame)
        if not frame.empty
    ]
    combined = (
        pd.concat(frames_to_combine, ignore_index=True)
        if frames_to_combine
        else _empty_reference_frame()
    )
    combined = combined[REFERENCE_COLUMNS].copy()

    comparison = _comparison_long_frame(combined)
    summary = _distribution_summary(combined)
    pairwise = _pairwise_distances(comparison)
    coverage = _coverage_table(combined)
    context_coverage = _context_coverage_table(combined)
    readiness = _metric_readiness_table(comparison)
    external_summary = _external_summary(pairwise)
    context_filter_audit = _context_filter_audit(comparison)
    audit_notes = _comparison_audit(load_audit, combined, comparison)

    outputs: dict[str, Path] = {
        "external_distribution_summary": write_csv(
            summary, results_path / "external_distribution_summary.csv"
        ),
        "external_pairwise_distance_summary": write_csv(
            pairwise, results_path / "external_pairwise_distance_summary.csv"
        ),
        "external_metric_readiness": write_csv(
            readiness, results_path / "external_metric_readiness.csv"
        ),
        "external_summary": write_csv(
            external_summary, results_path / "external_summary.csv"
        ),
        "external_reference_coverage": write_csv(
            coverage, results_path / "external_reference_coverage.csv"
        ),
        "external_context_coverage": write_csv(
            context_coverage, results_path / "external_context_coverage.csv"
        ),
        "external_reference_load_audit": write_csv(
            load_audit, results_path / "external_reference_load_audit.csv"
        ),
        "external_comparison_context_audit": write_csv(
            context_filter_audit,
            results_path / "external_comparison_context_audit.csv",
        ),
        "external_comparison_audit": write_csv(
            audit_notes, results_path / "external_comparison_audit.csv"
        ),
    }

    for metric, spec in METRIC_SPECS.items():
        figure_path = plot_external_metric_distribution(
            comparison=comparison,
            metric=metric,
            x_label=spec["x_label"],
            path=figures_path / spec["figure_name"],
            min_records=MIN_COMPARISON_RECORDS,
            sample_limit=PLOT_SAMPLE_LIMIT,
            random_state=MODEL_RANDOM_STATE,
            source_order=SOURCE_ORDER,
        )
        if figure_path is not None:
            outputs[f"{metric}_figure"] = figure_path

    return outputs
