from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

MODEL_COMPARISON_BLUE = "#1f77b4"

FIELD_DISPLAY_NAMES = {
    "network_type": "Network Type",
    "band": "Radio Band",
    "congestion_level": "Congestion Level",
    "tower_load": "Tower Load",
    "app_type": "Application Type",
    "movement_speed": "Mobility State",
    "weather": "Weather",
    "obstruction_level": "Obstruction Level",
    "time_of_day_bin": "Time of Day",
    "area_type": "Area Type",
    "video_quality_label": "Video Quality",
    "service_degraded": "Service Degraded",
}

MODEL_DISPLAY_NAMES = {
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "decision_tree_gini": "Decision Tree (Gini)",
    "decision_tree_entropy": "Decision Tree (Entropy)",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "adaboost": "AdaBoost",
    "hist_gradient_boosting": "Hist Gradient Boosting",
    "xgboost": "XGBoost",
}

MODEL_FAMILY_NAMES = {
    "knn": "Instance-Based",
    "naive_bayes": "Probabilistic",
    "decision_tree_gini": "Decision Tree",
    "decision_tree_entropy": "Decision Tree",
    "logistic_regression": "Linear Classifier",
    "random_forest": "Bagging Ensemble",
    "adaboost": "Boosting Ensemble",
    "hist_gradient_boosting": "Gradient Boosting",
    "xgboost": "Gradient Boosting",
}

SUPERVISED_SPLIT_DESIGN = "Session-grouped 80/20 holdout"
SUPERVISED_PREPROCESSING = (
    "Median imputation for numeric fields; mode imputation for categorical fields; "
    "standard scaling for numeric fields; one-hot encoding for categorical fields"
)

METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}

VALUE_DISPLAY_NAMES = {
    "LTE_Anchor": "LTE Anchor",
    "5G NSA": "5G NSA",
    "5G SA": "5G SA",
    "4G": "4G",
    "5G": "5G",
    "High": "High",
    "Medium": "Medium",
    "Low": "Low",
    "Streaming": "Streaming",
    "Gaming": "Gaming",
    "Browse": "Browsing",
    "Browsing": "Browsing",
    "Driving": "Driving",
    "Walking": "Walking",
    "Static": "Static",
    "morning": "Morning",
    "afternoon": "Afternoon",
    "evening": "Evening",
    "night": "Night",
    "Urban": "Urban",
    "Suburban": "Suburban",
    "Rural": "Rural",
    "<missing>": "Missing",
}


def ensure_figure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def save_figure(fig: Any, path: str | Path) -> Path:
    output_path = ensure_figure_parent(path)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp{output_path.suffix}"
    )
    fig.savefig(temporary_path, bbox_inches="tight")
    temporary_path.replace(output_path)
    return output_path


def display_field_name(name: Any) -> str:
    text = str(name).strip()
    if text in FIELD_DISPLAY_NAMES:
        return FIELD_DISPLAY_NAMES[text]
    return text.replace("_", " ").strip().title()


def display_model_name(name: Any) -> str:
    text = str(name).strip()
    if text in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[text]
    return text.replace("_", " ").strip().title()


def display_model_family(name: Any) -> str:
    text = str(name).strip()
    if text in MODEL_FAMILY_NAMES:
        return MODEL_FAMILY_NAMES[text]
    return "Other"


def display_metric_name(name: Any) -> str:
    text = str(name).strip()
    if text in METRIC_DISPLAY_NAMES:
        return METRIC_DISPLAY_NAMES[text]
    return text.replace("_", " ").strip().upper()


def display_value(value: Any) -> str:
    if pd.isna(value):
        return "Missing"

    text = str(value).strip()
    if not text:
        return "Missing"

    if text in VALUE_DISPLAY_NAMES:
        return VALUE_DISPLAY_NAMES[text]

    lower_text = text.lower()
    if lower_text in VALUE_DISPLAY_NAMES:
        return VALUE_DISPLAY_NAMES[lower_text]

    return text.replace("_", " ").strip().title()


def display_group_item(field_name: Any, value: Any, width: int = 36) -> str:
    label = f"{display_field_name(field_name)}: {display_value(value)}"
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def add_labels_to_rates(
    rate_tables: dict[str, pd.DataFrame],
    top_per_group: int = 5,
    max_rows: int = 24,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for group_name, table in rate_tables.items():
        if table.empty or group_name not in table.columns:
            continue

        working = table.copy()
        working = working.sort_values(
            ["degradation_rate", "row_count"],
            ascending=[False, False],
        ).head(top_per_group)

        for record in working.to_dict("records"):
            rows.append(
                {
                    "group_name": group_name,
                    "group_label": display_field_name(group_name),
                    "group_value": record[group_name],
                    "display_label": display_group_item(group_name, record[group_name]),
                    "row_count": int(record.get("row_count", 0)),
                    "degraded_count": int(record.get("degraded_count", 0)),
                    "degradation_rate": float(record.get("degradation_rate", 0.0)),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "group_name",
                "group_label",
                "group_value",
                "display_label",
                "row_count",
                "degraded_count",
                "degradation_rate",
            ]
        )

    output = pd.DataFrame(rows)
    output = output.sort_values(
        ["degradation_rate", "row_count"],
        ascending=[False, False],
    ).head(max_rows)

    return output.sort_values("degradation_rate", ascending=True).reset_index(drop=True)


def _wilson_score_interval(
    successes: int,
    trials: int,
    z_value: float = 1.959963984540054,
) -> tuple[float, float]:
    # Return the Wilson score interval for a binomial proportion.
    if trials <= 0:
        return 0.0, 0.0

    proportion = successes / trials
    denominator = 1.0 + z_value**2 / trials
    centre = proportion + z_value**2 / (2.0 * trials)
    margin = (
        z_value
        * (
            (proportion * (1.0 - proportion) / trials)
            + (z_value**2 / (4.0 * trials**2))
        )
        ** 0.5
    )

    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return max(0.0, lower), min(1.0, upper)


def _add_wilson_intervals(plot_frame: pd.DataFrame) -> pd.DataFrame:
    output = plot_frame.copy()
    intervals = [
        _wilson_score_interval(
            int(row["degraded_count"]),
            int(row["row_count"]),
        )
        for _, row in output.iterrows()
    ]
    if intervals:
        output["ci_lower"] = [lower for lower, _ in intervals]
        output["ci_upper"] = [upper for _, upper in intervals]
    else:
        output["ci_lower"] = pd.Series(dtype="float64")
        output["ci_upper"] = pd.Series(dtype="float64")
    return output


def _degradation_axis_limits(
    plot_frame: pd.DataFrame,
    overall_rate: float | None,
) -> tuple[float, float]:
    values = [
        float(plot_frame["ci_lower"].min()),
        float(plot_frame["ci_upper"].max()),
    ]
    if overall_rate is not None:
        values.append(float(overall_rate))

    lower = max(0.0, min(values) - 0.03)
    upper = min(1.0, max(values) + 0.03)

    lower = max(0.0, (int(lower / 0.05)) * 0.05)
    upper = min(1.0, ((int(upper / 0.05) + 1) * 0.05))

    if upper - lower < 0.12:
        centre = (upper + lower) / 2.0
        lower = max(0.0, centre - 0.06)
        upper = min(1.0, centre + 0.06)

    return lower, upper


def _format_rate_with_count(rate: float, row_count: int) -> str:
    return f"{rate:.3f} (n = {row_count:,})"


def plot_warehouse_degradation_signature(
    rate_tables: dict[str, pd.DataFrame],
    figure_path: str | Path,
    top_per_group: int = 5,
    max_rows: int = 24,
    overall_rate: float | None = None,
) -> pd.DataFrame:
    # Create a forest plot for warehouse degradation signatures.
    plot_frame = add_labels_to_rates(
        rate_tables=rate_tables,
        top_per_group=top_per_group,
        max_rows=max_rows,
    )
    if plot_frame.empty:
        return plot_frame

    plot_frame = _add_wilson_intervals(plot_frame)
    output_path = ensure_figure_parent(figure_path)

    row_count_max = float(plot_frame["row_count"].max())
    if row_count_max > 0:
        marker_sizes = (plot_frame["row_count"].astype(float) / row_count_max) * 620.0
        marker_sizes = marker_sizes.clip(lower=10.0)
    else:
        marker_sizes = pd.Series(24.0, index=plot_frame.index)

    y_positions = list(range(len(plot_frame)))
    height = max(5.6, 0.38 * len(plot_frame) + 1.9)
    fig, ax = plt.subplots(figsize=(10.8, height))

    x_min, x_max = _degradation_axis_limits(plot_frame, overall_rate)

    for position, (_, row) in enumerate(plot_frame.iterrows()):
        lower = float(row["ci_lower"])
        upper = float(row["ci_upper"])
        ax.hlines(
            y=position,
            xmin=lower,
            xmax=upper,
            color="0.20",
            linewidth=1.0,
            zorder=1,
        )
        ax.vlines(
            [lower, upper],
            ymin=position - 0.09,
            ymax=position + 0.09,
            color="0.20",
            linewidth=1.0,
            zorder=1,
        )

    ax.scatter(
        plot_frame["degradation_rate"].astype(float),
        y_positions,
        s=marker_sizes,
        color=MODEL_COMPARISON_BLUE,
        edgecolors=MODEL_COMPARISON_BLUE,
        linewidths=0.6,
        zorder=2,
    )

    if overall_rate is not None:
        ax.axvline(
            float(overall_rate),
            color="0.45",
            linestyle=(0, (4, 4)),
            linewidth=1.0,
            zorder=0,
        )

    label_x = 1.012
    for position, (_, row) in enumerate(plot_frame.iterrows()):
        ax.text(
            label_x,
            position,
            _format_rate_with_count(
                float(row["degradation_rate"]),
                int(row["row_count"]),
            ),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8.6,
            color="0.10",
        )

    note_lines = [
        "Marker area: subgroup sample size",
        "Whiskers: Wilson 95% confidence intervals",
    ]
    if overall_rate is not None:
        note_lines.append(
            f"Vertical dashed line: overall degradation rate ({overall_rate:.3f})"
        )
    note_lines.append("n: number of records")

    ax.text(
        0.56,
        1.04,
        "\n".join(note_lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.72",
            "linewidth": 0.6,
            "alpha": 0.95,
        },
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_frame["display_label"])
    ax.set_xlabel("Degradation rate")
    ax.set_ylabel("")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.7, len(plot_frame) - 0.3)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.29, right=0.78, bottom=0.08, top=0.86)
    save_figure(fig, output_path)
    plt.close(fig)

    return plot_frame.sort_values("degradation_rate", ascending=False).reset_index(
        drop=True
    )


def _scorecard_metric_columns(dataframe: pd.DataFrame) -> list[str]:
    preferred = ["f1", "balanced_accuracy", "roc_auc", "pr_auc", "recall", "precision"]
    return [column for column in preferred if column in dataframe.columns]


def _ranked_model_metric_columns(dataframe: pd.DataFrame) -> list[str]:
    preferred = [
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "recall",
        "precision",
        "accuracy",
    ]
    return [column for column in preferred if column in dataframe.columns]


def build_ranked_model_performance_table(
    model_comparison: pd.DataFrame,
) -> pd.DataFrame:
    if model_comparison.empty:
        return pd.DataFrame()

    metric_columns = _ranked_model_metric_columns(model_comparison)
    if not metric_columns:
        return pd.DataFrame()

    status = model_comparison.get(
        "training_status", pd.Series("ok", index=model_comparison.index)
    )
    frame = model_comparison.loc[
        status.astype("string").eq("ok"),
        ["model_name", *metric_columns],
    ].copy()
    if frame.empty:
        return frame

    for column in metric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    sort_columns = [
        column for column in ["f1", "balanced_accuracy", "pr_auc"] if column in frame
    ]
    if not sort_columns:
        sort_columns = metric_columns[:1]

    frame = frame.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    frame.insert(0, "overall_rank", range(1, len(frame) + 1))
    frame.insert(1, "model_family", frame["model_name"].map(display_model_family))
    frame.insert(2, "model", frame["model_name"].map(display_model_name))
    frame["split_design"] = SUPERVISED_SPLIT_DESIGN
    frame["preprocessing"] = SUPERVISED_PREPROCESSING

    output_columns = [
        "overall_rank",
        "model_family",
        "model",
        *metric_columns,
        "split_design",
        "preprocessing",
    ]
    return frame[output_columns].reset_index(drop=True)


def build_model_scorecard_frame(model_comparison: pd.DataFrame) -> pd.DataFrame:
    return build_ranked_model_performance_table(model_comparison)


def plot_ranked_model_metric(
    model_comparison: pd.DataFrame,
    figure_path: str | Path,
    metric: str = "f1",
) -> pd.DataFrame:
    if model_comparison.empty or metric not in model_comparison.columns:
        return pd.DataFrame()

    status = model_comparison.get(
        "training_status", pd.Series("ok", index=model_comparison.index)
    )
    plot_frame = model_comparison.loc[
        status.astype("string").eq("ok") & model_comparison[metric].notna()
    ].copy()
    if plot_frame.empty:
        return plot_frame

    plot_frame[metric] = pd.to_numeric(plot_frame[metric], errors="coerce")
    plot_frame = plot_frame.sort_values(metric, ascending=True)
    plot_frame["model_label"] = plot_frame["model_name"].map(display_model_name)

    output_path = ensure_figure_parent(figure_path)
    fig, ax = plt.subplots(figsize=(8.5, max(4.2, 0.45 * len(plot_frame))))
    ax.barh(plot_frame["model_label"], plot_frame[metric])

    for position, value in enumerate(plot_frame[metric].astype(float)):
        ax.text(value + 0.006, position, f"{value:.3f}", va="center", fontsize=8)

    ax.set_xlabel(display_metric_name(metric))
    ax.set_ylabel("")
    ax.set_xlim(0, min(1.08, max(0.1, float(plot_frame[metric].max()) + 0.12)))
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)

    return plot_frame.sort_values(metric, ascending=False).reset_index(drop=True)


INTERACTION_PAIR_LABELS = {
    ("app_type", "congestion_level"): "Application Type × Congestion Level",
    ("band", "congestion_level"): "Radio Band × Congestion Level",
    ("network_type", "movement_speed"): "Network Type × Mobility State",
}


def build_warehouse_interaction_matrix_table(
    crosstabs: pd.DataFrame,
) -> pd.DataFrame:
    # Build a table from warehouse drill-down crosstabs.
    required_columns = {
        "row_group",
        "column_group",
        "row_value",
        "column_value",
        "row_count",
        "degraded_count",
        "degradation_rate",
    }
    if crosstabs.empty or not required_columns.issubset(crosstabs.columns):
        return pd.DataFrame(
            columns=[
                "interaction_name",
                "row_group",
                "column_group",
                "row_label",
                "column_label",
                "row_count",
                "degraded_count",
                "degradation_rate",
            ]
        )

    rows: list[dict[str, Any]] = []
    for record in crosstabs.to_dict("records"):
        row_group = str(record["row_group"])
        column_group = str(record["column_group"])
    interaction_label = (
        f"{display_field_name(row_group)} × {display_field_name(column_group)}"
    )

    rows.append(
        {
            "interaction_name": INTERACTION_PAIR_LABELS.get(
                (row_group, column_group),
                interaction_label,
            ),
            "row_group": row_group,
            "column_group": column_group,
            "row_label": display_value(record["row_value"]),
            "column_label": display_value(record["column_value"]),
            "row_count": int(record.get("row_count", 0)),
            "degraded_count": int(record.get("degraded_count", 0)),
            "degradation_rate": float(record.get("degradation_rate", 0.0)),
        }
    )

    return pd.DataFrame(rows)
