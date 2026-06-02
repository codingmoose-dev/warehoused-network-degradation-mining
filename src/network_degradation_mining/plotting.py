from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
    "signal_strength_unclipped_dbm": "Unclipped Signal Strength (dBm)",
    "signal_strength_dbm": "Signal Strength (dBm)",
    "data_usage_mb": "Data Usage (MB)",
    "link_capacity_downlink_mbps": "Downlink Capacity (Mbps)",
    "link_capacity_upload_mbps": "Uplink Capacity (Mbps)",
    "offered_downlink_mbps": "Offered Downlink (Mbps)",
    "offered_upload_mbps": "Offered Uplink (Mbps)",
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

DECISION_PATH_RULE_FILL = "white"
DECISION_PATH_DEGRADED_FILL = "#D9D9D9"
DECISION_PATH_NOT_DEGRADED_FILL = "#1F1F1F"
DECISION_PATH_EDGE = "#111111"
DECISION_PATH_DARK_TEXT = "0.05"
DECISION_PATH_LIGHT_TEXT = "white"



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



def _wrap_decision_rule_label(text: Any, width: int = 68) -> str:
    wrapped_lines: list[str] = []
    for line in str(text).splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        wrapped_lines.extend(
            textwrap.wrap(clean_line, width=width, break_long_words=False)
        )
    return "\n".join(wrapped_lines)


def _select_representative_decision_rules(
    leaf_rules: pd.DataFrame,
    feature_context: str = "reduced_context",
    criterion: str = "gini",
    per_class: int = 3,
) -> pd.DataFrame:
    required_columns = {
        "feature_context",
        "criterion",
        "predicted_class",
        "degraded_rate_in_leaf",
        "node_sample_count",
    }
    missing_columns = required_columns.difference(leaf_rules.columns)
    if missing_columns:
        raise ValueError(
            "Missing decision-rule columns: "
            + ", ".join(sorted(missing_columns))
        )

    rules = leaf_rules[
        (leaf_rules["feature_context"] == feature_context)
        & (leaf_rules["criterion"] == criterion)
    ].copy()

    if rules.empty:
        raise ValueError(
            f"No rules found for feature_context={feature_context}, "
            f"criterion={criterion}."
        )

    rules["predicted_class"] = rules["predicted_class"].astype(int)
    rules["degraded_rate_in_leaf"] = pd.to_numeric(
        rules["degraded_rate_in_leaf"], errors="coerce"
    ).fillna(0.0)
    rules["node_sample_count"] = pd.to_numeric(
        rules["node_sample_count"], errors="coerce"
    ).fillna(0.0)

    rules["class_label"] = np.where(
        rules["predicted_class"].eq(1),
        "Degraded",
        "Not degraded",
    )
    rules["class_confidence"] = np.where(
        rules["predicted_class"].eq(1),
        rules["degraded_rate_in_leaf"],
        1.0 - rules["degraded_rate_in_leaf"],
    )
    rules["selection_score"] = rules["class_confidence"] * np.log1p(
        rules["node_sample_count"]
    )

    selected_frames: list[pd.DataFrame] = []
    for class_value in [1, 0]:
        class_rules = rules[rules["predicted_class"].eq(class_value)]
        class_rules = class_rules.sort_values(
            ["selection_score", "class_confidence", "node_sample_count"],
            ascending=[False, False, False],
        ).head(per_class)
        selected_frames.append(class_rules)

    selected = pd.concat(selected_frames, ignore_index=True)
    return selected.sort_values(
        ["predicted_class", "selection_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def plot_representative_decision_paths(
    leaf_rules: pd.DataFrame,
    path: str | Path,
    feature_context: str = "reduced_context",
    criterion: str = "gini",
    per_class: int = 3,
) -> Path:
    selected = _select_representative_decision_rules(
        leaf_rules=leaf_rules,
        feature_context=feature_context,
        criterion=criterion,
        per_class=per_class,
    )

    output_path = ensure_figure_parent(path)
    edge_color = "#111111"
    text_color = "#111111"
    box_fill = "white"
    separator_color = "0.35"

    row_count = len(selected)
    box_height = 0.68
    figure_height = max(4.0, 1.22 * row_count + 0.8)
    fig, ax = plt.subplots(figsize=(10.4, figure_height))
    ax.set_xlim(0.0, 0.83)
    ax.set_ylim(0.0, float(row_count))
    ax.axis("off")

    row_positions: list[tuple[int, float, float]] = []

    for row_index, (_, row) in enumerate(selected.iterrows()):
        y_position = row_count - row_index - 0.82
        predicted_class = int(row["predicted_class"])
        state_label = "Degraded" if predicted_class == 1 else "Not degraded"
        row_positions.append((predicted_class, y_position, box_height))

        rule_column = "rule_text_interpretable"
        if rule_column not in row.index or pd.isna(row.get(rule_column)):
            rule_column = "rule_text"

        rule_text = str(row.get(rule_column, "")).replace(" AND ", "\n")
        rule_text = _wrap_decision_rule_label(rule_text, width=46)

        rule_x = 0.03
        rule_y = y_position
        rule_w = 0.51
        rule_h = box_height

        ax.add_patch(
            plt.Rectangle(
                (rule_x, rule_y),
                rule_w,
                rule_h,
                linewidth=1.0,
                edgecolor=edge_color,
                facecolor=box_fill,
                zorder=1,
            )
        )

        ax.text(
            rule_x + 0.02,
            rule_y + rule_h / 2.0,
            rule_text,
            ha="left",
            va="center",
            fontsize=9.0,
            color=text_color,
            linespacing=1.18,
            zorder=2,
        )

        ax.annotate(
            "",
            xy=(0.62, y_position + rule_h / 2.0),
            xytext=(0.55, y_position + rule_h / 2.0),
            arrowprops={
                "arrowstyle": "-|>",
                "linewidth": 1.0,
                "color": edge_color,
                "mutation_scale": 12,
            },
        )

        leaf_x = 0.63
        leaf_y = y_position
        leaf_w = 0.17
        leaf_h = box_height

        ax.add_patch(
            plt.Rectangle(
                (leaf_x, leaf_y),
                leaf_w,
                leaf_h,
                linewidth=1.0,
                edgecolor=edge_color,
                facecolor=box_fill,
                zorder=1,
            )
        )

        class_confidence = float(row["class_confidence"])
        leaf_samples = int(float(row["node_sample_count"]))
        leaf_label = (
            f"{state_label}\n"
            f"Confidence = {class_confidence:.3f}\n"
            f"Leaf n = {leaf_samples:,}"
        )

        ax.text(
            leaf_x + leaf_w / 2.0,
            leaf_y + leaf_h / 2.0,
            leaf_label,
            ha="center",
            va="center",
            fontsize=9.3,
            color=text_color,
            linespacing=1.22,
            zorder=3,
        )

    for index in range(1, len(row_positions)):
        previous_class, previous_y, _ = row_positions[index - 1]
        current_class, current_y, current_h = row_positions[index]
        if previous_class != current_class:
            separator_y = (previous_y + current_y + current_h) / 2.0
            ax.hlines(
                y=separator_y,
                xmin=0.03,
                xmax=0.80,
                colors=separator_color,
                linewidth=0.9,
                linestyles=(0, (4, 4)),
                zorder=0,
            )
            break

    output_path = save_figure(fig, output_path)
    plt.close(fig)
    return output_path

def plot_pruned_decision_tree(
    model: Any,
    feature_names: list[str],
    path: str | Path,
) -> Path:
    from sklearn.tree import plot_tree as sklearn_plot_tree

    figure_width = max(12.0, min(26.0, 2.4 * max(model.get_depth(), 4)))
    fig, ax = plt.subplots(figsize=(figure_width, 9.0))
    sklearn_plot_tree(
        model,
        feature_names=feature_names,
        class_names=["not_degraded", "degraded"],
        filled=False,
        impurity=True,
        proportion=True,
        rounded=False,
        max_depth=3,
        fontsize=7,
        ax=ax,
    )
    ax.set_axis_off()
    output_path = save_figure(fig, path)
    plt.close(fig)
    return output_path


def plot_clustering_elbow_curve(validation: pd.DataFrame, path: str | Path) -> Path:
    data = validation[
        (validation["method"] == "kmeans") & validation["inertia"].notna()
    ].copy()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    if not data.empty:
        ax.plot(data["k"], data["inertia"], marker="o")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    output_path = save_figure(fig, path)
    plt.close(fig)
    return output_path


def plot_clustering_silhouette_scores(
    validation: pd.DataFrame, path: str | Path
) -> Path:
    data = validation[validation["silhouette_score"].notna()].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for method, group in data.groupby("method"):
        if group["k"].notna().any():
            ordered = group.sort_values("k")
            ax.plot(ordered["k"], ordered["silhouette_score"], marker="o", label=method)
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette score")
    if not data.empty:
        ax.legend(frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    output_path = save_figure(fig, path)
    plt.close(fig)
    return output_path


def plot_cluster_profiles(profile: pd.DataFrame, path: str | Path) -> Path:
    required = {"method", "cluster_label", "row_count", "service_degraded_mean"}
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    if required.issubset(profile.columns):
        data = profile[profile["method"] == "kmeans"].copy()
        data = data.sort_values("service_degraded_mean", ascending=True)
        labels = [f"C{int(value)}" for value in data["cluster_label"]]
        ax.barh(labels, data["service_degraded_mean"])
        ax.set_xlabel("Mean degradation label")
        ax.set_ylabel("KMeans cluster")
        for index, row in enumerate(data.to_dict("records")):
            ax.text(
                float(row["service_degraded_mean"]),
                index,
                f" n={int(row['row_count'])}",
                va="center",
                fontsize=8,
            )

    ax.grid(True, axis="x", linewidth=0.4, alpha=0.5)
    output_path = save_figure(fig, path)
    plt.close(fig)
    return output_path


def plot_top_association_rules(
    degradation_rules: pd.DataFrame,
    path: str | Path,
    top_n: int = 20,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.0, 7.0))

    if not degradation_rules.empty:
        data = degradation_rules.sort_values(
            ["lift", "confidence", "support"],
            ascending=[False, False, False],
        ).head(top_n)
        data = data.sort_values("lift", ascending=True)

        labels = [
            text.replace(";", "\n")
            for text in data["antecedent_text"].astype(str).tolist()
        ]
        ax.barh(labels, data["lift"])
        ax.set_xlabel("Lift")
        ax.set_ylabel("Antecedent itemset")

    ax.grid(True, axis="x", linewidth=0.4, alpha=0.5)
    output_path = save_figure(fig, path)
    plt.close(fig)
    return output_path


def plot_external_metric_distribution(
    plot_frame: pd.DataFrame,
    x_label: str,
    path: str | Path,
) -> Path | None:
    required = {"source_dataset", "value"}
    if not required.issubset(plot_frame.columns) or plot_frame.empty:
        return None

    working = plot_frame.copy()
    working["value"] = pd.to_numeric(working["value"], errors="coerce")
    working = working.dropna(subset=["source_dataset", "value"])
    if working["source_dataset"].nunique(dropna=True) < 2:
        return None

    ordered_sources = (
        working.groupby("source_dataset")["value"]
        .median()
        .sort_values()
        .index.tolist()
    )
    data = [
        working.loc[working["source_dataset"] == source, "value"].to_numpy()
        for source in ordered_sources
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    try:
        ax.boxplot(data, tick_labels=ordered_sources, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=ordered_sources, showfliers=False)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(x_label)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    output_path = save_figure(fig, path)
    plt.close(fig)
    return output_path

