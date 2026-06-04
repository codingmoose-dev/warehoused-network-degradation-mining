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
    "5G_NSA": "5G NSA",
    "5G_SA": "5G SA",
    "5g_nsa": "5G NSA",
    "5g_sa": "5G SA",
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
            "Missing decision-rule columns: " + ", ".join(sorted(missing_columns))
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


def _cluster_plot_frame(
    dataframe: pd.DataFrame,
    feature_block: str | None = None,
) -> pd.DataFrame:
    output = dataframe.copy()
    if feature_block and "feature_block" in output.columns:
        selected = output[output["feature_block"].astype("string").eq(feature_block)]
        if not selected.empty:
            output = selected.copy()
    return output


CLUSTER_METHOD_DISPLAY_NAMES = {
    "kmeans": "K-Means",
    "hierarchical": "Hierarchical",
    "gaussian_mixture": "Gaussian mixture",
}

CLUSTER_METHOD_COLORS = {
    "kmeans": "#1f77b4",
    "hierarchical": "#b24a2f",
    "gaussian_mixture": "#c79a00",
}

CLUSTER_METHOD_MARKERS = {
    "kmeans": "o",
    "hierarchical": "s",
    "gaussian_mixture": "^",
}

CLUSTER_METHOD_LINESTYLES = {
    "kmeans": "-",
    "hierarchical": "--",
    "gaussian_mixture": "-.",
}

CLUSTER_PROJECTION_CLUSTER_COLORS = {
    "C0": "#1f77b4",
    "C1": "#c79a00",
    "C2": "#b24a2f",
    "C3": "#6a3d9a",
    "C4": "#4d9221",
    "C5": "#8c510a",
}

CLUSTER_PROJECTION_STATUS_COLORS = {
    "Not degraded": "#1f77b4",
    "Degraded": "#b24a2f",
    "Unavailable": "0.55",
}

CLUSTER_PROJECTION_CLUSTER_MARKERS = {
    "C0": "o",
    "C1": "^",
    "C2": "s",
    "C3": "D",
    "C4": "P",
    "C5": "X",
}

CLUSTER_PROJECTION_STATUS_MARKERS = {
    "Not degraded": "o",
    "Degraded": "^",
    "Unavailable": "s",
}

CLUSTER_METRIC_DISPLAY_NAMES = {
    "service_degraded": "Degradation rate",
    "signal_strength_dbm": "Signal strength",
    "download_speed_mbps": "Download speed",
    "upload_speed_mbps": "Upload speed",
    "latency_ms": "Latency",
    "jitter_ms": "Jitter",
    "throughput_satisfaction_ratio": "Throughput satisfaction",
    "downlink_shortfall_fraction": "Downlink shortfall",
    "offered_downlink_mbps": "Offered downlink",
    "offered_upload_mbps": "Offered uplink",
    "link_capacity_downlink_mbps": "Downlink capacity",
    "distance_to_tower_km": "Tower distance",
    "distance_2d_m": "2D distance",
    "path_loss_db": "Path loss",
    "contextual_penalty_db": "Context penalty",
    "interval_handover_count": "Handover count",
    "activity_factor": "Activity factor",
    "data_usage_mb": "Data usage",
}

CLUSTER_EFFECT_COLUMNS = [
    "service_degraded",
    "signal_strength_dbm",
    "download_speed_mbps",
    "latency_ms",
    "throughput_satisfaction_ratio",
    "downlink_shortfall_fraction",
    "link_capacity_downlink_mbps",
    "path_loss_db",
    "distance_to_tower_km",
]


def _cluster_metric_label(metric: str) -> str:
    return CLUSTER_METRIC_DISPLAY_NAMES.get(
        metric,
        str(metric)
        .replace("_", " ")
        .replace(" dbm", " dBm")
        .replace(" mbps", " Mbps")
        .title(),
    )


def _selected_k_from_table(
    selection: pd.DataFrame, method: str = "kmeans"
) -> int | None:
    required = {"method", "selected_k"}
    if selection.empty or not required.issubset(selection.columns):
        return None
    candidates = selection[selection["method"].astype("string").eq(method)].copy()
    if "status" in candidates.columns:
        selected = candidates[candidates["status"].astype("string").eq("selected")]
        if not selected.empty:
            candidates = selected
    if candidates.empty:
        return None
    value = pd.to_numeric(candidates.iloc[0]["selected_k"], errors="coerce")
    if pd.isna(value):
        return None
    return int(value)


def _format_cluster_value(metric: str, value: Any) -> str:
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return "—"
    if metric in {
        "service_degraded",
        "throughput_satisfaction_ratio",
        "downlink_shortfall_fraction",
        "activity_factor",
        "row_fraction",
    }:
        return f"{100.0 * float(numeric_value):.1f}%"
    if metric in {"signal_strength_dbm", "path_loss_db", "contextual_penalty_db"}:
        return f"{float(numeric_value):.1f} dB"
    if metric in {
        "download_speed_mbps",
        "upload_speed_mbps",
        "offered_downlink_mbps",
        "offered_upload_mbps",
        "link_capacity_downlink_mbps",
    }:
        return f"{float(numeric_value):.1f} Mbps"
    if metric in {"latency_ms", "jitter_ms"}:
        return f"{float(numeric_value):.2f} ms"
    if metric == "distance_to_tower_km":
        return f"{float(numeric_value):.3f} km"
    if metric == "row_count":
        return f"{int(numeric_value):,}"
    return f"{float(numeric_value):.2f}"


def _profile_mean_column(profile: pd.DataFrame, metric: str) -> str | None:
    mean_column = f"{metric}_mean"
    if mean_column in profile.columns:
        return mean_column
    if metric in profile.columns:
        return metric
    return None


def _standardized_cluster_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    output = matrix.astype(float).copy()
    for column in output.columns:
        values = output[column]
        std = values.std(ddof=0)
        if pd.isna(std) or std == 0:
            output[column] = 0.0
        else:
            output[column] = (values - values.mean()) / std
    return output


def plot_clustering_elbow_curve(
    validation: pd.DataFrame,
    path: str | Path,
    selected_k: int | None = None,
    feature_block: str | None = None,
) -> Path:
    data = _cluster_plot_frame(validation, feature_block)
    data = data[(data["method"] == "kmeans") & data["inertia"].notna()].copy()
    data = data.sort_values("k")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    if not data.empty:
        ax.plot(data["k"], data["inertia"], marker="o", linewidth=1.7)
        if selected_k is not None and selected_k in set(data["k"].astype(int)):
            selected_row = data[data["k"].astype(int).eq(selected_k)].iloc[0]
            ax.scatter(
                [selected_k],
                [selected_row["inertia"]],
                s=80,
                edgecolor="0.1",
                linewidth=0.8,
                zorder=3,
            )
            ax.axvline(selected_k, linewidth=0.8, linestyle=(0, (4, 4)), color="0.35")
            ax.text(
                selected_k,
                float(selected_row["inertia"]),
                "  Selected K",
                va="center",
                fontsize=8.5,
                color="0.15",
            )
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.grid(axis="y", linewidth=0.4, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    output_path = save_figure(fig, path)
    plt.close(fig)
    return output_path


def plot_clustering_silhouette_scores(
    validation: pd.DataFrame,
    path: str | Path,
    selected_k: int | None = None,
    feature_block: str | None = None,
) -> Path:
    data = _cluster_plot_frame(validation, feature_block)
    data = data[data["silhouette_score"].notna()].copy()
    data = data[data["method"].isin(CLUSTER_METHOD_DISPLAY_NAMES)].copy()

    fig, ax = plt.subplots(figsize=(6.6, 4.0))

    for method in ["kmeans", "hierarchical", "gaussian_mixture"]:
        group = data[data["method"].eq(method)].dropna(subset=["k"]).sort_values("k")
        if group.empty:
            continue

        ax.plot(
            group["k"],
            group["silhouette_score"],
            marker=CLUSTER_METHOD_MARKERS[method],
            linestyle=CLUSTER_METHOD_LINESTYLES[method],
            linewidth=1.8,
            markersize=5.8,
            color=CLUSTER_METHOD_COLORS[method],
            label=CLUSTER_METHOD_DISPLAY_NAMES[method],
        )

    if selected_k is not None:
        ax.axvline(
            selected_k,
            linewidth=0.8,
            linestyle=(0, (4, 4)),
            color="0.35",
        )
        if not data.empty:
            ymax = float(data["silhouette_score"].max())
            ax.text(
                selected_k + 0.06,
                ymax - 0.005,
                "Selected K",
                fontsize=8.3,
                color="0.18",
                va="top",
            )

    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette score")
    if not data.empty:
        ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", linewidth=0.4, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_path = save_figure(fig, path)
    plt.close(fig)
    return output_path


def plot_cluster_profile_effect_sizes(
    effect_sizes: pd.DataFrame,
    path: str | Path,
) -> Path:
    data = effect_sizes[effect_sizes["metric"].isin(CLUSTER_EFFECT_COLUMNS)].copy()
    data = data.dropna(subset=["standardized_mean_difference"])
    data["metric_label"] = data["metric"].map(_cluster_metric_label)
    data["absolute_effect"] = data["standardized_mean_difference"].abs()
    data = data.sort_values(
        ["absolute_effect", "standardized_mean_difference"],
        ascending=[True, True],
    )

    fig_height = max(3.4, 0.42 * max(len(data), 1) + 1.0)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    if data.empty:
        ax.axis("off")
        ax.text(
            0.5, 0.5, "No effect-size records", ha="center", va="center", fontsize=9
        )
    else:
        y_positions = np.arange(len(data))
        values = data["standardized_mean_difference"].to_numpy(dtype=float)
        ax.hlines(y_positions, 0, values, linewidth=1.5, color="0.65", zorder=1)
        ax.scatter(
            values, y_positions, s=64, edgecolor="white", linewidth=0.7, zorder=2
        )
        ax.axvline(0, color="0.25", linewidth=0.8)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(data["metric_label"], fontsize=8.5)
        reference = (
            data["reference_cluster"].iloc[0]
            if "reference_cluster" in data.columns
            else np.nan
        )
        comparison = (
            data["comparison_cluster"].iloc[0]
            if "comparison_cluster" in data.columns
            else np.nan
        )
        axis_label = "Standardized mean difference"
        if pd.notna(reference) and pd.notna(comparison):
            axis_label = f"{axis_label} (C{int(comparison)} − C{int(reference)})"
        ax.set_xlabel(axis_label)
        ax.grid(axis="x", linewidth=0.4, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    output_path = save_figure(fig, path)
    plt.close(fig)
    return output_path


def _projection_axis_limits(values: pd.Series) -> tuple[float, float]:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return -1.0, 1.0

    lower = float(numeric_values.quantile(0.005))
    upper = float(numeric_values.quantile(0.995))

    if lower == upper:
        return lower - 1.0, upper + 1.0

    padding = 0.06 * (upper - lower)
    return lower - padding, upper + padding


def _draw_projection_panel(
    ax: Any,
    data: pd.DataFrame,
    group_column: str,
    group_order: list[str],
    color_map: dict[str, str],
    marker_map: dict[str, str],
    panel_label: str,
) -> None:
    for group in group_order:
        group_data = data[data[group_column].astype("string").eq(group)]
        if group_data.empty:
            continue
        
        marker = marker_map.get(group, "o")
        marker_size = 7.2 if marker != "o" else 5.2

        ax.scatter(
            group_data["pc1"],
            group_data["pc2"],
            s=marker_size,
            alpha=0.72,
            linewidths=0,
            marker=marker,
            color=color_map.get(group, "0.45"),
            label=group,
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=2.2, width=0.45, color="0.45")

    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("0.55")

    ax.legend(
        frameon=False,
        fontsize=7.8,
        loc="upper right",
        markerscale=1.7,
        handletextpad=0.25,
        borderaxespad=0.25,
    )

    ax.text(
        0.5,
        -0.13,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="0.08",
    )


def plot_cluster_projection_view(
    projection: pd.DataFrame,
    path: str | Path,
) -> Path:
    required_columns = {"pc1", "pc2", "cluster_name", "degradation_status"}

    if projection.empty or not required_columns.issubset(projection.columns):
        fig, ax = plt.subplots(figsize=(5.8, 3.2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No projection records",
            ha="center",
            va="center",
            fontsize=9,
        )
        output_path = save_figure(fig, path)
        plt.close(fig)
        return output_path

    data = projection.copy()
    data["pc1"] = pd.to_numeric(data["pc1"], errors="coerce")
    data["pc2"] = pd.to_numeric(data["pc2"], errors="coerce")
    data = data.dropna(subset=["pc1", "pc2"])

    if data.empty:
        fig, ax = plt.subplots(figsize=(5.8, 3.2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No valid projection coordinates",
            ha="center",
            va="center",
            fontsize=9,
        )
        output_path = save_figure(fig, path)
        plt.close(fig)
        return output_path

    cluster_order = sorted(data["cluster_name"].astype(str).unique())
    status_values = set(data["degradation_status"].astype(str))
    status_order = [
        status
        for status in ["Not degraded", "Degraded", "Unavailable"]
        if status in status_values
    ]

    x_limits = _projection_axis_limits(data["pc1"])
    y_limits = _projection_axis_limits(data["pc2"])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.7, 4.1),
        sharex=True,
        sharey=True,
    )

    _draw_projection_panel(
        axes[0],
        data,
        group_column="cluster_name",
        group_order=cluster_order,
        color_map=CLUSTER_PROJECTION_CLUSTER_COLORS,
        marker_map=CLUSTER_PROJECTION_CLUSTER_MARKERS,
        panel_label="(a) K-Means profiles",
    )

    _draw_projection_panel(
        axes[1],
        data,
        group_column="degradation_status",
        group_order=status_order,
        color_map=CLUSTER_PROJECTION_STATUS_COLORS,
        marker_map=CLUSTER_PROJECTION_STATUS_MARKERS,
        panel_label="(b) Service-degradation label",
    )

    for ax in axes:
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

    fig.subplots_adjust(
        left=0.045,
        right=0.99,
        bottom=0.16,
        top=0.98,
        wspace=0.08,
    )

    output_path = save_figure(fig, path)
    plt.close(fig)
    return output_path


def plot_cluster_gap_and_seed_agreement(
    null_baseline: pd.DataFrame,
    pairwise_stability: pd.DataFrame,
    path: str | Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
    ax_gap, ax_seed = axes

    if null_baseline.empty:
        ax_gap.axis("off")
        ax_gap.text(
            0.5,
            0.5,
            "No null-baseline records",
            ha="center",
            va="center",
            fontsize=9,
        )
    else:
        row = null_baseline.iloc[0]
        observed_gap = float(row["observed_degradation_rate_gap"])
        random_mean = float(row["random_gap_mean"])
        random_std = float(row["random_gap_std"])
        random_p95 = float(row["random_gap_p95"])
        random_p99 = float(row["random_gap_p99"])
        empirical_p = float(row["empirical_p_value_random_gap_ge_observed"])

        xmax = max(observed_gap, random_p99) * 1.10

        ax_gap.hlines(
            y=0.0,
            xmin=0.0,
            xmax=xmax,
            color="0.86",
            linewidth=5.0,
            zorder=0,
        )
        ax_gap.scatter(
            [random_mean],
            [0.0],
            s=54,
            color="0.65",
            edgecolor="0.35",
            linewidth=0.6,
            zorder=2,
            label="Random mean",
        )
        ax_gap.hlines(
            y=0.0,
            xmin=max(0.0, random_mean - random_std),
            xmax=random_mean + random_std,
            color="0.35",
            linewidth=1.2,
            zorder=1,
        )
        ax_gap.scatter(
            [random_p95],
            [0.0],
            s=50,
            marker="D",
            color="0.45",
            zorder=2,
            label="Random p95",
        )
        ax_gap.scatter(
            [random_p99],
            [0.0],
            s=58,
            marker="^",
            color="0.30",
            zorder=2,
            label="Random p99",
        )
        ax_gap.scatter(
            [observed_gap],
            [0.0],
            s=68,
            color="#b24a2f",
            edgecolor="0.15",
            linewidth=0.6,
            zorder=3,
            label="Observed gap",
        )
        ax_gap.vlines(
            observed_gap,
            ymin=-0.10,
            ymax=0.10,
            color="#b24a2f",
            linewidth=1.2,
            zorder=2,
        )
        ax_gap.text(
            observed_gap,
            0.14,
            f"Observed = {observed_gap:.3f}\nEmpirical p = {empirical_p:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="0.15",
        )

        ax_gap.set_xlim(0.0, xmax)
        ax_gap.set_ylim(-0.22, 0.32)
        ax_gap.set_yticks([])
        ax_gap.set_xlabel("Degradation-rate gap")
        ax_gap.legend(frameon=False, fontsize=8.0, loc="upper left")
        ax_gap.spines["top"].set_visible(False)
        ax_gap.spines["right"].set_visible(False)
        ax_gap.spines["left"].set_visible(False)

    if (
        pairwise_stability.empty
        or "adjusted_rand_index" not in pairwise_stability.columns
    ):
        ax_seed.axis("off")
        ax_seed.text(
            0.5,
            0.5,
            "No seed-agreement records",
            ha="center",
            va="center",
            fontsize=9,
        )
    else:
        ari_values = pd.to_numeric(
            pairwise_stability["adjusted_rand_index"],
            errors="coerce",
        ).dropna()

        if ari_values.empty:
            ax_seed.axis("off")
            ax_seed.text(
                0.5,
                0.5,
                "No valid ARI values",
                ha="center",
                va="center",
                fontsize=9,
            )
        else:
            ax_seed.boxplot(
                ari_values,
                vert=False,
                widths=0.45,
                patch_artist=True,
                boxprops={
                    "facecolor": "0.90",
                    "edgecolor": "0.35",
                    "linewidth": 0.9,
                },
                medianprops={"color": "0.20", "linewidth": 1.1},
                whiskerprops={"color": "0.35", "linewidth": 0.9},
                capprops={"color": "0.35", "linewidth": 0.9},
            )
            ax_seed.scatter(
                ari_values,
                np.ones(len(ari_values)),
                s=26,
                color="#1f77b4",
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

            mean_ari = float(ari_values.mean())
            min_ari = float(ari_values.min())
            max_ari = float(ari_values.max())

            ax_seed.axvline(
                mean_ari,
                color="#b24a2f",
                linewidth=1.1,
                linestyle=(0, (4, 3)),
            )
            ax_seed.text(
                mean_ari,
                1.24,
                f"Mean = {mean_ari:.3f}\nMin = {min_ari:.3f}\nMax = {max_ari:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color="0.15",
            )

            lower = max(0.0, min_ari - 0.01)
            upper = min(1.005, max_ari + 0.005)
            ax_seed.set_xlim(lower, upper)
            ax_seed.set_yticks([])
            ax_seed.set_xlabel("Pairwise adjusted Rand index")
            ax_seed.spines["top"].set_visible(False)
            ax_seed.spines["right"].set_visible(False)
            ax_seed.spines["left"].set_visible(False)
            ax_seed.grid(axis="x", linewidth=0.4, alpha=0.45)

    fig.tight_layout()
    output_path = save_figure(fig, path)
    plt.close(fig)
    return output_path


def plot_clustering_analysis_outputs(
    results_dir: str | Path,
    figures_dir: str | Path,
    feature_block: str = "reduced_context",
) -> dict[str, Path]:
    results_path = Path(results_dir)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    validation = pd.read_csv(results_path / "clustering_validation.csv")
    selection_path = results_path / "clustering_model_selection.csv"
    selection = (
        pd.read_csv(selection_path) if selection_path.exists() else pd.DataFrame()
    )
    selected_k = _selected_k_from_table(selection, method="kmeans")
    effect_sizes_path = results_path / "cluster_profile_effect_sizes.csv"
    projection_path = results_path / "cluster_projection_table.csv"
    null_baseline_path = results_path / "cluster_null_baseline.csv"
    pairwise_stability_path = results_path / "kmeans_pairwise_stability.csv"

    outputs: dict[str, Path] = {
        "elbow_curve": plot_clustering_elbow_curve(
            validation,
            figures_path / "elbow_curve.pdf",
            selected_k=selected_k,
            feature_block=feature_block,
        ),
        "silhouette_scores": plot_clustering_silhouette_scores(
            validation,
            figures_path / "silhouette_scores.pdf",
            selected_k=selected_k,
            feature_block=feature_block,
        ),
    }

    if effect_sizes_path.exists():
        effect_sizes = pd.read_csv(effect_sizes_path)
        outputs["cluster_profile_effect_sizes_figure"] = (
            plot_cluster_profile_effect_sizes(
                effect_sizes,
                figures_path / "cluster_profile_effect_sizes.pdf",
            )
        )

    if projection_path.exists():
        projection = pd.read_csv(projection_path)
        outputs["cluster_projection_view"] = plot_cluster_projection_view(
            projection,
            figures_path / "cluster_projection_view.pdf",
        )

    if null_baseline_path.exists() and pairwise_stability_path.exists():
        null_baseline = pd.read_csv(null_baseline_path)
        pairwise_stability = pd.read_csv(pairwise_stability_path)
        outputs["cluster_gap_and_seed_agreement"] = plot_cluster_gap_and_seed_agreement(
            null_baseline,
            pairwise_stability,
            figures_path / "cluster_gap_and_seed_agreement.pdf",
        )

    return outputs


def _display_association_itemset(text: Any, width: int = 44) -> str:
    labels: list[str] = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            field_name, value = item.split("=", 1)
            label = f"{display_field_name(field_name)}: {display_value(value)}"
        else:
            label = item.replace("_", " ").title()
        labels.append(
            "\n".join(textwrap.wrap(label, width=width, break_long_words=False))
        )
    return "\n".join(labels)


def _display_association_itemset_inline(text: Any, separator: str = " + ") -> str:
    labels: list[str] = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            field_name, value = item.split("=", 1)
            label = f"{display_field_name(field_name)}: {display_value(value)}"
        else:
            label = item.replace("_", " ").title()
        labels.append(label)
    return separator.join(labels)


def _association_antecedent_count(row: pd.Series) -> int:
    if "antecedent_count" in row.index and pd.notna(row["antecedent_count"]):
        return int(row["antecedent_count"])

    antecedent_text = str(row.get("antecedent_text", ""))
    return len([item for item in antecedent_text.split(";") if item.strip()])


def _scale_marker_sizes(
    values: pd.Series,
    minimum_size: float = 80.0,
    maximum_size: float = 560.0,
) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if numeric_values.empty:
        return pd.Series(dtype="float64")

    lower = float(numeric_values.min())
    upper = float(numeric_values.max())

    if lower == upper:
        midpoint = (minimum_size + maximum_size) / 2.0
        return pd.Series(midpoint, index=numeric_values.index)

    scaled = (numeric_values - lower) / (upper - lower)
    return minimum_size + scaled * (maximum_size - minimum_size)


def _collapse_rule_algorithm_labels(data: pd.DataFrame) -> pd.DataFrame:
    signature_columns = ["antecedent_text", "consequent_text"]
    if data.empty or not set(signature_columns).issubset(data.columns):
        return data

    if "algorithm" not in data.columns:
        return data.drop_duplicates(subset=signature_columns)

    algorithm_map = (
        data.groupby(signature_columns)["algorithm"]
        .apply(lambda values: ";".join(sorted(set(values.astype(str)))))
        .reset_index()
    )

    output = data.drop(columns=["algorithm"]).drop_duplicates(
        subset=signature_columns
    )
    output = output.merge(algorithm_map, on=signature_columns, how="left")
    return output


def _association_rule_plot_frame(
    degradation_rules: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    if degradation_rules.empty:
        return pd.DataFrame()

    required_columns = {
        "antecedent_text",
        "consequent_text",
        "support",
        "confidence",
        "lift",
    }
    if not required_columns.issubset(degradation_rules.columns):
        return pd.DataFrame()

    data = degradation_rules.copy()
    data = data[data["consequent_text"].astype(str).eq("service_degraded=1")]
    data = _collapse_rule_algorithm_labels(data)

    for column in ["support", "confidence", "lift"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["support", "confidence", "lift"])
    if data.empty:
        return data

    data["antecedent_count"] = data.apply(_association_antecedent_count, axis=1)
    data = data.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False],
    ).head(top_n)

    data = data.reset_index(drop=True)
    data.insert(0, "rule_id", [f"R{index + 1}" for index in range(len(data))])
    data["bubble_size"] = _scale_marker_sizes(data["lift"])
    return data


def build_association_rule_plot_index(
    degradation_rules: pd.DataFrame,
    top_n: int = 40,
    label_n: int = 0,
    transaction_count: int | None = None,
) -> pd.DataFrame:
    data = _association_rule_plot_frame(degradation_rules, top_n=top_n)

    columns = [
        "rule_id",
        "plot_rank",
        "is_labeled_in_bubble_plot",
        "algorithm",
        "antecedent_text",
        "antecedent_display",
        "consequent_text",
        "consequent_display",
        "support",
        "support_count",
        "confidence",
        "lift",
        "leverage",
        "conviction",
        "antecedent_count",
        "consequent_count",
        "bubble_size",
    ]

    if data.empty:
        return pd.DataFrame(columns=columns)

    output = data.copy().reset_index(drop=True)
    output.insert(1, "plot_rank", range(1, len(output) + 1))
    output.insert(2, "is_labeled_in_bubble_plot", output["plot_rank"].le(label_n))
    output["antecedent_display"] = output["antecedent_text"].map(
        _display_association_itemset_inline
    )
    output["consequent_display"] = output["consequent_text"].map(
        _display_association_itemset_inline
    )

    if transaction_count is None:
        output["support_count"] = pd.NA
    else:
        output["support_count"] = (
            pd.to_numeric(output["support"], errors="coerce") * int(transaction_count)
        ).round().astype("Int64")

    for column in ["leverage", "conviction", "consequent_count"]:
        if column not in output.columns:
            output[column] = pd.NA

    return output[columns]


def _association_axis_limits(
    values: pd.Series,
    padding_fraction: float,
) -> tuple[float, float]:
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    if numeric_values.empty:
        return 0.0, 1.0

    lower = float(numeric_values.min())
    upper = float(numeric_values.max())

    if lower == upper:
        return lower - 0.01, upper + 0.01

    padding = padding_fraction * (upper - lower)
    return lower - padding, upper + padding


def _association_length_style(antecedent_count: int) -> dict[str, Any]:
    if antecedent_count <= 1:
        return {
            "label": "1 item",
            "facecolor": "white",
            "edgecolor": "#244C5A",
            "alpha": 1.0,
            "linewidth": 1.0,
        }

    if antecedent_count == 2:
        return {
            "label": "2 items",
            "facecolor": "#8FAAB3",
            "edgecolor": "#244C5A",
            "alpha": 0.72,
            "linewidth": 0.9,
        }

    return {
        "label": "3 or more items",
        "facecolor": "#4E7F8A",
        "edgecolor": "#173B42",
        "alpha": 0.78,
        "linewidth": 0.9,
    }


def _bubble_size_for_lift(
    lift_value: float,
    lift_minimum: float,
    lift_maximum: float,
    minimum_size: float = 80.0,
    maximum_size: float = 560.0,
) -> float:
    if lift_minimum == lift_maximum:
        return (minimum_size + maximum_size) / 2.0

    scaled = (lift_value - lift_minimum) / (lift_maximum - lift_minimum)
    return minimum_size + scaled * (maximum_size - minimum_size)


def _add_association_bubble_legends(
    ax: Any,
    data: pd.DataFrame,
) -> tuple[Any, Any]:
    from matplotlib.lines import Line2D

    observed_lengths = sorted(data["antecedent_count"].astype(int).unique())
    length_handles = []

    for length in observed_lengths:
        style = _association_length_style(int(length))
        length_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=style["facecolor"],
                markeredgecolor=style["edgecolor"],
                markeredgewidth=style["linewidth"],
                markersize=7.5,
                alpha=style["alpha"],
                label=style["label"],
            )
        )

    length_legend = ax.legend(
        handles=length_handles,
        title="Antecedent length",
        frameon=True,
        fontsize=8.0,
        title_fontsize=8.3,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderpad=0.6,
        labelspacing=0.55,
    )
    length_legend.get_frame().set_edgecolor("0.72")
    length_legend.get_frame().set_linewidth(0.7)
    length_legend.get_frame().set_facecolor("white")
    ax.add_artist(length_legend)

    lift_minimum = float(data["lift"].min())
    lift_maximum = float(data["lift"].max())
    lift_values = sorted(
        {
            round(lift_minimum, 2),
            round(float(data["lift"].median()), 2),
            round(lift_maximum, 2),
        }
    )

    size_handles = []
    for lift_value in lift_values:
        marker_size = _bubble_size_for_lift(
            lift_value,
            lift_minimum,
            lift_maximum,
        )
        size_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="0.72",
                markeredgecolor="0.35",
                markeredgewidth=0.8,
                markersize=min(marker_size**0.5 * 0.72, 15.0),
                alpha=0.65,
                label=f"{lift_value:.2f}",
            )
        )

    size_legend = ax.legend(
        handles=size_handles,
        title="Lift",
        frameon=True,
        fontsize=8.0,
        title_fontsize=8.3,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.56),
        borderpad=0.6,
        labelspacing=1.10,
    )
    size_legend.get_frame().set_edgecolor("0.72")
    size_legend.get_frame().set_linewidth(0.7)
    size_legend.get_frame().set_facecolor("white")
    return length_legend, size_legend


def _annotate_association_rule_labels(
    ax: Any,
    data: pd.DataFrame,
    label_n: int,
) -> tuple[Any, ...]:
    if label_n <= 0 or "rule_id" not in data.columns:
        return tuple()

    label_frame = data.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False],
    ).head(label_n)

    offsets = [
        (7, 7),
        (7, -10),
        (-22, 7),
        (-22, -10),
        (10, 16),
        (-30, 16),
        (10, -18),
        (-30, -18),
        (0, 24),
        (0, -26),
    ]

    annotations: list[Any] = []
    for label_index, (_, row) in enumerate(label_frame.iterrows()):
        x_offset, y_offset = offsets[label_index % len(offsets)]
        annotation = ax.annotate(
            str(row["rule_id"]),
            xy=(float(row["support"]), float(row["confidence"])),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if x_offset >= 0 else "right",
            va="bottom" if y_offset >= 0 else "top",
            fontsize=7.5,
            color="0.08",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "0.45",
                "linewidth": 0.4,
                "alpha": 0.90,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "0.35",
                "linewidth": 0.45,
                "shrinkA": 0,
                "shrinkB": 4,
            },
            zorder=4,
            clip_on=False,
        )
        annotations.append(annotation)

    return tuple(annotations)


def plot_rule_quality_bubble_plot(
    degradation_rules: pd.DataFrame,
    path: str | Path,
    top_n: int = 40,
    label_n: int = 0,
) -> Path:
    output_path = ensure_figure_parent(path)
    data = _association_rule_plot_frame(degradation_rules, top_n=top_n)

    if data.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No selected degradation rules",
            ha="center",
            va="center",
            fontsize=9,
        )
        output_path = save_figure(fig, output_path)
        plt.close(fig)
        return output_path

    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    for antecedent_count in sorted(data["antecedent_count"].astype(int).unique()):
        subset = data[data["antecedent_count"].astype(int).eq(antecedent_count)]
        style = _association_length_style(int(antecedent_count))

        ax.scatter(
            subset["support"],
            subset["confidence"],
            s=subset["bubble_size"],
            facecolors=style["facecolor"],
            edgecolors=style["edgecolor"],
            linewidths=style["linewidth"],
            alpha=style["alpha"],
            zorder=2,
        )

    x_min, x_max = _association_axis_limits(data["support"], 0.08)
    y_min, y_max = _association_axis_limits(data["confidence"], 0.08)

    ax.set_xlim(max(0.0, x_min), x_max)
    ax.set_ylim(max(0.0, y_min), min(1.03, y_max))
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")

    ax.grid(True, linewidth=0.45, alpha=0.38)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    label_artists = _annotate_association_rule_labels(ax, data, label_n)
    legend_artists = _add_association_bubble_legends(ax, data)
    extra_artists = (*label_artists, *legend_artists)

    fig.subplots_adjust(left=0.10, right=0.76, bottom=0.13, top=0.96)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp{output_path.suffix}"
    )
    fig.savefig(
        temporary_path,
        bbox_inches="tight",
        bbox_extra_artists=extra_artists,
        pad_inches=0.08,
    )
    temporary_path.replace(output_path)
    plt.close(fig)
    return output_path


def plot_top_association_rules(
    degradation_rules: pd.DataFrame,
    path: str | Path,
    top_n: int = 20,
) -> Path:
    output_path = ensure_figure_parent(path)

    if degradation_rules.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No selected degradation rules",
            ha="center",
            va="center",
            fontsize=9,
        )
        output_path = save_figure(fig, output_path)
        plt.close(fig)
        return output_path

    data = degradation_rules.copy()
    if "consequent_text" in data.columns:
        data = data[data["consequent_text"].astype(str).eq("service_degraded=1")]

    data = _collapse_rule_algorithm_labels(data)
    for column in ["support", "confidence", "lift"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["support", "confidence", "lift"])
    data = data.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False],
    ).head(top_n)

    if data.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No selected degradation rules",
            ha="center",
            va="center",
            fontsize=9,
        )
        output_path = save_figure(fig, output_path)
        plt.close(fig)
        return output_path

    data = data.sort_values("lift", ascending=True)
    labels = [
        _display_association_itemset(text)
        for text in data["antecedent_text"].astype(str).tolist()
    ]

    baseline = 1.0
    fig_height = max(4.8, 0.55 * len(data) + 1.4)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))

    lift_values = data["lift"].astype(float)
    ax.barh(
        labels,
        lift_values - baseline,
        left=baseline,
        color=MODEL_COMPARISON_BLUE,
        edgecolor=MODEL_COMPARISON_BLUE,
        linewidth=0.4,
    )

    for position, value in enumerate(lift_values):
        ax.text(
            float(value) + 0.01,
            position,
            f"{float(value):.2f}",
            va="center",
            fontsize=8.4,
            color="0.10",
        )

    x_max = max(1.1, float(lift_values.max()) + 0.08)
    ax.axvline(
        baseline,
        color="0.35",
        linewidth=0.8,
        linestyle=(0, (4, 4)),
        zorder=0,
    )
    ax.set_xlabel("Lift")
    ax.set_ylabel("Antecedent itemset")
    ax.set_xlim(baseline, x_max)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_path = save_figure(fig, output_path)
    plt.close(fig)
    return output_path




EXTERNAL_BOXPLOT_MEDIAN_COLOR = "#6E6E6E"


def _external_source_order(
    source_dataset: Any,
    source_order: dict[str, int] | None,
) -> int:
    if source_order is None:
        return 99
    return source_order.get(str(source_dataset), 99)


def _external_sample_values(
    values: pd.Series,
    sample_limit: int,
    random_state: int,
) -> np.ndarray:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) <= sample_limit:
        return clean.to_numpy()
    return clean.sample(n=sample_limit, random_state=random_state).to_numpy()


def plot_external_metric_distribution(
    comparison: pd.DataFrame,
    metric: str,
    x_label: str,
    path: str | Path,
    min_records: int = 20,
    sample_limit: int = 15000,
    random_state: int = 42,
    source_order: dict[str, int] | None = None,
) -> Path | None:
    required_columns = {"metric", "source_dataset", "source_label", "value"}
    if comparison.empty or not required_columns.issubset(comparison.columns):
        return None

    metric_frame = comparison[comparison["metric"].astype(str).eq(str(metric))].copy()
    if metric_frame.empty:
        return None

    plot_rows: list[pd.DataFrame] = []
    group_columns = ["source_dataset", "source_label"]
    for _, group in metric_frame.groupby(group_columns, dropna=False):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if len(values) < min_records:
            continue

        sampled = _external_sample_values(values, sample_limit, random_state)
        representative = group.iloc[0]
        plot_rows.append(
            pd.DataFrame(
                {
                    "source_dataset": str(representative["source_dataset"]),
                    "source_label": str(representative["source_label"]),
                    "value": sampled,
                }
            )
        )

    if len(plot_rows) < 2:
        return None

    plot_frame = pd.concat(plot_rows, ignore_index=True)
    if "synnetqos" not in set(plot_frame["source_dataset"].astype(str)):
        return None

    reference_sources = set(plot_frame["source_dataset"].astype(str)) - {"synnetqos"}
    if not reference_sources:
        return None

    source_counts = plot_frame.groupby("source_label")["value"].size().to_dict()
    label_order = (
        plot_frame[["source_dataset", "source_label"]]
        .drop_duplicates()
        .assign(
            source_order=lambda frame: frame["source_dataset"].map(
                lambda value: _external_source_order(value, source_order)
            )
        )
        .sort_values(["source_order", "source_label"])
    )

    ordered_labels = label_order["source_label"].astype(str).tolist()
    data = [
        plot_frame.loc[
            plot_frame["source_label"].astype(str).eq(label), "value"
        ].to_numpy()
        for label in ordered_labels
    ]
    labels = [
        f"{label}\n(n={source_counts.get(label, 0):,})" for label in ordered_labels
    ]

    output_path = ensure_figure_parent(path)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    boxplot_kwargs: dict[str, Any] = {
        "showfliers": False,
        "medianprops": {
            "color": EXTERNAL_BOXPLOT_MEDIAN_COLOR,
            "linewidth": 1.6,
        },
    }
    try:
        ax.boxplot(data, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        ax.boxplot(data, labels=labels, **boxplot_kwargs)

    ax.set_xlabel("Dataset")
    ax.set_ylabel(x_label)
    ax.tick_params(axis="x", rotation=22)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)
    return output_path
