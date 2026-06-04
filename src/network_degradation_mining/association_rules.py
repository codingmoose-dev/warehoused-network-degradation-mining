from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from network_degradation_mining.io import ensure_directory, read_csv, write_csv
from network_degradation_mining.plotting import (
    build_association_rule_plot_index,
    plot_rule_quality_bubble_plot,
    plot_top_association_rules,
)

TARGET_POSITIVE_ITEM = "service_degraded=1"
TARGET_NEGATIVE_ITEM = "service_degraded=0"

MIN_SUPPORT = 0.05
MIN_CONFIDENCE = 0.60
MAX_ITEMSET_LENGTH = 3
MAX_DEGRADATION_RULES = 250
TOP_RULES_FOR_FIGURE = 20
TOP_RULES_FOR_BUBBLE_FIGURE = 80
LABEL_RULES_FOR_BUBBLE_FIGURE = 10
RUN_FP_GROWTH = True

LABEL_DEFINITION_FAMILIES = frozenset(
    {
        "latency_ms",
        "jitter_ms",
        "downlink_shortfall_fraction",
        "throughput_satisfaction_ratio",
        "dropped_connection",
        "video_quality_label",
    }
)

SUMMARY_COLUMNS = ["metric", "value"]
ITEMSET_COLUMNS = ["algorithm", "support", "itemsets", "item_count", "itemset_text"]
RULE_COLUMNS = [
    "algorithm",
    "antecedents",
    "consequents",
    "antecedent_text",
    "consequent_text",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
    "antecedent_count",
    "consequent_count",
]


def _load_mlxtend() -> tuple[Any, Any, Any, Any]:
    try:
        from mlxtend.frequent_patterns import (
            apriori,
            association_rules,
            fpgrowth,
        )
        from mlxtend.preprocessing import TransactionEncoder
    except Exception as exc:
        raise ImportError(
            "Association rule mining requires mlxtend. Install it with: "
            "pip install mlxtend"
        ) from exc

    return TransactionEncoder, apriori, fpgrowth, association_rules


def _item_family(item: Any) -> str:
    text = str(item).strip()
    return text.split("=", 1)[0] if "=" in text else text


def _item_value(item: Any) -> str:
    text = str(item).strip()
    return text.split("=", 1)[1] if "=" in text else ""


def _item_families(itemset: Any) -> set[str]:
    return {_item_family(item) for item in itemset}


def _itemset_to_text(itemset: Any) -> str:
    if isinstance(itemset, str):
        return itemset
    return ";".join(sorted(str(item) for item in itemset))


def _text_to_itemset(text: Any) -> set[str]:
    if pd.isna(text):
        return set()
    return {item.strip() for item in str(text).split(";") if item.strip()}


def _has_target_item(itemset: Any) -> bool:
    items = {str(item) for item in itemset}
    return TARGET_POSITIVE_ITEM in items or TARGET_NEGATIVE_ITEM in items


def _has_label_definition_item(itemset: Any) -> bool:
    return bool(_item_families(itemset) & LABEL_DEFINITION_FAMILIES)


def _parse_transactions(dataframe: pd.DataFrame) -> list[list[str]]:
    if "items" not in dataframe.columns:
        raise ValueError(
            "Association transaction table must contain an 'items' column."
        )

    transactions: list[list[str]] = []
    for value in dataframe["items"].fillna("").astype(str):
        items = sorted({item.strip() for item in value.split(";") if item.strip()})
        if items:
            transactions.append(items)

    if not transactions:
        raise ValueError("No non-empty transactions were found.")

    return transactions


def _validate_transaction_table(
    dataframe: pd.DataFrame,
    transactions: list[list[str]],
) -> None:
    if len(dataframe) != len(transactions):
        raise ValueError(
            "Association transaction parsing changed the row count: "
            f"input_rows={len(dataframe)}, transactions={len(transactions)}."
        )

    missing_target_rows = [
        index
        for index, transaction in enumerate(transactions)
        if TARGET_POSITIVE_ITEM not in transaction
        and TARGET_NEGATIVE_ITEM not in transaction
    ]
    if missing_target_rows:
        first_rows = ", ".join(str(index) for index in missing_target_rows[:5])
        raise ValueError(
            "Every association transaction must contain exactly one "
            "service-degradation target item. Missing target item in rows: "
            f"{first_rows}."
        )

    both_target_rows = [
        index
        for index, transaction in enumerate(transactions)
        if TARGET_POSITIVE_ITEM in transaction and TARGET_NEGATIVE_ITEM in transaction
    ]
    if both_target_rows:
        first_rows = ", ".join(str(index) for index in both_target_rows[:5])
        raise ValueError(
            "Each association transaction must contain only one service-degradation "
            f"target item. Both target items appear in rows: {first_rows}."
        )

    if "service_degraded" not in dataframe.columns:
        return

    expected_items = dataframe["service_degraded"].map(
        lambda value: TARGET_POSITIVE_ITEM if int(value) == 1 else TARGET_NEGATIVE_ITEM
    )
    mismatched_rows = [
        index
        for index, expected_item in expected_items.items()
        if expected_item not in transactions[index]
    ]
    if mismatched_rows:
        first_rows = ", ".join(str(index) for index in mismatched_rows[:5])
        raise ValueError(
            "The service_degraded column does not match the target item in rows: "
            f"{first_rows}."
        )


def _is_degradation_signature(row: pd.Series) -> bool:
    antecedents = {str(item) for item in row["antecedents"]}
    consequents = {str(item) for item in row["consequents"]}
    return (
        consequents == {TARGET_POSITIVE_ITEM}
        and not _has_target_item(antecedents)
        and not _has_label_definition_item(antecedents)
    )


def _prepare_itemsets(itemsets: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    if itemsets.empty:
        return pd.DataFrame(columns=ITEMSET_COLUMNS)

    output = itemsets.copy()
    output.insert(0, "algorithm", algorithm)
    output["item_count"] = output["itemsets"].map(len)
    output["itemset_text"] = output["itemsets"].map(_itemset_to_text)
    output = output.sort_values(
        ["support", "item_count", "itemset_text"], ascending=[False, False, True]
    )
    return output


def _prepare_rules(rules: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    if rules.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)

    output = rules.copy()
    output.insert(0, "algorithm", algorithm)
    output["antecedent_text"] = output["antecedents"].map(_itemset_to_text)
    output["consequent_text"] = output["consequents"].map(_itemset_to_text)
    output["antecedent_count"] = output["antecedents"].map(len)
    output["consequent_count"] = output["consequents"].map(len)
    output = output.sort_values(
        ["lift", "confidence", "support"],
        ascending=[False, False, False],
    )
    return output


def _mine_algorithm(
    encoded: pd.DataFrame,
    algorithm_name: str,
    miner: Any,
    association_rules_function: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    miner_kwargs: dict[str, Any] = {
        "min_support": MIN_SUPPORT,
        "use_colnames": True,
        "max_len": MAX_ITEMSET_LENGTH,
    }
    if algorithm_name == "apriori":
        miner_kwargs["low_memory"] = True

    itemsets = miner(encoded, **miner_kwargs)
    prepared_itemsets = _prepare_itemsets(itemsets, algorithm=algorithm_name)

    if itemsets.empty:
        return prepared_itemsets, pd.DataFrame(columns=RULE_COLUMNS)

    rules = association_rules_function(
        itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE,
    )
    prepared_rules = _prepare_rules(rules, algorithm=algorithm_name)
    return prepared_itemsets, prepared_rules


def _build_item_support_table(
    transactions: list[list[str]],
    transaction_count: int,
) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for transaction in transactions:
        for item in set(transaction):
            counts[item] = counts.get(item, 0) + 1

    rows = [
        {
            "item": item,
            "support_count": count,
            "support": count / transaction_count if transaction_count else 0.0,
            "item_family": _item_family(item),
        }
        for item, count in counts.items()
    ]

    if not rows:
        return pd.DataFrame(columns=["item", "support_count", "support", "item_family"])

    return pd.DataFrame(rows).sort_values(
        ["support", "item"], ascending=[False, True]
    )


def _build_item_definition_table(item_support: pd.DataFrame) -> pd.DataFrame:
    if item_support.empty:
        return pd.DataFrame(
            columns=[
                "item",
                "item_family",
                "item_value",
                "support_count",
                "support",
                "is_target_item",
                "is_label_definition_item",
                "rule_use",
            ]
        )

    output = item_support.copy()
    output["item_value"] = output["item"].map(_item_value)
    output["is_target_item"] = output["item"].isin(
        [TARGET_POSITIVE_ITEM, TARGET_NEGATIVE_ITEM]
    )
    output["is_label_definition_item"] = output["item_family"].isin(
        LABEL_DEFINITION_FAMILIES
    )
    output["rule_use"] = "eligible_context_antecedent"
    output.loc[output["is_target_item"], "rule_use"] = "target_item"
    output.loc[
        output["is_label_definition_item"], "rule_use"
    ] = "excluded_from_target_rule_antecedent"

    columns = [
        "item",
        "item_family",
        "item_value",
        "support_count",
        "support",
        "is_target_item",
        "is_label_definition_item",
        "rule_use",
    ]
    return output[columns].sort_values(["item_family", "item_value"]).reset_index(
        drop=True
    )


def _build_transaction_summary(
    dataframe: pd.DataFrame,
    transactions: list[list[str]],
) -> pd.DataFrame:
    item_counts = pd.Series([len(transaction) for transaction in transactions])
    unique_items = {item for transaction in transactions for item in transaction}
    target_counts = (
        dataframe["service_degraded"].value_counts().to_dict()
        if "service_degraded" in dataframe.columns
        else {}
    )
    row_count = len(dataframe)

    rows = [
        {"metric": "row_count", "value": int(row_count)},
        {"metric": "transaction_count", "value": int(len(transactions))},
        {"metric": "unique_item_count", "value": int(len(unique_items))},
        {"metric": "min_items_per_transaction", "value": int(item_counts.min())},
        {
            "metric": "median_items_per_transaction",
            "value": float(item_counts.median()),
        },
        {"metric": "max_items_per_transaction", "value": int(item_counts.max())},
        {
            "metric": "degraded_transaction_count",
            "value": int(target_counts.get(1, 0)),
        },
        {
            "metric": "not_degraded_transaction_count",
            "value": int(target_counts.get(0, 0)),
        },
        {
            "metric": "degraded_transaction_fraction",
            "value": float(target_counts.get(1, 0) / row_count) if row_count else 0.0,
        },
        {"metric": "min_support", "value": float(MIN_SUPPORT)},
        {"metric": "min_confidence", "value": float(MIN_CONFIDENCE)},
        {"metric": "max_itemset_length", "value": int(MAX_ITEMSET_LENGTH)},
        {"metric": "run_fp_growth", "value": bool(RUN_FP_GROWTH)},
    ]
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _build_rule_selection_summary(
    all_rules: pd.DataFrame,
    degradation_rules: pd.DataFrame,
    transaction_count: int,
) -> pd.DataFrame:
    if all_rules.empty or "algorithm" not in all_rules.columns:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "transaction_count",
                "all_rule_count",
                "exact_positive_consequent_count",
                "no_target_in_antecedent_count",
                "target_definition_antecedent_excluded_count",
                "selected_degradation_rule_count",
                "min_support",
                "min_confidence",
                "max_itemset_length",
            ]
        )

    rows: list[dict[str, Any]] = []
    for algorithm in sorted(all_rules["algorithm"].dropna().unique()):
        subset = all_rules[all_rules["algorithm"].eq(algorithm)].copy()
        exact_positive = subset[
            subset["consequents"].map(
                lambda itemset: {str(item) for item in itemset}
                == {TARGET_POSITIVE_ITEM}
            )
        ]
        no_target_in_antecedent = exact_positive[
            ~exact_positive["antecedents"].map(_has_target_item)
        ]
        non_definition = no_target_in_antecedent[
            ~no_target_in_antecedent["antecedents"].map(_has_label_definition_item)
        ]
        selected = degradation_rules[degradation_rules["algorithm"].eq(algorithm)]

        rows.append(
            {
                "algorithm": algorithm,
                "transaction_count": int(transaction_count),
                "all_rule_count": int(len(subset)),
                "exact_positive_consequent_count": int(len(exact_positive)),
                "no_target_in_antecedent_count": int(len(no_target_in_antecedent)),
                "target_definition_antecedent_excluded_count": int(
                    len(no_target_in_antecedent) - len(non_definition)
                ),
                "selected_degradation_rule_count": int(len(selected)),
                "min_support": float(MIN_SUPPORT),
                "min_confidence": float(MIN_CONFIDENCE),
                "max_itemset_length": int(MAX_ITEMSET_LENGTH),
            }
        )

    return pd.DataFrame(rows)


def _build_quality_summary(
    frequent_itemsets: pd.DataFrame,
    all_rules: pd.DataFrame,
    degradation_rules: pd.DataFrame,
    transaction_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    algorithms = sorted(set(frequent_itemsets.get("algorithm", pd.Series()).dropna()))

    for algorithm in algorithms:
        itemset_subset = frequent_itemsets[frequent_itemsets["algorithm"] == algorithm]
        rule_subset = all_rules[all_rules["algorithm"] == algorithm]
        degradation_subset = degradation_rules[
            degradation_rules["algorithm"] == algorithm
        ]

        rows.append(
            {
                "algorithm": algorithm,
                "transaction_count": int(transaction_count),
                "frequent_itemset_count": int(len(itemset_subset)),
                "rule_count": int(len(rule_subset)),
                "degradation_rule_count": int(len(degradation_subset)),
                "min_support": float(MIN_SUPPORT),
                "min_confidence": float(MIN_CONFIDENCE),
                "max_itemset_length": int(MAX_ITEMSET_LENGTH),
                "label_definition_families_excluded_from_degradation_rules": ";".join(
                    sorted(LABEL_DEFINITION_FAMILIES)
                ),
                "max_lift_degradation_rule": float(degradation_subset["lift"].max())
                if not degradation_subset.empty
                else float("nan"),
                "median_lift_degradation_rule": float(
                    degradation_subset["lift"].median()
                )
                if not degradation_subset.empty
                else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def _build_rule_quality_plot_index_summary(
    rule_quality_plot_index: pd.DataFrame,
    transaction_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"metric": "transaction_count", "value": int(transaction_count)},
        {"metric": "bubble_plot_top_n", "value": int(TOP_RULES_FOR_BUBBLE_FIGURE)},
        {"metric": "bubble_plot_label_n", "value": int(LABEL_RULES_FOR_BUBBLE_FIGURE)},
        {
            "metric": "bubble_plot_rule_count",
            "value": int(len(rule_quality_plot_index)),
        },
    ]

    if rule_quality_plot_index.empty:
        rows.extend(
            [
                {"metric": "bubble_plot_labeled_rule_count", "value": 0},
                {"metric": "bubble_plot_min_support", "value": float("nan")},
                {"metric": "bubble_plot_max_support", "value": float("nan")},
                {"metric": "bubble_plot_min_confidence", "value": float("nan")},
                {"metric": "bubble_plot_max_confidence", "value": float("nan")},
                {"metric": "bubble_plot_min_lift", "value": float("nan")},
                {"metric": "bubble_plot_max_lift", "value": float("nan")},
            ]
        )
        return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    labeled = rule_quality_plot_index[
        rule_quality_plot_index["is_labeled_in_bubble_plot"].astype(bool)
    ]
    rows.extend(
        [
            {"metric": "bubble_plot_labeled_rule_count", "value": int(len(labeled))},
            {
                "metric": "bubble_plot_min_support",
                "value": float(rule_quality_plot_index["support"].min()),
            },
            {
                "metric": "bubble_plot_max_support",
                "value": float(rule_quality_plot_index["support"].max()),
            },
            {
                "metric": "bubble_plot_min_confidence",
                "value": float(rule_quality_plot_index["confidence"].min()),
            },
            {
                "metric": "bubble_plot_max_confidence",
                "value": float(rule_quality_plot_index["confidence"].max()),
            },
            {
                "metric": "bubble_plot_min_lift",
                "value": float(rule_quality_plot_index["lift"].min()),
            },
            {
                "metric": "bubble_plot_max_lift",
                "value": float(rule_quality_plot_index["lift"].max()),
            },
        ]
    )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _serialise_itemset_columns(*frames: pd.DataFrame) -> None:
    for frame in frames:
        for column in ("itemsets", "antecedents", "consequents"):
            if column in frame.columns:
                frame[column] = frame[column].map(_itemset_to_text)


def run_association_rule_mining(
    association_table_path: str | Path,
    results_dir: str | Path,
    figures_dir: str | Path,
) -> dict[str, Path]:
    results_path = ensure_directory(results_dir)
    figures_path = ensure_directory(figures_dir)

    TransactionEncoder, apriori, fpgrowth, association_rules_function = _load_mlxtend()

    dataframe = read_csv(association_table_path, low_memory=False)
    transactions = _parse_transactions(dataframe)
    _validate_transaction_table(dataframe, transactions)
    transaction_count = len(transactions)

    encoder = TransactionEncoder()
    encoded_array = encoder.fit(transactions).transform(transactions)
    encoded = pd.DataFrame(encoded_array, columns=encoder.columns_)

    itemsets_frames: list[pd.DataFrame] = []
    rules_frames: list[pd.DataFrame] = []
    algorithm_miners: list[tuple[str, Any]] = [("apriori", apriori)]
    if RUN_FP_GROWTH:
        algorithm_miners.append(("fp_growth", fpgrowth))

    for algorithm_name, miner in algorithm_miners:
        itemsets, rules = _mine_algorithm(
            encoded=encoded,
            algorithm_name=algorithm_name,
            miner=miner,
            association_rules_function=association_rules_function,
        )
        itemsets_frames.append(itemsets)
        rules_frames.append(rules)

    frequent_itemsets = (
        pd.concat(itemsets_frames, ignore_index=True)
        if itemsets_frames
        else pd.DataFrame(columns=ITEMSET_COLUMNS)
    )
    all_rules = (
        pd.concat(rules_frames, ignore_index=True)
        if rules_frames
        else pd.DataFrame(columns=RULE_COLUMNS)
    )

    if not all_rules.empty:
        degradation_rules = all_rules[
            all_rules.apply(_is_degradation_signature, axis=1)
        ].copy()
        degradation_rules = degradation_rules.sort_values(
            ["lift", "confidence", "support"],
            ascending=[False, False, False],
        ).head(MAX_DEGRADATION_RULES)
    else:
        degradation_rules = pd.DataFrame(columns=RULE_COLUMNS)

    item_support = _build_item_support_table(transactions, transaction_count)
    item_definitions = _build_item_definition_table(item_support)
    transaction_summary = _build_transaction_summary(dataframe, transactions)
    rule_selection_summary = _build_rule_selection_summary(
        all_rules=all_rules,
        degradation_rules=degradation_rules,
        transaction_count=transaction_count,
    )
    quality_summary = _build_quality_summary(
        frequent_itemsets=frequent_itemsets,
        all_rules=all_rules,
        degradation_rules=degradation_rules,
        transaction_count=transaction_count,
    )

    _serialise_itemset_columns(frequent_itemsets, all_rules, degradation_rules)

    rule_quality_plot_index = build_association_rule_plot_index(
        degradation_rules,
        top_n=TOP_RULES_FOR_BUBBLE_FIGURE,
        label_n=LABEL_RULES_FOR_BUBBLE_FIGURE,
        transaction_count=transaction_count,
    )
    labeled_rule_quality_plot_index = rule_quality_plot_index[
        rule_quality_plot_index["is_labeled_in_bubble_plot"].astype(bool)
    ].copy()
    rule_quality_plot_index_summary = _build_rule_quality_plot_index_summary(
        rule_quality_plot_index=rule_quality_plot_index,
        transaction_count=transaction_count,
    )

    outputs: dict[str, Path] = {
        "frequent_itemsets": write_csv(
            frequent_itemsets, results_path / "frequent_itemsets.csv"
        ),
        "all_rules": write_csv(all_rules, results_path / "all_association_rules.csv"),
        "degradation_rules": write_csv(
            degradation_rules, results_path / "degradation_rules.csv"
        ),
        "rule_quality_summary": write_csv(
            quality_summary, results_path / "rule_quality_summary.csv"
        ),
        "association_item_support": write_csv(
            item_support, results_path / "association_item_support.csv"
        ),
        "association_item_definitions": write_csv(
            item_definitions, results_path / "association_item_definitions.csv"
        ),
        "association_transaction_summary": write_csv(
            transaction_summary, results_path / "association_transaction_summary.csv"
        ),
        "rule_selection_summary": write_csv(
            rule_selection_summary, results_path / "rule_selection_summary.csv"
        ),
        "rule_quality_plot_index": write_csv(
            rule_quality_plot_index, results_path / "rule_quality_plot_index.csv"
        ),
        "rule_quality_labeled_rules": write_csv(
            labeled_rule_quality_plot_index,
            results_path / "rule_quality_labeled_rules.csv",
        ),
        "rule_quality_plot_index_summary": write_csv(
            rule_quality_plot_index_summary,
            results_path / "rule_quality_plot_index_summary.csv",
        ),
        "rule_quality_bubble_plot": plot_rule_quality_bubble_plot(
            degradation_rules,
            figures_path / "rule_quality_bubble_plot.pdf",
            top_n=TOP_RULES_FOR_BUBBLE_FIGURE,
            label_n=LABEL_RULES_FOR_BUBBLE_FIGURE,
        ),
        "top_rules_by_lift": plot_top_association_rules(
            degradation_rules,
            figures_path / "top_rules_by_lift.pdf",
            top_n=TOP_RULES_FOR_FIGURE,
        ),
    }

    return outputs
