from __future__ import annotations

import pandas as pd

from network_degradation_mining import decision_tree_rules, supervised

TARGET_DERIVED_COLUMNS = {
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


def test_supervised_model_feature_selection_excludes_target_derived_columns() -> None:
    frame = pd.DataFrame(
        {
            column: [0, 1, 0, 1]
            for column in TARGET_DERIVED_COLUMNS
        }
    )
    frame["network_type"] = ["4G", "5G", "4G", "5G"]
    frame["signal_strength_dbm"] = [-90, -100, -91, -101]

    numeric, categorical = supervised._feature_columns(frame)
    selected = set(numeric) | set(categorical)

    assert selected.isdisjoint(TARGET_DERIVED_COLUMNS)
    assert "signal_strength_dbm" in selected
    assert "network_type" in selected


def test_decision_tree_rule_feature_selection_excludes_target_derived_columns() -> None:
    frame = pd.DataFrame(
        {
            column: [0, 1, 0, 1]
            for column in TARGET_DERIVED_COLUMNS
        }
    )
    frame["network_type"] = ["4G", "5G", "4G", "5G"]
    frame["signal_strength_dbm"] = [-90, -100, -91, -101]

    numeric, categorical = decision_tree_rules._feature_columns(
        frame, excluded_features=set(decision_tree_rules.BASE_EXCLUDED_COLUMNS)
    )
    selected = set(numeric) | set(categorical)

    assert selected.isdisjoint(TARGET_DERIVED_COLUMNS)
    assert "signal_strength_dbm" in selected
    assert "network_type" in selected
