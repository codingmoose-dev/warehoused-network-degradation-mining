from __future__ import annotations

import pandas as pd

from network_degradation_mining.features import add_degradation_labels


def test_degradation_labels_flag_expected_conditions() -> None:
    frame = pd.DataFrame(
        {
            "download_speed_mbps": [90, 10, 80, 70],
            "offered_downlink_mbps": [100, 100, 100, 100],
            "latency_ms": [30, 40, 200, 30],
            "jitter_ms": [0.2, 0.2, 0.2, 2.0],
            "app_type": ["Browse", "Browse", "Browse", "Browse"],
            "video_quality": [4, 4, 4, 4],
            "dropped_connection": [False, False, False, False],
            "anomalous": [False, False, False, False],
        }
    )

    labeled = add_degradation_labels(frame)

    assert labeled.loc[0, "service_degraded"] == 0
    assert labeled.loc[1, "downlink_service_shortfall"]
    assert labeled.loc[2, "high_latency"]
    assert labeled.loc[3, "high_jitter"]
    assert labeled["service_degraded"].sum() == 3
