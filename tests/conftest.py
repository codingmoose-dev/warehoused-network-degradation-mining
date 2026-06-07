from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from network_degradation_mining.io import write_csv  # noqa: E402
from network_degradation_mining.warehouse import build_warehouse_tables  # noqa: E402


@pytest.fixture()
def sample_source_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "measurement_id": ["m1", "m2", "m3", "m4", "m5", "m6"],
            "source_row_number": [1, 2, 3, 4, 5, 6],
            "session_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "tower_id": ["t1", "t1", "t1", "t2", "t2", "t2"],
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="min"),
            "session_step_index": [1, 2, 3, 1, 2, 3],
            "session_record_count": [3, 3, 3, 3, 3, 3],
            "deployment_area": ["A", "A", "A", "B", "B", "B"],
            "area_type": ["Urban", "Urban", "Urban", "Rural", "Rural", "Rural"],
            "ue_profile": ["phone", "phone", "phone", "phone", "phone", "phone"],
            "ue_capability_class": ["mid", "mid", "mid", "high", "high", "high"],
            "network_type": ["5G", "5G", "5G", "4G", "4G", "4G"],
            "operator_profile": ["op_a", "op_a", "op_a", "op_b", "op_b", "op_b"],
            "infrastructure_profile": [
                "macro",
                "macro",
                "macro",
                "micro",
                "micro",
                "micro",
            ],
            "propagation_model": ["umi", "umi", "umi", "uma", "uma", "uma"],
            "propagation_scenario": ["los", "los", "nlos", "los", "nlos", "nlos"],
            "los_state": ["LOS", "LOS", "NLOS", "LOS", "NLOS", "NLOS"],
            "carrier_frequency_ghz": [3.5, 3.5, 3.5, 2.1, 2.1, 2.1],
            "band": ["n78", "n78", "n78", "b3", "b3", "b3"],
            "app_type": [
                "Streaming",
                "Streaming",
                "Gaming",
                "Browse",
                "Browse",
                "Gaming",
            ],
            "movement_speed": [
                "Static",
                "Walking",
                "Walking",
                "Driving",
                "Driving",
                "Static",
            ],
            "weather": ["Clear", "Clear", "Rain", "Clear", "Rain", "Rain"],
            "obstruction_level": ["Low", "Low", "High", "Medium", "High", "High"],
            "is_indoor": [False, False, True, False, True, True],
            "congestion_level": ["Low", "High", "High", "Low", "High", "High"],
            "tower_load": ["Low", "High", "High", "Low", "High", "High"],
            "video_quality_label": ["Good", "Poor", "Good", "Good", "Good", "Good"],
            "vonr_enabled": [True, True, False, False, False, False],
            "anomalous": [False, False, False, False, False, True],
            "dropped_connection": [False, False, False, False, True, False],
            "synthetic_latitude": [23.0, 23.0, 23.0, 24.0, 24.0, 24.0],
            "synthetic_longitude": [90.0, 90.0, 90.0, 91.0, 91.0, 91.0],
            "signal_strength_dbm": [-82, -101, -110, -88, -115, -118],
            "los_probability": [0.8, 0.7, 0.3, 0.8, 0.2, 0.1],
            "distance_2d_m": [100, 150, 300, 120, 500, 700],
            "distance_3d_m": [101, 151, 301, 121, 501, 701],
            "breakpoint_distance_m": [200, 200, 200, 200, 200, 200],
            "path_loss_db": [90, 96, 110, 92, 115, 118],
            "deterministic_path_loss_db": [88, 94, 105, 91, 111, 116],
            "shadow_fading_db": [1, 1, 2, 1, 2, 2],
            "fast_fading_db": [0.1, 0.2, 0.4, 0.1, 0.5, 0.5],
            "obstruction_penalty_db": [0, 0, 6, 2, 6, 6],
            "weather_penalty_db": [0, 0, 2, 0, 2, 2],
            "mobility_penalty_db": [0, 1, 1, 3, 3, 0],
            "indoor_penalty_db": [0, 0, 5, 0, 5, 5],
            "contextual_penalty_db": [0, 1, 14, 5, 16, 13],
            "effective_tx_power_dbm": [23, 23, 23, 23, 23, 23],
            "signal_strength_unclipped_dbm": [-82, -101, -110, -88, -115, -118],
            "rsrp_clipped": [False, False, False, False, False, False],
            "link_capacity_downlink_mbps": [200, 140, 80, 120, 50, 40],
            "offered_downlink_mbps": [20, 100, 60, 15, 80, 90],
            "download_speed_mbps": [18, 20, 15, 14, 10, 12],
            "link_capacity_upload_mbps": [80, 70, 40, 60, 20, 20],
            "offered_upload_mbps": [5, 15, 10, 5, 15, 15],
            "upload_speed_mbps": [4.5, 10, 6, 4, 5, 5],
            "latency_ms": [30, 180, 220, 40, 250, 260],
            "jitter_ms": [0.2, 1.5, 1.8, 0.2, 2.0, 2.2],
            "ping_ms": [30, 180, 220, 40, 250, 260],
            "battery_level_percent": [80, 79, 78, 90, 89, 88],
            "temperature_c": [32, 32, 32, 31, 31, 31],
            "connected_duration_min": [1, 2, 3, 1, 2, 3],
            "interval_handover_count": [0, 1, 1, 0, 2, 1],
            "cumulative_handover_count": [0, 1, 2, 0, 2, 3],
            "activity_factor": [0.7, 0.9, 0.9, 0.6, 0.9, 0.8],
            "data_usage_mb": [10, 20, 20, 5, 25, 25],
            "distance_to_tower_km": [0.1, 0.15, 0.3, 0.12, 0.5, 0.7],
            "video_quality": [4, 1, 4, 4, 4, 4],
            "timestamp_parse_failed": [False, False, False, False, False, False],
        }
    )


@pytest.fixture()
def sample_warehouse_dir(tmp_path: Path, sample_source_frame: pd.DataFrame) -> Path:
    input_path = tmp_path / "synnetqos_core_clean.csv"
    warehouse_dir = tmp_path / "warehouse"
    results_dir = tmp_path / "warehouse_results"
    write_csv(sample_source_frame, input_path)
    build_warehouse_tables(input_path, warehouse_dir, results_dir)
    return warehouse_dir
