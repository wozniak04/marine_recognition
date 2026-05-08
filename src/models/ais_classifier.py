from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import create_logger

logger = create_logger(__name__)

EXTENDED_STATES = [
    "port_stay",
    "port_maneuver",
    "voyage",
    "anchor",
    "adrift",
    "turn",
    "shifting",
]

AIS_TO_BASIC = {
    "port_stay": "port_stay",
    "port_maneuver": "port_stay",
    "voyage": "voyage",
    "anchor": "anchor",
    "adrift": "adrift",
    "turn": "voyage",
    "shifting": "port_stay",
}


class AISRuleClassifier:
    """Rule-based classifier for AIS data with extended states and port awareness."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.voyage_speed_kn: float = p.get("voyage_speed_kn", 3.0)
        self.maneuver_speed_kn: float = p.get("maneuver_speed_kn", 2.0)
        self.stationary_speed_kn: float = p.get("stationary_speed_kn", 0.5)
        self.turn_dcog_threshold: float = p.get("turn_dcog_threshold", 30.0)
        self.drift_angle_threshold: float = p.get("drift_angle_threshold", 15.0)
        self.shifting_speed_min_kn: float = p.get("shifting_speed_min_kn", 1.0)
        self.shifting_speed_max_kn: float = p.get("shifting_speed_max_kn", 5.0)
        self.anchor_spread_max_m: float = p.get("anchor_spread_max_m", 200.0)

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        states = np.empty(n, dtype=object)
        confidences = np.zeros(n)

        sog = df["sog_knots"].values
        dcog = np.abs(df["dcog_deg"].fillna(0).values)
        in_port = df["in_port"].values if "in_port" in df.columns else np.zeros(n, dtype=bool)
        drift_angle = np.abs(df["drift_angle_deg"].fillna(0).values) if "drift_angle_deg" in df.columns else np.zeros(n)

        for i in range(n):
            s = sog[i]
            dc = dcog[i]
            da = drift_angle[i]
            ip = in_port[i]

            if ip and s < self.stationary_speed_kn:
                states[i] = "port_stay"
                confidences[i] = min(95.0, 50 + (self.stationary_speed_kn - s) * 80)
            elif ip and self.stationary_speed_kn <= s <= self.maneuver_speed_kn:
                states[i] = "port_maneuver"
                confidences[i] = 60 + min(30, s * 10)
            elif s >= self.voyage_speed_kn:
                if dc > self.turn_dcog_threshold:
                    states[i] = "turn"
                    confidences[i] = min(95.0, 50 + dc)
                else:
                    states[i] = "voyage"
                    confidences[i] = min(95.0, 50 + s * 5)
            elif self.shifting_speed_min_kn <= s <= self.shifting_speed_max_kn and ip:
                states[i] = "shifting"
                confidences[i] = 55 + min(35, s * 10)
            elif s < self.stationary_speed_kn:
                if ip:
                    states[i] = "port_stay"
                    confidences[i] = min(90.0, 50 + (self.stationary_speed_kn - s) * 80)
                else:
                    states[i] = "anchor"
                    confidences[i] = min(90.0, 50 + (self.stationary_speed_kn - s) * 80)
            elif da > self.drift_angle_threshold:
                states[i] = "adrift"
                confidences[i] = min(90.0, 50 + da)
            else:
                states[i] = "adrift"
                confidences[i] = 40.0

        df["predicted_state"] = states
        df["predicted_state_basic"] = [AIS_TO_BASIC[s] for s in states]
        df["confidence"] = confidences
        return df
