from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.geo import haversine
from src.utils.logger import create_logger

logger = create_logger(__name__)

STATE_PORT = "port_stay"
STATE_VOYAGE = "voyage"
STATE_ANCHOR = "anchor"
STATE_ADRIFT = "adrift"

ALL_STATES = [STATE_PORT, STATE_VOYAGE, STATE_ANCHOR, STATE_ADRIFT]


class ShipStateFSM:
    """Rule-based ship state classifier with port-aware logic."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.voyage_speed_kn: float = p.get("voyage_speed_kn", 3.0)
        self.adrift_speed_min_kn: float = p.get("adrift_speed_min_kn", 0.3)
        self.adrift_speed_max_kn: float = p.get("adrift_speed_max_kn", 3.0)
        self.stationary_speed_kn: float = p.get("stationary_speed_kn", 0.5)
        self.port_spread_max_m: float = p.get("port_spread_max_m", 15.0)
        self.anchor_spread_max_m: float = p.get("anchor_spread_max_m", 200.0)
        self.rolling_window: int = p.get("rolling_window", 30)
        self.min_episode_minutes: int = p.get("min_episode_minutes", 5)

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)

        gap_boundaries = self._find_gap_boundaries(df)
        df = self._compute_rolling_features(df, gap_boundaries)

        states, confidences = self._apply_rules(df)
        df["predicted_state"] = states
        df["confidence"] = confidences

        df["predicted_state"] = self._enforce_port_constraint(df)

        df["predicted_state"] = self._smooth_episodes(
            df["predicted_state"].values, gap_boundaries
        )

        return df

    def _find_gap_boundaries(self, df: pd.DataFrame) -> list[int]:
        if "is_gap" in df.columns:
            return df.index[df["is_gap"] == 1].tolist()
        return [0]

    def _compute_rolling_features(
        self, df: pd.DataFrame, gap_boundaries: list[int]
    ) -> pd.DataFrame:
        w = self.rolling_window

        segments = self._split_by_gaps(len(df), gap_boundaries)

        sog_col = df["sog_knots"].copy()
        rolling_sog = pd.Series(np.nan, index=df.index)
        for start, end in segments:
            seg = sog_col.iloc[start:end]
            rolling_sog.iloc[start:end] = (
                seg.rolling(window=w, center=True, min_periods=1).mean()
            )
        df["rolling_sog_kn"] = rolling_sog

        lat = df["LAT"].values
        lon = df["LON"].values
        n = len(df)
        spread = np.zeros(n)
        net_disp = np.zeros(n)
        half = w // 2

        for start, end in segments:
            for i in range(start, end):
                lo = max(start, i - half)
                hi = min(end, i + half + 1)
                if hi - lo < 2:
                    continue

                wlat = lat[lo:hi]
                wlon = lon[lo:hi]
                clat, clon = wlat.mean(), wlon.mean()
                dists = haversine(wlat, wlon, np.full(hi - lo, clat), np.full(hi - lo, clon))
                spread[i] = dists.max()

                net_disp[i] = haversine(
                    np.array([lat[lo]]), np.array([lon[lo]]),
                    np.array([lat[hi - 1]]), np.array([lon[hi - 1]]),
                )[0]

        df["rolling_spread_m"] = spread
        df["rolling_net_displacement_m"] = net_disp

        return df

    def _split_by_gaps(self, n: int, gap_boundaries: list[int]) -> list[tuple[int, int]]:
        boundaries = sorted(set(gap_boundaries) | {0})
        segments = []
        for idx, start in enumerate(boundaries):
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else n
            if end > start:
                segments.append((start, end))
        return segments

    def _apply_rules(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        states = np.empty(n, dtype=object)
        confidences = np.zeros(n)

        sog = df["rolling_sog_kn"].values
        spread = df["rolling_spread_m"].values
        net_disp = df["rolling_net_displacement_m"].values
        in_port = df["in_port"].values if "in_port" in df.columns else np.zeros(n)

        for i in range(n):
            state, conf = self._classify_point(sog[i], spread[i], net_disp[i], in_port[i])
            states[i] = state
            confidences[i] = conf

        return states, confidences

    def _classify_point(
        self,
        sog_kn: float,
        spread_m: float,
        net_disp_m: float,
        in_port: int,
    ) -> tuple[str, float]:
        if sog_kn >= self.voyage_speed_kn:
            excess = (sog_kn - self.voyage_speed_kn) / self.voyage_speed_kn
            conf = min(95.0, 70.0 + 25.0 * min(excess, 1.0))
            return STATE_VOYAGE, conf

        if self.adrift_speed_min_kn <= sog_kn <= self.adrift_speed_max_kn:
            if net_disp_m > spread_m * 0.3 and spread_m > self.anchor_spread_max_m:
                drift_signal = min(net_disp_m / max(spread_m, 1), 3.0) / 3.0
                conf = min(90.0, 55.0 + 35.0 * drift_signal)
                return STATE_ADRIFT, conf

        if sog_kn <= self.stationary_speed_kn:
            if spread_m <= self.port_spread_max_m:
                tightness = max(0, 1.0 - spread_m / self.port_spread_max_m)
                base = 75.0 if in_port else 60.0
                conf = min(95.0, base + 20.0 * tightness)
                return STATE_PORT, conf
            elif spread_m <= self.anchor_spread_max_m:
                anchor_signal = (spread_m - self.port_spread_max_m) / (
                    self.anchor_spread_max_m - self.port_spread_max_m
                )
                conf = min(90.0, 55.0 + 35.0 * min(anchor_signal, 1.0))
                return STATE_ANCHOR, conf

        if spread_m <= self.port_spread_max_m:
            return STATE_PORT, 45.0 if in_port else 40.0
        if spread_m <= self.anchor_spread_max_m:
            return STATE_ANCHOR, 40.0
        return STATE_ADRIFT, 35.0

    def _enforce_port_constraint(self, df: pd.DataFrame) -> np.ndarray:
        """Spread-based classification already handles port vs anchor distinction."""
        return df["predicted_state"].values.copy()

    def _smooth_episodes(
        self, states: np.ndarray, gap_boundaries: list[int]
    ) -> np.ndarray:
        result = states.copy()
        min_len = self.min_episode_minutes
        gap_set = set(gap_boundaries)

        i = 0
        while i < len(result):
            if i in gap_set and i > 0:
                i += 1
                continue

            j = i
            while j < len(result) and result[j] == result[i] and j not in gap_set:
                j += 1
            if j < len(result) and j in gap_set and result[j] == result[i]:
                while j < len(result) and result[j] == result[i]:
                    j += 1

            episode_len = j - i

            if episode_len < min_len and i > 0 and j < len(result):
                prev_state = result[i - 1]
                next_state = result[j]
                fill = prev_state if prev_state == next_state else prev_state
                result[i:j] = fill

            i = j

        return result
