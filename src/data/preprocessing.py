from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from src.utils.config import PreprocessingConfig, ProjectConfig
from src.utils.geo import bearing, haversine
from src.utils.logger import create_logger

logger = create_logger(__name__)


class DataPreprocessor:
    """Cleans raw GPS data and derives kinematic features (SOG, COG, ROT, acceleration)."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.prep_config = config.preprocessing if hasattr(config, "preprocessing") else PreprocessingConfig()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.copy()
            .pipe(self._clean)
            .pipe(self._compute_kinematics)
        )

    # ── Cleaning ─────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)

        df = df.dropna(subset=["LAT", "LON"]).reset_index(drop=True)

        lat_lo, lat_hi = self.prep_config.lat_range
        lon_lo, lon_hi = self.prep_config.lon_range
        df = df[df["LAT"].between(lat_lo, lat_hi) & df["LON"].between(lon_lo, lon_hi)]

        if self.prep_config.drop_zero_coords:
            df = df[(df["LAT"] != 0) & (df["LON"] != 0)]

        df["signaldate"] = pd.to_datetime(df["signaldate"])
        df = df.drop_duplicates(subset=["signaldate"])
        df = df.sort_values("signaldate").reset_index(drop=True)

        logger.info("Cleaned: %d -> %d rows (dropped %d)", n_before, len(df), n_before - len(df))
        return df

    # ── Kinematics ───────────────────────────────────────

    def _get_dt(self, df: pd.DataFrame) -> np.ndarray:
        if "dt" in df.columns: return df["dt"].values
        dt = np.empty(len(df))
        dt[0] = np.nan
        dt[1:] = df["signaldate"].diff().dt.total_seconds().values[1:]
        df["dt"] = dt
        return dt

    def _get_is_gap(self, df: pd.DataFrame) -> np.ndarray:
        if "is_gap" in df.columns: return df["is_gap"].values
        dt = self._get_dt(df)
        is_gap = np.zeros(len(df), dtype=bool)
        is_gap[0] = True
        is_gap[1:] = dt[1:] > 300  # 5 minutes
        is_gap_int = is_gap.astype(int)
        df["is_gap"] = is_gap_int
        return is_gap_int

    def _get_distance_m(self, df: pd.DataFrame) -> np.ndarray:
        if "distance_m" in df.columns: return df["distance_m"].values
        lat = df["LAT"].values
        lon = df["LON"].values
        dist = np.empty(len(df))
        dist[0] = np.nan
        dist[1:] = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])

        is_gap = self._get_is_gap(df).astype(bool)
        dist[is_gap] = np.nan
        df["distance_m"] = dist
        return dist

    def _get_sog_ms(self, df: pd.DataFrame) -> np.ndarray:
        if "sog_ms" in df.columns: return df["sog_ms"].values
        dist = self._get_distance_m(df)
        dt = self._get_dt(df)
        with np.errstate(divide="ignore", invalid="ignore"):
            sog_ms = dist / dt

        is_gap = self._get_is_gap(df).astype(bool)
        sog_ms[is_gap] = np.nan
        df["sog_ms"] = sog_ms
        return sog_ms

    def _get_sog_knots(self, df: pd.DataFrame) -> np.ndarray:
        if "sog_knots" in df.columns: return df["sog_knots"].values
        sog_ms = self._get_sog_ms(df)
        sog_knots = sog_ms * 1.943844
        df["sog_knots"] = sog_knots
        return sog_knots

    def _get_cog_deg(self, df: pd.DataFrame) -> np.ndarray:
        if "cog_deg" in df.columns: return df["cog_deg"].values
        lat = df["LAT"].values
        lon = df["LON"].values
        cog = np.empty(len(df))
        cog[0] = np.nan
        cog[1:] = bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])

        is_gap = self._get_is_gap(df).astype(bool)
        cog[is_gap] = np.nan
        df["cog_deg"] = cog
        return cog

    def _get_dcog_deg(self, df: pd.DataFrame) -> np.ndarray:
        if "dcog_deg" in df.columns: return df["dcog_deg"].values
        lat = df["LAT"].values
        lon = df["LON"].values
        raw_cog = np.empty(len(df))
        raw_cog[0] = np.nan
        raw_cog[1:] = bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])

        dcog = np.empty(len(df))
        dcog[:2] = np.nan
        dcog[2:] = (raw_cog[2:] - raw_cog[1:-1] + 180) % 360 - 180

        is_gap = self._get_is_gap(df).astype(bool)
        dcog[is_gap] = np.nan
        df["dcog_deg"] = dcog
        return dcog

    def _get_rot_deg_s(self, df: pd.DataFrame) -> np.ndarray:
        if "rot_deg_s" in df.columns: return df["rot_deg_s"].values
        dcog = self._get_dcog_deg(df)
        dt = self._get_dt(df)
        with np.errstate(divide="ignore", invalid="ignore"):
            rot = dcog / dt

        is_gap = self._get_is_gap(df).astype(bool)
        rot[is_gap] = np.nan
        df["rot_deg_s"] = rot
        return rot

    def _get_acceleration_ms2(self, df: pd.DataFrame) -> np.ndarray:
        if "acceleration_ms2" in df.columns: return df["acceleration_ms2"].values
        lat = df["LAT"].values
        lon = df["LON"].values
        dist = np.empty(len(df))
        dist[0] = np.nan
        dist[1:] = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        
        dt = self._get_dt(df)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_sog_ms = dist / dt
            
        dsog = np.empty(len(df))
        dsog[:2] = np.nan
        dsog[2:] = raw_sog_ms[2:] - raw_sog_ms[1:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = dsog / dt
            
        is_gap = self._get_is_gap(df).astype(bool)
        acc[is_gap] = np.nan
        df["acceleration_ms2"] = acc
        return acc

    def _compute_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            feature_cols = self.config.model.params.get("feature_cols", [])
        except AttributeError:
            feature_cols = []

        dispatch = {
            "dt": self._get_dt,
            "is_gap": self._get_is_gap,
            "distance_m": self._get_distance_m,
            "sog_ms": self._get_sog_ms,
            "sog_knots": self._get_sog_knots,
            "cog_deg": self._get_cog_deg,
            "dcog_deg": self._get_dcog_deg,
            "rot_deg_s": self._get_rot_deg_s,
            "acceleration_ms2": self._get_acceleration_ms2,
        }

        # Jeśli nie ma zdefiniowanych features w YAML, zabezpieczamy się wykonaniem wszystkich
        if not feature_cols:
            feature_cols = list(dispatch.keys())

        for feature in feature_cols:
            if feature in dispatch:
                dispatch[feature](df)

        if self.prep_config.apply_smoothing:
            w = self.prep_config.smoothing_window
            df["LAT_smooth"] = df["LAT"].rolling(window=w, center=True, min_periods=1).mean()
            df["LON_smooth"] = df["LON"].rolling(window=w, center=True, min_periods=1).mean()

        if "sog_knots" in df.columns:
            logger.info(
                "Kinematics computed: SOG range [%.2f, %.2f] knots",
                df["sog_knots"].min(skipna=True),
                df["sog_knots"].max(skipna=True),
            )
        else:
            logger.info("Kinematics computed selectively based on YAML.")

        return df
