from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import PreprocessingConfig
from src.utils.geo import bearing, haversine
from src.utils.logger import create_logger

logger = create_logger(__name__)


class DataPreprocessor:
    """Cleans raw GPS data and derives kinematic features (SOG, COG, ROT, acceleration)."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

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

        lat_lo, lat_hi = self.config.lat_range
        lon_lo, lon_hi = self.config.lon_range
        df = df[df["LAT"].between(lat_lo, lat_hi) & df["LON"].between(lon_lo, lon_hi)]

        if self.config.drop_zero_coords:
            df = df[(df["LAT"] != 0) & (df["LON"] != 0)]

        df["signaldate"] = pd.to_datetime(df["signaldate"])
        df = df.drop_duplicates(subset=["signaldate"])
        df = df.sort_values("signaldate").reset_index(drop=True)

        logger.info("Cleaned: %d -> %d rows (dropped %d)", n_before, len(df), n_before - len(df))
        return df

    # ── Kinematics ───────────────────────────────────────

    def _compute_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        lat = df["LAT"].values
        lon = df["LON"].values
        time = df["signaldate"]

        n = len(df)

        # Time delta in seconds
        dt = np.empty(n)
        dt[0] = np.nan
        dt[1:] = time.diff().dt.total_seconds().values[1:]

        # Distance between consecutive points
        dist = np.empty(n)
        dist[0] = np.nan
        dist[1:] = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])

        # Speed Over Ground
        with np.errstate(divide="ignore", invalid="ignore"):
            sog_ms = dist / dt
        sog_knots = sog_ms * 1.943844

        # Course Over Ground
        cog = np.empty(n)
        cog[0] = np.nan
        cog[1:] = bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])

        # Change in course (normalised to -180..180)
        dcog = np.empty(n)
        dcog[:2] = np.nan
        dcog[2:] = (cog[2:] - cog[1:-1] + 180) % 360 - 180

        # Rate of Turn (degrees per second)
        with np.errstate(divide="ignore", invalid="ignore"):
            rot = dcog / dt

        # Acceleration (m/s^2)
        dsog = np.empty(n)
        dsog[:2] = np.nan
        dsog[2:] = sog_ms[2:] - sog_ms[1:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = dsog / dt

        # Gap detection: large time gaps indicate separate tracks/datasets
        gap_threshold_s = 300  # 5 minutes
        is_gap = np.zeros(n, dtype=bool)
        is_gap[0] = True
        is_gap[1:] = dt[1:] > gap_threshold_s

        # Reset kinematics at gaps (they're meaningless across datasets)
        gap_indices = np.where(is_gap)[0]
        for gi in gap_indices:
            sog_ms[gi] = np.nan
            sog_knots[gi] = np.nan
            cog[gi] = np.nan
            dist[gi] = np.nan
            if gi < n:
                dcog[gi] = np.nan
                rot[gi] = np.nan
                acc[gi] = np.nan

        df["dt"] = dt
        df["distance_m"] = dist
        df["is_gap"] = is_gap.astype(int)
        df["sog_ms"] = sog_ms
        df["sog_knots"] = sog_knots
        df["cog_deg"] = cog
        df["dcog_deg"] = dcog
        df["rot_deg_s"] = rot
        df["acceleration_ms2"] = acc

        if self.config.apply_smoothing:
            w = self.config.smoothing_window
            df["LAT_smooth"] = df["LAT"].rolling(window=w, center=True, min_periods=1).mean()
            df["LON_smooth"] = df["LON"].rolling(window=w, center=True, min_periods=1).mean()

        logger.info(
            "Kinematics computed: SOG range [%.2f, %.2f] knots",
            np.nanmin(sog_knots),
            np.nanmax(sog_knots),
        )
        return df
