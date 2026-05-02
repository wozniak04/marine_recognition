from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.geo import haversine
from src.utils.logger import create_logger

logger = create_logger(__name__)


class AISPreprocessor:
    """Preprocess AIS data — SOG/COG/HDG come directly from transponder."""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(subset=["LAT", "LON"]).reset_index(drop=True)
        df = df.sort_values("signaldate").reset_index(drop=True)

        n = len(df)

        dt = np.empty(n)
        dt[0] = np.nan
        dt[1:] = df["signaldate"].diff().dt.total_seconds().values[1:]

        dist = np.empty(n)
        dist[0] = np.nan
        lat, lon = df["LAT"].values, df["LON"].values
        dist[1:] = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])

        cog = df["cog_deg"].values.astype(float)
        dcog = np.empty(n)
        dcog[:2] = np.nan
        dcog[2:] = (cog[2:] - cog[1:-1] + 180) % 360 - 180

        hdg = df["hdg_deg"].values.astype(float) if "hdg_deg" in df.columns else cog.copy()
        drift_angle = np.empty(n)
        drift_angle[0] = np.nan
        drift_angle[1:] = (cog[1:] - hdg[1:] + 180) % 360 - 180

        sog = df["sog_knots"].values.astype(float)
        sog_ms = sog / 1.943844

        with np.errstate(divide="ignore", invalid="ignore"):
            rot = dcog / dt

        dsog = np.empty(n)
        dsog[:2] = np.nan
        dsog[2:] = sog_ms[2:] - sog_ms[1:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = dsog / dt

        gap_threshold_s = 1800
        is_gap = np.zeros(n, dtype=bool)
        is_gap[0] = True
        is_gap[1:] = dt[1:] > gap_threshold_s

        gap_indices = np.where(is_gap)[0]
        for gi in gap_indices:
            dist[gi] = np.nan
            dcog[gi] = np.nan
            rot[gi] = np.nan
            acc[gi] = np.nan

        df["dt"] = dt
        df["distance_m"] = dist
        df["sog_ms"] = sog_ms
        df["dcog_deg"] = dcog
        df["rot_deg_s"] = rot
        df["acceleration_ms2"] = acc
        df["drift_angle_deg"] = drift_angle
        df["is_gap"] = is_gap.astype(int)

        logger.info(
            "AIS preprocessed: %d rows, SOG range [%.1f, %.1f] kn, %d gaps",
            n, np.nanmin(sog), np.nanmax(sog), is_gap.sum(),
        )
        return df
