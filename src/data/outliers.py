from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import OutlierConfig
from src.utils.geo import haversine
from src.utils.logger import create_logger

logger = create_logger(__name__)


class OutlierDetector:
    """Detects GPS outliers (position jumps / signal errors). Flags but does not remove."""

    def __init__(self, config: OutlierConfig | None = None) -> None:
        self.config = config or OutlierConfig()

    def detect(self, df: pd.DataFrame) -> pd.Series:
        flags = pd.Series(False, index=df.index)

        if "sog_ms" in df.columns:
            max_speed_ms = self.config.max_speed_knots * 0.514444
            speed_outlier = df["sog_ms"] > max_speed_ms
            flags |= speed_outlier
            logger.info(
                "Physical speed check (>%.0f kn): %d outliers",
                self.config.max_speed_knots,
                speed_outlier.sum(),
            )

        if "acceleration_ms2" in df.columns:
            acc_outlier = self._acceleration_outlier(df["acceleration_ms2"])
            flags |= acc_outlier

        method = self.config.method
        for col in self.config.columns:
            if col not in df.columns:
                continue
            if method in ("iqr", "combined"):
                flags |= self._iqr(df[col])
            if method in ("zscore", "combined"):
                flags |= self._zscore(df[col])
            if method in ("isolation_forest", "combined"):
                flags |= self._isolation_forest(df[[col]])

        pos_outliers = detect_position_outliers(df)
        flags |= pd.Series(pos_outliers, index=df.index)

        logger.info(
            "Total outliers flagged: %d / %d (%.2f%%)",
            flags.sum(),
            len(df),
            100 * flags.mean(),
        )
        return flags

    def _acceleration_outlier(self, series: pd.Series) -> pd.Series:
        abs_acc = series.abs()
        threshold = abs_acc.quantile(0.999)
        mask = (abs_acc > threshold) & series.notna()
        logger.info("Acceleration outlier check: %d flagged (threshold=%.4f m/s²)", mask.sum(), threshold)
        return mask

    def _iqr(self, series: pd.Series) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.config.iqr_multiplier * iqr
        upper = q3 + self.config.iqr_multiplier * iqr
        mask = ~series.between(lower, upper) & series.notna()
        logger.info(
            "IQR on %s: %d outliers (bounds [%.4f, %.4f])",
            series.name,
            mask.sum(),
            lower,
            upper,
        )
        return mask

    def _zscore(self, series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(False, index=series.index)
        z = ((series - mean) / std).abs()
        mask = (z > self.config.zscore_threshold) & series.notna()
        logger.info(
            "Z-score on %s: %d outliers (threshold=%.1f)",
            series.name,
            mask.sum(),
            self.config.zscore_threshold,
        )
        return mask

    def _isolation_forest(self, df: pd.DataFrame) -> pd.Series:
        from sklearn.ensemble import IsolationForest

        clean = df.dropna()
        if len(clean) < 10:
            return pd.Series(False, index=df.index)

        iso = IsolationForest(contamination=self.config.contamination, random_state=42)
        preds = iso.fit_predict(clean.values)
        mask = pd.Series(False, index=df.index)
        mask.loc[clean.index] = preds == -1
        logger.info(
            "Isolation Forest: %d outliers (contamination=%.3f)",
            mask.sum(),
            self.config.contamination,
        )
        return mask


def detect_teleports(
    df: pd.DataFrame,
    max_speed_knots: float = 35.0,
    max_jump_m: float = 2000.0,
) -> np.ndarray:
    """Detect position teleportation — impossible jumps between consecutive points.

    Returns boolean array where True = teleported point.
    """
    lat = df["LAT"].values
    lon = df["LON"].values
    n = len(df)
    is_teleport = np.zeros(n, dtype=bool)

    if n < 3:
        return is_teleport

    dt = np.ones(n) * 60.0
    if "dt" in df.columns:
        dt_vals = df["dt"].values
        valid = ~np.isnan(dt_vals) & (dt_vals > 0)
        dt[valid] = dt_vals[valid]

    for i in range(1, n):
        if np.isnan(lat[i]) or np.isnan(lon[i]):
            is_teleport[i] = True
            continue

        dist = haversine(
            np.array([lat[i - 1]]), np.array([lon[i - 1]]),
            np.array([lat[i]]), np.array([lon[i]]),
        )[0]

        if dist > max_jump_m and dt[i] < 120:
            is_teleport[i] = True
            continue

        max_dist = max_speed_knots * 0.514444 * max(dt[i], 1.0)
        if dist > max_dist * 1.5:
            is_teleport[i] = True

    flagged = is_teleport.sum()
    if flagged > 0:
        logger.info("Teleport detection: %d points flagged (max_speed=%.0f kn)", flagged, max_speed_knots)
    return is_teleport


def fix_teleports(df: pd.DataFrame, teleport_mask: np.ndarray) -> pd.DataFrame:
    """Replace teleported positions with linear interpolation from valid neighbors."""
    if not teleport_mask.any():
        return df

    df = df.copy()
    lat = df["LAT"].values.copy()
    lon = df["LON"].values.copy()

    valid = ~teleport_mask & ~np.isnan(lat)
    valid_idx = np.where(valid)[0]

    if len(valid_idx) < 2:
        return df

    for i in np.where(teleport_mask)[0]:
        left = valid_idx[valid_idx < i]
        right = valid_idx[valid_idx > i]

        if len(left) > 0 and len(right) > 0:
            li, ri = left[-1], right[0]
            frac = (i - li) / (ri - li)
            lat[i] = lat[li] + frac * (lat[ri] - lat[li])
            lon[i] = lon[li] + frac * (lon[ri] - lon[li])
        elif len(left) > 0:
            lat[i] = lat[left[-1]]
            lon[i] = lon[left[-1]]
        elif len(right) > 0:
            lat[i] = lat[right[0]]
            lon[i] = lon[right[0]]

    df["LAT"] = lat
    df["LON"] = lon

    logger.info("Fixed %d teleported positions via interpolation", teleport_mask.sum())
    return df


def detect_position_outliers(
    df: pd.DataFrame,
    window: int = 21,
    deviation_factor: float = 5.0,
    min_threshold_m: float = 500.0,
    jump_return_m: float = 300.0,
) -> np.ndarray:
    """Detect GPS outliers from trajectory consistency.

    Two checks:
    1. Rolling median deviation — point far from its surrounding median position.
    2. Jump-and-return — point jumps away from both neighbors while neighbors are close.
    """
    lat = df["LAT"].values.astype(float)
    lon = df["LON"].values.astype(float)
    n = len(df)
    is_outlier = np.zeros(n, dtype=bool)

    if n < 5:
        return is_outlier

    s = pd.Series(lat)
    rolling_lat = s.rolling(window, center=True, min_periods=3).median().values
    rolling_lon = pd.Series(lon).rolling(window, center=True, min_periods=3).median().values

    valid = ~(np.isnan(rolling_lat) | np.isnan(rolling_lon) | np.isnan(lat) | np.isnan(lon))
    if not valid.any():
        return is_outlier

    dist_to_median = np.full(n, np.nan)
    idx = np.where(valid)[0]
    dist_to_median[idx] = haversine(lat[idx], lon[idx], rolling_lat[idx], rolling_lon[idx])

    lat_spread = pd.Series(lat).rolling(window, center=True, min_periods=3).std().values * 111_000
    lon_factor = np.cos(np.radians(np.nanmedian(lat)))
    lon_spread = pd.Series(lon).rolling(window, center=True, min_periods=3).std().values * 111_000 * lon_factor
    local_spread = np.sqrt(np.nan_to_num(lat_spread, nan=0) ** 2 + np.nan_to_num(lon_spread, nan=0) ** 2)

    threshold = np.maximum(local_spread * deviation_factor, min_threshold_m)
    median_outlier = valid & (dist_to_median > threshold)
    is_outlier |= median_outlier

    if median_outlier.any():
        logger.info("Position median deviation: %d outliers", median_outlier.sum())

    for i in range(1, n - 1):
        if np.isnan(lat[i]) or np.isnan(lat[i - 1]) or np.isnan(lat[i + 1]):
            continue
        d_prev = haversine(
            np.array([lat[i - 1]]), np.array([lon[i - 1]]),
            np.array([lat[i]]), np.array([lon[i]]),
        )[0]
        d_next = haversine(
            np.array([lat[i]]), np.array([lon[i]]),
            np.array([lat[i + 1]]), np.array([lon[i + 1]]),
        )[0]
        d_neighbors = haversine(
            np.array([lat[i - 1]]), np.array([lon[i - 1]]),
            np.array([lat[i + 1]]), np.array([lon[i + 1]]),
        )[0]

        if (
            d_prev > jump_return_m
            and d_next > jump_return_m
            and d_neighbors < min(d_prev, d_next) * 0.5
        ):
            is_outlier[i] = True

    jump_count = is_outlier.sum() - median_outlier.sum()
    if jump_count > 0:
        logger.info("Jump-and-return detection: %d outliers", jump_count)

    return is_outlier


def detect_land_outliers(
    df: pd.DataFrame,
    coastal_tolerance_km: float = 5.0,
    n_directions: int = 16,
) -> np.ndarray:
    """Flag GPS points deep inland using global land mask.

    Points on land are only flagged if no water cell is found within coastal_tolerance_km
    in any direction. Coastal/port points pass through.
    """
    try:
        from global_land_mask import globe
    except ImportError:
        logger.warning("global-land-mask not installed, skipping land outlier detection")
        return np.zeros(len(df), dtype=bool)

    lat = df["LAT"].values.astype(float)
    lon = df["LON"].values.astype(float)
    n = len(df)
    is_outlier = np.zeros(n, dtype=bool)

    valid = ~(np.isnan(lat) | np.isnan(lon))
    if not valid.any():
        return is_outlier

    on_land = np.zeros(n, dtype=bool)
    v_idx = np.where(valid)[0]
    on_land[v_idx] = globe.is_land(lat[v_idx], lon[v_idx])

    land_idx = np.where(on_land)[0]
    if len(land_idx) == 0:
        return is_outlier

    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    radius_deg = coastal_tolerance_km / 111.0

    unique_cells = {}
    precision = 3
    for i in land_idx:
        key = (round(lat[i], precision), round(lon[i], precision))
        if key not in unique_cells:
            unique_cells[key] = []
        unique_cells[key].append(i)

    inland_count = 0
    for (clat, clon), indices in unique_cells.items():
        cos_lat = max(np.cos(np.radians(clat)), 0.1)
        is_coastal = False
        for angle in angles:
            check_lat = clat + radius_deg * np.sin(angle)
            check_lon = clon + radius_deg * np.cos(angle) / cos_lat
            if not globe.is_land(check_lat, check_lon):
                is_coastal = True
                break
        if not is_coastal:
            for i in indices:
                is_outlier[i] = True
            inland_count += len(indices)

    if inland_count > 0:
        logger.info(
            "Land outlier detection: %d points deep inland (>%gkm from water)",
            inland_count,
            coastal_tolerance_km,
        )
    return is_outlier
