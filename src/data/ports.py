from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.utils.config import PortConfig
from src.utils.logger import create_logger

logger = create_logger(__name__)

EARTH_RADIUS_M = 6_371_000


@dataclass
class PortMatch:
    in_port: bool
    locode: str | None
    distance_m: float


class PortMatcher:
    """Matches GPS positions to nearest port using BallTree with haversine metric."""

    def __init__(self, ports_df: pd.DataFrame, config: PortConfig | None = None) -> None:
        self.config = config or PortConfig()
        self.ports_df = ports_df.copy()

        coords_rad = np.radians(ports_df[["LAT", "LON"]].values)
        self._tree = BallTree(coords_rad, metric="haversine")
        self._locodes = ports_df["LOCODE"].values

        logger.info("PortMatcher built with %d ports (radius=%.0fm)", len(ports_df), self.config.radius_meters)

    def query_single(self, lat: float, lon: float) -> PortMatch:
        point = np.radians([[lat, lon]])
        dist_rad, idx = self._tree.query(point, k=1)
        dist_m = float(dist_rad[0, 0]) * EARTH_RADIUS_M
        locode = str(self._locodes[idx[0, 0]])
        return PortMatch(
            in_port=dist_m <= self.config.radius_meters,
            locode=locode if dist_m <= self.config.radius_meters else None,
            distance_m=dist_m,
        )

    def query_batch(self, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
        coords_rad = np.radians(np.column_stack([lats, lons]))
        dist_rad, idx = self._tree.query(coords_rad, k=1)
        dist_m = dist_rad[:, 0] * EARTH_RADIUS_M
        locodes = self._locodes[idx[:, 0]]
        in_port = dist_m <= self.config.radius_meters

        result = pd.DataFrame({
            "in_port": in_port.astype(int),
            "nearest_locode": locodes,
            "port_distance_m": dist_m,
        })
        result.loc[~in_port, "nearest_locode"] = None

        logger.info(
            "Port matching: %d/%d points in port (%.1f%%)",
            in_port.sum(),
            len(lats),
            100 * in_port.mean(),
        )
        return result


def discover_ports(
    df: pd.DataFrame,
    sog_threshold_kn: float = 0.5,
    min_duration_minutes: int = 30,
    max_spread_m: float = 50.0,
    merge_radius_m: float = 500.0,
) -> pd.DataFrame:
    """Discover port locations from stationary clusters in GPS data.

    Returns DataFrame with columns: LAT, LON, LOCODE, duration_min, n_points.
    """
    lat = df["LAT"].values
    lon = df["LON"].values
    sog = df["sog_knots"].values if "sog_knots" in df.columns else np.zeros(len(df))
    time = pd.to_datetime(df["signaldate"])

    stationary = sog <= sog_threshold_kn
    is_gap = df["is_gap"].values.astype(bool) if "is_gap" in df.columns else np.zeros(len(df), dtype=bool)

    clusters: list[dict] = []
    i = 0
    n = len(df)
    while i < n:
        if not stationary[i] or is_gap[i]:
            i += 1
            continue

        start = i
        while i < n and stationary[i] and not (i > start and is_gap[i]):
            i += 1
        end = i

        duration_min = (time.iloc[end - 1] - time.iloc[start]).total_seconds() / 60
        if duration_min < min_duration_minutes:
            continue

        seg_lat = lat[start:end]
        seg_lon = lon[start:end]
        clat, clon = np.nanmean(seg_lat), np.nanmean(seg_lon)

        dlat = np.radians(seg_lat - clat)
        dlon = np.radians(seg_lon - clon) * np.cos(np.radians(clat))
        dists = EARTH_RADIUS_M * np.sqrt(dlat**2 + dlon**2)
        spread = np.nanmax(dists)

        if spread > max_spread_m:
            continue

        clusters.append({
            "LAT": clat,
            "LON": clon,
            "duration_min": duration_min,
            "n_points": end - start,
            "spread_m": spread,
        })

    if not clusters:
        logger.info("No port clusters discovered")
        return pd.DataFrame(columns=["LAT", "LON", "LOCODE", "duration_min", "n_points"])

    cdf = pd.DataFrame(clusters)

    merged = _merge_nearby_clusters(cdf, merge_radius_m)

    merged["LOCODE"] = [f"DISC{i:03d}" for i in range(len(merged))]

    logger.info(
        "Discovered %d port locations from %d stationary clusters",
        len(merged), len(clusters),
    )
    for _, row in merged.iterrows():
        logger.info(
            "  %s: (%.4f, %.4f) duration=%.0fmin points=%d spread=%.1fm",
            row["LOCODE"], row["LAT"], row["LON"],
            row["duration_min"], row["n_points"], row["spread_m"],
        )

    return merged


def _merge_nearby_clusters(df: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    if len(df) <= 1:
        return df

    coords = np.radians(df[["LAT", "LON"]].values)
    tree = BallTree(coords, metric="haversine")
    radius_rad = radius_m / EARTH_RADIUS_M

    visited = set()
    merged: list[dict] = []

    for i in range(len(df)):
        if i in visited:
            continue

        neighbors = tree.query_radius(coords[i:i+1], r=radius_rad)[0]
        group = [j for j in neighbors if j not in visited]
        visited.update(group)

        if not group:
            continue

        sub = df.iloc[group]
        weights = sub["n_points"].values.astype(float)
        merged.append({
            "LAT": np.average(sub["LAT"].values, weights=weights),
            "LON": np.average(sub["LON"].values, weights=weights),
            "duration_min": sub["duration_min"].sum(),
            "n_points": sub["n_points"].sum(),
            "spread_m": sub["spread_m"].max(),
        })

    return pd.DataFrame(merged)


def build_port_matcher(
    ports_db: pd.DataFrame,
    discovered: pd.DataFrame,
    config: PortConfig | None = None,
) -> PortMatcher:
    """Build PortMatcher from database ports + discovered ports."""
    frames = []
    if len(ports_db) > 0:
        db = ports_db[["LAT", "LON", "LOCODE"]].copy()
        frames.append(db)
    if len(discovered) > 0:
        disc = discovered[["LAT", "LON", "LOCODE"]].copy()
        frames.append(disc)

    if not frames:
        empty = pd.DataFrame({"LAT": [0.0], "LON": [0.0], "LOCODE": ["NONE"]})
        return PortMatcher(empty, config)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined port database: %d db + %d discovered = %d total",
                len(ports_db), len(discovered), len(combined))
    return PortMatcher(combined, config)
