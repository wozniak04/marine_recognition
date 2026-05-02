from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.utils.config import PortConfig
from src.utils.logger import create_logger

logger = create_logger(__name__)


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
        dist_m = float(dist_rad[0, 0]) * 6_371_000
        locode = str(self._locodes[idx[0, 0]])
        return PortMatch(
            in_port=dist_m <= self.config.radius_meters,
            locode=locode if dist_m <= self.config.radius_meters else None,
            distance_m=dist_m,
        )

    def query_batch(self, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
        """Return DataFrame with columns: in_port, nearest_locode, port_distance_m."""
        coords_rad = np.radians(np.column_stack([lats, lons]))
        dist_rad, idx = self._tree.query(coords_rad, k=1)
        dist_m = dist_rad[:, 0] * 6_371_000
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
