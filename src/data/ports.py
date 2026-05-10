"""Port matching — loads all CSV files from data/ports/ directory."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6_371_000


class PortMatcher:
    def __init__(self, ports_df: pd.DataFrame, radius_m: float = 5000.0) -> None:
        self.radius_m = radius_m
        self.ports_df = ports_df.copy()
        coords_rad = np.radians(ports_df[["LAT", "LON"]].values)
        self._tree = BallTree(coords_rad, metric="haversine")
        self._locodes = ports_df["LOCODE"].values

    def match(self, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
        n = len(lats)
        in_port = np.zeros(n, dtype=int)
        nearest_locode = np.array([None] * n, dtype=object)
        port_dist = np.full(n, np.nan)

        valid = ~(np.isnan(lats) | np.isnan(lons))
        if valid.any():
            coords_rad = np.radians(np.column_stack([lats[valid], lons[valid]]))
            dist_rad, idx = self._tree.query(coords_rad, k=1)
            dist_m = dist_rad[:, 0] * EARTH_RADIUS_M
            matched = dist_m <= self.radius_m
            valid_idx = np.where(valid)[0]
            in_port[valid_idx[matched]] = 1
            nearest_locode[valid_idx[matched]] = self._locodes[idx[matched, 0]]
            port_dist[valid_idx] = dist_m

        return pd.DataFrame(
            {
                "in_port": in_port,
                "nearest_locode": nearest_locode,
                "port_distance_m": port_dist,
            }
        )


def load_port_matcher(ports_dir: str, radius_m: float = 5000.0) -> PortMatcher:
    """Load all CSV files from ports_dir, combine, and build a matcher."""
    ports_path = Path(ports_dir)
    frames = []

    if ports_path.is_file():
        csv_files = [ports_path]
    else:
        csv_files = sorted(ports_path.glob("*.csv"))

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if "LAT" in df.columns and "LON" in df.columns and "LOCODE" in df.columns:
            frames.append(df[["LAT", "LON", "LOCODE"]].dropna(subset=["LAT", "LON"]))
            print(f"  Loaded {len(frames[-1])} ports from {csv_file.name}")

    if not frames:
        raise FileNotFoundError(f"No valid port CSV files found in {ports_dir}")

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["LOCODE"])
    print(f"  Total unique ports: {len(combined)}")
    return PortMatcher(combined, radius_m)
