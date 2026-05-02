from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import FeatureConfig
from src.utils.geo import haversine
from src.utils.logger import create_logger

logger = create_logger(__name__)


FEATURE_NAMES = [
    "sog_mean",
    "sog_std",
    "sog_min",
    "sog_max",
    "sog_median",
    "sog_range",
    "dcog_abs_mean",
    "dcog_abs_std",
    "dcog_abs_max",
    "rot_abs_mean",
    "rot_abs_std",
    "acc_abs_mean",
    "acc_abs_std",
    "spread_m",
    "net_displacement_m",
    "sinuosity",
    "path_length_m",
    "lat_std",
    "lon_std",
    "in_port_frac",
    "port_dist_min",
    "port_dist_mean",
]


def extract_window_features(
    df: pd.DataFrame,
    config: FeatureConfig | None = None,
    label_col: str | None = "operation_id",
) -> pd.DataFrame:
    """Extract statistical features from sliding windows over the track.

    Returns a DataFrame where each row = one window with feature columns + optional label.
    """
    config = config or FeatureConfig()
    w = config.window_size
    stride = config.stride

    gap_col = df["is_gap"].values if "is_gap" in df.columns else np.zeros(len(df))
    gap_indices = set(np.where(gap_col == 1)[0])

    segments = _find_segments(len(df), gap_indices)

    rows: list[dict] = []
    for seg_start, seg_end in segments:
        for i in range(seg_start, seg_end - w + 1, stride):
            window = df.iloc[i : i + w]
            feats = _compute_features(window)

            if label_col and label_col in df.columns:
                feats["label"] = int(window[label_col].mode().iloc[0])

            feats["window_center_idx"] = i + w // 2
            rows.append(feats)

    result = pd.DataFrame(rows)
    logger.info(
        "Extracted %d windows (size=%d, stride=%d) with %d features",
        len(result),
        w,
        stride,
        len(FEATURE_NAMES),
    )
    return result


def _find_segments(n: int, gap_indices: set[int]) -> list[tuple[int, int]]:
    segments = []
    start = 0
    for i in sorted(gap_indices):
        if i > 0 and i > start:
            segments.append((start, i))
        start = i
    segments.append((start, n))
    return segments


def _compute_features(window: pd.DataFrame) -> dict[str, float]:
    sog = window["sog_knots"].dropna()
    dcog = window["dcog_deg"].dropna().abs()
    rot = window["rot_deg_s"].dropna().abs()
    acc = window["acceleration_ms2"].dropna().abs()
    lat = window["LAT"].values
    lon = window["LON"].values

    feats: dict[str, float] = {}

    # SOG stats
    feats["sog_mean"] = sog.mean() if len(sog) > 0 else 0.0
    feats["sog_std"] = sog.std() if len(sog) > 1 else 0.0
    feats["sog_min"] = sog.min() if len(sog) > 0 else 0.0
    feats["sog_max"] = sog.max() if len(sog) > 0 else 0.0
    feats["sog_median"] = sog.median() if len(sog) > 0 else 0.0
    feats["sog_range"] = feats["sog_max"] - feats["sog_min"]

    # COG change stats
    feats["dcog_abs_mean"] = dcog.mean() if len(dcog) > 0 else 0.0
    feats["dcog_abs_std"] = dcog.std() if len(dcog) > 1 else 0.0
    feats["dcog_abs_max"] = dcog.max() if len(dcog) > 0 else 0.0

    # Rate of turn
    feats["rot_abs_mean"] = rot.mean() if len(rot) > 0 else 0.0
    feats["rot_abs_std"] = rot.std() if len(rot) > 1 else 0.0

    # Acceleration
    feats["acc_abs_mean"] = acc.mean() if len(acc) > 0 else 0.0
    feats["acc_abs_std"] = acc.std() if len(acc) > 1 else 0.0

    # Spatial: spread and displacement
    n = len(lat)
    if n >= 2:
        clat, clon = lat.mean(), lon.mean()
        dists = haversine(lat, lon, np.full(n, clat), np.full(n, clon))
        feats["spread_m"] = float(dists.max())
        feats["net_displacement_m"] = float(
            haversine(np.array([lat[0]]), np.array([lon[0]]),
                      np.array([lat[-1]]), np.array([lon[-1]]))[0]
        )
        path_len = window["distance_m"].dropna().sum()
        feats["path_length_m"] = float(path_len)
        feats["sinuosity"] = (
            path_len / feats["net_displacement_m"]
            if feats["net_displacement_m"] > 1.0
            else 1.0
        )
    else:
        feats["spread_m"] = 0.0
        feats["net_displacement_m"] = 0.0
        feats["path_length_m"] = 0.0
        feats["sinuosity"] = 1.0

    # Position variability
    feats["lat_std"] = float(np.std(lat))
    feats["lon_std"] = float(np.std(lon))

    # Port proximity
    if "in_port" in window.columns:
        feats["in_port_frac"] = float(window["in_port"].mean())
    else:
        feats["in_port_frac"] = 0.0

    if "port_distance_m" in window.columns:
        pdist = window["port_distance_m"].dropna()
        feats["port_dist_min"] = float(pdist.min()) if len(pdist) > 0 else 999999.0
        feats["port_dist_mean"] = float(pdist.mean()) if len(pdist) > 0 else 999999.0
    else:
        feats["port_dist_min"] = 999999.0
        feats["port_dist_mean"] = 999999.0

    return feats
