"""Preprocessing pipeline: kinematics computation + outlier detection + rolling features.

Based on the reference implementation from message.txt.
Extended with configurable rolling features for state classification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from geopy.distance import geodesic

HIGHEST_POSSIBLE_SOG = 60.0
HIGHEST_POSSIBLE_ACCELERATION = 0.2
HIGHEST_POSSIBLE_TURN = 1.0

EARTH_RADIUS_M = 6_371_000
GAP_THRESHOLD_S = 300


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    try:
        p_lat, p_lon, c_lat, c_lon = np.radians(
            [float(lat1), float(lon1), float(lat2), float(lon2)]
        )
        d_lon = c_lon - p_lon
        y = np.sin(d_lon) * np.cos(c_lat)
        x = np.cos(p_lat) * np.sin(c_lat) - np.sin(p_lat) * np.cos(c_lat) * np.cos(
            d_lon
        )
        return (np.degrees(np.arctan2(y, x)) + 360) % 360
    except Exception:
        return np.nan


def haversine(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def preprocess(df: pd.DataFrame, rolling_window: int = 45) -> pd.DataFrame:
    """Full preprocessing: kinematics, outlier detection, gap marking, rolling features."""
    df = df.copy()
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df["signaldate"] = pd.to_datetime(df["signaldate"])
    df = df.sort_values(by="signaldate").reset_index(drop=True)

    df = _compute_kinematics_and_outliers(df)
    df = _mark_gaps(df)
    df = _compute_rolling_features(df, rolling_window=rolling_window)

    return df


def _compute_kinematics_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SOG, COG, acceleration, ROT and flag outliers.

    Each point is compared against the last VALID point, so outliers
    don't corrupt subsequent calculations.
    """
    out_sog, out_cog, out_accel, out_rot, is_outlier = [], [], [], [], []

    last_valid_row = None
    last_valid_sog = 0.0
    last_valid_cog = np.nan

    for index, row in df.iterrows():
        if pd.isna(row["LAT"]) or pd.isna(row["LON"]):
            out_sog.append(np.nan)
            out_cog.append(np.nan)
            out_accel.append(np.nan)
            out_rot.append(np.nan)
            is_outlier.append(1)
            continue

        if last_valid_row is None:
            out_sog.append(0.0)
            out_cog.append(np.nan)
            out_accel.append(0.0)
            out_rot.append(0.0)
            is_outlier.append(0)
            last_valid_row = row
            last_valid_sog = 0.0
            continue

        time_diff = (row["signaldate"] - last_valid_row["signaldate"]).total_seconds()

        if time_diff <= 0:
            out_sog.append(last_valid_sog)
            out_cog.append(last_valid_cog)
            out_accel.append(0.0)
            out_rot.append(0.0)
            is_outlier.append(1)
            continue

        try:
            prev_pos = (float(last_valid_row["LAT"]), float(last_valid_row["LON"]))
            curr_pos = (float(row["LAT"]), float(row["LON"]))
            dist_meters = geodesic(prev_pos, curr_pos).meters
        except Exception:
            dist_meters = 0.0

        current_sog = (dist_meters / time_diff) * 1.94384
        current_cog = calculate_bearing(
            prev_pos[0], prev_pos[1], curr_pos[0], curr_pos[1]
        )

        curr_sog_ms = current_sog * 0.51444
        prev_sog_ms = last_valid_sog * 0.51444
        current_accel = (curr_sog_ms - prev_sog_ms) / time_diff

        p_cog = last_valid_cog if not pd.isna(last_valid_cog) else current_cog
        delta_cog = (current_cog - p_cog + 180) % 360 - 180
        current_rot = (delta_cog / time_diff) * 60

        outlier_flag = 0
        if current_sog > HIGHEST_POSSIBLE_SOG:
            outlier_flag = 1
        elif abs(current_accel) > HIGHEST_POSSIBLE_ACCELERATION:
            outlier_flag = 1
        elif current_sog > 1.0 and (
            current_sog * abs(current_rot) / 1100
        ) > HIGHEST_POSSIBLE_TURN:
            outlier_flag = 1

        out_sog.append(current_sog)
        out_cog.append(current_cog)
        out_accel.append(current_accel)
        out_rot.append(current_rot)
        is_outlier.append(outlier_flag)

        if outlier_flag == 0:
            last_valid_row = row
            last_valid_sog = current_sog
            last_valid_cog = current_cog

    df["sog_knots"] = out_sog
    df["cog"] = out_cog
    df["acceleration"] = out_accel
    df["rot"] = out_rot
    df["outlier_gps"] = is_outlier

    return df


def _mark_gaps(df: pd.DataFrame) -> pd.DataFrame:
    time = pd.to_datetime(df["signaldate"])
    dt = time.diff().dt.total_seconds().values
    is_gap = np.zeros(len(df), dtype=int)
    is_gap[0] = 1
    is_gap[1:] = (np.nan_to_num(dt[1:], nan=9999) > GAP_THRESHOLD_S).astype(int)
    df["is_gap"] = is_gap
    return df


def _compute_rolling_features(
    df: pd.DataFrame, rolling_window: int = 45
) -> pd.DataFrame:
    """Compute rolling features over a centered window.

    Spatial features (haversine per-point):
    - rolling_spread_m: maks. odleglosc punktu od srodka ciezkosci okna.
      port_stay < 13m, anchor < 200m, adrift/voyage > 200m.
    - rolling_net_displacement_m: odleglosc pierwszy-ostatni punkt w oknie.
      Duza przy duzym spread = dryf, mala = kotwica.

    Aggregated features (rolling mean/std of per-point kinematics):
    - rolling_sog_kn: srednia SOG w oknie
    - rolling_sog_std: odchylenie std SOG — stabilnosc predkosci
    - rolling_accel_abs: sredni |acceleration| — aktywnosc manewrowa
    - rolling_rot_abs: sredni |ROT| — intensywnosc skretow
    - rolling_cog_std: circular std kursu — stabilnosc kierunku
    """
    w = rolling_window
    gap_boundaries = df.index[df["is_gap"] == 1].tolist()
    segments = _split_by_gaps(len(df), gap_boundaries)

    sog = df["sog_knots"].fillna(0)
    accel = df["acceleration"].fillna(0).abs()
    rot = df["rot"].fillna(0).abs()

    rolling_sog = pd.Series(np.nan, index=df.index)
    rolling_sog_std = pd.Series(0.0, index=df.index)
    rolling_accel = pd.Series(0.0, index=df.index)
    rolling_rot = pd.Series(0.0, index=df.index)
    rolling_cog_std = pd.Series(0.0, index=df.index)

    for start, end in segments:
        seg_sog = sog.iloc[start:end]
        rolling_sog.iloc[start:end] = seg_sog.rolling(w, center=True, min_periods=1).mean()
        rolling_sog_std.iloc[start:end] = seg_sog.rolling(w, center=True, min_periods=1).std().fillna(0)

        seg_accel = accel.iloc[start:end]
        rolling_accel.iloc[start:end] = seg_accel.rolling(w, center=True, min_periods=1).mean()

        seg_rot = rot.iloc[start:end]
        rolling_rot.iloc[start:end] = seg_rot.rolling(w, center=True, min_periods=1).mean()

        seg_cog = df["cog"].iloc[start:end].ffill().fillna(0)
        rolling_cog_std.iloc[start:end] = _circular_rolling_std(seg_cog.values, w)

    df["rolling_sog_kn"] = rolling_sog
    df["rolling_sog_std"] = rolling_sog_std
    df["rolling_accel_abs"] = rolling_accel
    df["rolling_rot_abs"] = rolling_rot
    df["rolling_cog_std"] = rolling_cog_std

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
            dists = haversine(
                wlat, wlon, np.full(hi - lo, clat), np.full(hi - lo, clon)
            )
            spread[i] = dists.max()
            net_disp[i] = haversine(
                np.array([lat[lo]]),
                np.array([lon[lo]]),
                np.array([lat[hi - 1]]),
                np.array([lon[hi - 1]]),
            )[0]

    df["rolling_spread_m"] = spread
    df["rolling_net_displacement_m"] = net_disp

    return df


def _circular_rolling_std(angles_deg: np.ndarray, window: int) -> np.ndarray:
    rad = np.radians(angles_deg)
    sin_vals = np.sin(rad)
    cos_vals = np.cos(rad)
    n = len(angles_deg)
    result = np.zeros(n)
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        s_mean = sin_vals[lo:hi].mean()
        c_mean = cos_vals[lo:hi].mean()
        r = np.sqrt(s_mean**2 + c_mean**2)
        result[i] = np.degrees(np.sqrt(-2 * np.log(max(r, 1e-10))))

    return result


def _split_by_gaps(n: int, gap_boundaries: list[int]) -> list[tuple[int, int]]:
    boundaries = sorted(set(gap_boundaries) | {0})
    segments = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else n
        if end > start:
            segments.append((start, end))
    return segments
