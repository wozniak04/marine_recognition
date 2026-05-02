from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

EARTH_RADIUS_M = 6_371_000


def haversine(
    lat1: ArrayLike,
    lon1: ArrayLike,
    lat2: ArrayLike,
    lon2: ArrayLike,
) -> np.ndarray:
    """Distance in metres between two (lat, lon) points. Vectorised."""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing(
    lat1: ArrayLike,
    lon1: ArrayLike,
    lat2: ArrayLike,
    lon2: ArrayLike,
) -> np.ndarray:
    """Initial bearing (COG) in degrees [0, 360). Vectorised."""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    lon1, lon2 = np.radians(lon1), np.radians(lon2)
    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    return np.degrees(np.arctan2(x, y)) % 360


def knots_to_ms(knots: float) -> float:
    return knots * 0.514444


def ms_to_knots(ms: float) -> float:
    return ms * 1.943844
