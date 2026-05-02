from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_gps_track() -> pd.DataFrame:
    """Synthetic GPS track: port stay -> voyage -> anchor."""
    rng = np.random.RandomState(42)
    n = 120

    base_time = pd.Timestamp("2024-09-09 08:00:00")
    times = [base_time + pd.Timedelta(minutes=i) for i in range(n)]

    lat = np.zeros(n)
    lon = np.zeros(n)
    operation_id = np.zeros(n, dtype=int)

    # Port stay (0-39): stationary near Constanta port
    for i in range(40):
        lat[i] = 44.1000 + rng.normal(0, 0.00001)
        lon[i] = 28.6580 + rng.normal(0, 0.00001)
        operation_id[i] = 1

    # Voyage (40-79): moving south-east at ~10 knots
    for i in range(40, 80):
        step = i - 40
        lat[i] = 44.1000 - step * 0.005
        lon[i] = 28.6580 + step * 0.005
        operation_id[i] = 4

    # Anchor (80-119): stationary with small oscillations
    for i in range(80, 120):
        lat[i] = 43.9000 + rng.normal(0, 0.0002)
        lon[i] = 28.8580 + rng.normal(0, 0.0002)
        operation_id[i] = 2

    return pd.DataFrame({
        "signaldate": times,
        "LAT": lat,
        "LON": lon,
        "operation_id": operation_id,
    })


@pytest.fixture
def sample_ports_df() -> pd.DataFrame:
    return pd.DataFrame({
        "LOCODE": ["ROCON", "TRIST", "GRPIR"],
        "LAT": [44.1000, 41.0000, 37.9500],
        "LON": [28.6580, 29.0000, 23.6300],
    })
