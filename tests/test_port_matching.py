from __future__ import annotations

import numpy as np

from src.data.ports import PortMatcher
from src.utils.config import PortConfig


class TestPortMatcher:
    def test_point_at_port(self, sample_ports_df):
        matcher = PortMatcher(sample_ports_df, PortConfig(radius_meters=5000))
        result = matcher.query_single(44.1000, 28.6580)
        assert result.in_port is True
        assert result.locode == "ROCON"
        assert result.distance_m < 100

    def test_point_far_from_port(self, sample_ports_df):
        matcher = PortMatcher(sample_ports_df, PortConfig(radius_meters=2000))
        result = matcher.query_single(50.0, 10.0)
        assert result.in_port is False
        assert result.locode is None

    def test_batch_query(self, sample_ports_df):
        matcher = PortMatcher(sample_ports_df, PortConfig(radius_meters=5000))
        lats = np.array([44.1000, 50.0, 41.0000])
        lons = np.array([28.6580, 10.0, 29.0000])
        result = matcher.query_batch(lats, lons)

        assert len(result) == 3
        assert result["in_port"].iloc[0] == 1
        assert result["in_port"].iloc[1] == 0
        assert result["in_port"].iloc[2] == 1
