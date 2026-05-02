from __future__ import annotations

import numpy as np

from src.data.preprocessing import DataPreprocessor
from src.utils.geo import bearing, haversine


class TestHaversine:
    def test_zero_distance(self):
        d = haversine(np.array([44.1]), np.array([28.6]), np.array([44.1]), np.array([28.6]))
        assert d[0] == 0.0

    def test_known_distance(self):
        # Gdansk to Warsaw ~ 300 km
        d = haversine(
            np.array([54.3520]),
            np.array([18.6466]),
            np.array([52.2297]),
            np.array([21.0122]),
        )
        assert 280_000 < d[0] < 320_000

    def test_vectorized(self):
        lats1 = np.array([0.0, 0.0])
        lons1 = np.array([0.0, 0.0])
        lats2 = np.array([1.0, 2.0])
        lons2 = np.array([0.0, 0.0])
        d = haversine(lats1, lons1, lats2, lons2)
        assert d[1] > d[0]


class TestBearing:
    def test_north(self):
        b = bearing(np.array([0.0]), np.array([0.0]), np.array([1.0]), np.array([0.0]))
        assert abs(b[0] - 0.0) < 0.01

    def test_east(self):
        b = bearing(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([1.0]))
        assert abs(b[0] - 90.0) < 0.01

    def test_south(self):
        b = bearing(np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert abs(b[0] - 180.0) < 0.01


class TestDataPreprocessor:
    def test_output_columns(self, sample_gps_track):
        preprocessor = DataPreprocessor()
        result = preprocessor.process(sample_gps_track)
        expected_cols = ["sog_ms", "sog_knots", "cog_deg", "dcog_deg", "rot_deg_s", "acceleration_ms2", "dt", "distance_m"]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_preserves_rows(self, sample_gps_track):
        preprocessor = DataPreprocessor()
        result = preprocessor.process(sample_gps_track)
        assert len(result) == len(sample_gps_track)

    def test_port_stay_low_speed(self, sample_gps_track):
        preprocessor = DataPreprocessor()
        result = preprocessor.process(sample_gps_track)
        port_speeds = result.loc[result["operation_id"] == 1, "sog_knots"].dropna()
        assert port_speeds.mean() < 1.0

    def test_voyage_high_speed(self, sample_gps_track):
        preprocessor = DataPreprocessor()
        result = preprocessor.process(sample_gps_track)
        voyage_speeds = result.loc[result["operation_id"] == 4, "sog_knots"].dropna()
        assert voyage_speeds.mean() > 5.0

    def test_sorted_by_time(self, sample_gps_track):
        shuffled = sample_gps_track.sample(frac=1, random_state=0)
        preprocessor = DataPreprocessor()
        result = preprocessor.process(shuffled)
        assert result["signaldate"].is_monotonic_increasing
