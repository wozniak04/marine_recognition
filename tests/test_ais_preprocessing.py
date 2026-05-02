from __future__ import annotations

import pandas as pd
import pytest

from src.data.ais_preprocessing import AISPreprocessor


@pytest.fixture
def sample_ais_data() -> pd.DataFrame:
    return pd.DataFrame({
        "signaldate": pd.to_datetime([
            "2024-09-09T12:45:00",
            "2024-09-09T12:50:00",
            "2024-09-09T12:55:00",
            "2024-09-09T13:00:00",
            "2024-09-09T13:05:00",
        ]),
        "LAT": [44.10, 44.11, 44.12, 44.13, 44.14],
        "LON": [28.65, 28.66, 28.67, 28.68, 28.69],
        "sog_knots": [5.0, 8.0, 12.0, 10.0, 0.3],
        "cog_deg": [90.0, 95.0, 100.0, 105.0, 110.0],
        "hdg_deg": [88.0, 93.0, 98.0, 103.0, 108.0],
    })


class TestAISPreprocessor:
    def test_output_columns(self, sample_ais_data: pd.DataFrame) -> None:
        proc = AISPreprocessor()
        result = proc.process(sample_ais_data)
        expected = {"dt", "distance_m", "sog_ms", "dcog_deg", "rot_deg_s", "acceleration_ms2", "drift_angle_deg", "is_gap"}
        assert expected.issubset(set(result.columns))

    def test_preserves_rows(self, sample_ais_data: pd.DataFrame) -> None:
        proc = AISPreprocessor()
        result = proc.process(sample_ais_data)
        assert len(result) == len(sample_ais_data)

    def test_drift_angle(self, sample_ais_data: pd.DataFrame) -> None:
        proc = AISPreprocessor()
        result = proc.process(sample_ais_data)
        assert result["drift_angle_deg"].iloc[1] == pytest.approx(2.0, abs=0.1)

    def test_gap_detection(self) -> None:
        df = pd.DataFrame({
            "signaldate": pd.to_datetime([
                "2024-09-09T12:00:00",
                "2024-09-09T12:05:00",
                "2024-09-09T14:00:00",
            ]),
            "LAT": [44.10, 44.11, 44.12],
            "LON": [28.65, 28.66, 28.67],
            "sog_knots": [5.0, 8.0, 3.0],
            "cog_deg": [90.0, 95.0, 100.0],
            "hdg_deg": [88.0, 93.0, 98.0],
        })
        proc = AISPreprocessor()
        result = proc.process(df)
        assert result["is_gap"].iloc[2] == 1
