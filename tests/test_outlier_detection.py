from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.outliers import OutlierDetector
from src.utils.config import OutlierConfig


class TestOutlierDetector:
    def test_iqr_flags_extreme_speed(self):
        config = OutlierConfig(method="iqr", max_speed_knots=30.0)
        detector = OutlierDetector(config)

        sog = np.concatenate([np.ones(100) * 5.0, [100.0]])
        df = pd.DataFrame({"sog_ms": sog})

        flags = detector.detect(df)
        assert flags.iloc[-1] is np.True_
        assert flags.iloc[0] is np.False_

    def test_physical_speed_limit(self):
        config = OutlierConfig(method="iqr", max_speed_knots=30.0)
        detector = OutlierDetector(config)

        df = pd.DataFrame({"sog_ms": [5.0, 10.0, 20.0]})
        flags = detector.detect(df)
        # 20 m/s = ~38.9 knots > 30 knots limit
        assert flags.iloc[2] is np.True_
        assert flags.iloc[0] is np.False_

    def test_no_outliers_in_clean_data(self):
        config = OutlierConfig(method="iqr", max_speed_knots=30.0)
        detector = OutlierDetector(config)

        df = pd.DataFrame({"sog_ms": np.ones(100) * 5.0})
        flags = detector.detect(df)
        assert flags.sum() == 0
