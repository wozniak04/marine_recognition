from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import OutlierConfig
from src.utils.logger import create_logger

logger = create_logger(__name__)


class OutlierDetector:
    """Detects GPS outliers (position jumps / signal errors). Flags but does not remove."""

    def __init__(self, config: OutlierConfig | None = None) -> None:
        self.config = config or OutlierConfig()

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Return boolean Series where True = outlier."""
        flags = pd.Series(False, index=df.index)

        if "sog_ms" in df.columns:
            max_speed_ms = self.config.max_speed_knots * 0.514444
            speed_outlier = df["sog_ms"] > max_speed_ms
            flags |= speed_outlier
            logger.info(
                "Physical speed check (>%.0f kn): %d outliers",
                self.config.max_speed_knots,
                speed_outlier.sum(),
            )

        if "acceleration_ms2" in df.columns:
            acc_outlier = self._acceleration_outlier(df["acceleration_ms2"])
            flags |= acc_outlier

        method = self.config.method
        for col in self.config.columns:
            if col not in df.columns:
                continue
            if method in ("iqr", "combined"):
                flags |= self._iqr(df[col])
            if method in ("zscore", "combined"):
                flags |= self._zscore(df[col])
            if method in ("isolation_forest", "combined"):
                flags |= self._isolation_forest(df[[col]])

        logger.info(
            "Total outliers flagged: %d / %d (%.2f%%)",
            flags.sum(),
            len(df),
            100 * flags.mean(),
        )
        return flags

    def _acceleration_outlier(self, series: pd.Series) -> pd.Series:
        """Flag points with physically impossible acceleration (GPS jumps)."""
        abs_acc = series.abs()
        threshold = abs_acc.quantile(0.999)
        mask = (abs_acc > threshold) & series.notna()
        logger.info("Acceleration outlier check: %d flagged (threshold=%.4f m/s²)", mask.sum(), threshold)
        return mask

    def _iqr(self, series: pd.Series) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - self.config.iqr_multiplier * iqr
        upper = q3 + self.config.iqr_multiplier * iqr
        mask = ~series.between(lower, upper) & series.notna()
        logger.info(
            "IQR on %s: %d outliers (bounds [%.4f, %.4f])",
            series.name,
            mask.sum(),
            lower,
            upper,
        )
        return mask

    def _zscore(self, series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(False, index=series.index)
        z = ((series - mean) / std).abs()
        mask = (z > self.config.zscore_threshold) & series.notna()
        logger.info(
            "Z-score on %s: %d outliers (threshold=%.1f)",
            series.name,
            mask.sum(),
            self.config.zscore_threshold,
        )
        return mask

    def _isolation_forest(self, df: pd.DataFrame) -> pd.Series:
        from sklearn.ensemble import IsolationForest

        clean = df.dropna()
        if len(clean) < 10:
            return pd.Series(False, index=df.index)

        iso = IsolationForest(contamination=self.config.contamination, random_state=42)
        preds = iso.fit_predict(clean.values)
        mask = pd.Series(False, index=df.index)
        mask.loc[clean.index] = preds == -1
        logger.info(
            "Isolation Forest: %d outliers (contamination=%.3f)",
            mask.sum(),
            self.config.contamination,
        )
        return mask
