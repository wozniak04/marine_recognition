from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.lstm import LSTMTrainer
from src.utils.config import OPERATION_ID_TO_STATE

STATE_ORDER = sorted(OPERATION_ID_TO_STATE.values())
STATE_TO_INT = {s: i for i, s in enumerate(STATE_ORDER)}
INT_TO_STATE = {i: s for s, i in STATE_TO_INT.items()}

DEFAULT_FEATURE_COLS = ["sog_knots", "delta_lat", "delta_lon"]

GAP_THRESHOLD_S = 300


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    time = pd.to_datetime(df["signaldate"])
    dt = time.diff().dt.total_seconds().values
    is_gap = np.zeros(len(df), dtype=bool)
    is_gap[0] = True
    is_gap[1:] = np.nan_to_num(dt[1:], nan=9999) > GAP_THRESHOLD_S
    df["is_gap"] = is_gap.astype(int)

    lat = df["LAT"].values
    lon = df["LON"].values
    delta_lat = np.zeros(len(lat))
    delta_lon = np.zeros(len(lon))
    delta_lat[1:] = lat[1:] - lat[:-1]
    delta_lon[1:] = lon[1:] - lon[:-1]

    gap_idx = np.where(is_gap)[0]
    delta_lat[gap_idx] = 0.0
    delta_lon[gap_idx] = 0.0

    df["delta_lat"] = delta_lat
    df["delta_lon"] = delta_lon
    return df


@register_experiment("lstm")
class LSTMExperiment(BaseExperiment):

    @classmethod
    def name(cls) -> str:
        return "lstm"

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        train_df = prepare_features(train_df)
        val_df = prepare_features(val_df)

        feature_cols = self.config.model.params.get("feature_cols", DEFAULT_FEATURE_COLS)
        feature_cols = [c for c in feature_cols if c in train_df.columns]

        X_train = train_df[feature_cols].fillna(0).values
        X_val = val_df[feature_cols].fillna(0).values

        op_to_int = {v: STATE_TO_INT[s] for v, s in OPERATION_ID_TO_STATE.items()}
        y_train = train_df["operation_id"].map(op_to_int).fillna(0).astype(int).values
        y_val = val_df["operation_id"].map(op_to_int).fillna(0).astype(int).values

        gap_train = train_df["is_gap"].values
        gap_val = val_df["is_gap"].values

        class_counts = np.bincount(y_train, minlength=len(STATE_ORDER))
        weights = np.sqrt(len(y_train) / (len(STATE_ORDER) * np.maximum(class_counts, 1)))
        class_weights = weights / weights.sum() * len(STATE_ORDER)

        trainer = LSTMTrainer(self.config.model.params)
        trainer.feature_cols = feature_cols
        history = trainer.fit(X_train, y_train, X_val, y_val, gap_train, gap_val, class_weights)

        trainer.save(self.models_dir)

        self.logger.info(
            "Training complete: %d epochs, best val_acc=%.4f",
            len(history["train_loss"]),
            max(history["val_acc"]) if history["val_acc"] else 0.0,
        )

        return {"trainer": trainer, "feature_cols": feature_cols}

    def predict(self, trained: Any, df: pd.DataFrame) -> pd.DataFrame:
        trainer: LSTMTrainer = trained["trainer"]
        feature_cols: list[str] = trained["feature_cols"]

        df = prepare_features(df)

        X = df[feature_cols].fillna(0).values
        gap_mask = df["is_gap"].values

        state_indices, confidences = trainer.predict(X, gap_mask)

        result = df.copy()
        result["predicted_state"] = [INT_TO_STATE.get(i, "anchor") for i in state_indices]
        result["confidence"] = confidences

        return result
