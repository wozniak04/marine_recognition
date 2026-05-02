from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.lstm import LSTMTrainer
from src.models.rule_engine import ShipStateFSM
from src.utils.config import OPERATION_ID_TO_STATE

STATE_ORDER = sorted(OPERATION_ID_TO_STATE.values())
STATE_TO_INT = {s: i for i, s in enumerate(STATE_ORDER)}
INT_TO_STATE = {i: s for s, i in STATE_TO_INT.items()}

FEATURE_COLS = ["sog_knots", "dcog_deg", "acceleration_ms2", "distance_m", "cog_deg", "rot_deg_s"]


@register_experiment("lstm")
class LSTMExperiment(BaseExperiment):

    @classmethod
    def name(cls) -> str:
        return "lstm"

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        fsm = ShipStateFSM({"rolling_window": 30})
        train_df = train_df.copy().reset_index(drop=True)
        val_df = val_df.copy().reset_index(drop=True)
        train_df = fsm._compute_rolling_features(train_df, fsm._find_gap_boundaries(train_df))
        val_df = fsm._compute_rolling_features(val_df, fsm._find_gap_boundaries(val_df))

        feature_cols = self.config.model.params.get("feature_cols", FEATURE_COLS)
        if "rolling_spread_m" in train_df.columns:
            feature_cols = [c for c in [*feature_cols, "rolling_spread_m"] if c in train_df.columns]

        X_train = train_df[feature_cols].fillna(0).values
        X_val = val_df[feature_cols].fillna(0).values

        y_train = train_df["operation_id"].map(
            {v: STATE_TO_INT[s] for v, s in OPERATION_ID_TO_STATE.items()}
        ).fillna(0).astype(int).values
        y_val = val_df["operation_id"].map(
            {v: STATE_TO_INT[s] for v, s in OPERATION_ID_TO_STATE.items()}
        ).fillna(0).astype(int).values

        gap_train = train_df["is_gap"].values if "is_gap" in train_df.columns else None
        gap_val = val_df["is_gap"].values if "is_gap" in val_df.columns else None

        class_counts = np.bincount(y_train, minlength=len(STATE_ORDER))
        class_weights = len(y_train) / (len(STATE_ORDER) * np.maximum(class_counts, 1))
        class_weights = class_weights / class_weights.sum() * len(STATE_ORDER)

        trainer = LSTMTrainer(self.config.model.params)
        trainer.feature_cols = feature_cols
        history = trainer.fit(X_train, y_train, X_val, y_val, gap_train, gap_val, class_weights)

        trainer.save(self.models_dir)

        self.logger.info(
            "Training complete: %d epochs, best val_loss=%.4f",
            len(history["train_loss"]),
            min(history["val_loss"]) if history["val_loss"] else float("inf"),
        )

        return {"trainer": trainer, "feature_cols": feature_cols}

    def predict(self, trained: Any, df: pd.DataFrame) -> pd.DataFrame:
        trainer: LSTMTrainer = trained["trainer"]
        feature_cols: list[str] = trained["feature_cols"]

        fsm = ShipStateFSM({"rolling_window": 30})
        df = df.copy().reset_index(drop=True)
        df = fsm._compute_rolling_features(df, fsm._find_gap_boundaries(df))

        X = df[feature_cols].fillna(0).values
        gap_mask = df["is_gap"].values if "is_gap" in df.columns else None

        state_indices, confidences = trainer.predict(X, gap_mask)

        result = df.copy()
        result["predicted_state"] = [INT_TO_STATE.get(i, "anchor") for i in state_indices]
        result["confidence"] = confidences

        return result
