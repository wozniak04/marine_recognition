from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.hmm import HMM_TO_STATE, OPERATION_TO_HMM, ShipHMM
from src.models.rule_engine import ShipStateFSM
from src.utils.config import OPERATION_ID_TO_STATE

FEATURE_COLS = ["sog_knots", "dcog_deg", "acceleration_ms2"]


@register_experiment("hmm")
class HMMExperiment(BaseExperiment):

    @classmethod
    def name(cls) -> str:
        return "hmm"

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        fsm = ShipStateFSM({"rolling_window": 30})
        train_df = train_df.copy().reset_index(drop=True)
        train_df = fsm._compute_rolling_features(train_df, fsm._find_gap_boundaries(train_df))

        feature_cols = self.config.model.params.get("feature_cols", FEATURE_COLS)
        if "rolling_spread_m" in train_df.columns:
            feature_cols = [*feature_cols, "rolling_spread_m"]

        X_train, y_train, lengths_train = self._prepare_sequences(train_df, feature_cols)

        hmm = ShipHMM(self.config.model.params)
        hmm.feature_cols = feature_cols
        hmm.fit_supervised(X_train, y_train, lengths_train)

        hmm.save(self.models_dir)

        tm = hmm.transition_matrix_labeled()
        if tm:
            self.logger.info("Transition matrix (readable):")
            for from_s, targets in tm.items():
                self.logger.info("  %s -> %s", from_s, targets)

        return {"hmm": hmm, "feature_cols": feature_cols}

    def predict(self, trained: Any, df: pd.DataFrame) -> pd.DataFrame:
        hmm: ShipHMM = trained["hmm"]
        feature_cols: list[str] = trained["feature_cols"]

        fsm = ShipStateFSM({"rolling_window": 30})
        df = df.copy().reset_index(drop=True)
        df = fsm._compute_rolling_features(df, fsm._find_gap_boundaries(df))

        X, _, lengths = self._prepare_sequences(df, feature_cols)

        state_indices, confidences = hmm.predict(X, lengths)

        result = df.copy()
        result["predicted_state"] = [HMM_TO_STATE[i] for i in state_indices]
        result["confidence"] = confidences

        return result

    def _prepare_sequences(
        self, df: pd.DataFrame, feature_cols: list[str]
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        gap_col = df["is_gap"].values if "is_gap" in df.columns else np.zeros(len(df))
        gap_indices = np.where(gap_col == 1)[0]

        boundaries = sorted(set(gap_indices.tolist()) | {0})
        segments = []
        for idx, start in enumerate(boundaries):
            end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(df)
            if end > start:
                segments.append((start, end))

        all_X = []
        all_y = []
        lengths = []

        for start, end in segments:
            seg = df.iloc[start:end]
            X_seg = seg[feature_cols].fillna(0).values
            all_X.append(X_seg)
            lengths.append(len(X_seg))

            if "operation_id" in seg.columns:
                y_seg = seg["operation_id"].map(OPERATION_TO_HMM).fillna(0).astype(int).values
                all_y.append(y_seg)

        X = np.vstack(all_X)
        y = np.concatenate(all_y) if all_y else np.zeros(len(X), dtype=int)
        return X, y, lengths
