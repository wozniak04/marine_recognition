from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_engineering import FEATURE_NAMES, extract_window_features
from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.classical import ClassicalModel
from src.utils.config import OPERATION_ID_TO_STATE, ProjectConfig


STATE_TO_INT = {s: i for i, s in enumerate(sorted(OPERATION_ID_TO_STATE.values()))}
INT_TO_STATE = {i: s for s, i in STATE_TO_INT.items()}


def _register_classical(model_key: str):
    """Register a classical ML experiment under the given model key."""

    @register_experiment(model_key)
    class _ClassicalExperiment(BaseExperiment):

        @classmethod
        def name(cls) -> str:
            return model_key

        def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
            train_feats = extract_window_features(train_df, self.config.features)
            val_feats = extract_window_features(val_df, self.config.features)

            X_train, y_train = self._prepare_xy(train_feats)
            X_val, y_val = self._prepare_xy(val_feats)

            model = ClassicalModel(model_key, self.config.model.params)
            model.fit(X_train, y_train, feature_names=FEATURE_NAMES)

            val_pred = model.predict(X_val)
            val_acc = float(np.mean(val_pred == y_val))
            self.logger.info("Validation accuracy: %.4f", val_acc)

            importance = model.feature_importance()
            if importance:
                top5 = list(importance.items())[:5]
                self.logger.info("Top features: %s", top5)

            model.save(self.models_dir)

            return {"model": model, "train_feats": train_feats, "val_feats": val_feats}

        def predict(self, trained: Any, df: pd.DataFrame) -> pd.DataFrame:
            model: ClassicalModel = trained["model"]

            feats = extract_window_features(df, self.config.features, label_col="operation_id")
            X, _ = self._prepare_xy(feats)

            predictions = model.predict(X)
            probas = model.predict_proba(X)

            pred_states = [INT_TO_STATE[p] for p in predictions]
            confidences = [float(probas[i, p]) * 100.0 for i, p in enumerate(predictions)]

            center_indices = feats["window_center_idx"].values.astype(int)

            result = df.copy()
            result["predicted_state"] = "anchor"
            result["confidence"] = 50.0

            valid_mask = center_indices < len(result)
            valid_centers = center_indices[valid_mask]
            result.iloc[valid_centers, result.columns.get_loc("predicted_state")] = [
                pred_states[i] for i in range(len(valid_centers))
            ]
            result.iloc[valid_centers, result.columns.get_loc("confidence")] = [
                confidences[i] for i in range(len(valid_centers))
            ]

            result["predicted_state"] = result["predicted_state"].ffill().bfill()
            result["confidence"] = result["confidence"].ffill().bfill()

            return result

        def _prepare_xy(self, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            X = feats[FEATURE_NAMES].fillna(0).values
            y = feats["label"].map(
                {v: STATE_TO_INT[s] for v, s in OPERATION_ID_TO_STATE.items()}
            ).values
            return X, y

    _ClassicalExperiment.__name__ = f"{model_key.title().replace('_', '')}Experiment"
    _ClassicalExperiment.__qualname__ = _ClassicalExperiment.__name__
    return _ClassicalExperiment


RandomForestExperiment = _register_classical("random_forest")
XGBoostExperiment = _register_classical("xgboost")
LightGBMExperiment = _register_classical("lightgbm")
