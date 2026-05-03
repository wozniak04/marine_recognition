from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.utils.logger import create_logger

logger = create_logger(__name__)


class ClassicalModel:
    """Unified wrapper for sklearn-compatible classifiers (RF, XGBoost, LightGBM)."""

    def __init__(self, model_name: str, params: dict[str, Any] | None = None) -> None:
        self.model_name = model_name
        self.params = params or {}
        self.model = self._build_model()
        self.feature_names: list[str] = []

    def _build_model(self):
        p = self.params.copy()

        if self.model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=p.pop("n_estimators", 200),
                max_depth=p.pop("max_depth", None),
                class_weight=p.pop("class_weight", "balanced"),
                random_state=p.pop("random_state", 42),
                n_jobs=-1,
                **p,
            )

        if self.model_name == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=p.pop("n_estimators", 200),
                max_depth=p.pop("max_depth", 6),
                learning_rate=p.pop("learning_rate", 0.1),
                random_state=p.pop("random_state", 42),
                eval_metric="mlogloss",
                n_jobs=1,
                **p,
            )

        if self.model_name == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=p.pop("n_estimators", 200),
                max_depth=p.pop("max_depth", -1),
                learning_rate=p.pop("learning_rate", 0.1),
                class_weight=p.pop("class_weight", "balanced"),
                random_state=p.pop("random_state", 42),
                verbose=-1,
                n_jobs=1,
                **p,
            )

        raise ValueError(f"Unknown model: {self.model_name}")

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> None:
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        logger.info("Training %s on %d samples, %d features", self.model_name, X.shape[0], X.shape[1])
        self.model.fit(X, y)
        logger.info("Training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def feature_importance(self) -> dict[str, float]:
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
            return dict(sorted(
                zip(self.feature_names, imp),
                key=lambda x: x[1],
                reverse=True,
            ))
        return {}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "model.joblib")
        with open(path / "feature_names.json", "w") as f:
            json.dump(self.feature_names, f)
        logger.info("Model saved to %s", path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path / "model.joblib")
        with open(path / "feature_names.json") as f:
            self.feature_names = json.load(f)
        logger.info("Model loaded from %s", path)
