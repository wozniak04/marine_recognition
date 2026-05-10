"""Decision Tree classifier for ship state classification."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.models.rule_based import ALL_STATES, OPERATION_ID_TO_STATE

DEFAULT_FEATURES = [
    "rolling_sog_kn",
    "rolling_sog_std",
    "rolling_spread_m",
    "rolling_net_displacement_m",
    "rolling_accel_abs",
    "rolling_rot_abs",
    "rolling_cog_std",
]

STATE_TO_INT = {s: i for i, s in enumerate(ALL_STATES)}
INT_TO_STATE = {i: s for i, s in enumerate(ALL_STATES)}


class TreeClassifier:
    def __init__(
        self,
        feature_cols: list[str] | None = None,
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        min_episode_minutes: int = 50,
    ) -> None:
        self.feature_cols = feature_cols or DEFAULT_FEATURES
        self.min_episode_minutes = min_episode_minutes
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=42,
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> None:
        cols = [c for c in self.feature_cols if c in df.columns]
        states = df["operation_id"].map(OPERATION_ID_TO_STATE)
        mask = states != "port_stay"
        X = df.loc[mask, cols].fillna(0).values
        y = states[mask].map(STATE_TO_INT).values
        self.feature_cols = cols
        self.model.fit(X, y)
        self._fitted = True
        print(f"  Decision tree: depth={self.model.get_depth()}, leaves={self.model.get_n_leaves()}, features={cols}")
        print(f"  Trained on {len(X)} rows (excluded {(~mask).sum()} port_stay rows)")

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].fillna(0).values

        pred_int = self.model.predict(X)
        proba = self.model.predict_proba(X)
        confidences = np.max(proba, axis=1) * 100.0

        df["predicted_state"] = [INT_TO_STATE[i] for i in pred_int]
        df["confidence"] = confidences

        gap_boundaries = df.index[df["is_gap"] == 1].tolist() if "is_gap" in df.columns else [0]
        df["predicted_state"] = self._smooth_episodes(
            df["predicted_state"].values, gap_boundaries
        )
        return df

    def _smooth_episodes(
        self, states: np.ndarray, gap_boundaries: list[int]
    ) -> np.ndarray:
        result = states.copy()
        min_len = self.min_episode_minutes
        gap_set = set(gap_boundaries)

        i = 0
        while i < len(result):
            if i in gap_set and i > 0:
                i += 1
                continue
            j = i
            while j < len(result) and result[j] == result[i] and j not in gap_set:
                j += 1
            if j < len(result) and j in gap_set and result[j] == result[i]:
                while j < len(result) and result[j] == result[i]:
                    j += 1

            episode_len = j - i
            if episode_len < min_len and i > 0 and j < len(result):
                result[i:j] = result[i - 1]

            i = j

        return result

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "tree_model.joblib")
        meta = {
            "feature_cols": self.feature_cols,
            "min_episode_minutes": self.min_episode_minutes,
        }
        with open(path / "tree_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> TreeClassifier:
        with open(path / "tree_meta.json") as f:
            meta = json.load(f)
        obj = cls(
            feature_cols=meta["feature_cols"],
            min_episode_minutes=meta.get("min_episode_minutes", 50),
        )
        obj.model = joblib.load(path / "tree_model.joblib")
        obj._fitted = True
        return obj
