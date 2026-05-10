"""Rule-based ship state classifier driven by configurable rules.

Rules are defined in YAML config as an ordered list. Each rule specifies
conditions on feature columns and a resulting state. First matching rule wins.
Easy to extend — add new features in preprocessing, reference them in config.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ALL_STATES = ["port_stay", "voyage", "anchor", "adrift"]

OPERATION_ID_TO_STATE = {1: "port_stay", 2: "anchor", 3: "adrift", 4: "voyage"}

DEFAULT_RULES = [
    {
        "state": "voyage",
        "confidence": 85,
        "conditions": {"rolling_sog_kn": {"min": 3.0}},
    },
    {
        "state": "adrift",
        "confidence": 70,
        "conditions": {
            "rolling_sog_kn": {"min": 0.3, "max": 3.0},
            "rolling_spread_m": {"min": 200.0},
            "rolling_net_displacement_m": {"min_ratio": {"of": "rolling_spread_m", "value": 0.3}},
        },
    },
    {
        "state": "anchor",
        "confidence": 70,
        "conditions": {
            "rolling_sog_kn": {"max": 0.5},
            "rolling_spread_m": {"max": 200.0},
        },
    },
    {
        "state": "anchor",
        "confidence": 40,
        "conditions": {"rolling_spread_m": {"max": 200.0}},
    },
    {
        "state": "adrift",
        "confidence": 35,
        "conditions": {},
    },
]


class RuleClassifier:
    def __init__(self, rules: list[dict] | None = None, min_episode_minutes: int = 50) -> None:
        self.rules = rules or DEFAULT_RULES
        self.min_episode_minutes = min_episode_minutes

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        states, confidences = self._apply_rules(df)
        df["predicted_state"] = states
        df["confidence"] = confidences

        gap_boundaries = df.index[df["is_gap"] == 1].tolist() if "is_gap" in df.columns else [0]
        df["predicted_state"] = self._smooth_episodes(
            df["predicted_state"].values, gap_boundaries
        )
        return df

    def _apply_rules(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n = len(df)
        states = np.full(n, "adrift", dtype=object)
        confidences = np.full(n, 35.0)

        feature_cache = {}
        for col in df.columns:
            if df[col].dtype in (np.float64, np.int64, float, int):
                feature_cache[col] = df[col].fillna(0).values

        for i in range(n):
            for rule in self.rules:
                if self._check_conditions(rule.get("conditions", {}), feature_cache, i):
                    states[i] = rule["state"]
                    confidences[i] = float(rule.get("confidence", 50))
                    break

        return states, confidences

    def _check_conditions(
        self, conditions: dict, features: dict[str, np.ndarray], idx: int
    ) -> bool:
        if not conditions:
            return True

        for col, constraint in conditions.items():
            if col not in features:
                return False

            val = features[col][idx]

            if isinstance(constraint, dict):
                if "min" in constraint and val < constraint["min"]:
                    return False
                if "max" in constraint and val > constraint["max"]:
                    return False
                if "min_ratio" in constraint:
                    ratio_cfg = constraint["min_ratio"]
                    ref_col = ratio_cfg["of"]
                    ratio_val = ratio_cfg["value"]
                    if ref_col not in features:
                        return False
                    ref_val = features[ref_col][idx]
                    if val < ref_val * ratio_val:
                        return False
            else:
                if val != constraint:
                    return False

        return True

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
        data = {
            "rules": self.rules,
            "min_episode_minutes": self.min_episode_minutes,
        }
        with open(path / "rule_params.json", "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> RuleClassifier:
        with open(path / "rule_params.json") as f:
            data = json.load(f)
        if "rules" in data:
            return cls(rules=data["rules"], min_episode_minutes=data.get("min_episode_minutes", 50))
        return cls(min_episode_minutes=data.get("min_episode_minutes", 50))
