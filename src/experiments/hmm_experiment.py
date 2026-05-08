from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.hmm import HMM_TO_STATE, OPERATION_TO_HMM, ShipHMM
from src.models.rule_engine import ShipStateFSM
from src.utils.config import OPERATION_ID_TO_STATE

GAP_THRESHOLD_S = 300

DEFAULT_FEATURE_COLS = [
    "sog_knots",
    "rolling_spread_m",
    "rolling_net_displacement_m",
]


def recompute_gaps_and_features(
    df: pd.DataFrame, params: dict[str, Any] | None = None
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)

    time = pd.to_datetime(df["signaldate"])
    dt = time.diff().dt.total_seconds().values
    is_gap = np.zeros(len(df), dtype=bool)
    is_gap[0] = True
    is_gap[1:] = np.nan_to_num(dt[1:], nan=9999) > GAP_THRESHOLD_S
    df["is_gap"] = is_gap.astype(int)

    fsm = ShipStateFSM(params)
    gap_boundaries = fsm._find_gap_boundaries(df)
    df = fsm._compute_rolling_features(df, gap_boundaries)

    return df


@register_experiment("hmm")
class HMMExperiment(BaseExperiment):

    @classmethod
    def name(cls) -> str:
        return "hmm"

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        train_df = recompute_gaps_and_features(train_df, self.config.model.params)

        feature_cols = self.config.model.params.get("feature_cols", DEFAULT_FEATURE_COLS)
        feature_cols = [c for c in feature_cols if c in train_df.columns]

        train_df = self._oversample_minority(train_df)

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

        df = recompute_gaps_and_features(df, self.config.model.params)

        X, _, lengths = self._prepare_sequences(df, feature_cols)
        lengths = self._split_long_segments(df, lengths)

        in_port = df["in_port"].values if "in_port" in df.columns else None
        state_indices, confidences = hmm.predict(X, lengths, in_port=in_port)

        result = df.copy()
        result["predicted_state"] = [HMM_TO_STATE[i] for i in state_indices]
        result["confidence"] = confidences

        return result

    @staticmethod
    def _split_long_segments(
        df: pd.DataFrame, lengths: list[int], speed_threshold: float = 2.0
    ) -> list[int]:
        """Break long segments at regime transitions.

        Boundaries are created at:
        - Speed regime changes (stationary ↔ moving)
        - Port proximity changes (in_port 0 ↔ 1)

        Shorter segments let Viterbi start fresh at natural state boundaries
        instead of locking into one state for thousands of points.
        """
        sog = df["sog_knots"].fillna(0).values if "sog_knots" in df.columns else None
        in_port = df["in_port"].values if "in_port" in df.columns else None

        if sog is None and in_port is None:
            return lengths

        new_lengths = []
        offset = 0
        for seg_len in lengths:
            if seg_len <= 50:
                new_lengths.append(seg_len)
                offset += seg_len
                continue

            boundaries = set()
            seg_end = offset + seg_len

            if sog is not None:
                seg_sog = sog[offset:seg_end]
                moving = seg_sog > speed_threshold
                for i in range(1, seg_len):
                    if moving[i] != moving[i - 1]:
                        boundaries.add(i)

            if in_port is not None:
                seg_port = in_port[offset:seg_end]
                for i in range(1, seg_len):
                    if seg_port[i] != seg_port[i - 1]:
                        boundaries.add(i)

            if not boundaries:
                new_lengths.append(seg_len)
                offset += seg_len
                continue

            cuts = sorted(boundaries)
            prev = 0
            for cut in cuts:
                chunk = cut - prev
                if chunk > 0:
                    new_lengths.append(chunk)
                prev = cut
            remainder = seg_len - prev
            if remainder > 0:
                new_lengths.append(remainder)

            offset += seg_len

        return new_lengths

    def _oversample_minority(self, df: pd.DataFrame) -> pd.DataFrame:
        """Duplicate minority-class episodes so HMM sees balanced state distributions."""
        if "operation_id" not in df.columns:
            return df

        state_counts = df["operation_id"].value_counts()
        if len(state_counts) <= 1:
            return df

        median_count = int(state_counts.median())
        episodes = (df["operation_id"] != df["operation_id"].shift()).cumsum()
        df = df.copy()
        df["_ep"] = episodes

        max_oversample = 3
        parts = [df]
        for state_id, count in state_counts.items():
            if count >= median_count * 0.5:
                continue
            ratio = min(max_oversample, max(1, int(round(median_count / count)))) - 1
            state_eps = df[df["operation_id"] == state_id]["_ep"].unique()
            for _ in range(ratio):
                for ep in state_eps:
                    parts.append(df[df["_ep"] == ep])
            self.logger.info(
                "Oversampled state %s: %d → %dx (%d episodes)",
                OPERATION_ID_TO_STATE.get(state_id, state_id),
                count,
                ratio + 1,
                len(state_eps),
            )

        result = pd.concat(parts, ignore_index=True).drop(columns=["_ep"])
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
