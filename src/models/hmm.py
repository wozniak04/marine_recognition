from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from hmmlearn.hmm import GaussianHMM

from src.utils.logger import create_logger

logger = create_logger(__name__)

STATE_ORDER = ["adrift", "anchor", "port_stay", "voyage"]
OPERATION_TO_HMM = {3: 0, 2: 1, 1: 2, 4: 3}
HMM_TO_STATE = {i: s for i, s in enumerate(STATE_ORDER)}


class ShipHMM:
    """Gaussian HMM for ship state sequence modelling with port-aware post-processing."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.n_states = 4
        self.n_iter = p.get("n_iter", 200)
        self.covariance_type = p.get("covariance_type", "full")
        self.feature_cols = p.get("feature_cols", [
            "sog_knots", "rolling_spread_m", "rolling_net_displacement_m",
        ])
        self.model: GaussianHMM | None = None
        self._transition_matrix: np.ndarray | None = None

    def fit_supervised(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lengths: list[int],
    ) -> None:
        means = np.zeros((self.n_states, X.shape[1]))
        covars = np.zeros((self.n_states, X.shape[1], X.shape[1]))

        for state_idx in range(self.n_states):
            mask = y == state_idx
            if mask.sum() < 2:
                means[state_idx] = X.mean(axis=0)
                covars[state_idx] = np.eye(X.shape[1])
                continue
            state_data = X[mask]
            means[state_idx] = state_data.mean(axis=0)
            covars[state_idx] = np.cov(state_data.T) + np.eye(X.shape[1]) * 1e-6

        startprob = np.zeros(self.n_states)
        transmat = np.zeros((self.n_states, self.n_states))

        offset = 0
        for seg_len in lengths:
            seg_y = y[offset : offset + seg_len]
            if len(seg_y) > 0:
                startprob[seg_y[0]] += 1
            for j in range(len(seg_y) - 1):
                transmat[seg_y[j], seg_y[j + 1]] += 1
            offset += seg_len

        startprob = startprob / max(startprob.sum(), 1)
        for i in range(self.n_states):
            row_sum = transmat[i].sum()
            if row_sum > 0:
                transmat[i] /= row_sum
            else:
                transmat[i] = 1.0 / self.n_states

        alpha = 0.9
        for i in range(self.n_states):
            transmat[i] = alpha * np.eye(self.n_states)[i] + (1 - alpha) * transmat[i]
            transmat[i] /= transmat[i].sum()

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            init_params="",
            params="st",
        )
        self.model.startprob_ = startprob
        self.model.transmat_ = transmat
        self.model.means_ = means
        self.model.covars_ = covars

        self.model.fit(X, lengths)
        self._transition_matrix = self.model.transmat_

        logger.info("HMM trained: %d iterations, score=%.2f", self.n_iter, self.model.score(X, lengths))
        logger.info("Transition matrix:\n%s", np.array2string(self._transition_matrix, precision=3))

    def predict(
        self,
        X: np.ndarray,
        lengths: list[int] | None = None,
        in_port: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model not fitted")

        if lengths is None:
            lengths = [len(X)]

        state_indices = self.model.predict(X, lengths)

        posteriors = self.model.predict_proba(X, lengths)
        confidences = np.max(posteriors, axis=1) * 100.0

        if in_port is not None:
            state_indices = self._apply_port_constraint(state_indices, in_port, X)

        return state_indices, confidences

    def _apply_port_constraint(
        self, states: np.ndarray, in_port: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        """Reclassify anchor→port_stay when in port with tight positioning."""
        result = states.copy()
        port_idx = STATE_ORDER.index("port_stay")
        anchor_idx = STATE_ORDER.index("anchor")

        spread_col = None
        for ci, col in enumerate(self.feature_cols):
            if "spread" in col:
                spread_col = ci
                break
        if spread_col is None:
            return result

        port_spread_max = 11.0
        for i in range(len(result)):
            if result[i] == anchor_idx and in_port[i]:
                if X[i, spread_col] < port_spread_max:
                    result[i] = port_idx
        return result

    @property
    def transition_matrix(self) -> np.ndarray | None:
        return self._transition_matrix

    def transition_matrix_labeled(self) -> dict[str, dict[str, float]] | None:
        if self._transition_matrix is None:
            return None
        result = {}
        for i, from_state in enumerate(STATE_ORDER):
            result[from_state] = {}
            for j, to_state in enumerate(STATE_ORDER):
                result[from_state][to_state] = round(float(self._transition_matrix[i, j]), 4)
        return result

    def save(self, path: Path) -> None:
        import joblib

        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path / "hmm_model.joblib")

        meta = {"feature_cols": self.feature_cols}
        with open(path / "hmm_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        if self._transition_matrix is not None:
            with open(path / "transition_matrix.json", "w") as f:
                json.dump(self.transition_matrix_labeled(), f, indent=2)
        logger.info("HMM saved to %s", path)

    def load(self, path: Path) -> None:
        import joblib

        self.model = joblib.load(path / "hmm_model.joblib")
        self._transition_matrix = self.model.transmat_

        meta_path = path / "hmm_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self.feature_cols = meta.get("feature_cols", self.feature_cols)

        logger.info("HMM loaded from %s", path)
