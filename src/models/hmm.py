from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from hmmlearn.hmm import GaussianHMM

from src.utils.logger import create_logger

logger = create_logger(__name__)

STATE_ORDER = ["adrift", "anchor", "port_stay", "voyage"]
OPERATION_TO_HMM = {3: 0, 2: 1, 1: 2, 4: 3}  # operation_id -> HMM state index
HMM_TO_STATE = {i: s for i, s in enumerate(STATE_ORDER)}


class ShipHMM:
    """Gaussian HMM for ship state sequence modelling."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        self.n_states = 4
        self.n_iter = p.get("n_iter", 100)
        self.covariance_type = p.get("covariance_type", "full")
        self.feature_cols = p.get("feature_cols", [
            "sog_knots", "dcog_deg", "acceleration_ms2", "rolling_spread_m",
        ])
        self.model: GaussianHMM | None = None
        self._transition_matrix: np.ndarray | None = None

    def fit_supervised(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lengths: list[int],
    ) -> None:
        """Initialize HMM parameters from labeled data, then refine with EM."""
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

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            init_params="",
            params="stmc",
        )
        self.model.startprob_ = startprob
        self.model.transmat_ = transmat
        self.model.means_ = means
        self.model.covars_ = covars

        self.model.fit(X, lengths)
        self._transition_matrix = self.model.transmat_

        logger.info("HMM trained: %d iterations, score=%.2f", self.n_iter, self.model.score(X, lengths))
        logger.info("Transition matrix:\n%s", np.array2string(self._transition_matrix, precision=3))

    def predict(self, X: np.ndarray, lengths: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Predict states using Viterbi decoding. Returns (state_indices, confidences)."""
        if self.model is None:
            raise RuntimeError("Model not fitted")

        if lengths is None:
            lengths = [len(X)]

        state_indices = self.model.predict(X, lengths)

        posteriors = self.model.predict_proba(X, lengths)
        confidences = np.max(posteriors, axis=1) * 100.0

        return state_indices, confidences

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
        if self._transition_matrix is not None:
            with open(path / "transition_matrix.json", "w") as f:
                json.dump(self.transition_matrix_labeled(), f, indent=2)
        logger.info("HMM saved to %s", path)

    def load(self, path: Path) -> None:
        import joblib

        self.model = joblib.load(path / "hmm_model.joblib")
        self._transition_matrix = self.model.transmat_
        logger.info("HMM loaded from %s", path)
