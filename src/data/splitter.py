from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.utils.config import TrainingConfig
from src.utils.logger import create_logger

logger = create_logger(__name__)


def identify_episodes(df: pd.DataFrame, state_col: str = "operation_id") -> pd.Series:
    """Assign an episode_id to each contiguous block of the same state."""
    return (df[state_col] != df[state_col].shift()).cumsum()


def split_by_episodes(
    df: pd.DataFrame,
    config: TrainingConfig,
    state_col: str = "operation_id",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Stratified episode split: ensures each split has all available state classes."""
    episodes = identify_episodes(df, state_col)
    df = df.copy()
    df["_episode_id"] = episodes

    episode_info = (
        df.groupby("_episode_id")[state_col]
        .first()
        .reset_index()
        .rename(columns={state_col: "state"})
    )

    rng = np.random.RandomState(seed)

    episodes_by_class: dict[int, list[int]] = defaultdict(list)
    for _, row in episode_info.iterrows():
        episodes_by_class[row["state"]].append(row["_episode_id"])

    for eps_list in episodes_by_class.values():
        rng.shuffle(eps_list)

    train_eps: list[int] = []
    val_eps: list[int] = []
    test_eps: list[int] = []

    for state, eps_list in episodes_by_class.items():
        n = len(eps_list)
        if n >= 3:
            n_test = max(1, int(n * config.test_size))
            n_val = max(1, int(n * config.validation_size))
            n_train = n - n_test - n_val
            if n_train < 1:
                n_train = 1
                n_val = max(1, (n - 1) // 2)
                n_test = n - n_train - n_val
            test_eps.extend(eps_list[:n_test])
            val_eps.extend(eps_list[n_test : n_test + n_val])
            train_eps.extend(eps_list[n_test + n_val :])
        elif n == 2:
            train_eps.append(eps_list[0])
            test_eps.append(eps_list[1])
        else:
            train_eps.append(eps_list[0])

    splits = {
        "train": df[df["_episode_id"].isin(train_eps)].drop(columns=["_episode_id"]),
        "val": df[df["_episode_id"].isin(val_eps)].drop(columns=["_episode_id"]),
        "test": df[df["_episode_id"].isin(test_eps)].drop(columns=["_episode_id"]),
    }

    for name, split_df in splits.items():
        states = sorted(split_df[state_col].unique()) if state_col in split_df.columns else []
        logger.info("Split %s: %d rows, states=%s", name, len(split_df), states)

    logger.info(
        "Stratified episode split: train=%d val=%d test=%d (total episodes=%d)",
        len(train_eps),
        len(val_eps),
        len(test_eps),
        len(episode_info),
    )
    return splits


def split_temporal(
    df: pd.DataFrame,
    config: TrainingConfig,
) -> dict[str, pd.DataFrame]:
    """Split by time: first chunk = train, middle = val, last = test."""
    n = len(df)
    n_test = int(n * config.test_size)
    n_val = int(n * config.validation_size)

    return {
        "train": df.iloc[: n - n_val - n_test].copy(),
        "val": df.iloc[n - n_val - n_test : n - n_test].copy(),
        "test": df.iloc[n - n_test :].copy(),
    }
