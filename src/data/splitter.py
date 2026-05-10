"""Train/val/test splitting by episodes with stratification."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def identify_episodes(df: pd.DataFrame, state_col: str = "operation_id") -> pd.Series:
    return (df[state_col] != df[state_col].shift()).cumsum()


def split_by_episodes(
    df: pd.DataFrame,
    state_col: str = "operation_id",
    test_size: float = 0.2,
    val_size: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
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

    train_eps, val_eps, test_eps = [], [], []

    for state, eps_list in episodes_by_class.items():
        n = len(eps_list)
        if n >= 3:
            n_test = max(1, int(n * test_size))
            n_val = max(1, int(n * val_size))
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
        print(f"  Split {name}: {len(split_df)} rows, states={states}")

    return splits
