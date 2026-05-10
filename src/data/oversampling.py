"""Oversampling minority class episodes for balanced training."""
from __future__ import annotations

import pandas as pd

from src.models.rule_based import OPERATION_ID_TO_STATE


def oversample_minority(df: pd.DataFrame, max_ratio: int = 3) -> pd.DataFrame:
    if "operation_id" not in df.columns:
        return df

    state_counts = df["operation_id"].value_counts()
    if len(state_counts) <= 1:
        return df

    median_count = int(state_counts.median())
    episodes = (df["operation_id"] != df["operation_id"].shift()).cumsum()
    df = df.copy()
    df["_ep"] = episodes

    parts = [df]
    for state_id, count in state_counts.items():
        if count >= median_count * 0.5:
            continue
        ratio = min(max_ratio, max(1, int(round(median_count / count)))) - 1
        state_eps = df[df["operation_id"] == state_id]["_ep"].unique()
        for _ in range(ratio):
            for ep in state_eps:
                parts.append(df[df["_ep"] == ep])
        state_name = OPERATION_ID_TO_STATE.get(state_id, str(state_id))
        print(f"  Oversampled {state_name}: {count} -> {ratio + 1}x ({len(state_eps)} episodes)")

    return pd.concat(parts, ignore_index=True).drop(columns=["_ep"])
