from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

STATE_COLORS = {
    "port_stay": "#2196F3",
    "anchor": "#FF9800",
    "adrift": "#F44336",
    "voyage": "#4CAF50",
    "port_maneuver": "#9C27B0",
    "turn": "#E91E63",
    "shifting": "#00BCD4",
}


def plot_timeseries_states(
    df: pd.DataFrame,
    state_col: str = "predicted_state",
    save_path: Path | None = None,
    title: str = "Ship State Classification",
) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    time = pd.to_datetime(df["signaldate"])
    states = df[state_col].values
    unique_states = sorted(set(states))

    for state in unique_states:
        mask = states == state
        color = STATE_COLORS.get(state, "#999999")

        if "sog_knots" in df.columns:
            axes[0].scatter(time[mask], df["sog_knots"].values[mask], c=color, s=1, label=state, alpha=0.7)
        if "cog_deg" in df.columns:
            axes[1].scatter(time[mask], df["cog_deg"].values[mask], c=color, s=1, alpha=0.7)
        if "dcog_deg" in df.columns:
            axes[2].scatter(time[mask], df["dcog_deg"].values[mask], c=color, s=1, alpha=0.7)

    axes[0].set_ylabel("SOG (knots)")
    axes[0].set_title(title)
    axes[0].legend(markerscale=8, loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("COG (degrees)")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("dCOG (degrees)")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    cm: pd.DataFrame,
    save_path: Path | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    cm_norm = cm.div(cm.sum(axis=1), axis=0).fillna(0)

    sns.heatmap(
        cm_norm,
        annot=cm.values,
        fmt="d",
        cmap="Blues",
        xticklabels=cm.columns,
        yticklabels=cm.index,
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Proportion"},
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_comparison(
    results: dict[str, dict],
    save_path: Path | None = None,
) -> plt.Figure:
    models = list(results.keys())
    splits = ["train", "val", "test"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.25

    for i, split in enumerate(splits):
        accs = [results[m].get(split, {}).get("overall_accuracy", 0) for m in models]
        bars = ax.bar(x + i * width, accs, width, label=split, alpha=0.85)
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{acc:.1%}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison — Overall Accuracy")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_per_class_f1(
    results: dict[str, dict],
    split: str = "val",
    save_path: Path | None = None,
) -> plt.Figure:
    models = list(results.keys())
    states = ["port_stay", "voyage", "anchor", "adrift"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(states))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        per_class = results[model].get(split, {}).get("per_class", {})
        f1s = [per_class.get(s, {}).get("f1", 0) for s in states]
        ax.bar(x + i * width, f1s, width, label=model.replace("_", " "), alpha=0.85)

    ax.set_xlabel("State")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-Class F1 Score ({split} set)")
    ax.set_xticks(x + width * len(models) / 2)
    ax.set_xticklabels(states)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_state_distribution(
    df: pd.DataFrame,
    state_col: str = "predicted_state",
    save_path: Path | None = None,
    title: str = "State Distribution",
) -> plt.Figure:
    counts = df[state_col].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [STATE_COLORS.get(s, "#999999") for s in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.85)

    for bar, count in zip(bars, counts.values):
        pct = count / len(df) * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + len(df) * 0.01,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
