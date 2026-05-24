"""Confusion matrix and metric plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = [
    "in_port", "in_port_shifting", "in_port_arrival", "in_port_departure",
    "voyage", "at_sea_turn", "anchor", "adrift",
]


def plot_confusion_matrix(metrics: dict, split_name: str, output_dir: Path, suffix: str = "") -> None:
    cm_dict = metrics.get("confusion_matrix", {})
    if not cm_dict:
        return

    labels = [c for c in CLASS_NAMES if c in cm_dict]
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    for i, true_cls in enumerate(labels):
        for j, pred_cls in enumerate(labels):
            matrix[i, j] = cm_dict.get(true_cls, {}).get(pred_cls, 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {split_name} (acc={metrics['overall_accuracy']:.4f})")

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > matrix.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"confusion_matrix_{split_name}{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"  Plot saved: {path}")


def plot_all_splits(results: dict, output_dir: Path, suffix: str = "") -> None:
    for split_name, metrics in results.items():
        plot_confusion_matrix(metrics, split_name, output_dir, suffix)
