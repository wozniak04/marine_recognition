"""Evaluation metrics for ship state classification."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = [
    "in_port", "in_port_shifting", "in_port_arrival", "in_port_departure",
    "voyage", "at_sea_turn", "anchor", "adrift",
]


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    overall_acc = float(np.mean(y_true == y_pred))

    per_class = {}
    for cls in CLASS_NAMES:
        tb = (y_true == cls).astype(int)
        pb = (y_pred == cls).astype(int)

        tp = int(((tb == 1) & (pb == 1)).sum())
        tn = int(((tb == 0) & (pb == 0)).sum())
        fp = int(((tb == 0) & (pb == 1)).sum())
        fn = int(((tb == 1) & (pb == 0)).sum())

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    cm = pd.DataFrame(0, index=CLASS_NAMES, columns=CLASS_NAMES)
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val in CLASS_NAMES and pred_val in CLASS_NAMES:
            cm.loc[true_val, pred_val] += 1

    return {
        "overall_accuracy": round(overall_acc, 4),
        "per_class": per_class,
        "confusion_matrix": cm.to_dict(),
    }


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def print_metrics(metrics: dict, split_name: str = "") -> None:
    prefix = f"[{split_name}] " if split_name else ""
    print(f"{prefix}Overall accuracy: {metrics['overall_accuracy']:.4f}")
    for cls, m in metrics["per_class"].items():
        print(
            f"  {cls}: prec={m['precision']:.4f} rec={m['recall']:.4f} f1={m['f1']:.4f}"
        )
