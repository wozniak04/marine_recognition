from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import create_logger

logger = create_logger(__name__)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy = (TP + TN) / Total as defined in the project brief."""
    return float(np.mean(y_true == y_pred))


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    """Per-class TP, TN, accuracy, precision, recall, F1."""
    results: dict[str, dict[str, float]] = {}

    for cls in class_names:
        true_binary = (y_true == cls).astype(int)
        pred_binary = (y_pred == cls).astype(int)

        tp = int(((true_binary == 1) & (pred_binary == 1)).sum())
        tn = int(((true_binary == 0) & (pred_binary == 0)).sum())
        fp = int(((true_binary == 0) & (pred_binary == 1)).sum())
        fn = int(((true_binary == 1) & (pred_binary == 0)).sum())

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[cls] = {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    return results


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Build a confusion matrix as a DataFrame."""
    matrix = pd.DataFrame(0, index=class_names, columns=class_names)
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val in class_names and pred_val in class_names:
            matrix.loc[true_val, pred_val] += 1
    return matrix


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    overall_acc = compute_accuracy(y_true, y_pred)
    per_class = per_class_accuracy(y_true, y_pred, class_names)
    cm = confusion_matrix(y_true, y_pred, class_names)

    logger.info("Overall accuracy: %.4f", overall_acc)
    for cls, m in per_class.items():
        logger.info(
            "  %s: acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
            cls,
            m["accuracy"],
            m["precision"],
            m["recall"],
            m["f1"],
        )

    return {
        "overall_accuracy": round(overall_acc, 4),
        "per_class": per_class,
        "confusion_matrix": cm.to_dict(),
    }


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", path)
