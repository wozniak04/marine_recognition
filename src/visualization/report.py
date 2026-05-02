from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import confusion_matrix
from src.utils.logger import create_logger
from src.visualization.plots import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_per_class_f1,
    plot_state_distribution,
    plot_timeseries_states,
)

logger = create_logger(__name__)

CLASS_NAMES = ["port_stay", "voyage", "anchor", "adrift"]


def generate_experiment_report(experiment_name: str) -> None:
    reports_dir = Path("reports") / experiment_name
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pred_path = reports_dir / "test_predictions.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path)

        if "predicted_state" in df.columns:
            plot_timeseries_states(
                df,
                state_col="predicted_state",
                save_path=figures_dir / "timeseries_predicted.png",
                title=f"{experiment_name} — Predicted States",
            )
            logger.info("Saved timeseries plot")

            plot_state_distribution(
                df,
                state_col="predicted_state",
                save_path=figures_dir / "state_distribution.png",
                title=f"{experiment_name} — State Distribution",
            )
            logger.info("Saved state distribution plot")

        if "state" in df.columns and "predicted_state" in df.columns:
            cm = confusion_matrix(df["state"].values, df["predicted_state"].values, CLASS_NAMES)
            plot_confusion_matrix(
                cm,
                save_path=figures_dir / "confusion_matrix.png",
                title=f"{experiment_name} — Confusion Matrix (Test)",
            )
            logger.info("Saved confusion matrix")

    logger.info("Report generated for %s in %s", experiment_name, figures_dir)


def generate_comparison_report() -> None:
    reports_dir = Path("reports")
    figures_dir = reports_dir / "comparison"
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    for exp_dir in sorted(reports_dir.iterdir()):
        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        with metrics_file.open() as f:
            all_results[exp_dir.name] = json.load(f)

    if not all_results:
        logger.warning("No experiment results found")
        return

    plot_model_comparison(all_results, save_path=figures_dir / "accuracy_comparison.png")
    logger.info("Saved accuracy comparison plot")

    plot_per_class_f1(all_results, split="val", save_path=figures_dir / "f1_val.png")
    plot_per_class_f1(all_results, split="test", save_path=figures_dir / "f1_test.png")
    logger.info("Saved per-class F1 plots")

    logger.info("Comparison report generated in %s", figures_dir)
