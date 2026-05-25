"""Train classifier on labeled preprocessed data.

Supports rule-based and decision tree models (selected via config model.name).

Usage:
    python train.py --config configs/rule_based.yaml
    python train.py --config configs/decision_tree.yaml
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.oversampling import oversample_minority
from src.data.ports import load_port_matcher
from src.data.splitter import split_by_episodes
from src.evaluation.metrics import compute_all_metrics, print_metrics, save_metrics
from src.evaluation.plots import plot_all_splits
from src.models.rule_based import OPERATION_ID_TO_STATE, RuleClassifier
from src.models.decision_tree import TreeClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument("--config", required=True, help="YAML config file")
    return parser.parse_args()


def build_classifier(cfg: Config):
    model_name = cfg.model.get("name", "rule_based")

    if model_name == "decision_tree":
        feature_cols = cfg.model.get("feature_cols", None)
        if isinstance(feature_cols, Config):
            feature_cols = None
        max_depth = cfg.model.get("max_depth", None)
        min_samples_leaf = cfg.model.get("min_samples_leaf", 5)
        min_ep = cfg.model.get("min_episode_minutes", {"at_sea_turn": 1, "at_port_shifting": 1, "other": 20})
        return TreeClassifier(
            feature_cols=feature_cols,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_episode_minutes=min_ep,
        )

    rules = cfg.model.get("rules", None)
    if isinstance(rules, Config):
        rules = None
    min_ep = cfg.model.get("min_episode_minutes", {"at_sea_turn": 1, "at_port_shifting": 1, "other": 20})
    return RuleClassifier(rules=rules, min_episode_minutes=min_ep)


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    input_path = Path(cfg.paths.preprocessed_dir) / "preprocessed.csv"
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    df["signaldate"] = pd.to_datetime(df["signaldate"])

    if "operation_id" not in df.columns:
        print("ERROR: No 'operation_id' column — need labeled data for training")
        return

    df["state"] = df["operation_id"].map(OPERATION_ID_TO_STATE)

    ports_dir = Path(cfg.paths.ports_dir)
    port_radius = cfg.port.get("radius_meters", 3000.0)
    if ports_dir.exists():
        print("Matching ports...")
        matcher = load_port_matcher(str(ports_dir), port_radius)
        port_cols = ["in_port", "nearest_locode", "port_distance_m"]
        df = df.drop(columns=[c for c in port_cols if c in df.columns])
        port_info = matcher.match(df["LAT"].values, df["LON"].values)
        df = df.join(port_info)
        in_port_count = int(df["in_port"].sum())
        print(f"  In port: {in_port_count} / {len(df)}")

    seed = cfg.get("random_seed", 42)
    test_size = cfg.training.get("test_size", 0.2)
    val_size = cfg.training.get("validation_size", 0.15)

    print("Splitting data...")
    splits = split_by_episodes(df, seed=seed, test_size=test_size, val_size=val_size)

    max_ratio = cfg.training.get("oversampling_max_ratio", 3)
    splits["train"] = oversample_minority(splits["train"], max_ratio=max_ratio)

    model_name = cfg.model.get("name", "rule_based")
    print(f"Training model: {model_name}")
    classifier = build_classifier(cfg)

    if isinstance(classifier, TreeClassifier):
        classifier.fit(splits["train"])

    results_model = {}
    results_pipeline = {}
    for split_name, split_df in splits.items():
        predicted = classifier.classify(split_df)

        model_true = predicted["state"].values.copy()
        model_true[model_true == "port_stay"] = "anchor"
        model_metrics = compute_all_metrics(model_true, predicted["predicted_state"].values)
        results_model[split_name] = model_metrics

        if "in_port" in predicted.columns:
            port_mask = (predicted["in_port"].values == 1) & (predicted["predicted_state"].isin(["anchor", "adrift"]))
            predicted.loc[port_mask, "predicted_state"] = "port_stay"
            predicted.loc[port_mask, "confidence"] = 100.0
        pipeline_metrics = compute_all_metrics(
            predicted["state"].values, predicted["predicted_state"].values
        )
        results_pipeline[split_name] = pipeline_metrics

        print_metrics(model_metrics, f"{split_name} model")
        print_metrics(pipeline_metrics, f"{split_name} pipeline")

    model_dir = Path(cfg.paths.models_dir) / cfg.experiment_name
    classifier.save(model_dir)

    reports_dir = Path(cfg.paths.reports_dir) / cfg.experiment_name
    save_metrics({"model": results_model, "pipeline": results_pipeline}, reports_dir / "metrics.json")

    print("\nGenerating confusion matrix plots...")
    plot_all_splits(results_model, reports_dir, suffix="_model")
    plot_all_splits(results_pipeline, reports_dir, suffix="_pipeline")

    test_predicted = classifier.classify(splits["test"])
    if "in_port" in test_predicted.columns:
        port_mask = test_predicted["in_port"].values == 1
        test_predicted.loc[port_mask, "predicted_state"] = "port_stay"
        test_predicted.loc[port_mask, "confidence"] = 100.0
    test_predicted.to_csv(reports_dir / "test_predictions.csv", index=False)

    print(f"\nModel saved to {model_dir}")
    print(f"Metrics saved to {reports_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
