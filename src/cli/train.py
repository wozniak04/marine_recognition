"""CLI: Train a model using experiment config.

Usage: uv run python -m src.cli.train --config configs/rule_based_gps.yaml
"""
from __future__ import annotations

import argparse

from src.utils.config import ProjectConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)

    # Import experiments to trigger registration
    import src.experiments.classical_ml  # noqa: F401
    import src.experiments.extended_ais  # noqa: F401
    import src.experiments.hmm_experiment  # noqa: F401
    import src.experiments.lstm_experiment  # noqa: F401
    import src.experiments.rule_based  # noqa: F401

    from src.experiments.registry import build_experiment

    experiment = build_experiment(config.model.name, config)
    results = experiment.run()

    print("\n=== Results ===")
    for key, value in results.items():
        if isinstance(value, dict) and "overall_accuracy" in value:
            print(f"  {key}: accuracy={value['overall_accuracy']:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
