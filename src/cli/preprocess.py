"""CLI: Preprocess raw GPS data and compute kinematic features.

Usage: uv run python -m src.cli.preprocess --config configs/base.yaml
"""
from __future__ import annotations

import argparse

from src.data.dataset import load_classified_data, load_port_database
from src.data.outliers import OutlierDetector
from src.data.ports import PortMatcher
from src.data.preprocessing import DataPreprocessor
from src.utils.config import ProjectConfig
from src.utils.logger import create_logger
from src.utils.seeding import set_global_seed

logger = create_logger("preprocess")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GPS data")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    set_global_seed(config.random_seed)

    processed_dir, _, _ = config.experiment_paths()

    df = load_classified_data(config)
    logger.info("Raw data: %d rows, columns: %s", len(df), list(df.columns))

    preprocessor = DataPreprocessor(config.preprocessing)
    df = preprocessor.process(df)

    outlier_detector = OutlierDetector(config.outlier)
    df["outlier_gps"] = outlier_detector.detect(df).astype(int)

    ports_df = load_port_database(config)
    matcher = PortMatcher(ports_df, config.port)
    port_info = matcher.query_batch(df["LAT"].values, df["LON"].values)
    df = df.join(port_info)

    output_path = processed_dir / "preprocessed.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved preprocessed data to %s (%d rows)", output_path, len(df))


if __name__ == "__main__":
    main()
