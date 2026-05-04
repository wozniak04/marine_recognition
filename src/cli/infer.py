"""CLI: Run inference with a pre-trained LSTM model on new data.

Full pipeline: raw CSV -> preprocess -> predict -> output CSV

Usage:
    uv run python -m src.cli.infer --model models/lstm_gps_v1 --input data.csv --output results.csv
    uv run python -m src.cli.infer --model models/lstm_gps_v1 --input data.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.dataset import load_gps_csv
from src.data.outliers import OutlierDetector
from src.data.ports import PortMatcher
from src.data.preprocessing import DataPreprocessor
from src.experiments.lstm_experiment import INT_TO_STATE, prepare_features
from src.models.lstm import LSTMTrainer
from src.utils.config import OutlierConfig, PortConfig, PreprocessingConfig, ProjectConfig
from src.utils.logger import create_logger

logger = create_logger("infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSTM inference on new GPS data")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model directory")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV (raw GPS data)")
    parser.add_argument("--output", type=str, default=None, help="Path for output CSV (default: reports/<model_name>/inference_output.csv)")
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML for preprocessing params")
    parser.add_argument("--ports-csv", type=str, default="data/ports/EU_Port_Codes.csv", help="Path to port database CSV")
    return parser.parse_args()


def run_pipeline(
    model_dir: Path,
    input_path: Path,
    output_path: Path,
    config: ProjectConfig | None = None,
    ports_csv: Path | None = None,
) -> pd.DataFrame:
    logger.info("Loading model from %s", model_dir)
    trainer = LSTMTrainer.load(model_dir)

    logger.info("Loading input data from %s", input_path)
    df = load_gps_csv(input_path)
    logger.info("Loaded %d rows", len(df))

    preproc_config = config.preprocessing if config else PreprocessingConfig()
    preprocessor = DataPreprocessor(preproc_config)
    df = preprocessor.process(df)

    outlier_config = config.outlier if config else OutlierConfig(iqr_multiplier=3.0, columns=[])
    detector = OutlierDetector(outlier_config)
    df["outlier_gps"] = detector.detect(df).astype(int)

    ports_path = ports_csv or (config.paths.ports_csv if config else Path("data/ports/EU_Port_Codes.csv"))
    if ports_path.exists():
        port_config = config.port if config else PortConfig(radius_meters=5000.0)
        ports_df = pd.read_csv(ports_path).dropna(subset=["LAT", "LON"]).reset_index(drop=True)
        matcher = PortMatcher(ports_df, port_config)
        port_info = matcher.query_batch(df["LAT"].values, df["LON"].values)
        df = df.join(port_info)
    else:
        logger.warning("Port database not found at %s, skipping port matching", ports_path)

    df = prepare_features(df)

    feature_cols = trainer.feature_cols
    X = df[feature_cols].fillna(0).values
    gap_mask = df["is_gap"].values if "is_gap" in df.columns else None

    logger.info("Running inference with features=%s, window=%d", feature_cols, trainer.window_size)
    state_indices, confidences = trainer.predict(X, gap_mask)

    df["predicted_state"] = [INT_TO_STATE.get(i, "anchor") for i in state_indices]
    df["confidence"] = confidences

    output = pd.DataFrame({
        "signaldate": df["signaldate"],
        "LAT": df["LAT"],
        "LON": df["LON"],
        "Outlier GPS": df["outlier_gps"],
        "In Port": (df["predicted_state"] == "port_stay").astype(int),
        "At Sea Voyage": (df["predicted_state"] == "voyage").astype(int),
        "At Sea Anchor": (df["predicted_state"] == "anchor").astype(int),
        "At Sea Adrift GPS": (df["predicted_state"] == "adrift").astype(int),
        "Confidence Rate": confidences.round(1) if hasattr(confidences, "round") else [round(c, 1) for c in confidences],
    })

    if "nearest_locode" in df.columns:
        departure, destination = _compute_port_locodes(df)
        output["Departure Port Locode"] = departure
        output["Destination Port Locode"] = destination

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    logger.info("Results saved to %s (%d rows)", output_path, len(output))

    return output


def _compute_port_locodes(df: pd.DataFrame) -> tuple[list, list]:
    states = df["predicted_state"].values
    locodes = df.get("nearest_locode")
    n = len(df)
    departure = [None] * n
    destination = [None] * n

    if locodes is None:
        return departure, destination

    voyage_segments: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if states[i] == "voyage":
            start = i
            while i < n and states[i] == "voyage":
                i += 1
            voyage_segments.append((start, i))
        else:
            i += 1

    for seg_start, seg_end in voyage_segments:
        dep_locode = None
        for j in range(seg_start - 1, -1, -1):
            if states[j] == "port_stay" and pd.notna(locodes.iloc[j]):
                dep_locode = locodes.iloc[j]
                break

        dest_locode = None
        for j in range(seg_end, n):
            if states[j] == "port_stay" and pd.notna(locodes.iloc[j]):
                dest_locode = locodes.iloc[j]
                break

        for j in range(seg_start, seg_end):
            departure[j] = dep_locode
            destination[j] = dest_locode

    return departure, destination


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model)
    input_path = Path(args.input)

    config = None
    if args.config:
        config = ProjectConfig.from_yaml(args.config)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("reports") / model_dir.name / "inference_output.csv"

    ports_csv = Path(args.ports_csv)

    run_pipeline(model_dir, input_path, output_path, config, ports_csv)


if __name__ == "__main__":
    main()
