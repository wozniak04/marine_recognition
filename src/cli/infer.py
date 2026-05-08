"""CLI: Run inference with a pre-trained model on new data.

Supports LSTM, HMM, and rule-based models. Model type is auto-detected
from saved artifacts in the model directory.

Usage:
    uv run python -m src.cli.infer --model models/lstm_gps_v1 --input data.csv
    uv run python -m src.cli.infer --model models/hmm_gps_v1 --input data.csv
    uv run python -m src.cli.infer --model models/rule_based_gps_v1 --input data.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import load_ais_csv, load_gps_csv
from src.data.outliers import OutlierDetector, detect_teleports, fix_teleports
from src.data.ports import PortMatcher, build_port_matcher, discover_ports
from src.data.preprocessing import DataPreprocessor
from src.utils.config import OutlierConfig, PortConfig, PreprocessingConfig, ProjectConfig
from src.utils.logger import create_logger

logger = create_logger("infer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on new data")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model directory")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output", type=str, default=None, help="Path for output CSV")
    parser.add_argument("--config", type=str, default=None, help="Optional config YAML for preprocessing params")
    parser.add_argument("--ports-csv", type=str, default="data/ports/EU_Port_Codes.csv", help="Path to port database CSV")
    parser.add_argument("--ais", action="store_true", help="Input is AIS data (not GPS)")
    return parser.parse_args()


def detect_model_type(model_dir: Path) -> str:
    if (model_dir / "lstm_model.pt").exists():
        return "lstm"
    if (model_dir / "hmm_model.joblib").exists():
        return "hmm"
    if (model_dir / "rule_params.json").exists():
        return "rule_based"
    raise FileNotFoundError(
        f"Cannot detect model type in {model_dir}. "
        "Expected one of: lstm_model.pt, hmm_model.joblib, rule_params.json"
    )


def load_and_preprocess(
    input_path: Path,
    is_ais: bool,
    config: ProjectConfig | None,
    ports_csv: Path,
) -> pd.DataFrame:
    if is_ais:
        from src.data.ais_preprocessing import AISPreprocessor
        df = load_ais_csv(input_path)
        df = AISPreprocessor().process(df)
    else:
        df = load_gps_csv(input_path)
        preproc_config = config.preprocessing if config else PreprocessingConfig()
        df = DataPreprocessor(preproc_config).process(df)

    logger.info("Loaded %d rows from %s", len(df), input_path)

    outlier_config = config.outlier if config else OutlierConfig(iqr_multiplier=3.0, max_speed_knots=30.0, columns=[])
    teleport_mask = detect_teleports(df, max_speed_knots=outlier_config.max_speed_knots)
    if teleport_mask.any():
        logger.info("Fixing %d teleport points", teleport_mask.sum())
        df = fix_teleports(df, teleport_mask)

    detector = OutlierDetector(outlier_config)
    df["outlier_gps"] = detector.detect(df).astype(int)

    if ports_csv.exists():
        port_config = config.port if config else PortConfig(radius_meters=5000.0)
        ports_df = pd.read_csv(ports_csv).dropna(subset=["LAT", "LON"]).reset_index(drop=True)
        discovered = discover_ports(df)
        matcher = build_port_matcher(ports_df, discovered, port_config)
        port_info = matcher.query_batch(df["LAT"].values, df["LON"].values)
        df = df.join(port_info)
    else:
        logger.warning("Port database not found at %s, skipping", ports_csv)

    return df


def infer_lstm(model_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    from src.experiments.lstm_experiment import INT_TO_STATE, prepare_features
    from src.models.lstm import LSTMTrainer

    trainer = LSTMTrainer.load(model_dir)
    df = prepare_features(df)

    feature_cols = trainer.feature_cols
    X = df[feature_cols].fillna(0).values
    gap_mask = df["is_gap"].values if "is_gap" in df.columns else None

    logger.info("LSTM inference: features=%s, window=%d", feature_cols, trainer.window_size)
    state_indices, confidences = trainer.predict(X, gap_mask)

    df["predicted_state"] = [INT_TO_STATE.get(i, "anchor") for i in state_indices]
    df["confidence"] = confidences
    return df


def infer_hmm(model_dir: Path, df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    from src.models.hmm import HMM_TO_STATE, ShipHMM
    from src.models.rule_engine import ShipStateFSM

    hmm = ShipHMM(params)
    hmm.load(model_dir)

    df = df.copy().reset_index(drop=True)
    time = pd.to_datetime(df["signaldate"])
    dt = time.diff().dt.total_seconds().values
    is_gap = np.zeros(len(df), dtype=bool)
    is_gap[0] = True
    is_gap[1:] = np.nan_to_num(dt[1:], nan=9999) > 300
    df["is_gap"] = is_gap.astype(int)

    fsm = ShipStateFSM(params)
    gap_boundaries = fsm._find_gap_boundaries(df)
    df = fsm._compute_rolling_features(df, gap_boundaries)

    feature_cols = hmm.feature_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    gap_indices = np.where(df["is_gap"].values == 1)[0]
    boundaries = sorted(set(gap_indices.tolist()) | {0})
    segments = []
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(df)
        if end > start:
            segments.append((start, end))

    all_X = []
    lengths = []
    for start, end in segments:
        seg = df.iloc[start:end]
        all_X.append(seg[feature_cols].fillna(0).values)
        lengths.append(end - start)

    X = np.vstack(all_X)
    in_port = df["in_port"].values if "in_port" in df.columns else None

    logger.info("HMM inference: features=%s, %d segments", feature_cols, len(segments))
    state_indices, confidences = hmm.predict(X, lengths, in_port=in_port)

    df["predicted_state"] = [HMM_TO_STATE[i] for i in state_indices]
    df["confidence"] = confidences
    return df


def infer_rule_based(model_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    from src.models.rule_engine import ShipStateFSM

    params_path = model_dir / "rule_params.json"
    with open(params_path) as f:
        params = json.load(f)

    logger.info("Rule-based inference: params=%s", params)
    fsm = ShipStateFSM(params)
    return fsm.classify(df)


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame({
        "signaldate": df["signaldate"],
        "LAT": df["LAT"],
        "LON": df["LON"],
        "Outlier GPS": df.get("outlier_gps", 0),
        "In Port": (df["predicted_state"] == "port_stay").astype(int),
        "At Sea Voyage": (df["predicted_state"] == "voyage").astype(int),
        "At Sea Anchor": (df["predicted_state"] == "anchor").astype(int),
        "At Sea Adrift GPS": (df["predicted_state"] == "adrift").astype(int),
        "Confidence Rate": df["confidence"].round(1),
    })

    if "nearest_locode" in df.columns:
        departure, destination = _compute_port_locodes(df)
        output["Departure Port Locode"] = departure
        output["Destination Port Locode"] = destination

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

    config = ProjectConfig.from_yaml(args.config) if args.config else None
    ports_csv = Path(args.ports_csv)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("reports") / model_dir.name / "inference_output.csv"

    model_type = detect_model_type(model_dir)
    logger.info("Detected model type: %s", model_type)

    df = load_and_preprocess(input_path, args.ais, config, ports_csv)

    if model_type == "lstm":
        df = infer_lstm(model_dir, df)
    elif model_type == "hmm":
        params = config.model.params if config else {}
        df = infer_hmm(model_dir, df, params)
    elif model_type == "rule_based":
        df = infer_rule_based(model_dir, df)

    output = build_output(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    logger.info("Results saved to %s (%d rows)", output_path, len(output))


if __name__ == "__main__":
    main()
