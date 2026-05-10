"""Classify new (unclassified) GPS data using a trained model.

Runs full pipeline: preprocessing (kinematics, outliers, rolling features),
port matching, classification. Works on raw CSV without operation_id.

Supports rule-based and decision tree models (auto-detected from saved files).

Usage:
    python classify.py --config configs/rule_based.yaml --input data/raw/new_data.csv
    python classify.py --config configs/decision_tree.yaml --input data/raw/new_data.csv --output results/
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.ports import load_port_matcher
from src.data.preprocessing import preprocess
from src.models.decision_tree import TreeClassifier
from src.models.rule_based import RuleClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify GPS data")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--input", required=True, help="Input CSV (raw, no labels needed)")
    parser.add_argument("--output", default=None, help="Output CSV path or directory")
    return parser.parse_args()


def load_classifier(model_dir: Path):
    if (model_dir / "tree_model.joblib").exists():
        return TreeClassifier.load(model_dir)
    return RuleClassifier.load(model_dir)


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "signaldate": df["signaldate"],
            "LAT": df["LAT"],
            "LON": df["LON"],
            "Outlier GPS": df["outlier_gps"] if "outlier_gps" in df.columns else 0,
            "In Port": (df["predicted_state"] == "port_stay").astype(int),
            "At Sea Voyage": (df["predicted_state"] == "voyage").astype(int),
            "At Sea Anchor": (df["predicted_state"] == "anchor").astype(int),
            "At Sea Adrift GPS": (df["predicted_state"] == "adrift").astype(int),
            "Confidence Rate": df["confidence"].round(1),
        }
    )

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

    voyage_segments = []
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
    cfg = Config.from_yaml(args.config)

    model_dir = Path(cfg.paths.models_dir) / cfg.experiment_name
    print(f"Loading model from {model_dir}...")
    classifier = load_classifier(model_dir)
    model_type = "decision_tree" if isinstance(classifier, TreeClassifier) else "rule_based"
    print(f"  Model type: {model_type}")

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)

    print("Preprocessing (kinematics + outliers + rolling features)...")
    rolling_window = cfg.preprocessing.get("rolling_window", 45)
    df = preprocess(df, rolling_window=rolling_window)
    outlier_count = int(df["outlier_gps"].sum())
    print(f"  Outliers: {outlier_count} / {len(df)}")

    ports_dir = cfg.paths.ports_dir
    port_radius = cfg.port.get("radius_meters", 5000.0)
    if Path(ports_dir).exists():
        print("Matching ports...")
        matcher = load_port_matcher(ports_dir, port_radius)
        port_info = matcher.match(df["LAT"].values, df["LON"].values)
        df = df.join(port_info)
        print(f"  In port: {int(df['in_port'].sum())} / {len(df)}")

    print("Classifying...")
    df = classifier.classify(df)

    if "in_port" in df.columns:
        port_mask = (df["in_port"].values == 1) & (df["predicted_state"].isin(["anchor", "adrift"]))
        df.loc[port_mask, "predicted_state"] = "port_stay"
        df.loc[port_mask, "confidence"] = 100.0

    output = build_output(df)

    if args.output:
        output_path = Path(args.output)
        if output_path.suffix == "":
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / "output.csv"
    else:
        output_path = Path(cfg.paths.reports_dir) / cfg.experiment_name / "output.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Results saved to {output_path} ({len(output)} rows)")


if __name__ == "__main__":
    main()
