"""Preprocess raw GPS data: compute kinematics, detect outliers, match ports.

Usage:
    python preprocess.py --config configs/default.yaml
    python preprocess.py --config configs/default.yaml --input data/raw/custom.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data.ports import load_port_matcher
from src.data.preprocessing import preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GPS data")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--input", nargs="+", default=None, help="Override input files")
    parser.add_argument("--output", default=None, help="Override output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    if args.input:
        input_files = args.input
    else:
        raw_dir = Path(cfg.paths.raw_dir)
        input_files = [str(raw_dir / f) for f in cfg.data.input_files]

    frames = []
    for f in input_files:
        print(f"Loading {f}...")
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    print(f"Total rows: {len(df)}")

    print("Preprocessing (kinematics + outliers + rolling features)...")
    rolling_window = cfg.preprocessing.get("rolling_window", 45)
    reverse = cfg.preprocessing.get("reverse", False)
    df = preprocess(df, rolling_window=rolling_window, reverse=reverse)
    outlier_count = int(df["outlier_gps"].sum())
    print(f"Outliers detected: {outlier_count} / {len(df)} ({100*outlier_count/len(df):.2f}%)")

    ports_dir = cfg.paths.ports_dir
    port_radius = cfg.port.get("radius_meters", 5000.0)
    if Path(ports_dir).exists():
        print(f"Matching ports from {ports_dir} (radius={port_radius}m)...")
        matcher = load_port_matcher(ports_dir, port_radius)
        port_info = matcher.match(df["LAT"].values, df["LON"].values)
        df = df.join(port_info)
        in_port_count = int(df["in_port"].sum())
        print(f"Points in port: {in_port_count} / {len(df)} ({100*in_port_count/len(df):.1f}%)")
    else:
        print(f"Port directory not found at {ports_dir}, skipping port matching")

    output = Path(args.output or str(Path(cfg.paths.preprocessed_dir) / "preprocessed.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved to {output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
