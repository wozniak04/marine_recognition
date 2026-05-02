"""CLI: Compare all experiment results.

Usage: uv run python -m src.cli.compare
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    reports_dir = Path("reports")

    experiments: list[dict] = []
    for exp_dir in sorted(reports_dir.iterdir()):
        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        with metrics_file.open() as f:
            metrics = json.load(f)

        experiments.append({"name": exp_dir.name, "metrics": metrics})

    if not experiments:
        print("No experiments found in reports/")
        return

    states = ["port_stay", "voyage", "anchor", "adrift"]
    splits = ["train", "val", "test"]

    print("\n" + "=" * 90)
    print("MODEL COMPARISON — Overall Accuracy")
    print("=" * 90)
    header = f"{'Model':<25}"
    for split in splits:
        header += f"{'':>3}{split:>10}"
    print(header)
    print("-" * 90)

    for exp in experiments:
        row = f"{exp['name']:<25}"
        for split in splits:
            if split in exp["metrics"]:
                acc = exp["metrics"][split]["overall_accuracy"]
                row += f"{'':>3}{acc:>10.4f}"
            else:
                row += f"{'':>3}{'N/A':>10}"
        print(row)

    print("\n" + "=" * 90)
    print("PER-CLASS F1 SCORES (Validation Set)")
    print("=" * 90)
    header = f"{'Model':<25}"
    for s in states:
        header += f"  {s:>12}"
    print(header)
    print("-" * 90)

    for exp in experiments:
        row = f"{exp['name']:<25}"
        val = exp["metrics"].get("val", {})
        per_class = val.get("per_class", {})
        for s in states:
            if s in per_class:
                f1 = per_class[s]["f1"]
                row += f"  {f1:>12.4f}"
            else:
                row += f"  {'N/A':>12}"
        print(row)

    print("\n" + "=" * 90)
    print("PER-CLASS F1 SCORES (Test Set)")
    print("=" * 90)
    header = f"{'Model':<25}"
    for s in states:
        header += f"  {s:>12}"
    print(header)
    print("-" * 90)

    for exp in experiments:
        row = f"{exp['name']:<25}"
        test = exp["metrics"].get("test", {})
        per_class = test.get("per_class", {})
        for s in states:
            if s in per_class:
                f1 = per_class[s]["f1"]
                row += f"  {f1:>12.4f}"
            else:
                row += f"  {'N/A':>12}"
        print(row)

    print()


if __name__ == "__main__":
    main()
