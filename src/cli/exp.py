import argparse
from pathlib import Path

from src.data.dataset import read_dataset
from src.utils.logger import create_logger
from src.model.nn import train_classifier

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data for experiment.")
    
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    logger = create_logger("Experiments")
    
    df = read_dataset("data/processed/Ship_Operation_example_dataset_classified_2.csv")
    r = train_classifier(
        df,
        [32, 16],
        feature_names=['traveled', 'dVelocity_m_s', 'dCOG', 'acc', 'omega', 'is_first_in_segment'],
        target_col='operation_id',
        target_classes=[1, 2, 4],
        class_names=["in_port", "at_sea_anchor", "at_sea_voyage"],
        # class_names=["in_port", "at_sea_anchor", "at_sea_drift", "at_sea_voyage"],
        plot_path="visualization/plot.png"
    )
    
    return 0
    
if __name__ == "__main__":
    raise SystemExit(main())