import argparse
from pathlib import Path

from src.data.dataset import read_dataset
from src.utils.logger import create_logger
from src.data.preproccesing import DataPreprocessor

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data for experiment.")
    parser.add_argument("--dataset", required=True, help="Path to raw dataset file")
    
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    data_path = Path(args.dataset)
    logger = create_logger("preprocess_data")
    
    df = read_dataset(data_path)
    preprocessor = DataPreprocessor(drop_prev=True)
    df_clean = preprocessor.process(df)
    
    df_clean.to_csv(f"data/processed/{data_path.stem}.csv", index=False)
    logger.info(f"Preprocessing wykonany dla: {data_path}")
    return 0
    
if __name__ == "__main__":
    raise SystemExit(main())