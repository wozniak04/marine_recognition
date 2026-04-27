import argparse

from src.data.dataset import read_dataset
from src.utils.config import ProjectConfig
from src.utils.logger import create_logger

from src.experiments.registry import Experiment

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data for experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML file with configuration")
    
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    config = ProjectConfig.from_yaml(args.config)
    logger = create_logger("preprocess_data")
    
    df = read_dataset(config.paths.raw_csv)
    experiment = Experiment.build(config.model.name, config)
    experiment.preprocess(df)
    logger.info(f"Preprocessing wykonany dla: {config.experiment_name}")
    return 0
    
if __name__ == "__main__":
    raise SystemExit(main())