import pandas as pd

from abc import ABC, abstractmethod
from typing import Any

from src.utils.config import ProjectConfig
from src.utils.logger import create_logger

from src.data.preprocessing import DataPreprocessor

class ExperimentBase(ABC):
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.logger = create_logger(self.config.experiment_name)
        
    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError
    
    def build_model(self) -> Any:
        raise NotImplementedError
    
    def preprocess(self, df: pd.DataFrame) -> None:
        preprocessor = DataPreprocessor(drop_prev=True)
        df_clean = preprocessor.process(df)
        
        df_clean.to_csv(self.config.paths.processed_dir / "Ship_operation.csv", index=False)
