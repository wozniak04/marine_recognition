from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

class PathConfig(BaseModel):
    raw_csv: Path
    processed_dir: Path
    
class DataConfig(BaseModel):
    target_column: str
    feature_columns: list[str]
    class_names: list[str]
    test_size: float = Field(gt=0.0, lt=1.0)
    validation_size: float = Field(gt=0.0, lt=1.0)
    
class PreprocessingConfig(BaseModel):
    strategy: str
    
class ModelConfig(BaseModel):
    name: Literal["brak"]
    
class ProjectConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    
    project_name: str
    experiment_name: str
    random_seed: int = 10
    
    paths: PathConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> ProjectConfig:
        path = Path(config_path)
        
        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return cls(**config)
    
    