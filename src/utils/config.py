from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PathsConfig(BaseModel):
    raw_dir: Path = Path("data/raw/classified")
    ports_csv: Path = Path("data/ports/EU_Port_Codes.csv")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")


class DataConfig(BaseModel):
    source: Literal["gps", "ais", "gps+ais"] = "gps"
    input_files: list[str] = []
    labeled: bool = False
    sampling_interval_seconds: int = 60


class PreprocessingConfig(BaseModel):
    lat_range: tuple[float, float] = (-90.0, 90.0)
    lon_range: tuple[float, float] = (-180.0, 180.0)
    drop_zero_coords: bool = True
    smoothing_window: int = 3
    apply_smoothing: bool = False


class OutlierConfig(BaseModel):
    method: Literal["iqr", "zscore", "isolation_forest", "combined"] = "iqr"
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0
    contamination: float = 0.01
    max_speed_knots: float = 30.0
    columns: list[str] = Field(default_factory=lambda: ["sog_ms"])


class PortConfig(BaseModel):
    radius_meters: float = 2000.0


class FeatureConfig(BaseModel):
    window_size: int = 10
    stride: int = 1
    aggregations: list[str] = Field(
        default_factory=lambda: ["mean", "std", "min", "max", "median"]
    )
    feature_columns: list[str] | None = None


class ModelConfig(BaseModel):
    name: Literal[
        "rule_based",
        "random_forest",
        "xgboost",
        "lightgbm",
        "hmm",
        "lstm",
        "cnn_1d",
        "ensemble",
        "extended_ais",
        "hierarchical_lstm",
    ] = "rule_based"
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingConfig(BaseModel):
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    validation_size: float = Field(0.15, gt=0.0, lt=1.0)
    split_strategy: Literal["episode", "temporal", "random"] = "episode"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    class_weights: Literal["balanced", "none"] = "balanced"


OPERATION_ID_TO_STATE: dict[int, str] = {
    1: "port_stay",
    2: "anchor",
    3: "adrift",
    4: "voyage",
}

STATE_TO_COLUMNS: dict[str, str] = {
    "port_stay": "In Port",
    "voyage": "At Sea Voyage",
    "anchor": "At Sea Anchor",
    "adrift": "At Sea Adrift GPS",
}


class ProjectConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    project_name: str = "marine_recognition"
    experiment_name: str
    variant: Literal["A", "B"] = "A"
    random_seed: int = 42

    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    outlier: OutlierConfig = OutlierConfig()
    port: PortConfig = PortConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    class_names: list[str] = Field(
        default_factory=lambda: ["port_stay", "voyage", "anchor", "adrift"]
    )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> ProjectConfig:
        path = Path(config_path)
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def experiment_paths(self) -> tuple[Path, Path, Path]:
        """Return (processed, models, reports) dirs scoped to this experiment."""
        processed = self.paths.processed_dir / self.experiment_name
        models = self.paths.models_dir / self.experiment_name
        reports = self.paths.reports_dir / self.experiment_name
        for d in (processed, models, reports):
            d.mkdir(parents=True, exist_ok=True)
        return processed, models, reports
