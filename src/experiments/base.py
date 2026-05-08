from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dataset import load_classified_data, load_port_database
from src.data.outliers import OutlierDetector
from src.data.ports import PortMatcher
from src.data.preprocessing import DataPreprocessor
from src.data.splitter import split_by_episodes
from src.evaluation.metrics import compute_all_metrics, save_metrics
from src.utils.config import OPERATION_ID_TO_STATE, ProjectConfig
from src.utils.logger import create_logger
from src.utils.seeding import set_global_seed


class BaseExperiment(ABC):
    """Base class for all ship state detection experiments."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.logger = create_logger(config.experiment_name)
        self.processed_dir, self.models_dir, self.reports_dir = config.experiment_paths()

    @classmethod
    @abstractmethod
    def name(cls) -> str: ...

    @abstractmethod
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Any: ...

    @abstractmethod
    def predict(self, model: Any, df: pd.DataFrame) -> pd.DataFrame: ...

    def run(self) -> dict:
        set_global_seed(self.config.random_seed)

        df = self._load_and_preprocess()

        df["state"] = df["operation_id"].map(OPERATION_ID_TO_STATE)

        splits = split_by_episodes(df, self.config.training, state_col="operation_id", seed=self.config.random_seed)

        model = self.train(splits["train"], splits["val"])

        results = {}
        for split_name, split_df in splits.items():
            predicted = self.predict(model, split_df)
            metrics = compute_all_metrics(
                predicted["state"].values,
                predicted["predicted_state"].values,
                self.config.class_names,
            )
            results[split_name] = metrics
            self.logger.info("%s accuracy: %.4f", split_name, metrics["overall_accuracy"])

        save_metrics(results, self.reports_dir / "metrics.json")

        test_predicted = self.predict(model, splits["test"])
        test_predicted.to_csv(self.reports_dir / "test_predictions.csv", index=False)

        full_predicted = self.predict(model, df)
        self._export_csv(full_predicted)

        return results

    def _load_and_preprocess(self) -> pd.DataFrame:
        df = load_classified_data(self.config)

        preprocessor = DataPreprocessor(self.config)
        df = preprocessor.process(df)

        detector = OutlierDetector(self.config.outlier)
        df["outlier_gps"] = detector.detect(df).astype(int)

        ports_df = load_port_database(self.config)
        matcher = PortMatcher(ports_df, self.config.port)
        port_info = matcher.query_batch(df["LAT"].values, df["LON"].values)
        df = df.join(port_info)

        df.to_csv(self.processed_dir / "preprocessed.csv", index=False)
        return df

    def _export_csv(self, df: pd.DataFrame) -> Path:
        """Generate the final CSV matching the project brief format."""
        departure, destination = self._compute_port_locodes(df)

        output = pd.DataFrame({
            "signaldate": df["signaldate"],
            "LAT": df["LAT"],
            "LON": df["LON"],
            "Outlier GPS": df["outlier_gps"],
            "In Port": (df["predicted_state"] == "port_stay").astype(int),
            "At Sea Voyage": (df["predicted_state"] == "voyage").astype(int),
            "At Sea Anchor": (df["predicted_state"] == "anchor").astype(int),
            "At Sea Adrift GPS": (df["predicted_state"] == "adrift").astype(int),
            "Confidence Rate": df["confidence"].round(1),
            "Departure Port Locode": departure,
            "Destination Port Locode": destination,
        })

        path = self.reports_dir / "output.csv"
        output.to_csv(path, index=False)
        self.logger.info("Exported final CSV to %s (%d rows)", path, len(output))
        return path

    def _compute_port_locodes(self, df: pd.DataFrame) -> tuple[list, list]:
        """Determine departure/destination port for each voyage segment."""
        states = df["predicted_state"].values
        locodes = df.get("nearest_locode")
        n = len(df)

        departure = [None] * n
        destination = [None] * n

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

        if locodes is None:
            return departure, destination

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
