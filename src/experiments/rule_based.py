from __future__ import annotations

from typing import Any

import pandas as pd

from src.experiments.base import BaseExperiment
from src.experiments.registry import register_experiment
from src.models.rule_engine import ShipStateFSM
from src.utils.config import ProjectConfig


@register_experiment("rule_based")
class RuleBasedExperiment(BaseExperiment):

    @classmethod
    def name(cls) -> str:
        return "rule_based"

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> ShipStateFSM:
        fsm = ShipStateFSM(self.config.model.params)
        self.logger.info("Rule-based model: no training needed, using configured thresholds")
        self.logger.info("Params: %s", self.config.model.params)
        return fsm

    def predict(self, model: Any, df: pd.DataFrame) -> pd.DataFrame:
        fsm: ShipStateFSM = model
        return fsm.classify(df)
