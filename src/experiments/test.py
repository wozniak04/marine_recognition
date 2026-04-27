from __future__ import annotations

from src.experiments.pipeline import ExperimentBase
from src.experiments.registry import register_experiment

@register_experiment("brak")
class BrakExperiment(ExperimentBase):
    @classmethod
    def name(cls) -> str:
        return "brak"
    
    def build_model(self) -> BrakExperiment:
        return BrakExperiment()