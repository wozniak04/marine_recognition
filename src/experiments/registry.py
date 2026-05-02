from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.experiments.base import BaseExperiment
    from src.utils.config import ProjectConfig

REGISTRY: dict[str, type[BaseExperiment]] = {}


def register_experiment(name: str):
    def decorator(cls: type[BaseExperiment]) -> type[BaseExperiment]:
        REGISTRY[name] = cls
        return cls
    return decorator


def build_experiment(name: str, config: ProjectConfig) -> BaseExperiment:
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown experiment: {name}. Available: {available}")
    return REGISTRY[name](config)
