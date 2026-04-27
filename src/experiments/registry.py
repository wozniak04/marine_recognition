from src.experiments.pipeline import ExperimentBase

REGISTRY: dict[str, type[ExperimentBase]] = {}

def register_experiment(name: str):
    def decorator(cls: type[ExperimentBase]) -> type[ExperimentBase]:
        REGISTRY[name] = cls
        return cls

    return decorator

class Experiment:
    @staticmethod
    def build(name: str, config) -> ExperimentBase:
        if name not in REGISTRY:
            available = ", ".join(sorted(REGISTRY))
            raise ValueError(f"Nieznany eksperyment: {name}. Dostępne eksperymenty: {available}")
        return REGISTRY[name](config)