"""Model entry points; modules are imported only when selected."""

from importlib import import_module


MODEL_REGISTRY = {
    "RF": "src.model.RF",
    "GCN_RF": "src.model.GCN_RF",
    "random_score": "src.model.random_score",
    "single_feature": "src.model.single_feature",
    "isolation_forest": "src.model.isolation_forest",
    "AE": "src.model.AE",
    "GAE": "src.model.GAE",
}


def validate_model_name(name):
    if name not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY)
        raise ValueError(f"Model '{name}' is not supported. Supported models: {supported}")
    return name


def get_model_runner(name):
    validate_model_name(name)
    return import_module(MODEL_REGISTRY[name]).run
