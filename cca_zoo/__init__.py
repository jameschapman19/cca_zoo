import importlib.metadata

__version__ = importlib.metadata.version("cca_zoo")

__all__ = [
    "datasets",
    "deep",
    "linear",
    "model_selection",
    "visualisation",
    "preprocessing",
    "sequential",
    "nonparametric",
    "probabilistic",
]
