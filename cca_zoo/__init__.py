"""Multiview canonical correlation analysis with a scikit-learn API."""

import importlib.metadata

__version__: str = importlib.metadata.version("cca_zoo")

__all__ = [
    "datasets",
    "deep",
    "gam",
    "gp",
    "linear",
    "metrics",
    "model_selection",
    "nonparametric",
    "preprocessing",
    "probabilistic",
    "sparse",
    "stochastic",
    "tree",
]
