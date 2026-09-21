"""CCA-Zoo: Multiview Canonical Correlation Analysis library.

A scikit-learn style library implementing a wide range of multiview
Canonical Correlation Analysis methods including linear, kernel,
deep learning, tree-based, GAM-based, and probabilistic variants.
"""

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
