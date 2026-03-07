"""CCA-Zoo: Multiview Canonical Correlation Analysis library.

A scikit-learn style library implementing a wide range of multiview
Canonical Correlation Analysis methods including linear, kernel,
deep learning, and probabilistic variants.
"""

import importlib.metadata

__version__: str = importlib.metadata.version("cca_zoo")

__all__ = [
    "datasets",
    "deep",
    "linear",
    "model_selection",
    "nonparametric",
    "probabilistic",
]
