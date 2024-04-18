"""
Utilities for visualising CCA results.
"""
import importlib.util

if importlib.util.find_spec("seaborn") is None or importlib.util.find_spec(
    "matplotlib.pyplot"
):
    print(
        "Warning: seaborn or matplotlib are not installed. Some functionality in cca_zoo.visualisation may be limited."
    )
from .correlation import CorrelationHeatmapDisplay
from .covariance import CovarianceHeatmapDisplay
from .explained_covariance import ExplainedCovarianceDisplay
from .explained_variance import ExplainedVarianceDisplay
from .inference import WeightInferenceDisplay
from .representations import (
    RepresentationScatterDisplay,
    JointRepresentationScatterDisplay,
    SeparateRepresentationScatterDisplay,
    SeparateJointRepresentationDisplay,
    PairRepresentationScatterDisplay,
    TSNERepresentationDisplay,
    UMAPRepresentationDisplay,
)
from .weights import WeightHeatmapDisplay

__all__ = [
    "ExplainedVarianceDisplay",
    "RepresentationScatterDisplay",
    "JointRepresentationScatterDisplay",
    "SeparateRepresentationScatterDisplay",
    "SeparateJointRepresentationDisplay",
    "PairRepresentationScatterDisplay",
    "ExplainedCovarianceDisplay",
    "WeightHeatmapDisplay",
    "CorrelationHeatmapDisplay",
    "CovarianceHeatmapDisplay",
    "WeightInferenceDisplay",
]
