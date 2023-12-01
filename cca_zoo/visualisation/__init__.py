"""
Utilities for visualising CCA results.
"""
try:
    import seaborn
    import matplotlib.pyplot as plt
except ImportError as error:
    print(
        "Warning: seaborn or matplotlib are not installed. Some functionality in cca_zoo.visualisation may be limited."
    )
from .correlation import CorrelationHeatmapDisplay
from .covariance import CovarianceHeatmapDisplay
from .explained_covariance import ExplainedCovarianceDisplay
from .explained_variance import ExplainedVarianceDisplay
from .inference import WeightInferenceDisplay
from .scores import (
    ScoreScatterDisplay,
    JointScoreScatterDisplay,
    SeparateScoreScatterDisplay,
    SeparateJointScoreDisplay,
    PairScoreScatterDisplay,
)
from .tsne_scores import TSNEScoreDisplay
from .umap_scores import UMAPScoreDisplay
from .weights import WeightHeatmapDisplay

__all__ = [
    "ExplainedVarianceDisplay",
    "ScoreScatterDisplay",
    "JointScoreScatterDisplay",
    "SeparateScoreScatterDisplay",
    "SeparateJointScoreDisplay",
    "PairScoreScatterDisplay",
    "ExplainedCovarianceDisplay",
    "WeightHeatmapDisplay",
    "CorrelationHeatmapDisplay",
    "CovarianceHeatmapDisplay",
    "TSNEScoreDisplay",
    "UMAPScoreDisplay",
    "WeightInferenceDisplay",
]
