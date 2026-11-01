"""Metrics on latent scores and loadings, in the style of :mod:`sklearn.metrics`."""

from __future__ import annotations

from cca_zoo.metrics._correlation import (
    average_pairwise_correlations,
    factor_loadings,
    pairwise_correlations,
)
from cca_zoo.metrics._redundancy import (
    adequacy_coefficient,
    redundancy_index,
    total_redundancy,
)

__all__ = [
    "adequacy_coefficient",
    "average_pairwise_correlations",
    "factor_loadings",
    "pairwise_correlations",
    "redundancy_index",
    "total_redundancy",
]
