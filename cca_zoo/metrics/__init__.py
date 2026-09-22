"""Metrics for evaluating fitted multiview CCA models.

Functions take already-computed arrays (latent scores, loadings, a
correlation matrix) rather than a fitted model or raw views, the same
convention ``sklearn.metrics`` uses. ``BaseModel``'s own
``pairwise_correlations``/``average_pairwise_correlations``/
``get_factor_loadings`` methods are the usual way to get those inputs from
a fitted model and a set of views.
"""

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
    "pairwise_correlations",
    "average_pairwise_correlations",
    "factor_loadings",
    "adequacy_coefficient",
    "redundancy_index",
    "total_redundancy",
]
