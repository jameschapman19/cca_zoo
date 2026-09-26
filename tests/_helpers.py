"""Helpers shared across test modules."""

from __future__ import annotations

import numpy as np

from cca_zoo.metrics import average_pairwise_correlations, pairwise_correlations


def canonical_correlations(model: object, views: list[np.ndarray]) -> np.ndarray:
    """Per-dimension mean pairwise canonical correlation of a fitted model."""
    return average_pairwise_correlations(pairwise_correlations(model.transform(views)))  # type: ignore[attr-defined]
