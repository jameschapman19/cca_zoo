"""Canonical gradient boosting: supervised learners that borrow CCA.

Gradient boosting fits each new tree to the current loss gradients using
axis-aligned splits. The estimators here additionally give each tree the
directions of feature space that are maximally *canonically correlated*
with the gradient matrix, so a single split can cut along an oblique
steepest-descent direction.
"""

from __future__ import annotations

from cca_zoo.boosting._canonical_boosting import (
    CanonicalBoostingClassifier,
    CanonicalBoostingRegressor,
)

__all__ = ["CanonicalBoostingClassifier", "CanonicalBoostingRegressor"]
