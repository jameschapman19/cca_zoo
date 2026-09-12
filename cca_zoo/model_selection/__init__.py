"""Model selection utilities for multiview CCA models."""

from __future__ import annotations

from cca_zoo.model_selection._search import (
    GridSearchCV,
    MultiviewWrapper,
    RandomizedSearchCV,
)
from cca_zoo.model_selection._significance import (
    PermutationTestResult,
    permutation_test_significance,
    procrustes_rotation,
)

__all__ = [
    "GridSearchCV",
    "MultiviewWrapper",
    "RandomizedSearchCV",
    "PermutationTestResult",
    "permutation_test_significance",
    "procrustes_rotation",
]
