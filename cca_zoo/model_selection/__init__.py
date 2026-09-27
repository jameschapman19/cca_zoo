"""Hyperparameter search and permutation tests for multiview models."""

from __future__ import annotations

from cca_zoo.model_selection._search import (
    GridSearchCV,
    MultiviewWrapper,
    RandomizedSearchCV,
)
from cca_zoo.model_selection._significance import (
    PermutationTestResult,
    permutation_test_significance,
)
from cca_zoo.model_selection._validation import (
    cross_val_score,
    cross_validate,
    learning_curve,
    validation_curve,
)

__all__ = [
    "GridSearchCV",
    "MultiviewWrapper",
    "RandomizedSearchCV",
    "PermutationTestResult",
    "permutation_test_significance",
    "cross_val_score",
    "cross_validate",
    "learning_curve",
    "validation_curve",
]
