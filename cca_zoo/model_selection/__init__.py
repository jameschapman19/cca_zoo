"""Hyperparameter search and permutation tests for multiview models."""

from __future__ import annotations

from cca_zoo.model_selection import _search
from cca_zoo.model_selection._search import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
)
from cca_zoo.model_selection._significance import (
    PermutationTestResult,
    permutation_test_significance,
)
from cca_zoo.model_selection._validation import (
    cross_val_predict,
    cross_val_score,
    cross_validate,
    learning_curve,
    validation_curve,
)

__all__ = [
    "GridSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
    "PermutationTestResult",
    "RandomizedSearchCV",
    "cross_val_predict",
    "cross_val_score",
    "cross_validate",
    "learning_curve",
    "permutation_test_significance",
    "validation_curve",
]

if hasattr(_search, "OptunaSearchCV"):
    OptunaSearchCV = _search.OptunaSearchCV
    __all__.append("OptunaSearchCV")
else:

    def __getattr__(name: str) -> object:
        if name == "OptunaSearchCV":
            raise ImportError(
                "OptunaSearchCV requires optuna-integration: "
                "pip install 'cca-zoo[optuna]'."
            )
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
