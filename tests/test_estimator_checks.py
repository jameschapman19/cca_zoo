"""scikit-learn's estimator checks, run on every model through a two-view adapter.

sklearn's checks fit and transform a single array ``X``. The adapter splits
it into two views and otherwise passes it straight through, so input
validation, fitted-state checks, pickling, cloning, idempotence and
invariance to sample order are the models' own.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import (
    check_estimator_repr,
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
    parametrize_with_checks,
)
from sklearn.utils.validation import check_is_fitted, validate_data

from cca_zoo._base import BaseModel
from cca_zoo.linear import RidgeCCA
from cca_zoo.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    MultiviewWrapper,
    RandomizedSearchCV,
)
from cca_zoo.preprocessing import PerViewTransformer
from tests._helpers import MODEL_CLASSES, SLOW_MODULES, make_model


class TwoViewAdapter(TransformerMixin, BaseEstimator):
    """A multiview model as a transformer of one array.

    Four or more features are split into two views; fewer are used as both
    views. Scores are concatenated.
    """

    def __init__(self, estimator: BaseModel) -> None:
        self.estimator = estimator

    def _views(self, X: Any, reset: bool) -> list[Any]:
        X = X if hasattr(X, "shape") else np.asarray(X)
        if X.ndim != 2:
            return [X, X]  # the model rejects it
        if reset:
            self.n_features_in_ = X.shape[1]
        else:
            validate_data(self, X, reset=False, skip_check_array=True)
        half = X.shape[1] // 2
        return [X[:, :half], X[:, half:]] if X.shape[1] >= 4 else [X, X]

    def fit(self, X: ArrayLike, y: None = None) -> TwoViewAdapter:
        """Fit the model on the views of ``X``."""
        self.estimator_ = clone(self.estimator).fit(self._views(X, reset=True))
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """The views' scores, side by side."""
        check_is_fitted(self)
        return np.hstack(self.estimator_.transform(self._views(X, reset=False)))

    def score(self, X: ArrayLike, y: None = None) -> float:
        """The model's score on the views of ``X``."""
        check_is_fitted(self)
        return self.estimator_.score(self._views(X, reset=False))


_MARS_SMALL_DATA = (
    "MARS needs more samples per view than the check's data to place a knot"
)
_EXPECTED_FAILURES = {
    "MARSCCA": {
        "check_estimators_nan_inf": _MARS_SMALL_DATA,
        "check_fit2d_1feature": _MARS_SMALL_DATA,
        "check_n_features_in_after_fitting": _MARS_SMALL_DATA,
    },
}


def _expected_failures(adapter: TwoViewAdapter) -> dict[str, str]:
    return _EXPECTED_FAILURES.get(type(adapter.estimator).__name__, {})


def _adapters(slow: bool) -> list[TwoViewAdapter]:
    return [
        TwoViewAdapter(make_model(cls))
        for cls in MODEL_CLASSES
        if cls.__name__ != "PartialCCA"  # fit requires the partials
        and (cls.__module__.rsplit(".", 1)[0] in SLOW_MODULES) == slow
    ]


@parametrize_with_checks(
    _adapters(slow=False), expected_failed_checks=_expected_failures
)
def test_sklearn_estimator_checks(estimator: TwoViewAdapter, check: Any) -> None:
    """Each model passes scikit-learn's estimator checks."""
    check(estimator)


@pytest.mark.slow
@parametrize_with_checks(
    _adapters(slow=True), expected_failed_checks=_expected_failures
)
def test_sklearn_estimator_checks_slow(estimator: TwoViewAdapter, check: Any) -> None:
    """Each model with a slow backend passes scikit-learn's estimator checks."""
    check(estimator)


# Meta-estimators fit on views, so sklearn's fitting checks cannot build their
# data; its construction checks apply unchanged.
_META_ESTIMATORS = [
    GridSearchCV(RidgeCCA(), param_grid={"c": [0.1]}),
    RandomizedSearchCV(RidgeCCA(), param_distributions={"c": [0.1]}),
    HalvingGridSearchCV(RidgeCCA(), param_grid={"c": [0.1]}),
    HalvingRandomSearchCV(RidgeCCA(), param_distributions={"c": [0.1]}),
    MultiviewWrapper(RidgeCCA(), n_features_per_view=[2, 2]),
    PerViewTransformer(StandardScaler()),
]
_CONSTRUCTION_CHECKS = [
    check_no_attributes_set_in_init,
    check_get_params_invariance,
    check_set_params,
    check_estimator_repr,
    check_parameters_default_constructible,
]


@pytest.mark.parametrize(
    "estimator", _META_ESTIMATORS, ids=[type(e).__name__ for e in _META_ESTIMATORS]
)
@pytest.mark.parametrize(
    "check", _CONSTRUCTION_CHECKS, ids=[c.__name__ for c in _CONSTRUCTION_CHECKS]
)
def test_meta_estimator_construction(estimator: BaseEstimator, check: Any) -> None:
    """The searches and adapters follow sklearn's constructor conventions."""
    check(type(estimator).__name__, estimator)
