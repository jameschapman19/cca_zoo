"""scikit-learn's estimator checks, on every model through MultiviewWrapper.

sklearn's checks fit and transform one array. MultiviewWrapper splits it into
views, so the checks test the wrapper and, through it, each model: input
validation, fitted-state errors, pickling, cloning, idempotence and
invariance to sample order.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import (
    check_estimator_repr,
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
    parametrize_with_checks,
)

from cca_zoo._base import BaseModel
from cca_zoo.linear import RidgeCCA
from cca_zoo.model_selection import (
    GridSearchCV,
    MultiviewWrapper,
    RandomizedSearchCV,
)
from cca_zoo.preprocessing import PerViewTransformer
from tests._helpers import MODEL_CLASSES, SLOW_MODULES, make_model


class TwoViewAdapter(MultiviewWrapper):
    """MultiviewWrapper for sklearn's checks, whose data vary in width.

    Four or more features are halved into two views; fewer are used as both.
    """

    def __init__(self, estimator: BaseModel) -> None:
        self.estimator = estimator

    def _view_widths(self, n_features: int) -> list[int]:
        half = n_features // 2
        return [half, n_features - half] if n_features >= 4 else [n_features]

    def _split_views(self, X: ArrayLike, reset: bool) -> list[np.ndarray]:
        views = super()._split_views(X, reset)
        return views if len(views) == 2 else views * 2


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
