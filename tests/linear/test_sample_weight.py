"""sample_weight on the models fitted from second moments, as sklearn defines it."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils.validation import has_fit_parameter

from cca_zoo._base import BaseModel
from cca_zoo.linear import (
    CCA,
    GCCA,
    GRCCA,
    MCCA,
    PLS,
    GraphicalLassoCCA,
    PartialCCA,
    RidgeCCA,
)
from cca_zoo.model_selection import GridSearchCV
from tests._helpers import linear_views

_WEIGHTED = [
    CCA(2),
    RidgeCCA(2, shrinkage=0.3),
    PLS(2),
    MCCA(2, shrinkage=0.2),
    GCCA(2, shrinkage=0.2),
    GRCCA(2),
]
_IDS = [type(m).__name__ for m in _WEIGHTED]


def _assert_equal_up_to_sign(a: list[np.ndarray], b: list[np.ndarray]) -> None:
    for x, y in zip(a, b):
        signs = np.sign(np.sum(x * y, axis=0))
        np.testing.assert_allclose(x, y * signs, atol=1e-8)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("model", _WEIGHTED, ids=_IDS)
def test_integer_weights_repeat_rows(model: BaseModel, center: bool) -> None:
    """Weighting a sample by k is fitting it k times."""
    train, test = linear_views(0, 40), linear_views(1, 20)
    weights = np.random.default_rng(2).integers(0, 4, 40)
    repeated = [np.repeat(v, weights, axis=0) for v in train]
    model.set_params(center=center)
    weighted = model.fit(train, sample_weight=weights).transform(test)
    _assert_equal_up_to_sign(weighted, model.fit(repeated).transform(test))


@pytest.mark.parametrize("model", _WEIGHTED, ids=_IDS)
def test_unit_weights_change_nothing(model: BaseModel) -> None:
    """Weights of one are no weights."""
    train, test = linear_views(0, 40), linear_views(1, 20)
    weighted = model.fit(train, sample_weight=np.ones(40)).transform(test)
    _assert_equal_up_to_sign(weighted, model.fit(train).transform(test))


def test_weights_are_validated() -> None:
    """Negative weights, or too little total weight for a covariance, raise."""
    views = linear_views(0, 40)
    with pytest.raises(ValueError, match="Negative values"):
        CCA().fit(views, sample_weight=-np.ones(40))
    with pytest.raises(ValueError, match="sum to more than 1"):
        CCA().fit(views, sample_weight=np.full(40, 0.02))


def test_row_based_estimators_take_no_weights() -> None:
    """The graphical lasso and the partials' regression use the rows themselves."""
    assert not has_fit_parameter(GraphicalLassoCCA(), "sample_weight")
    assert not has_fit_parameter(PartialCCA(), "sample_weight")


def test_search_routes_weights_to_each_fold() -> None:
    """A search slices sample_weight with the views, as sklearn's do.

    Its scores are unweighted correlations, and sklearn says so.
    """
    views = linear_views(0, 60)
    weights = np.random.default_rng(3).uniform(0.5, 2.0, 60)
    search = GridSearchCV(RidgeCCA(), {"shrinkage": [0.1, 0.5]}, cv=3)
    with pytest.warns(UserWarning, match="does not support sample_weight"):
        search.fit(views, sample_weight=weights)
    refit = RidgeCCA(shrinkage=search.best_params_["shrinkage"])
    expected = refit.fit(views, sample_weight=weights).transform(views)
    _assert_equal_up_to_sign(search.transform(views), expected)
