"""What a fit must not depend on, checked on every model.

The views' covariances do not change when the data set is stacked on
itself, its rows are shuffled or a whole view changes units; reordering a
view's columns only reorders its weights. Each property compares the
subspaces the latent scores span, since the Eckart-Young models fix only
the subspace. A model exempt from one names why.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._base import BaseModel
from tests._helpers import (
    MODEL_CLASSES,
    assert_same_subspace,
    linear_views,
    make_model,
    model_params,
)

_N = 150

# Models whose fit legitimately depends on the property, and why.
_RANDOM_TREES = "a random start and row and column subsampling: seeds alone differ"
_MONTE_CARLO = "its posterior is estimated by sampling"
_SHRINKAGE = "shrinkage=0.1 blends towards PLS, which depends on units"
_RANDOM_SUBSETS = "it fits random subsets of the rows"
_STOCHASTIC = {
    "XGBoostCCA": _RANDOM_TREES,
    "LightGBMCCA": _RANDOM_TREES,
    "CatBoostCCA": _RANDOM_TREES,
    "ProbabilisticCCA": _MONTE_CARLO,
    "VariationalBayesCCA": _MONTE_CARLO,
}
_FOLDS = "the cross-validation folds are contiguous blocks of rows"
_STACKING_EXEMPT = {
    **_STOCHASTIC,
    "ManifoldCCA": "a duplicated sample is its copy's nearest neighbour",
    "MARSCCA": "earth's minspan and endspan grow with the number of samples",
    "SAR": "BIC's penalty grows with log(n)",
    "RANSACCCA": _RANDOM_SUBSETS,
    "TrimmedCCA": _RANDOM_SUBSETS,
    "StochasticCCAEY": "an epoch's steps grow with the number of samples",
    "RidgeCCACV": _FOLDS,
}
_ROW_ORDER_EXEMPT = {
    **_STOCHASTIC,
    "RidgeCCACV": _FOLDS,
    "RANSACCCA": _RANDOM_SUBSETS,
    "TrimmedCCA": _RANDOM_SUBSETS,
}
_COLUMN_ORDER_EXEMPT = {
    **_STOCHASTIC,
    "ProjectionPursuitCCA": "each feature's random start weight follows its position",
    "StochasticCCAEY": "each feature's random start weight follows its position",
    "ElasticNetCCA": (
        "each feature's random start weight follows its position, and the "
        "penalised loss has more than one optimum"
    ),
}
_UNITS_EXEMPT = {
    "ProbabilisticCCA": _MONTE_CARLO,
    "VariationalBayesCCA": _MONTE_CARLO,
    "ElasticNetCCA": "alpha is an absolute L1 penalty, as in sklearn's ElasticNet",
    "MultiTaskElasticNetCCA": "alpha is an absolute penalty, as in sklearn",
    "IPLSCCA": "alpha is an absolute penalty, as in sklearn's Lasso",
    "ADMMCCA": "alpha is an absolute L1 penalty",
    "KCCA": _SHRINKAGE,
    "KGCCA": _SHRINKAGE,
    "KTCCA": _SHRINKAGE,
    "RANSACCCA": _SHRINKAGE,
    "TrimmedCCA": _SHRINKAGE,
}


def _scores(
    cls: type[BaseModel], fit_on: list[np.ndarray], score: list[np.ndarray]
) -> list[np.ndarray]:
    model = make_model(cls)
    model.set_params(n_components=1 if cls.__name__ == "TrimmedCCA" else 2)
    if cls.__name__ == "PartialCCA":
        # A partial variable of each row's own data follows the rows.
        model.fit(fit_on, partials=np.abs(fit_on[0]).sum(axis=1, keepdims=True))
        return model.transform(
            score, partials=np.abs(score[0]).sum(axis=1, keepdims=True)
        )
    return model.fit(fit_on).transform(score)


def _ids(exempt: dict[str, str]) -> list[object]:
    return model_params([c for c in MODEL_CLASSES if c.__name__ not in exempt])


@pytest.mark.parametrize("cls", _ids(_STACKING_EXEMPT))
def test_stacking_the_data_changes_nothing(cls: type[BaseModel]) -> None:
    """Stacking the data on itself keeps every covariance."""
    views = linear_views(0, _N, (8, 6))
    stacked = [np.vstack([v, v]) for v in views]
    assert_same_subspace(_scores(cls, views, views), _scores(cls, stacked, views))


@pytest.mark.parametrize("cls", _ids(_ROW_ORDER_EXEMPT))
def test_row_order_changes_nothing(cls: type[BaseModel]) -> None:
    """Samples are exchangeable."""
    views = linear_views(0, _N, (8, 6))
    order = np.random.default_rng(1).permutation(_N)
    shuffled = [v[order] for v in views]
    assert_same_subspace(_scores(cls, views, views), _scores(cls, shuffled, views))


@pytest.mark.parametrize("cls", _ids(_COLUMN_ORDER_EXEMPT))
def test_column_order_only_reorders_the_weights(cls: type[BaseModel]) -> None:
    """Features are exchangeable."""
    views = linear_views(0, _N, (8, 6))
    rng = np.random.default_rng(1)
    permuted = [v[:, rng.permutation(v.shape[1])] for v in views]
    assert_same_subspace(_scores(cls, views, views), _scores(cls, permuted, permuted))


@pytest.mark.parametrize("cls", _ids(_UNITS_EXEMPT))
def test_a_views_units_change_nothing(cls: type[BaseModel]) -> None:
    """Rescaling a whole view keeps every correlation."""
    views = linear_views(0, _N, (8, 6))
    rescaled = [views[0] * 1000.0, views[1] / 1000.0]
    assert_same_subspace(_scores(cls, views, views), _scores(cls, rescaled, rescaled))
