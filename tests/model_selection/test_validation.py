"""Tests for the multiview cross-validation functions."""

from __future__ import annotations

import numpy as np
import sklearn.model_selection as skms

from cca_zoo.linear import CCA, RidgeCCA
from cca_zoo.model_selection import (
    cross_val_predict,
    cross_val_score,
    cross_validate,
    learning_curve,
    validation_curve,
)
from cca_zoo.model_selection._search import _MultiviewWrapper


def _views(n: int = 60) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n, 1))
    return [
        z @ rng.standard_normal((1, p)) + rng.standard_normal((n, p)) for p in (5, 4)
    ]


def test_cross_val_score_matches_sklearn_on_the_wrapper() -> None:
    """The function is sklearn's on the stacked views."""
    views = _views()
    wrapper = _MultiviewWrapper(CCA(), n_features_per_view=[5, 4])
    expected = skms.cross_val_score(wrapper, np.hstack(views), cv=3)
    np.testing.assert_allclose(cross_val_score(CCA(), views, cv=3), expected)


def test_cross_val_score_passes_groups() -> None:
    """Keyword arguments such as groups reach sklearn."""
    groups = np.repeat(np.arange(3), 20)
    scores = cross_val_score(CCA(), _views(), cv=skms.GroupKFold(3), groups=groups)
    assert scores.shape == (3,)


def test_cross_validate_returns_fitted_multiview_estimators() -> None:
    """return_estimator gives the multiview estimators, not the wrappers."""
    results = cross_validate(CCA(), _views(), cv=3, return_estimator=True)
    assert all(isinstance(est, CCA) for est in results["estimator"])
    assert results["test_score"].shape == (3,)


def test_cross_val_predict_transforms_each_fold_with_the_others_model() -> None:
    """Each view's scores are out-of-fold transforms."""
    views = _views()
    predicted = cross_val_predict(CCA(n_components=2), views, cv=3)
    for train, test in skms.KFold(3).split(views[0]):
        model = CCA(n_components=2).fit([v[train] for v in views])
        for scores, expected in zip(
            predicted, model.transform([v[test] for v in views])
        ):
            np.testing.assert_allclose(scores[test], expected)


def test_learning_curve_shapes() -> None:
    """One row of scores per training size."""
    sizes, train, test = learning_curve(CCA(), _views(), train_sizes=[0.5, 1.0], cv=3)
    assert sizes.shape == (2,)
    assert train.shape == test.shape == (2, 3)


def test_validation_curve_varies_one_views_parameter() -> None:
    """A per-view name varies that view's value only."""
    views = _views()
    _, test = validation_curve(RidgeCCA(), views, "shrinkage__1", [0.0, 1.0], cv=3)
    for value, row in zip([0.0, 1.0], test):
        expected = cross_val_score(RidgeCCA(shrinkage=[0.0, value]), views, cv=3)
        np.testing.assert_allclose(row, expected)
