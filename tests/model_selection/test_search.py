"""The multiview hyperparameter searches."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import sklearn.model_selection as skms
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cca_zoo.linear import CCA, RidgeCCA
from cca_zoo.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    cross_validate,
)
from cca_zoo.model_selection._search import _MultiviewWrapper
from cca_zoo.preprocessing import PerViewTransformer

_GRID = {"shrinkage__0": [0.0, 0.5], "shrinkage__1": [0.1, 0.9]}
_SEARCHES = [
    (GridSearchCV, {"param_grid": _GRID}),
    (
        RandomizedSearchCV,
        {"param_distributions": _GRID, "n_iter": 3, "random_state": 0},
    ),
    (HalvingGridSearchCV, {"param_grid": _GRID, "min_resources": 20}),
    (
        HalvingRandomSearchCV,
        {"param_distributions": _GRID, "min_resources": 20, "random_state": 0},
    ),
]


@pytest.mark.parametrize(
    ("search", "kwargs"), _SEARCHES, ids=[s.__name__ for s, _ in _SEARCHES]
)
def test_search_over_per_view_parameters(
    search: type, kwargs: dict[str, Any], two_views: list[np.ndarray]
) -> None:
    """Each view's value is searched independently and reported without prefixes."""
    fitted = search(RidgeCCA(), cv=2, **kwargs).fit(two_views)
    assert set(fitted.best_params_) == {"shrinkage__0", "shrinkage__1"}
    assert fitted.best_estimator_.shrinkage == [
        fitted.best_params_["shrinkage__0"],
        fitted.best_params_["shrinkage__1"],
    ]
    assert not any("estimator__" in key for key in fitted.cv_results_)
    assert len(fitted.transform(two_views)) == 2
    assert isinstance(fitted.score(two_views), float)


@pytest.mark.parametrize(
    ("ours", "theirs"),
    [
        (GridSearchCV, skms.GridSearchCV),
        (RandomizedSearchCV, skms.RandomizedSearchCV),
        (HalvingGridSearchCV, skms.HalvingGridSearchCV),
        (HalvingRandomSearchCV, skms.HalvingRandomSearchCV),
    ],
    ids=lambda c: c.__name__,
)
def test_parameters_are_the_upstream_searchs(ours: type, theirs: type) -> None:
    """Each search takes exactly the parameters of the sklearn search it extends."""
    assert ours._get_param_names() == theirs._get_param_names()


@pytest.mark.slow
def test_optuna_search_over_per_view_parameters(two_views: list[np.ndarray]) -> None:
    """OptunaSearchCV searches per-view values and returns the multiview model."""
    distributions = pytest.importorskip("optuna.distributions")
    from cca_zoo.model_selection import OptunaSearchCV

    space = {
        f"shrinkage__{i}": distributions.FloatDistribution(0.0, 1.0) for i in (0, 1)
    }
    fitted = OptunaSearchCV(RidgeCCA(), space, n_trials=4, cv=2, random_state=0)
    fitted.fit(two_views)
    assert fitted.best_estimator_.shrinkage == [
        fitted.best_params_["shrinkage__0"],
        fitted.best_params_["shrinkage__1"],
    ]
    assert len(fitted.transform(two_views)) == 2
    assert isinstance(fitted.score(two_views), float)


def test_grid_search_is_sklearns_on_the_stacked_views(
    two_views: list[np.ndarray],
) -> None:
    """Scores and choice match sklearn's GridSearchCV on the stacked views."""
    ours = GridSearchCV(
        RidgeCCA(), param_grid={"shrinkage": [0.0, 0.3, 0.9]}, cv=3
    ).fit(two_views)
    theirs = skms.GridSearchCV(
        _MultiviewWrapper(RidgeCCA(), n_features_per_view=[10, 8]),
        param_grid={"estimator__shrinkage": [0.0, 0.3, 0.9]},
        cv=3,
    ).fit(np.hstack(two_views))
    np.testing.assert_allclose(
        ours.cv_results_["mean_test_score"], theirs.cv_results_["mean_test_score"]
    )
    assert ours.best_index_ == theirs.best_index_
    assert ours.n_splits_ == 3


def test_grid_search_over_a_list_of_grids(two_views: list[np.ndarray]) -> None:
    """Disjoint grids are searched in turn."""
    gs = GridSearchCV(
        RidgeCCA(), param_grid=[{"shrinkage": [0.0]}, {"n_components": [1, 2]}], cv=2
    ).fit(two_views)
    assert len(gs.cv_results_["params"]) == 3


def test_per_view_override_keeps_the_other_views(two_views: list[np.ndarray]) -> None:
    """Views missing from the grid keep the estimator's value."""
    gs = GridSearchCV(
        RidgeCCA(shrinkage=0.3), param_grid={"shrinkage__0": [0.0, 0.9]}, cv=2
    )
    assert gs.fit(two_views).best_estimator_.shrinkage[1] == 0.3


def test_per_view_value_overrides_a_whole_model_value(
    two_views: list[np.ndarray],
) -> None:
    """A grid point setting both keeps the whole-model value for the other views."""
    gs = GridSearchCV(
        RidgeCCA(shrinkage=0.9),
        param_grid={"shrinkage": [0.1], "shrinkage__1": [0.5]},
        cv=2,
    )
    assert gs.fit(two_views).best_estimator_.shrinkage == [0.1, 0.5]


def test_per_view_index_beyond_the_views_raises(two_views: list[np.ndarray]) -> None:
    """A view index beyond the data names the problem."""
    gs = GridSearchCV(RidgeCCA(), param_grid={"shrinkage__5": [0.1]}, cv=2)
    with pytest.raises(ValueError, match="only 2 views"):
        gs.fit(two_views)


def test_transform_needs_refit(two_views: list[np.ndarray]) -> None:
    """With refit=False there is no best estimator to transform with."""
    gs = GridSearchCV(CCA(), param_grid={"n_components": [1]}, cv=2, refit=False)
    with pytest.raises(AttributeError, match="refit=False"):
        gs.fit(two_views).transform(two_views)


def test_callable_refit_sees_unprefixed_cv_results(
    two_views: list[np.ndarray],
) -> None:
    """A refit callable gets the same parameter names as ``cv_results_``."""
    seen: list[set[str]] = []

    def smallest_c(cv_results: dict[str, Any]) -> int:
        seen.append(set(cv_results))
        return int(np.argmin(cv_results["param_shrinkage"]))

    gs = GridSearchCV(
        RidgeCCA(), param_grid={"shrinkage": [0.5, 0.0, 0.9]}, cv=3, refit=smallest_c
    ).fit(two_views)
    assert "param_shrinkage" in seen[0]
    assert gs.best_params_["shrinkage"] == 0.0


def test_per_view_names_reach_into_a_pipeline(two_views: list[np.ndarray]) -> None:
    """A per-view name addresses a pipeline step's parameter."""
    pipeline = Pipeline(
        [
            ("scale", PerViewTransformer(StandardScaler())),
            ("cca", RidgeCCA(shrinkage=0.3)),
        ]
    )
    gs = GridSearchCV(pipeline, param_grid={"cca__shrinkage__0": [0.0, 0.9]}, cv=2)
    assert gs.fit(two_views).best_estimator_["cca"].shrinkage[1] == 0.3


def test_scoring_gets_the_estimator_and_views(two_views: list[np.ndarray]) -> None:
    """A scoring callable sees the multiview model and the held-out views."""
    seen: list[tuple[type, int]] = []

    def n_views(estimator: RidgeCCA, views: list[np.ndarray]) -> float:
        seen.append((type(estimator), len(views)))
        return 0.0

    GridSearchCV(
        RidgeCCA(), param_grid={"shrinkage": [0.0]}, cv=2, scoring=n_views
    ).fit(two_views)
    cross_validate(RidgeCCA(), two_views, cv=2, scoring={"n": n_views})
    assert set(seen) == {(RidgeCCA, 2)}
