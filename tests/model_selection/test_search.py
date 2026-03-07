"""Tests for cca_zoo.model_selection.GridSearchCV."""

from __future__ import annotations

import numpy as np

from cca_zoo.linear import CCA, MCCA, rCCA
from cca_zoo.model_selection import GridSearchCV

# ---------------------------------------------------------------------------
# Basic fit
# ---------------------------------------------------------------------------


def test_grid_search_fit_completes(two_views: list[np.ndarray]) -> None:
    """GridSearchCV.fit completes without error on two-view data."""
    gs = GridSearchCV(
        CCA(),
        param_grid={"latent_dimensions": [1, 2]},
        cv=2,
    )
    fitted = gs.fit(two_views)
    assert fitted is gs


def test_grid_search_fit_returns_self(two_views: list[np.ndarray]) -> None:
    """GridSearchCV.fit returns self."""
    gs = GridSearchCV(
        CCA(),
        param_grid={"latent_dimensions": [1]},
        cv=2,
    )
    result = gs.fit(two_views)
    assert result is gs


# ---------------------------------------------------------------------------
# best_params_
# ---------------------------------------------------------------------------


def test_best_params_accessible_after_fit(two_views: list[np.ndarray]) -> None:
    """best_params_ is set after fit and contains the searched parameter key."""
    gs = GridSearchCV(
        CCA(),
        param_grid={"latent_dimensions": [1, 2]},
        cv=2,
    )
    gs.fit(two_views)
    assert hasattr(gs, "best_params_")
    assert "latent_dimensions" in gs.best_params_


def test_best_params_value_in_grid(two_views: list[np.ndarray]) -> None:
    """best_params_ value is one of the grid values."""
    grid_values = [1, 2]
    gs = GridSearchCV(
        CCA(),
        param_grid={"latent_dimensions": grid_values},
        cv=2,
    )
    gs.fit(two_views)
    assert gs.best_params_["latent_dimensions"] in grid_values


def test_best_params_no_prefix(two_views: list[np.ndarray]) -> None:
    """best_params_ keys should NOT have 'estimator__' prefix."""
    gs = GridSearchCV(
        CCA(),
        param_grid={"latent_dimensions": [1, 2]},
        cv=2,
    )
    gs.fit(two_views)
    for key in gs.best_params_:
        assert not key.startswith("estimator__"), (
            f"Key '{key}' should not have 'estimator__' prefix"
        )


# ---------------------------------------------------------------------------
# best_score_
# ---------------------------------------------------------------------------


def test_best_score_is_float(two_views: list[np.ndarray]) -> None:
    """best_score_ is a Python float."""
    gs = GridSearchCV(
        CCA(),
        param_grid={"latent_dimensions": [1, 2]},
        cv=2,
    )
    gs.fit(two_views)
    assert isinstance(gs.best_score_, float)


def test_best_score_in_valid_range(two_views: list[np.ndarray]) -> None:
    """best_score_ is a valid correlation value in [-1, 1]."""
    gs = GridSearchCV(
        CCA(),
        param_grid={"latent_dimensions": [1, 2]},
        cv=2,
    )
    gs.fit(two_views)
    assert -1.0 <= gs.best_score_ <= 1.0


# ---------------------------------------------------------------------------
# best_estimator_
# ---------------------------------------------------------------------------


def test_best_estimator_accessible_after_fit(two_views: list[np.ndarray]) -> None:
    """best_estimator_ is set after fit when refit=True (default)."""
    gs = GridSearchCV(
        CCA(),
        param_grid={"latent_dimensions": [1, 2]},
        cv=2,
        refit=True,
    )
    gs.fit(two_views)
    assert hasattr(gs, "best_estimator_")


def test_best_estimator_is_fitted(two_views: list[np.ndarray]) -> None:
    """best_estimator_ is fitted and can transform data."""
    gs = GridSearchCV(
        CCA(),
        param_grid={"latent_dimensions": [1, 2]},
        cv=2,
    )
    gs.fit(two_views)
    result = gs.best_estimator_.transform(two_views)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# cv_results_
# ---------------------------------------------------------------------------


def test_cv_results_accessible_after_fit(two_views: list[np.ndarray]) -> None:
    """cv_results_ is a dict accessible after fit."""
    gs = GridSearchCV(
        CCA(),
        param_grid={"latent_dimensions": [1, 2]},
        cv=2,
    )
    gs.fit(two_views)
    assert hasattr(gs, "cv_results_")
    assert isinstance(gs.cv_results_, dict)
    assert "mean_test_score" in gs.cv_results_


# ---------------------------------------------------------------------------
# Multiple parameters in grid
# ---------------------------------------------------------------------------


def test_grid_search_multi_param(two_views: list[np.ndarray]) -> None:
    """GridSearchCV handles a grid with multiple parameters."""
    gs = GridSearchCV(
        rCCA(),
        param_grid={"latent_dimensions": [1, 2], "c": [0.0, 0.1]},
        cv=2,
    )
    gs.fit(two_views)
    assert "latent_dimensions" in gs.best_params_
    assert "c" in gs.best_params_


# ---------------------------------------------------------------------------
# List of grids
# ---------------------------------------------------------------------------


def test_grid_search_list_of_grids(two_views: list[np.ndarray]) -> None:
    """GridSearchCV accepts a list of parameter dicts (disjoint grids)."""
    gs = GridSearchCV(
        CCA(),
        param_grid=[
            {"latent_dimensions": [1]},
            {"latent_dimensions": [2]},
        ],
        cv=2,
    )
    gs.fit(two_views)
    assert gs.best_params_["latent_dimensions"] in [1, 2]


# ---------------------------------------------------------------------------
# score method after fit
# ---------------------------------------------------------------------------


def test_score_after_fit(two_views: list[np.ndarray]) -> None:
    """GridSearchCV.score works on new data after fit."""
    rng = np.random.default_rng(10)
    test_views = [rng.standard_normal((20, 10)), rng.standard_normal((20, 8))]
    gs = GridSearchCV(
        CCA(),
        param_grid={"latent_dimensions": [1, 2]},
        cv=2,
    )
    gs.fit(two_views)
    s = gs.score(test_views)
    assert isinstance(s, float)
    assert -1.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# n_jobs
# ---------------------------------------------------------------------------


def test_grid_search_n_jobs(two_views: list[np.ndarray]) -> None:
    """GridSearchCV works with n_jobs=1."""
    gs = GridSearchCV(
        CCA(),
        param_grid={"latent_dimensions": [1, 2]},
        cv=2,
        n_jobs=1,
    )
    gs.fit(two_views)
    assert hasattr(gs, "best_score_")


# ---------------------------------------------------------------------------
# Three-view model
# ---------------------------------------------------------------------------


def test_grid_search_three_view_model(three_views: list[np.ndarray]) -> None:
    """GridSearchCV works with a multi-view model (MCCA) on three views."""
    gs = GridSearchCV(
        MCCA(),
        param_grid={"latent_dimensions": [1, 2]},
        cv=2,
    )
    gs.fit(three_views)
    assert isinstance(gs.best_score_, float)
    assert "latent_dimensions" in gs.best_params_
