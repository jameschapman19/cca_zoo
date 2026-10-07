"""RidgeCCACV agrees with refitting RidgeCCA over the shrinkage grid."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import RidgeCCA, RidgeCCACV
from cca_zoo.model_selection import GridSearchCV

GRID = [0.001, 0.01, 0.1, 0.5, 1.0]


@pytest.fixture
def views() -> list[np.ndarray]:
    """Two views sharing two latent factors."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((120, 2))
    return [
        z @ rng.standard_normal((2, p)) + rng.standard_normal((120, p))
        for p in (60, 40)
    ]


@pytest.mark.parametrize("center", [True, False])
def test_cv_scores_equal_grid_search_scores(
    views: list[np.ndarray], center: bool
) -> None:
    """Sharing the eigendecomposition across shrinkages changes no score."""
    model = RidgeCCACV(2, center=center, shrinkages=GRID, cv=4).fit(views)
    search = GridSearchCV(RidgeCCA(2, center=center), {"shrinkage": GRID}, cv=4).fit(
        views
    )
    np.testing.assert_allclose(
        model.cv_scores_, search.cv_results_["mean_test_score"], atol=1e-9
    )
    assert model.shrinkage_ == search.best_params_["shrinkage"]


def test_final_fit_is_ridge_cca_at_the_chosen_shrinkage(
    views: list[np.ndarray],
) -> None:
    """The refit uses all the data, as RidgeCCA at that shrinkage."""
    model = RidgeCCACV(2, shrinkages=GRID).fit(views)
    reference = RidgeCCA(2, shrinkage=model.shrinkage_).fit(views)
    for w, w_ref in zip(model.weights_, reference.weights_):
        np.testing.assert_allclose(w, w_ref, atol=1e-10)


def test_requires_two_views(views: list[np.ndarray]) -> None:
    """A third view is rejected by name."""
    with pytest.raises(ValueError, match="exactly 2 views"):
        RidgeCCACV().fit([*views, views[0]])


def _planted(rhos: list[float], n: int, p: int = 30, seed: int = 0) -> list[np.ndarray]:
    """Views whose population canonical correlations are ``rhos``, rest noise."""
    rng = np.random.default_rng(seed)
    k = len(rhos)
    joint = rng.standard_normal((n, p + p))
    shared = rng.standard_normal((n, k))
    out = []
    for view in (joint[:, :p], joint[:, p:]):
        loadings = np.linalg.qr(rng.standard_normal((p, k)))[0].T
        out.append(
            view
            + (shared * np.sqrt(np.asarray(rhos) / (1 - np.asarray(rhos)))) @ loadings
        )
    return out


@pytest.mark.parametrize("rank", [1, 3])
def test_auto_recovers_the_number_of_planted_components(rank: int) -> None:
    """With enough samples, ``n_components="auto"`` keeps exactly the real ones."""
    views = _planted([0.8, 0.7, 0.6][:rank], n=2000)
    model = RidgeCCACV("auto").fit(views)
    assert model.n_components_ == rank
    assert model.cv_component_scores_.shape == (len(model.cv_scores_), 10)


def test_auto_warns_and_keeps_one_component_on_pure_noise() -> None:
    """Nothing clears the chance threshold: one component, and a warning."""
    views = _planted([], n=300, seed=3)
    with pytest.warns(UserWarning, match="exceeds chance"):
        model = RidgeCCACV("auto").fit(views)
    assert model.n_components_ == 1
