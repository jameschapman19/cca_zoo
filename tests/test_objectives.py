"""Fitted weights satisfy the optimality conditions of each model's stated objective.

The property suites check what a model's output does; these check that the
code optimises the objective its docstring writes down, built here from the
formula rather than from the model's own code.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import GCCA, GRCCA, MCCA
from tests._helpers import principal_cosines


def _views(n: int = 80) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n, 2))
    views = [
        z @ rng.standard_normal((2, p)) + rng.standard_normal((n, p)) for p in (6, 5, 4)
    ]
    return [v - v.mean(axis=0) for v in views]


def _regularised_covariance(view: np.ndarray, shrinkage: float) -> np.ndarray:
    n, p = view.shape
    return (1 - shrinkage) * view.T @ view / (n - 1) + shrinkage * np.eye(p)


@pytest.mark.parametrize("shrinkage", [0.0, 0.5, 1.0])
def test_gcca_weights_regress_the_top_eigenvectors_on_each_view(
    shrinkage: float,
) -> None:
    """T: top eigenvectors of sum_i X_i C_i^-1 X_i'; w_i = C_i^-1 X_i' T / (n - 1)."""
    views = _views()
    n = len(views[0])
    model = GCCA(2, shrinkage=shrinkage).fit(views)
    inverses = [np.linalg.inv(_regularised_covariance(v, shrinkage)) for v in views]
    q = sum(v @ c @ v.T for v, c in zip(views, inverses))
    eigenvalues = np.linalg.eigvalsh(q)
    # The summed scores are Q T / (n - 1), so T is them at unit variance.
    summed = sum(v @ w for v, w in zip(views, model.weights_))
    t = summed / summed.std(axis=0, ddof=1)
    np.testing.assert_allclose(q @ t, t * eigenvalues[-2:][::-1], rtol=1e-6)
    for view, inverse, weights in zip(views, inverses, model.weights_):
        np.testing.assert_allclose(weights, inverse @ view.T @ t / (n - 1), atol=1e-8)


def _group_penalty(groups: np.ndarray, lam: float, mu: float) -> np.ndarray:
    """Tuzhilina et al.'s penalty as a quadratic form, one group at a time.

    lam * sum_g sum_j (w_j - mean_g w)^2 + mu * sum_g p_g (mean_g w)^2.
    """
    p = len(groups)
    penalty = np.zeros((p, p))
    for g in np.unique(groups):
        members = np.flatnonzero(groups == g)
        mean = np.zeros(p)
        mean[members] = 1 / len(members)
        for j in members:
            deviation = -mean.copy()
            deviation[j] += 1
            penalty += lam * np.outer(deviation, deviation)
        penalty += mu * len(members) * np.outer(mean, mean)
    return penalty


@pytest.mark.parametrize(("shrinkage", "mu"), [(0.5, 0.0), (0.5, 0.3), (0.8, 2.0)])
def test_grcca_solves_the_group_penalised_eigenproblem(
    shrinkage: float, mu: float
) -> None:
    """The weights are the top eigenvectors of A w = rho (Sigma + P) w."""
    views = _views()[:2]
    groups = [np.arange(v.shape[1]) % 2 for v in views]
    model = GRCCA(2, shrinkage=shrinkage, mu=mu, feature_groups=groups).fit(views)
    lam = shrinkage / (1 - shrinkage)
    n = len(views[0])
    sigma = [v.T @ v / (n - 1) for v in views]
    b = [s + _group_penalty(g, lam, mu * lam) for s, g in zip(sigma, groups)]
    cross = views[0].T @ views[1] / (n - 1)
    # Two views: w_1 are the top left singular vectors of B_1^-1/2 S_12 B_2^-1/2.
    roots = [np.linalg.inv(np.linalg.cholesky(bi)).T for bi in b]
    u, _, vt = np.linalg.svd(roots[0].T @ cross @ roots[1])
    for weights, expected in zip(
        model.weights_, [roots[0] @ u[:, :2], roots[1] @ vt[:2].T]
    ):
        assert np.all(principal_cosines(weights, expected) > 1 - 1e-8)


def test_grcca_with_mu_one_is_ridge_mcca() -> None:
    """Penalising group means as much as deviations is the plain ridge."""
    views = _views()
    groups = [np.arange(v.shape[1]) % 2 for v in views]
    grcca = GRCCA(2, shrinkage=0.4, mu=1.0, feature_groups=groups).fit(views)
    mcca = MCCA(2, shrinkage=0.4, pca=False).fit(views)
    for a, b in zip(grcca.weights_, mcca.weights_):
        assert np.all(principal_cosines(a, b) > 1 - 1e-8)
