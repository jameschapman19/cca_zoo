"""Fitted weights satisfy the optimality conditions of each model's stated objective.

The property suites check what a model's output does; these check that the
code optimises the objective its docstring writes down, built here from the
formula rather than from the model's own code.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import GCCA


def _views(n: int = 80) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n, 2))
    views = [
        z @ rng.standard_normal((2, p)) + rng.standard_normal((n, p))
        for p in (6, 5, 4)
    ]
    return [v - v.mean(axis=0) for v in views]


def _regularised_covariance(view: np.ndarray, shrinkage: float) -> np.ndarray:
    n, p = view.shape
    return (1 - shrinkage) * view.T @ view / (n - 1) + shrinkage * np.eye(p)


@pytest.mark.parametrize("shrinkage", [0.0, 0.5, 1.0])
def test_gcca_weights_regress_the_top_eigenvectors_on_each_view(
    shrinkage: float,
) -> None:
    """T spans the top eigenvectors of sum_i X_i C_i^-1 X_i', w_i = C_i^-1 X_i' T/(n-1)."""
    views = _views()
    n = len(views[0])
    model = GCCA(2, shrinkage=shrinkage).fit(views)
    inverses = [np.linalg.inv(_regularised_covariance(v, shrinkage)) for v in views]
    q = sum(v @ c @ v.T for v, c in zip(views, inverses))
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    # The summed scores are Q T / (n - 1), so T is them at unit variance.
    summed = sum(v @ w for v, w in zip(views, model.weights_))
    t = summed / summed.std(axis=0, ddof=1)
    np.testing.assert_allclose(q @ t, t * eigenvalues[-2:][::-1], rtol=1e-6)
    for view, inverse, weights in zip(views, inverses, model.weights_):
        np.testing.assert_allclose(weights, inverse @ view.T @ t / (n - 1), atol=1e-8)
