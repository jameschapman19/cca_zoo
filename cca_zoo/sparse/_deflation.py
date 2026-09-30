"""What the alternating sparse CCA models share: deflation and their updates' parts.

Each model fits one component at a time by alternating per-view updates,
then deflates the views by that component's scores before the next. The
models differ only in the update, which each writes out in its own ``fit``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from cca_zoo._utils._linalg import deflate, loading, svd_whiten, undeflated_weights


class Deflation:
    """Projection deflation across components, as in PLS.

    Iterating yields, for each component, the views with every earlier
    component's scores projected out; :meth:`record` adds a component's
    weights on those views. :meth:`weights` maps the recorded weights back
    to the original views as ``W (P'W)^-1``, sklearn's PLS ``x_rotations_``,
    so that ``transform`` reproduces the scores found on the deflated views.

    Args:
        views: The centred training views.
        n_components: Number of components.
    """

    def __init__(self, views: list[np.ndarray], n_components: int) -> None:
        self.views = views
        self.n_components = n_components
        shapes = [(v.shape[1], n_components) for v in views]
        self._weights = [np.zeros(shape) for shape in shapes]
        self._loadings = [np.zeros(shape) for shape in shapes]
        self._recorded = 0

    def __iter__(self) -> Iterator[list[np.ndarray]]:
        for _ in range(self.n_components):
            yield self.views

    def record(self, weights: list[np.ndarray]) -> None:
        """Add a component's weights on the current views, then deflate by it."""
        for i, (view, w) in enumerate(zip(self.views, weights)):
            self._weights[i][:, self._recorded] = w
            self._loadings[i][:, self._recorded] = loading(view, w)
        self.views = deflate(self.views, weights)
        self._recorded += 1

    def weights(self) -> list[np.ndarray]:
        """Each view's weights on the original view, ``W (P'W)^-1``."""
        return [undeflated_weights(w, p) for w, p in zip(self._weights, self._loadings)]


def pls_direction(
    views: list[np.ndarray], rng: np.random.Generator
) -> list[np.ndarray]:
    """Unit weights maximising the summed cross-covariance of the scores.

    The PLS direction, which sklearn's PLS also starts from: the leading
    eigenvector of the between-view blocks ``X_i' X_j``, found by Lanczos
    from a random start. Alternating updates started at random can stall
    at all-zero weights when the first target barely correlates with a view.
    """
    splits = np.cumsum([v.shape[1] for v in views])[:-1]

    def cross_covariance(stacked: np.ndarray) -> np.ndarray:
        scores = [v @ w for v, w in zip(views, np.split(stacked, splits))]
        total = sum(scores)
        return np.concatenate([v.T @ (total - s) for v, s in zip(views, scores)])

    size = sum(v.shape[1] for v in views)
    operator = LinearOperator((size, size), matvec=cross_covariance, dtype=float)
    _, vector = eigsh(operator, k=1, which="LA", v0=rng.standard_normal(size))
    return [w / max(np.linalg.norm(w), 1e-12) for w in np.split(vector[:, 0], splits)]


def ridge_cca_direction(
    views: list[np.ndarray], shrinkage: float = 0.5
) -> list[np.ndarray]:
    """Unit weights of the leading ridge-regularised CCA direction.

    The start for the models whose updates are regressions: from the PLS
    direction, a view dominated by variance the others do not share gives
    them a target they barely correlate with, and a penalised regression
    selects nothing. The leading right singular vector of the stacked
    ridge-whitened views is MAXVAR's first direction on those views, and with
    ``shrinkage`` between CCA and PLS it exists at any sample size.
    """
    whitened = [svd_whiten(v, shrinkage) for v in views]
    _, _, vt = np.linalg.svd(np.hstack([x for x, _ in whitened]), full_matrices=False)
    splits = np.cumsum([x.shape[1] for x, _ in whitened])[:-1]
    directions = [w @ u for (_, w), u in zip(whitened, np.split(vt[0], splits))]
    return [w / max(np.linalg.norm(w), 1e-12) for w in directions]


def others_score(
    views: list[np.ndarray], weights: list[np.ndarray], i: int
) -> np.ndarray:
    """The sum of every view's score but view ``i``'s, at unit variance.

    Unit variance, not unit norm, so that a penalty on view ``i``'s update
    means the same at any sample size.
    """
    total: np.ndarray = np.asarray(
        sum(v @ w for j, (v, w) in enumerate(zip(views, weights)) if j != i)
    )
    std = total.std()
    return total / std if std > 1e-12 else total


def elastic_net(
    alpha: float, l1_ratio: float, tol: float, random_state: int | None
) -> LinearRegression | Ridge | Lasso | ElasticNet:
    """The regression one view's update solves, without intercept.

    Least squares at ``alpha=0``, otherwise Ridge, Lasso or ElasticNet by
    ``l1_ratio``.
    """
    if alpha == 0.0:
        return LinearRegression(fit_intercept=False)
    if l1_ratio == 0.0:
        return Ridge(alpha=alpha, fit_intercept=False, tol=tol)
    if l1_ratio == 1.0:
        return Lasso(
            alpha=alpha,
            fit_intercept=False,
            warm_start=True,
            tol=tol,
            random_state=random_state,
            selection="random",
        )
    return ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=False,
        warm_start=True,
        tol=tol,
        random_state=random_state,
        selection="random",
    )
