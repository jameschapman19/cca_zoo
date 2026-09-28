"""Canonical quantile regression."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import null_space
from sklearn.linear_model import QuantileRegressor
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._linalg import covariance
from cca_zoo._utils._param_constraints import POSITIVE_EPS, POSITIVE_INT, RANDOM_STATE
from cca_zoo.linear._ridge_cca import RidgeCCA


def _check_loss(residual: np.ndarray, quantile: float) -> float:
    """Quantile regression's check loss, averaged over the samples."""
    return float(np.mean(residual * (quantile - (residual < 0))))


def _unconditional_loss(
    scores: np.ndarray, quantile: float
) -> tuple[float, np.ndarray]:
    """The check loss of ``scores`` about their quantile, and its sample weights.

    The loss is positively homogeneous in the scores, so for ``scores = Y a``
    its tangent plane at ``a`` is ``(Y.T @ weights / n) @ a``.
    """
    centred = scores - np.quantile(scores, quantile)
    return _check_loss(centred, quantile), quantile - (centred < 0)


def _canonical_direction(
    X: np.ndarray,
    Y: np.ndarray,
    start: np.ndarray,
    earlier: list[np.ndarray],
    quantile: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """One local solution from ``start``: ``(a, b, 1 - R1, n_iter)``.

    Each iteration replaces the normalisation, the unconditional check loss
    of ``Y a`` equal to 1, by its tangent plane at the current ``a``, and
    adds that ``Y a`` is uncorrelated with the ``earlier`` components. Then
    ``a = particular + free @ z``, for the minimum-norm ``particular`` meeting
    these linear constraints and a basis ``free`` of the directions keeping
    them, and the check loss of ``Y a - X b - c`` is a quantile regression of
    ``Y particular`` on ``[-Y free, X]`` with coefficients ``(z, b)``, solved
    exactly. ``n_iter`` exceeds ``max_iter`` when ``a`` has not settled.
    """
    cov_y = covariance(Y)
    regressor = QuantileRegressor(quantile=quantile, alpha=0.0)
    a = start / _unconditional_loss(Y @ start, quantile)[0]
    for n_iter in range(1, max_iter + 1):
        _, weights = _unconditional_loss(Y @ a, quantile)
        constraints = np.column_stack(
            [Y.T @ weights / len(Y), *(cov_y @ w for w in earlier)]
        )
        target = np.zeros(constraints.shape[1])
        target[0] = 1.0
        particular = np.linalg.lstsq(constraints.T, target, rcond=None)[0]
        free = null_space(constraints.T)
        regressor.fit(np.hstack([-Y @ free, X]), Y @ particular)
        z, b = np.split(regressor.coef_, [free.shape[1]])
        updated = particular + free @ z
        scale, _ = _unconditional_loss(Y @ updated, quantile)
        updated, b, intercept = updated / scale, b / scale, regressor.intercept_ / scale
        change = np.linalg.norm(updated - a)
        a = updated
        if change < tol:
            break
    else:
        n_iter = max_iter + 1
    return a, b, _check_loss(Y @ a - X @ b - intercept, quantile), n_iter


class QuantileCCA(BaseModel):
    r"""Canonical quantile regression: CCA with quantile regression's check loss.

    For covariates $X$ (the first view) and responses $Y$ (the second), each
    component finds the combination $Y a$ whose $\tau$-quantile a linear
    function of $X$ explains best, in Koenker and Machado's $R^1(\tau)$, the
    quantile analogue of $R^2$:

    $$
    \max_a R^1(\tau) = 1 - \frac{
        \min_{b, c} \sum_i \rho_\tau(y_i^\top a - x_i^\top b - c)
    }{
        \min_c \sum_i \rho_\tau(y_i^\top a - c)
    },
    $$

    with $\rho_\tau(u) = u (\tau - \mathbb{1}[u < 0])$, and each component's
    $Y a$ uncorrelated with the earlier ones'. With the squared loss this is
    CCA, which maximises $R^2$, and for Gaussian data every quantile gives
    CCA's directions; where the relationship changes across the
    distribution, as with heteroscedastic noise, the directions change with
    the quantile.

    The denominator is fixed at 1. Each iteration replaces that constraint by
    its tangent plane at the current $a$, which leaves a linear quantile
    regression in $b$, $c$ and the free part of $a$, solved exactly by
    :class:`~sklearn.linear_model.QuantileRegressor`. The problem is not
    convex, so each component is solved from ``n_init`` starts, ridge CCA's
    direction (the answer for Gaussian data) and random ones, keeping the
    largest $R^1(\tau)$.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        quantile: The quantile $\tau$, in ``(0, 1)``. Default is 0.5, the
            median.
        n_init: Number of starts per component. Default is 5.
        max_iter: Maximum iterations per start. Default is 100.
        tol: Tolerance on the change in $a$. Default is 1e-6.
        random_state: Seed for the random starts. Default is None.

    Attributes:
        weights_: ``[b, a]``: the covariate and response weights, shapes
            (n_features_0, n_components) and (n_features_1, n_components).
        r1_: $R^1(\tau)$ of each component.
        n_iter_: Most iterations run by any kept solution.

    References:
        Koenker, R., & Machado, J. A. F. (1999). Goodness of fit and related
        inference processes for quantile regression. Journal of the American
        Statistical Association, 94(448), 1296-1310.

        Portnoy, S. (2022). Canonical quantile regression. Journal of
        Multivariate Analysis, 192.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import QuantileCCA
        >>> rng = np.random.default_rng(0)
        >>> X = rng.standard_normal((100, 4))
        >>> Y = X[:, :2] + 0.5 * rng.standard_normal((100, 2))
        >>> model = QuantileCCA(quantile=0.9).fit([X, Y])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "quantile": [Interval(Real, 0, 1, closed="neither")],
        "n_init": POSITIVE_INT,
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        quantile: float = 0.5,
        n_init: int = 5,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.quantile = quantile
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> QuantileCCA:
        """Fit the model.

        Args:
            views: ``[X, Y]``: covariates and responses, each of shape
                (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self.

        Raises:
            ValueError: If there are not exactly two views.
        """
        views_ = self._setup_fit(views)
        if self.n_views_ != 2:
            raise ValueError(
                f"QuantileCCA requires exactly 2 views, covariates and responses, "
                f"got {self.n_views_}."
            )
        X, Y = views_
        ridge = RidgeCCA(self.n_components, center=False, shrinkage=0.1)
        ridge.fit([X, Y])
        rng = check_random_state(self.random_state)

        weights_x: list[np.ndarray] = []
        weights_y: list[np.ndarray] = []
        r1: list[float] = []
        self.n_iter_ = 0
        for k in range(self.n_components):
            starts = [ridge.weights_[1][:, k]] + [
                rng.standard_normal(Y.shape[1]) for _ in range(self.n_init - 1)
            ]
            solutions = [
                _canonical_direction(
                    X, Y, start, weights_y, self.quantile, self.max_iter, self.tol
                )
                for start in starts
            ]
            a, b, unexplained, n_iter = min(solutions, key=lambda s: s[2])
            self.n_iter_ = max(self.n_iter_, n_iter)
            r1.append(1.0 - unexplained)
            weights_x.append(b)
            weights_y.append(a)
        warn_if_not_converged(self, self.n_iter_ <= self.max_iter)
        self.n_iter_ = min(self.n_iter_, self.max_iter)
        self.r1_: np.ndarray = np.array(r1)

        self.weights_: list[np.ndarray] = [
            np.column_stack(weights_x),
            np.column_stack(weights_y),
        ]
        return self._finish_fit(views_)
