"""Sparse CCA by coordinate descent on the elastic-net-penalised EY loss."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._ey import cheap_orthonormal_projection_weights, ey_loss
from cca_zoo._utils._param_constraints import RANDOM_STATE
from cca_zoo._utils._validation import perview_parameter
from cca_zoo.sparse._coordinate_descent import (
    PENALTY_PATH,
    ey_quartic,
    minimise_quartic,
    others_covariance,
)


class ElasticNetCCA(BaseModel):
    r"""Sparse multiview CCA by coordinate descent on the elastic-net EY loss.

    Minimises, over $Z_i = X_i W_i$,

    $$
    \mathcal{L}_{EY}(Z_1, \dots, Z_M)
        + \sum_i \left( \alpha_i \rho_i \|W_i\|_1
        + \tfrac{1}{2} \alpha_i (1-\rho_i) \|W_i\|_F^2 \right),
    $$

    with $\rho$ = ``l1_ratio``, by cyclic coordinate descent as in
    :class:`~sklearn.linear_model.ElasticNet`; each coordinate's quartic
    restriction is minimised exactly. The loss is not jointly convex, so the
    result can depend on ``random_state``.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        alpha: Penalty strength. Per-view. Default is 1.0.
        l1_ratio: L1 share of the penalty in ``[0, 1]``. Per-view. Default
            is 0.5.
        max_iter: Maximum coordinate-descent sweeps at each stage of the
            penalty path. Default is 1000, as sklearn's ElasticNet.
        tol: Tolerance on the change in the objective. Default is 1e-6.
        random_state: Seed for the initial weights. Default is None.
        positive: Whether to constrain the weights to be non-negative.
            Default is False.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Coordinate-descent sweeps at the full penalty.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.sparse import ElasticNetCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 20))
        >>> X2 = rng.standard_normal((200, 15))
        >>> model = ElasticNetCCA(n_components=2, alpha=[0.1, 0.5]).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "l1_ratio": [Interval(Real, 0, 1, closed="both"), "array-like"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "positive": ["boolean"],
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        alpha: float | list[float] = 1.0,
        l1_ratio: float | list[float] = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: int | None = None,
        positive: bool = False,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.positive = positive

    def fit(self, views: list[ArrayLike], y: None = None) -> ElasticNetCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        alpha = perview_parameter("alpha", self.alpha, 1.0, self.n_views_)
        l1_ratio = perview_parameter("l1_ratio", self.l1_ratio, 0.5, self.n_views_)
        rng = np.random.default_rng(self.random_state)
        weights = cheap_orthonormal_projection_weights(
            views_, self.n_components, None, rng
        )
        for scale in PENALTY_PATH:
            self.n_iter_, converged = self._coordinate_descent(
                views_, weights, [a * scale for a in alpha], l1_ratio
            )
        warn_if_not_converged(self, converged)
        # All-zero weights have loss zero, so a fit ending above it has none.
        if self._objective(views_, weights, alpha, l1_ratio) > 0.0:
            weights = [np.zeros_like(w) for w in weights]
        self.weights_: list[np.ndarray] = weights
        self._fit_maps_and_importances(views_)
        return self

    def _coordinate_descent(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        alpha: list[float],
        l1_ratio: list[float],
    ) -> tuple[int, bool]:
        """Cyclic sweeps setting each weight to its exact minimiser, in place.

        Returns:
            The sweeps run, and whether the objective settled within ``tol``.
        """
        n = views[0].shape[0]
        k = self.n_components
        a0 = 1.0 / (len(views) * (n - 1))
        lasso = [a * r for a, r in zip(alpha, l1_ratio)]
        ridge = [a * (1.0 - r) for a, r in zip(alpha, l1_ratio)]
        scores = [v @ w for v, w in zip(views, weights)]
        total = np.sum(scores, axis=0)
        previous = np.inf
        for n_iter in range(1, self.max_iter + 1):
            for i, (view, w) in enumerate(zip(views, weights)):
                v_other = others_covariance(scores, i, a0)
                for j, xj in enumerate(view.T):
                    a = xj @ xj
                    if a < 1e-12:
                        continue
                    for c in range(k):
                        c4, c3, c2, c1 = ey_quartic(
                            xj, a, a0, scores[i], total, v_other, w[j], c, k
                        )
                        new = minimise_quartic(
                            c4, c3, c2 + ridge[i] / 2, c1, lasso[i], self.positive
                        )
                        step = new - w[j, c]
                        w[j, c] = new
                        scores[i][:, c] += xj * step
                        total[:, c] += xj * step
            objective = self._objective(views, weights, alpha, l1_ratio)
            if abs(previous - objective) < self.tol:
                return n_iter, True
            previous = objective
        return self.max_iter, False

    @staticmethod
    def _objective(
        views: list[np.ndarray],
        weights: list[np.ndarray],
        alpha: list[float],
        l1_ratio: list[float],
    ) -> float:
        """The EY loss plus each view's elastic-net penalty."""
        penalty = sum(
            a * r * np.sum(np.abs(w)) + a * (1.0 - r) * np.sum(w**2) / 2
            for w, a, r in zip(weights, alpha, l1_ratio)
        )
        scores = [v @ w for v, w in zip(views, weights)]
        return float(ey_loss(scores)["objective"] + penalty)
