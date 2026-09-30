"""Row-sparse CCA by coordinate descent on the EY loss."""

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
    others_covariance,
)


class MultiTaskElasticNetCCA(BaseModel):
    r"""Row-sparse multiview CCA with a multi-task elastic-net penalty on the EY loss.

    As :class:`~cca_zoo.sparse.ElasticNetCCA` with
    :class:`~sklearn.linear_model.MultiTaskElasticNet`'s penalty,

    $$
    \mathcal{L}_{EY}(Z_1, \dots, Z_M)
        + \sum_i \left( \alpha_i \rho_i \|W_i\|_{2,1}
        + \tfrac{1}{2} \alpha_i (1-\rho_i) \|W_i\|_F^2 \right),
    $$

    so each feature is used by every component or by none. Each row is
    updated by a proximal-gradient step with backtracking.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        alpha: Penalty strength. Per-view. Default is 1.0.
        l1_ratio: Row-group share of the penalty in ``[0, 1]``. Per-view.
            Default is 0.5.
        max_iter: Maximum coordinate-descent sweeps at each stage of the
            penalty path. Default is 1000, as sklearn's ElasticNet.
        tol: Tolerance on the change in the objective. Default is 1e-6.
        random_state: Seed for the initial weights. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Proximal sweeps at the full penalty.

    References:
        Chapman, J., Wang, H.-T., Wells, L., & Wiesner, J. (2021). CCA-Zoo: A
        collection of Regularized, Deep Learning based, Kernel, and
        Probabilistic CCA methods in a scikit-learn style framework. Journal
        of Open Source Software, 6(68), 3823.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.sparse import MultiTaskElasticNetCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 20))
        >>> X2 = rng.standard_normal((200, 15))
        >>> model = MultiTaskElasticNetCCA(n_components=2, alpha=0.1).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "l1_ratio": [Interval(Real, 0, 1, closed="both"), "array-like"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
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
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> MultiTaskElasticNetCCA:
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
            self.n_iter_, converged = self._proximal_descent(
                views_, weights, [a * scale for a in alpha], l1_ratio
            )
        warn_if_not_converged(self, converged)
        # All-zero weights have loss zero, so a fit ending above it has none.
        if self._objective(views_, weights, alpha, l1_ratio) > 0.0:
            weights = [np.zeros_like(w) for w in weights]
        self.weights_: list[np.ndarray] = weights
        self._fit_maps_and_importances(views_)
        return self

    def _proximal_descent(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        alpha: list[float],
        l1_ratio: list[float],
    ) -> tuple[int, bool]:
        """Cyclic proximal-gradient steps on each feature's row, in place.

        A row's restriction is a coupled quartic with no closed-form
        minimiser, so each row takes a group-soft-thresholded gradient step,
        backtracking until the EY loss is below its quadratic upper bound at
        the step. Accepting any step that lowers the penalised objective
        instead lets a row jump to zero past a nonzero minimum.

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
        loss = ey_loss(scores)["objective"]
        previous = np.inf
        for n_iter in range(1, self.max_iter + 1):
            for i, (view, w) in enumerate(zip(views, weights)):
                v_other = others_covariance(scores, i, a0)
                for j, xj in enumerate(view.T):
                    a = xj @ xj
                    if a < 1e-12:
                        continue
                    row = w[j].copy()
                    gradient = np.empty(k)
                    for c in range(k):
                        c4, c3, c2, c1 = ey_quartic(
                            xj, a, a0, scores[i], total, v_other, row, c, k
                        )
                        x = row[c]
                        gradient[c] = 4 * c4 * x**3 + 3 * c3 * x**2 + 2 * c2 * x + c1
                    curvature = max(a0 * a, 1e-6)
                    for _ in range(_MAX_BACKTRACK):
                        denom = curvature + ridge[i]
                        step = (
                            _group_soft_threshold(
                                (curvature * row - gradient) / denom, lasso[i] / denom
                            )
                            - row
                        )
                        change = np.outer(xj, step)
                        scores[i] += change
                        total += change
                        trial = ey_loss(scores)["objective"]
                        bound = loss + gradient @ step + curvature * (step @ step) / 2
                        if trial <= bound + 1e-12:
                            w[j] = row + step
                            loss = trial
                            break
                        scores[i] -= change
                        total -= change
                        curvature *= 2.0
            objective = loss + self._penalty(weights, alpha, l1_ratio)
            if abs(previous - objective) < self.tol:
                return n_iter, True
            previous = objective
        return self.max_iter, False

    @staticmethod
    def _penalty(
        weights: list[np.ndarray], alpha: list[float], l1_ratio: list[float]
    ) -> float:
        """Each view's row-group elastic-net penalty, summed."""
        return float(
            sum(
                a * r * np.sum(np.linalg.norm(w, axis=1))
                + a * (1.0 - r) * np.sum(w**2) / 2
                for w, a, r in zip(weights, alpha, l1_ratio)
            )
        )

    def _objective(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        alpha: list[float],
        l1_ratio: list[float],
    ) -> float:
        """The EY loss plus the penalty."""
        scores = [v @ w for v, w in zip(views, weights)]
        return float(
            ey_loss(scores)["objective"] + self._penalty(weights, alpha, l1_ratio)
        )


# Most halvings of a row's step before its update is abandoned.
_MAX_BACKTRACK = 40


def _group_soft_threshold(u: np.ndarray, threshold: float) -> np.ndarray:
    """``u`` shrunk towards zero by ``threshold`` in norm, the group lasso's prox."""
    norm = float(np.linalg.norm(u))
    return max(0.0, 1.0 - threshold / norm) * u if norm > 1e-15 else np.zeros_like(u)
