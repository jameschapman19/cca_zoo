"""Eckart-Young CCA, ridge-blended with PLSEY."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._ey import (
    cheap_orthonormal_projection_weights,
    ey_centred,
    ey_cross_covariance,
    weight_gram_mean,
)
from cca_zoo._utils._param_constraints import RANDOM_STATE


class CCAEY(BaseModel):
    r"""Multiview CCA by minimising the Eckart-Young loss, with a ridge blend.

    For $Z_i = X_i W_i$, with $C$ and $V$ the mean pairwise cross-covariance
    and mean auto-covariance (:mod:`cca_zoo._utils._ey`) and
    $B = \frac{1}{M} \sum_i W_i^\top W_i$,

    $$
    V_s = (1 - s) V + s B, \qquad
    \mathcal{L}_{EY}(s) = -2 \operatorname{tr}(C - s V) + \operatorname{tr}(V_s V_s)
    $$

    for ``shrinkage`` $s$, as in :class:`~cca_zoo.linear.RidgeCCA`: 0 is CCA
    and 1 is :class:`~cca_zoo.linear.gradient.PLSEY`. The loss is minimised by
    full-batch L-BFGS-B with its exact gradient; see
    :class:`~cca_zoo.linear.gradient.StochasticCCAEY` for mini-batches. With
    few samples per feature, ``shrinkage=0`` is ill-conditioned; use 0.1 to
    0.3.

    A sample lacks a view when its whole row is ``NaN``. Every covariance in the
    loss is then taken over the samples that have the views concerned, so a
    sample contributes through each view it has. Only whole views may be
    missing, and the views must be linked by more than ``n_components`` shared
    samples (see :class:`~cca_zoo.linear.IncompleteGCCA`). The weights are
    fitted for the missing-at-random case.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's covariance towards the identity,
            in ``[0, 1]``: 0 is CCA and 1 is PLS. Default is 0.
        max_iter: Maximum L-BFGS-B iterations. Default is 1000.
        tol: L-BFGS-B ``ftol``. Default is 1e-8.
        random_state: Seed for the initial weights. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: L-BFGS-B iterations run.

    References:
        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import CCAEY
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((1000, 20))
        >>> X2 = rng.standard_normal((1000, 15))
        >>> X3 = rng.standard_normal((1000, 10))
        >>> model = CCAEY(n_components=4, random_state=0).fit([X1, X2, X3])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "shrinkage": [Interval(Real, 0, 1, closed="both")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "random_state": RANDOM_STATE,
    }

    _supports_missing_views: ClassVar[bool] = True

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float = 0.0,
        max_iter: int = 1000,
        tol: float = 1e-8,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.shrinkage = shrinkage
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> CCAEY:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_, observed = self._setup_missing(views)
        filled = [np.nan_to_num(v) for v in views_]
        rng = np.random.default_rng(self.random_state)
        start = self._initial_weights(filled, rng)
        self.weights_, self.n_iter_ = self._minimise(
            filled, start, self.max_iter, observed
        )
        warn_if_not_converged(self, self.n_iter_ < self.max_iter)
        self._fit_maps_and_importances(views_)
        return self

    def _minimise(
        self,
        views: list[np.ndarray],
        start: list[np.ndarray],
        max_iter: int,
        observed: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], int]:
        """The weights minimising the loss on ``views`` by L-BFGS-B from ``start``.

        L-BFGS-B stops on an absolute gradient, and a view's gradient scales
        with its units, so the search runs over ``u_i = s_i w_i``, with
        ``s_i`` the root mean eigenvalue of the view's constraint matrix: the
        same loss, in units where every view's gradient is comparable.

        Returns:
            The weights, and the iterations run.
        """
        c = self.shrinkage
        seen = (
            np.ones((views[0].shape[0], len(views)), bool)
            if observed is None
            else observed
        )
        scales = [
            np.sqrt((1 - c) * np.mean(v[rows].var(axis=0, ddof=1)) + c) or 1.0
            for v, rows in zip(views, seen.T)
        ]
        shapes = [w.shape for w in start]
        splits = np.cumsum([w.size for w in start])[:-1]

        def weights_of(u: np.ndarray) -> list[np.ndarray]:
            parts = np.split(u, splits)
            return [p.reshape(shape) / s for p, shape, s in zip(parts, shapes, scales)]

        def loss_and_gradient(u: np.ndarray) -> tuple[float, np.ndarray]:
            weights = weights_of(u)
            scores = [v @ w for v, w in zip(views, weights)]
            gradients = self._derivative(views, scores, weights, observed)
            return self._objective(views, scores, weights, observed), np.concatenate(
                [(g / s).ravel() for g, s in zip(gradients, scales)]
            )

        result = minimize(
            loss_and_gradient,
            np.concatenate([(w * s).ravel() for w, s in zip(start, scales)]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": self.tol},
        )
        return weights_of(result.x), int(result.nit)

    def _initial_weights(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Weights giving unit-variance, uncorrelated projections on the full data."""
        return cheap_orthonormal_projection_weights(views, self.n_components, None, rng)

    def _sample_weight(self, representations: list[np.ndarray]) -> np.ndarray:
        """Each sample's weight in the loss's moments: equal, for CCA."""
        return np.ones(len(representations[0]))

    def _derivative(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
        observed: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        r"""Gradient of $\mathcal{L}_{EY}(c)$ in each view's weights.

        The chain rule through the embeddings plus the direct term from $B$,
        with the sample weights held fixed. Each embedding is centred on its
        view's observed samples, so its gradient is centred on them too.
        """
        m = len(views)
        c = self.shrinkage
        sample_weight = self._sample_weight(representations)
        centred, divisor = ey_centred(representations, sample_weight, observed)
        inverse = np.divide(1.0, divisor, out=np.zeros_like(divisor), where=divisor > 0)
        _, v_data = ey_cross_covariance(representations, sample_weight, observed)
        v_blend = (1 - c) * v_data + c * weight_gram_mean(weights)
        seen = np.ones((len(views[0]), m)) if observed is None else observed
        gradients = []
        for i, (view, z, w) in enumerate(zip(views, centred, weights)):
            embedding = (c * z + (1 - c) * (z @ v_blend)) * inverse[i, i]
            others = sum(zj * inverse[i, j] for j, zj in enumerate(centred))
            grad_z = (
                (4.0 / m)
                * (seen[:, [i]] * sample_weight[:, None])
                * (embedding - others)
            )
            weight_i = seen[:, i] * sample_weight
            grad_z = grad_z - weight_i[:, None] * (grad_z.sum(axis=0) / weight_i.sum())
            gradients.append(view.T @ grad_z + (4.0 * c / m) * (w @ v_blend))
        return gradients

    def _objective(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
        observed: np.ndarray | None = None,
    ) -> float:
        r"""$\mathcal{L}_{EY}(c)$."""
        del views
        c = self.shrinkage
        C, v_data = ey_cross_covariance(
            representations, self._sample_weight(representations), observed
        )
        v_blend = (1 - c) * v_data + c * weight_gram_mean(weights)
        return float(-2.0 * np.trace(C - c * v_data) + np.trace(v_blend @ v_blend))
