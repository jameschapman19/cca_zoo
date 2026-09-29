"""Eckart-Young CCA, ridge-blended with PLSEY."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._utils._ey import (
    cheap_orthonormal_projection_weights,
    ey_cross_covariance,
    weight_gram_mean,
)
from cca_zoo.linear.gradient._base import BaseFullBatchEYModel


class CCAEY(BaseFullBatchEYModel):
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
        **BaseFullBatchEYModel._parameter_constraints,
        "shrinkage": [Interval(Real, 0, 1, closed="both")],
    }

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
        super().__init__(
            n_components=n_components,
            center=center,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.shrinkage = shrinkage

    def fit(self, views: list[ArrayLike], y: None = None) -> CCAEY:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        self.weights_ = self._fit_lbfgsb(views_, rng)
        return self._finish_fit(views_)

    def _weight_scale(self, view: np.ndarray) -> float:
        """Root mean eigenvalue of ``(1 - shrinkage) cov + shrinkage I``."""
        variance = float(np.mean(view.var(axis=0, ddof=1)))
        return float(np.sqrt((1 - self.shrinkage) * variance + self.shrinkage))

    def _initial_weights(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Weights giving unit-variance, uncorrelated projections on the full data."""
        return cheap_orthonormal_projection_weights(views, self.n_components, None, rng)

    def _derivative(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> list[np.ndarray]:
        r"""Gradient of $\mathcal{L}_{EY}(c)$ in each view's weights.

        The chain rule through the embeddings plus the direct term from $B$.
        """
        m = len(views)
        n = views[0].shape[0]
        c = self.shrinkage
        centred_reps = [z - z.mean(axis=0) for z in representations]
        total = sum(centred_reps)
        _, v_data = ey_cross_covariance(representations)
        b = weight_gram_mean(weights)
        v_blend = (1 - c) * v_data + c * b
        scale = 4.0 / (m * (n - 1))
        grads = []
        for k, (view, zk) in enumerate(zip(views, centred_reps)):
            view_c = view - view.mean(axis=0)
            z_term = scale * (c * zk + (1 - c) * (zk @ v_blend) - total)
            grad = view_c.T @ z_term + (4.0 * c / m) * (weights[k] @ v_blend)
            grads.append(grad)
        return grads

    def _objective(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> float:
        r"""$\mathcal{L}_{EY}(c)$."""
        del views
        c = self.shrinkage
        C, v_data = ey_cross_covariance(representations)
        b = weight_gram_mean(weights)
        v_blend = (1 - c) * v_data + c * b
        reward = C - c * v_data
        return float(-2.0 * np.trace(reward) + np.trace(v_blend @ v_blend))
