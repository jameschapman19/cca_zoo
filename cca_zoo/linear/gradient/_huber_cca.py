"""Bounded-influence Eckart-Young CCA."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._utils._ey import cheap_orthonormal_projection_weights
from cca_zoo.linear.gradient._base import BaseFullBatchEYModel


def _huber_sample_weight(representations: list[np.ndarray], delta: float) -> np.ndarray:
    """Huber weights capping each sample's leverage.

    A sample's leverage is the norm of its standardised embeddings over every
    view and component. Samples within ``delta`` times the median leverage
    keep weight 1; the rest are weighted by ``cutoff / leverage``.

    Args:
        representations: One array of shape (n_samples, k) per view.
        delta: Cutoff as a multiple of the median leverage.

    Returns:
        Weights in ``(0, 1]``, shape (n_samples,).
    """
    standardised = [
        (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-12) for z in representations
    ]
    leverage = np.sqrt(sum((s**2).sum(axis=1) for s in standardised))
    cutoff = delta * np.median(leverage) + 1e-12
    result: np.ndarray = np.minimum(1.0, cutoff / (leverage + 1e-12))
    return result


def _weighted_ey(
    representations: list[np.ndarray], sample_weight: np.ndarray
) -> tuple[float, list[np.ndarray]]:
    """Sample-weighted EY loss and its gradient in each embedding.

    As :func:`~cca_zoo._utils._ey.ey_loss` with weighted moments and
    ``n - 1`` replaced by ``sum(sample_weight) - 1``. The weights are held
    fixed, as in IRLS.

    Args:
        representations: One array of shape (n_samples, k) per view.
        sample_weight: Weights, shape (n_samples,).

    Returns:
        ``(objective, grads)``, with one gradient of shape (n_samples, k) per
        view.
    """
    m = len(representations)
    w = sample_weight
    w_sum = w.sum()
    means = [np.average(z, axis=0, weights=w) for z in representations]
    centred = [z - mu for z, mu in zip(representations, means)]
    total = sum(centred)

    k = centred[0].shape[1]
    C = np.zeros((k, k))
    V = np.zeros((k, k))
    for zi in centred:
        wzi = w[:, None] * zi
        V += wzi.T @ zi / (w_sum - 1)
        for zj in centred:
            C += wzi.T @ zj / (w_sum - 1)
    C /= m
    V /= m
    objective = float(-2.0 * np.trace(C) + np.trace(V @ V))

    scale = 4.0 / (m * (w_sum - 1))
    grad_z = [scale * w[:, None] * (zc @ V - total) for zc in centred]
    return objective, grad_z


class HuberCCA(BaseFullBatchEYModel):
    """Bounded-influence CCA by Huber reweighting of the EY loss.

    As :class:`~cca_zoo.linear.gradient.CCAEY`, but each sample's
    contribution to the EY moments is reweighted by a Huber weight of its
    leverage, recomputed at every evaluation, so high-leverage samples have
    bounded influence. Fitted by full-batch L-BFGS-B.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        delta: Huber cutoff as a multiple of the median sample leverage;
            smaller is more robust. Values below 1 downweight most of the
            data. Default is 4.0.
        max_iter: Maximum L-BFGS-B iterations. Default is 1000.
        tol: L-BFGS-B ``ftol``. Default is 1e-8.
        random_state: Seed for the initial weights. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Filzmoser, P., Dehon, C., & Croux, C. (2000). Outlier resistant
        estimators for canonical correlation analysis. In COMPSTAT:
        Proceedings in Computational Statistics 2000 (pp. 301-306).
        Physica-Verlag.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.linear import HuberCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((1000, 20))
        >>> X2 = rng.standard_normal((1000, 15))
        >>> model = HuberCCA(n_components=4, random_state=0).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseFullBatchEYModel._parameter_constraints,
        "delta": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        delta: float = 4.0,
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
        self.delta = delta

    def fit(self, views: list[ArrayLike], y: None = None) -> HuberCCA:
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
        return self

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
        """Gradient of the weighted EY loss in each view's weights.

        The weighted-mean-centred view contracted with :func:`_weighted_ey`'s
        embedding gradient.
        """
        del weights
        sample_weight = _huber_sample_weight(representations, self.delta)
        _, grad_z = _weighted_ey(representations, sample_weight)
        grads = []
        for view, gz in zip(views, grad_z):
            view_mean = np.average(view, axis=0, weights=sample_weight)
            view_cw = view - view_mean
            grads.append(view_cw.T @ gz)
        return grads

    def _objective(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> float:
        """The weighted EY loss."""
        del views, weights
        sample_weight = _huber_sample_weight(representations, self.delta)
        objective, _ = _weighted_ey(representations, sample_weight)
        return objective
