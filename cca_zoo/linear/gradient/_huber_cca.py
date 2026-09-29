"""Bounded-influence Eckart-Young CCA."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo.linear.gradient._cca_ey import CCAEY


def _huber_sample_weight(representations: list[np.ndarray], delta: float) -> np.ndarray:
    """Huber weights capping each sample's leverage.

    A sample's leverage is the Mahalanobis norm of its embeddings, summed in
    square over views, so it does not depend on how the components are
    rotated. Samples within ``delta`` times the median leverage keep weight
    1; the rest are weighted by ``cutoff / leverage``.

    Args:
        representations: One array of shape (n_samples, k) per view.
        delta: Cutoff as a multiple of the median leverage.

    Returns:
        Weights in ``(0, 1]``, shape (n_samples,).
    """
    leverage_sq = np.zeros(len(representations[0]))
    for z in representations:
        centred = z - z.mean(axis=0)
        precision = np.linalg.pinv(centred.T @ centred / len(z))
        leverage_sq += np.sum((centred @ precision) * centred, axis=1)
    leverage = np.sqrt(leverage_sq)
    cutoff = delta * np.median(leverage) + 1e-12
    result: np.ndarray = np.minimum(1.0, cutoff / (leverage + 1e-12))
    return result


class HuberCCA(CCAEY):
    """Bounded-influence CCA by Huber reweighting of the EY loss.

    :class:`~cca_zoo.linear.gradient.CCAEY`, with each sample's contribution
    to the EY moments reweighted by a Huber weight of its leverage,
    recomputed at every evaluation, so high-leverage samples have bounded
    influence. Fitted by full-batch L-BFGS-B, with each evaluation's weights
    held fixed in its gradient, as in iteratively reweighted least squares.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's covariance towards the identity,
            in ``[0, 1]``: 0 is CCA and 1 is PLS. Default is 0.
        delta: Huber cutoff as a multiple of the median sample leverage;
            smaller is more robust. Values below 1 downweight most of the
            data. Default is 4.0.
        max_iter: Maximum L-BFGS-B iterations. Default is 1000.
        tol: L-BFGS-B ``ftol``. Default is 1e-8.
        random_state: Seed for the initial weights. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: L-BFGS-B iterations run.

    References:
        Filzmoser, P., Dehon, C., & Croux, C. (2000). Outlier resistant
        estimators for canonical correlation analysis. In COMPSTAT:
        Proceedings in Computational Statistics 2000 (pp. 301-306).
        Physica-Verlag.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import HuberCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((1000, 20))
        >>> X2 = rng.standard_normal((1000, 15))
        >>> model = HuberCCA(n_components=4, random_state=0).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **CCAEY._parameter_constraints,
        "delta": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float = 0.0,
        delta: float = 4.0,
        max_iter: int = 1000,
        tol: float = 1e-8,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            shrinkage=shrinkage,
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
        return super().fit(views, y)

    def _sample_weight(self, representations: list[np.ndarray]) -> np.ndarray:
        """Each sample's Huber weight of its leverage."""
        return _huber_sample_weight(representations, self.delta)
