"""Sparse CCA by penalized matrix decomposition (Witten et al., 2009)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._linalg import soft_threshold
from cca_zoo._utils._param_constraints import (
    FRACTION_PER_VIEW,
    POSITIVE_EPS,
    POSITIVE_INT,
    RANDOM_STATE,
)
from cca_zoo._utils._validation import perview_parameter
from cca_zoo.sparse._deflation import Deflation, others_score, pls_direction


class PMDCCA(BaseModel):
    r"""Sparse CCA by penalized matrix decomposition.

    Maximises cross-view covariance subject to L1 and L2 constraints on each
    weight vector:

    $$
    \max_{\mathbf{w}_1, \mathbf{w}_2} \mathbf{w}_1^\top X_1^\top X_2 \mathbf{w}_2
    \quad\text{subject to}\quad
    \|\mathbf{w}_i\|_1 \le b_i \sqrt{p_i},\ \|\mathbf{w}_i\|_2 = 1.
    $$

    Each update soft-thresholds with the level found by bisection.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        l1_bound: Bound $b_i$ on the L1 norm, as a fraction of
            ``sqrt(n_features_i)``, in ``(0, 1]``;
            1 imposes no sparsity. Per-view. Default is 1.
        max_iter: Maximum iterations per latent dimension. Default is 500.
        tol: Convergence tolerance on the change in weights. Default is 1e-6.
        random_state: Seed for the start. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Most iterations run by any component.

    References:
        Witten, D. M., Tibshirani, R., & Hastie, T. (2009). A penalized matrix
        decomposition, with applications to sparse principal components and
        canonical correlation analysis. Biostatistics, 10(3), 515-534.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.sparse import PMDCCA
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 10)), rng.standard_normal((50, 8))
        >>> model = PMDCCA(l1_bound=0.5, random_state=0).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "l1_bound": FRACTION_PER_VIEW,
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        l1_bound: float | list[float] = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.l1_bound = l1_bound
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> PMDCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        bounds = [
            b * np.sqrt(p)
            for b, p in zip(
                perview_parameter("l1_bound", self.l1_bound, 1.0, self.n_views_),
                self.n_features_per_view_,
            )
        ]
        self.n_iter_: int = 0
        converged = True
        deflation = Deflation(views_, self.n_components)
        for deflated in deflation:
            w = pls_direction(deflated, rng)
            for n_iter in range(1, self.max_iter + 1):
                previous = [wi.copy() for wi in w]
                # Power iteration, soft-thresholded to each view's L1 bound.
                for i, (view, bound) in enumerate(zip(deflated, bounds)):
                    w[i] = _threshold_to_l1_bound(
                        view.T @ others_score(deflated, w, i), bound
                    )
                if max(np.linalg.norm(a - b) for a, b in zip(w, previous)) < self.tol:
                    break
            else:
                converged = False
            self.n_iter_ = max(self.n_iter_, n_iter)
            deflation.record(w)
        warn_if_not_converged(self, converged)
        self.weights_: list[np.ndarray] = deflation.weights()
        self._fit_maps_and_importances(views_)
        return self


def _threshold_to_l1_bound(x: np.ndarray, l1_bound: float) -> np.ndarray:
    """``x`` soft-thresholded to the unit vector with L1 norm ``l1_bound``.

    Witten et al.'s search for the threshold, on the L1/L2 ratio of the
    thresholded vector, which is invariant to the scale of ``x``. The ratio
    is 0 once the threshold reaches ``max|x|`` and exceeds the bound at 0,
    so Brent's method has a valid bracket.
    """
    norm_x = np.linalg.norm(x)
    if norm_x <= 1e-12:
        return np.zeros_like(x)
    if np.linalg.norm(x, 1) / norm_x <= l1_bound:
        return np.asarray(x / norm_x)

    def l1_over_l2_minus_bound(delta: float) -> float:
        thresholded = soft_threshold(x, delta)
        norm_t = np.linalg.norm(thresholded)
        ratio = np.linalg.norm(thresholded, 1) / norm_t if norm_t > 1e-12 else 0.0
        return ratio - l1_bound

    delta = brentq(l1_over_l2_minus_bound, 0.0, np.abs(x).max(), xtol=1e-10)
    result = soft_threshold(x, delta)
    return result / max(float(np.linalg.norm(result)), 1e-12)
