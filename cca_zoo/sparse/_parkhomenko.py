"""Sparse CCA by soft-thresholded power iteration (Parkhomenko et al., 2009)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._linalg import soft_threshold
from cca_zoo._utils._param_constraints import (
    NONNEGATIVE_PER_VIEW,
    POSITIVE_EPS,
    POSITIVE_INT,
    RANDOM_STATE,
)
from cca_zoo._utils._validation import perview_parameter
from cca_zoo.sparse._deflation import Deflation, others_score, pls_direction


class ParkhomenkoCCA(BaseModel):
    r"""Sparse CCA by soft-thresholded power iteration on standardised views.

    Uses the paper's diagonal approximation to the within-view covariances,
    which amounts to standardising each feature, then iterates

    $$
    \mathbf{w}_i \leftarrow
        S_{\alpha_i}\bigl(\tfrac{1}{n} \tilde X_i^\top \bar{\mathbf{s}}_{\neg i}\bigr),
    $$

    with $S_\alpha$ the soft-threshold, $\tilde X_i$ the standardised view and
    $\bar{\mathbf{s}}_{\neg i}$ the other views' summed score at unit variance:
    each feature's correlation with that score is soft-thresholded at
    $\alpha_i$. Weights are returned on the original feature scale.

    One of three sparse PLS power iterations, with :class:`PMDCCA` and
    :class:`SpanCCA`: each multiplies by the other views' summed score, then
    thresholds, and they differ only in the threshold.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        alpha: Soft-threshold on each feature's correlation with the other
            views' score; 1 or more zeroes every weight. Per-view. Default
            is 0.1.
        max_iter: Maximum iterations per latent dimension. Default is 500.
        tol: Convergence tolerance on the change in weights. Default is 1e-6.
        random_state: Seed for the start. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Most iterations run by any component.

    References:
        Parkhomenko, E., Tritchler, D., & Beyene, J. (2009). Sparse canonical
        correlation analysis with application to genomic data integration.
        Statistical Applications in Genetics and Molecular Biology, 8(1).

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.sparse import ParkhomenkoCCA
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 10)), rng.standard_normal((50, 8))
        >>> model = ParkhomenkoCCA(alpha=0.1, random_state=0).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": NONNEGATIVE_PER_VIEW,
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        alpha: float | list[float] = 0.1,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> ParkhomenkoCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        alphas = perview_parameter("alpha", self.alpha, 0.1, self.n_views_)
        self.n_iter_: int = 0
        converged = True
        deflation = Deflation(views_, self.n_components)
        for deflated in deflation:
            # The diagonal approximation to each covariance: standardise.
            scales = [
                np.where(s < 1e-12, 1.0, s) for s in (v.std(axis=0) for v in deflated)
            ]
            standardised = [v / s for v, s in zip(deflated, scales)]
            w = pls_direction(standardised, rng)
            for n_iter in range(1, self.max_iter + 1):
                previous = [wi.copy() for wi in w]
                # Each feature's correlation with the others' score,
                # soft-thresholded at the view's alpha.
                for i, (view, alpha) in enumerate(zip(standardised, alphas)):
                    correlations = view.T @ others_score(standardised, w, i) / len(view)
                    raw = soft_threshold(correlations, alpha)
                    w[i] = raw / max(np.linalg.norm(raw), 1e-12)
                if max(np.linalg.norm(a - b) for a, b in zip(w, previous)) < self.tol:
                    break
            else:
                converged = False
            self.n_iter_ = max(self.n_iter_, n_iter)
            # Back to the original feature scale, at unit norm.
            w = [wi / s for wi, s in zip(w, scales)]
            deflation.record([wi / max(np.linalg.norm(wi), 1e-12) for wi in w])
        warn_if_not_converged(self, converged)
        self.weights_: list[np.ndarray] = deflation.weights()
        self._fit_maps_and_importances(views_)
        return self
