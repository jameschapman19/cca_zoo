"""Generalized CCA."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import psd_inverse_sqrt
from cca_zoo._utils._param_constraints import POSITIVE_EPS, RIDGE_PARAMETER
from cca_zoo._utils._validation import perview_parameter


class GCCA(BaseModel):
    r"""Generalized CCA: views correlated with a shared latent variable.

    $$
    \max_{w_i, T} \sum_i \mu_i w_i^\top X_i^\top T
    \quad \text{subject to} \quad T^\top T = I.
    $$

    $T$ holds the top eigenvectors of
    $\sum_i \mu_i X_i ((1 - c_i) X_i^\top X_i + c_i I)^{-1} X_i^\top$,
    found from the SVD of the stacked whitened views, and
    $w_i = X_i^+ T$.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        c: Ridge blend in ``[0, 1]``. Per-view. Default is 0.
        view_weights: Weight $\mu_i$ of each view; None weights them
            equally. Default is None.
        eps: Floor on the within-view eigenvalues. Default is 1e-6.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Tenenhaus, A., & Tenenhaus, M. (2011). Regularized generalized
        canonical correlation analysis. Psychometrika, 76(2), 257-284.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import GCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> X3 = rng.standard_normal((50, 6))
        >>> model = GCCA(n_components=2).fit([X1, X2, X3])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "c": RIDGE_PARAMETER,
        "eps": POSITIVE_EPS,
    }

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        c: float | list[float] = 0.0,
        view_weights: list[float] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.c = c
        self.view_weights = view_weights
        self.eps = eps

    def fit(self, views: list[ArrayLike], y: None = None) -> GCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        c_ = perview_parameter("c", self.c, 0.0, self.n_views_)
        mu = perview_parameter("view_weights", self.view_weights, 1.0, self.n_views_)

        # Q = sum_i mu_i X_i cov_i^{-1} X_i^T is H H^T for the stacked
        # whitened views H = [sqrt(mu_i) X_i cov_i^{-1/2}], so its top
        # eigenvectors are H's top left singular vectors: an n x sum(p_i)
        # SVD in place of an n x n eigenproblem.
        stacked = np.hstack(
            [
                np.sqrt(mi)
                * v
                @ psd_inverse_sqrt(
                    (1.0 - ci) * np.cov(v, rowvar=False) + ci * np.eye(v.shape[1]),
                    self.eps,
                )
                for v, ci, mi in zip(views_, c_, mu)
            ]
        )
        T = np.linalg.svd(stacked, full_matrices=False)[0][:, : self.n_components]
        self.weights_: list[np.ndarray] = [np.linalg.pinv(v) @ T for v in views_]
        return self
