"""GCCA — Generalised Canonical Correlation Analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._validation import perview_parameter


class GCCA(BaseModel):
    r"""Generalised Canonical Correlation Analysis.

    Finds linear projections of multiple (>=2) views that maximise their
    joint correlation with a shared auxiliary latent vector:

    .. math::

        \max_{\mathbf{w}_i, T}
            \sum_{i=1}^M \mathbf{w}_i^\top X_i^\top T

        \text{subject to }
        T^\top T = I

    The solution is obtained by constructing the weighted projection matrix:

    .. math::

        Q = \sum_{i=1}^M \mu_i X_i
            \bigl((1-c_i) X_i^\top X_i + c_i I\bigr)^{-1} X_i^\top

    and computing its top-k eigenvectors :math:`V`, then recovering the
    per-view weights as :math:`\mathbf{w}_i = X_i^+ V`.

    References:
        Tenenhaus, A., & Tenenhaus, M. (2011). Regularized generalized
        canonical correlation analysis. *Psychometrika*, 76(2), 257–284.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.
        c: Ridge regularisation parameter(s) in ``[0, 1]``.  Default is 0.
        view_weights: Per-view weights :math:`\mu_i` in the GCCA objective.
            Default is equal weights (1 for all views).
        eps: Regularisation floor for within-view matrices.  Default is 1e-6.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> X3 = rng.standard_normal((50, 6))
        >>> model = GCCA(latent_dimensions=2).fit([X1, X2, X3])
        >>> scores = model.transform([X1, X2, X3])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        c: float | list[float] = 0.0,
        view_weights: list[float] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.c = c
        self.view_weights = view_weights
        self.eps = eps

    def fit(self, views: list[ArrayLike], y: None = None) -> GCCA:
        """Fit the GCCA model.

        Args:
            views: List of arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        c_ = perview_parameter("c", self.c, 0.0, self.n_views_)
        mu = perview_parameter("view_weights", self.view_weights, 1.0, self.n_views_)

        # Build Q = sum_i mu_i X_i (cov_i)^{-1} X_i^T
        Q = np.zeros((self.n_samples_, self.n_samples_))
        for i, (v, ci, mi) in enumerate(zip(views_, c_, mu)):
            cov_i = (1.0 - ci) * np.cov(v, rowvar=False) + ci * np.eye(v.shape[1])
            min_eig = np.linalg.eigvalsh(cov_i).min()
            if min_eig < self.eps:
                cov_i += (self.eps - min_eig) * np.eye(cov_i.shape[0])
            Q += mi * (v @ np.linalg.inv(cov_i) @ v.T)

        _, eigvecs = gevp(Q, None, self.latent_dimensions)
        T = eigvecs[:, : self.latent_dimensions]  # (n_samples, k)
        self.weights_: list[np.ndarray] = [np.linalg.pinv(v) @ T for v in views_]
        return self
