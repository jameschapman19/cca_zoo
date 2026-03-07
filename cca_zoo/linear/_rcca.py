"""Regularised CCA (rCCA) — canonical ridge via SVD for exactly two views."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import svd_whiten
from cca_zoo._utils._validation import perview_parameter


class rCCA(BaseModel):
    r"""Regularised Canonical Correlation Analysis (canonical ridge).

    Finds the pair of linear projections of two views that maximise their
    correlation subject to regularised within-view variance constraints:

    .. math::

        \max_{\mathbf{w}_1, \mathbf{w}_2}
            \mathbf{w}_1^\top X_1^\top X_2 \mathbf{w}_2

        \text{subject to }
        \mathbf{w}_i^\top
        \bigl((1 - c_i) X_i^\top X_i + c_i I\bigr) \mathbf{w}_i = 1

    The solution is found by whitening each view with its regularised
    covariance matrix and computing the SVD of the resulting cross-covariance.

    :class:`CCA` (``c=0``) and :class:`PLS` (``c=1``) are special cases.

    References:
        Vinod, H. D. (1976). Canonical ridge and econometrics of joint
        production. *Journal of Econometrics*, 4(2), 147–166.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.
        c: Ridge regularisation parameter(s) in ``[0, 1]``.  A single float
            is applied to both views; a list ``[c1, c2]`` applies per-view
            regularisation.  Default is 0 (standard CCA).

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = rCCA(latent_dimensions=2, c=0.1).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        c: float | list[float] = 0.0,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.c = c

    def fit(self, views: list[ArrayLike], y: None = None) -> rCCA:
        """Fit the rCCA model.

        Args:
            views: List of exactly two arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If the number of views is not exactly 2.
            ValueError: If views have inconsistent numbers of samples.
        """
        views = self._setup_fit(views)
        if self.n_views_ != 2:
            raise ValueError(
                f"rCCA requires exactly 2 views, got {self.n_views_}. "
                "Use MCCA for more than 2 views."
            )
        c_ = perview_parameter("c", self.c, 0.0, 2)
        X1, X2 = views
        # Whiten each view with its regularised covariance
        X1_w, W1 = svd_whiten(X1, c_[0])
        X2_w, W2 = svd_whiten(X2, c_[1])
        # SVD of the cross-covariance of whitened views
        k = min(self.latent_dimensions, X1_w.shape[1], X2_w.shape[1])
        cross_cov = X1_w.T @ X2_w / (X1.shape[0] - 1)
        U, _, Vt = np.linalg.svd(cross_cov, full_matrices=False)
        U = U[:, :k]
        Vt = Vt[:k, :]
        self.weights_: list[np.ndarray] = [W1 @ U, W2 @ Vt.T]
        return self
