"""KGCCA — Kernel Generalised Canonical Correlation Analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._validation import perview_parameter


class KGCCA(BaseModel):
    r"""Kernel Generalised Canonical Correlation Analysis.

    Kernelised version of GCCA.  The shared latent vector is found by
    solving the eigenvalue problem on the weighted sum of kernel projection
    matrices:

    .. math::

        Q = \sum_{i=1}^M \mu_i K_i
            \bigl(c_i K_i + (1 - c_i) K_i^2\bigr)^{-1} K_i

    and the dual variables (kernel coefficients) are recovered as
    :math:`\boldsymbol{\alpha}_i = K_i^+ T` where :math:`T` is the matrix
    of top-k eigenvectors of :math:`Q`.

    References:
        Tenenhaus, A., Philippe, C., & Frouin, V. (2015). Kernel generalized
        canonical correlation analysis. *Computational Statistics & Data
        Analysis*, 90, 114–131.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.
        c: Regularisation parameter(s). Default is 0.1.
        kernel: Kernel name(s). Default is ``"linear"``.
        gamma: Gamma for RBF/polynomial kernel.
        degree: Degree for polynomial kernel.
        coef0: coef0 for polynomial/sigmoid kernel.
        kernel_params: Extra per-view kernel keyword arguments.
        view_weights: Per-view weights. Default is equal weights.
        eps: Regularisation floor. Default is 1e-6.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((30, 5))
        >>> X2 = rng.standard_normal((30, 5))
        >>> X3 = rng.standard_normal((30, 5))
        >>> model = KGCCA(latent_dimensions=2).fit([X1, X2, X3])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        c: float | list[float] = 0.1,
        kernel: str | list[str] = "linear",
        gamma: float | list[float | None] | None = None,
        degree: float | list[float] = 1.0,
        coef0: float | list[float] = 1.0,
        kernel_params: dict[str, object] | list[dict[str, object]] | None = None,
        view_weights: list[float] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.view_weights = view_weights
        self.eps = eps

    def fit(self, views: list[ArrayLike], y: None = None) -> KGCCA:
        """Fit the KGCCA model.

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
        c_ = perview_parameter("c", self.c, 0.1, self.n_views_)
        mu = perview_parameter("view_weights", self.view_weights, 1.0, self.n_views_)
        kernel_ = perview_parameter("kernel", self.kernel, "linear", self.n_views_)
        gamma_ = perview_parameter("gamma", self.gamma, None, self.n_views_)
        degree_ = perview_parameter("degree", self.degree, 1.0, self.n_views_)
        coef0_ = perview_parameter("coef0", self.coef0, 1.0, self.n_views_)
        kp_ = perview_parameter("kernel_params", self.kernel_params, {}, self.n_views_)

        self.train_views_: list[np.ndarray] = views_
        kernels = [
            pairwise_kernels(
                v,
                metric=kernel_[i],
                gamma=gamma_[i],
                degree=degree_[i],
                coef0=coef0_[i],
                filter_params=True,
                **(kp_[i] if kp_[i] else {}),
            )
            for i, v in enumerate(views_)
        ]
        # Build Q (n x n)
        Q = np.zeros((self.n_samples_, self.n_samples_))
        for i, (K, ci, mi) in enumerate(zip(kernels, c_, mu)):
            B_i = ci * K + (1.0 - ci) * K @ K
            min_eig = np.linalg.eigvalsh(B_i).min()
            if min_eig < self.eps:
                B_i += (self.eps - min_eig) * np.eye(B_i.shape[0])
            Q += mi * K @ np.linalg.inv(B_i) @ K

        _, eigvecs = gevp(Q, None, self.latent_dimensions)
        T = eigvecs[:, : self.latent_dimensions]
        self.weights_: list[np.ndarray] = [np.linalg.pinv(K) @ T for K in kernels]
        # Store kernel parameters for transform
        self._kernel = kernel_
        self._gamma = gamma_
        self._degree = degree_
        self._coef0 = coef0_
        self._kp = kp_
        return self

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Transform new views using fitted kernel dual variables.

        Args:
            views: List of arrays, each (n_samples_test, n_features_i).

        Returns:
            List of arrays, each (n_samples_test, latent_dimensions).

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
        """
        check_is_fitted(self)
        from cca_zoo._utils._validation import validate_views

        validated = validate_views(views)
        result = []
        for i, v in enumerate(validated):
            K_test = pairwise_kernels(
                self.train_views_[i],
                Y=v,
                metric=self._kernel[i],
                gamma=self._gamma[i],
                degree=self._degree[i],
                coef0=self._coef0[i],
                filter_params=True,
                **(self._kp[i] if self._kp[i] else {}),
            )
            result.append(K_test.T @ self.weights_[i])
        return result
