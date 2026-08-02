"""KCCA — Kernel Canonical Correlation Analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._validation import perview_parameter


class KCCA(BaseModel):
    r"""Kernel Canonical Correlation Analysis.

    Extends MCCA to nonlinear relationships by mapping each view into a
    reproducing kernel Hilbert space via a kernel function $k_i$.  The
    dual variables (kernel coefficients) $\boldsymbol{\alpha}_i$ are
    found by solving the kernelised generalised eigenvalue problem:

    $$
    A \boldsymbol{\alpha} = \lambda B \boldsymbol{\alpha}
    $$

    where:

    * $A$ is the between-kernel cross-covariance block matrix.
    * $B = \mathrm{block\_diag}\bigl(
          c_i K_i + (1 - c_i) K_i^2
      \bigr)$ is the regularised within-kernel matrix.

    References:
        Hardoon, D. R., Szedmak, S., & Shawe-Taylor, J. (2004). Canonical
        correlation analysis: An overview with application to learning methods.
        *Neural Computation*, 16(12), 2639–2664.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.
        c: Regularisation parameter(s) in ``[0, 1]``. Default is 0.1.
        kernel: Kernel name(s) or callable(s) passed to
            :func:`sklearn.metrics.pairwise_kernels`. Default is ``"linear"``.
        gamma: Gamma parameter(s) for the RBF/polynomial kernel.
        degree: Degree parameter(s) for the polynomial kernel.
        coef0: coef0 parameter(s) for the polynomial/sigmoid kernel.
        kernel_params: Extra per-view keyword arguments for the kernel.
        eps: Regularisation floor for the B matrix. Default is 1e-3.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((30, 5))
        >>> X2 = rng.standard_normal((30, 5))
        >>> model = KCCA(latent_dimensions=2, c=0.1).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
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
        eps: float = 1e-3,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.eps = eps

    def fit(self, views: list[ArrayLike], y: None = None) -> KCCA:
        """Fit the KCCA model.

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
        kernel_ = perview_parameter("kernel", self.kernel, "linear", self.n_views_)
        gamma_ = perview_parameter("gamma", self.gamma, None, self.n_views_)
        degree_ = perview_parameter("degree", self.degree, 1.0, self.n_views_)
        coef0_ = perview_parameter("coef0", self.coef0, 1.0, self.n_views_)
        kp_ = perview_parameter("kernel_params", self.kernel_params, {}, self.n_views_)

        self.train_views_: list[np.ndarray] = views_
        kernels = self._compute_kernels(views_, kernel_, gamma_, degree_, coef0_, kp_)
        A = self._build_A(kernels)
        B = self._build_B(kernels, c_)
        splits = np.cumsum([k.shape[1] for k in kernels])
        _, eigvecs = gevp(A, B, self.latent_dimensions)
        self.weights_: list[np.ndarray] = list(np.split(eigvecs, splits[:-1], axis=0))
        # Store kernel parameters for transform
        self._kernel: list[str] = kernel_
        self._gamma: list[float | None] = gamma_
        self._degree: list[float] = degree_
        self._coef0: list[float] = coef0_
        self._kp: list[dict[str, object]] = kp_
        return self

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Transform new views using the fitted kernel dual variables.

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

    def _compute_kernels(
        self,
        views: list[np.ndarray],
        kernel: list[str],
        gamma: list[float | None],
        degree: list[float],
        coef0: list[float],
        kp: list[dict[str, object]],
    ) -> list[np.ndarray]:
        """Compute training kernel matrices.

        Args:
            views: Training view arrays.
            kernel: Per-view kernel names.
            gamma: Per-view gamma values.
            degree: Per-view degree values.
            coef0: Per-view coef0 values.
            kp: Per-view extra kernel parameters.

        Returns:
            List of kernel matrices, each (n_samples, n_samples).
        """
        return [
            pairwise_kernels(
                v,
                metric=kernel[i],
                gamma=gamma[i],
                degree=degree[i],
                coef0=coef0[i],
                filter_params=True,
                **(kp[i] if kp[i] else {}),
            )
            for i, v in enumerate(views)
        ]

    def _build_A(self, kernels: list[np.ndarray]) -> np.ndarray:
        """Build the between-kernel covariance block matrix.

        Args:
            kernels: List of kernel matrices.

        Returns:
            Block covariance matrix.
        """
        all_k = np.hstack(kernels)
        A = np.cov(all_k, rowvar=False)
        A -= block_diag(*[np.cov(k, rowvar=False) for k in kernels])
        return A / len(kernels)

    def _build_B(self, kernels: list[np.ndarray], c: list[float]) -> np.ndarray:
        """Build the regularised within-kernel block matrix.

        Args:
            kernels: List of kernel matrices.
            c: Per-view regularisation parameters.

        Returns:
            Block-diagonal positive-definite matrix.
        """
        blocks = [
            c[i] * kernels[i] + (1.0 - c[i]) * kernels[i] @ kernels[i]
            for i in range(len(kernels))
        ]
        B: np.ndarray = np.asarray(block_diag(*blocks))
        min_eig = np.linalg.eigvalsh(B).min()
        if min_eig < self.eps:
            B += (self.eps - min_eig) * np.eye(B.shape[0])
        return np.asarray(B / len(kernels))
