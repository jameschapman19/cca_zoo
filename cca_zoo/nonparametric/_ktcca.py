"""KTCCA — Kernel Tensor Canonical Correlation Analysis."""

from __future__ import annotations

import numpy as np
import tensorly as tl
from numpy.typing import ArrayLike
from scipy.linalg import sqrtm
from sklearn.metrics import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from tensorly.decomposition import parafac

from cca_zoo._base import BaseModel
from cca_zoo._utils._validation import perview_parameter


class KTCCA(BaseModel):
    r"""Kernel Tensor Canonical Correlation Analysis.

    Extends TCCA to nonlinear relationships by computing the cross-moment
    tensor from whitened kernel matrices rather than from the raw views.
    Each kernel matrix :math:`K_i` is whitened using its regularised
    self-product, then PARAFAC is applied to the resulting cross-moment
    tensor.

    References:
        Kim, T.-K., Wong, S.-F., & Cipolla, R. (2007). Tensor canonical
        correlation analysis for action classification. *CVPR 2007*. IEEE.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.
        c: Regularisation parameter(s). Default is 0.1.
        kernel: Kernel name(s). Default is ``"linear"``.
        gamma: Gamma for RBF/polynomial kernel.
        degree: Degree for polynomial kernel.
        coef0: coef0 for polynomial/sigmoid kernel.
        kernel_params: Extra per-view keyword arguments for the kernel.
        eps: Regularisation floor. Default is 1e-3.
        random_state: Seed for PARAFAC. Default is None.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((20, 5))
        >>> X2 = rng.standard_normal((20, 5))
        >>> X3 = rng.standard_normal((20, 5))
        >>> model = KTCCA(latent_dimensions=1, random_state=0).fit([X1, X2, X3])
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
        random_state: int | None = None,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.eps = eps
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> KTCCA:
        """Fit the KTCCA model.

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
        # Store parameters for transform
        self._kernel = kernel_
        self._gamma = gamma_
        self._degree = degree_
        self._coef0 = coef0_
        self._kp = kp_

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
        whitened, self._cov_invsqrt = self._whiten_kernels(kernels, c_)

        # Build cross-moment tensor
        M: np.ndarray | None = None
        for i, wk in enumerate(whitened):
            if M is None:
                M = wk
            else:
                for _ in range(len(M.shape) - 1):
                    wk = np.expand_dims(wk, 1)
                M = np.expand_dims(M, -1) @ wk
        assert M is not None
        M = np.mean(M, 0)

        tl.set_backend("numpy")
        parafac_result = parafac(
            M,
            self.latent_dimensions,
            verbose=False,
            random_state=self.random_state,
        )
        self.weights_: list[np.ndarray] = [
            self._cov_invsqrt[i] @ fac for i, fac in enumerate(parafac_result.factors)
        ]
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

    def _whiten_kernels(
        self,
        kernels: list[np.ndarray],
        c: list[float],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Whiten kernel matrices using their regularised self-products.

        Args:
            kernels: List of kernel matrices.
            c: Per-view regularisation parameters.

        Returns:
            Tuple of (whitened_kernels, inverse_sqrt_matrices).
        """
        whitened = []
        cov_invsqrt = []
        for i, K in enumerate(kernels):
            cov = (1.0 - c[i]) * K @ K + c[i] * K
            min_eig = np.linalg.eigvalsh(cov).min()
            if min_eig < self.eps:
                cov += (self.eps - min_eig) * np.eye(cov.shape[0])
            invsqrt = np.linalg.inv(sqrtm(cov).real)
            whitened.append(K @ invsqrt)
            cov_invsqrt.append(invsqrt)
        return whitened, cov_invsqrt
