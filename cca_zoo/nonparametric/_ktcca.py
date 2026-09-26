"""KTCCA — Kernel Tensor Canonical Correlation Analysis."""

from __future__ import annotations

import numpy as np
import tensorly as tl
from numpy.typing import ArrayLike
from sklearn.metrics import pairwise_kernels
from tensorly.decomposition import parafac

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import cross_moment_tensor, psd_inverse_sqrt
from cca_zoo._utils._validation import perview_parameter


class KTCCA(BaseModel):
    r"""Kernel Tensor Canonical Correlation Analysis.

    Extends TCCA to nonlinear relationships by computing the cross-moment
    tensor from whitened kernel matrices rather than from the raw views.
    Each kernel matrix $K_i$ is whitened using its regularised
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

    Examples:
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

        M = cross_moment_tensor(whitened)

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

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        kernel = pairwise_kernels(
            centred,
            self.train_views_[view],
            metric=self._kernel[view],
            gamma=self._gamma[view],
            degree=self._degree[view],
            coef0=self._coef0[view],
            filter_params=True,
            **(self._kp[view] if self._kp[view] else {}),
        )
        scores: np.ndarray = kernel @ self.weights_[view]
        return scores

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
            invsqrt = psd_inverse_sqrt(cov, self.eps)
            whitened.append(K @ invsqrt)
            cov_invsqrt.append(invsqrt)
        return whitened, cov_invsqrt
