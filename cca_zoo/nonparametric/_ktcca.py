"""Kernel tensor CCA."""

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
    """Kernel tensor CCA.

    :class:`~cca_zoo.linear.TCCA` on whitened kernel matrices: PARAFAC of
    their cross-moment tensor gives the dual coefficients.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        c: Ridge blend in ``[0, 1]``. Per-view. Default is 0.1.
        kernel: Kernel name or callable for
            :func:`~sklearn.metrics.pairwise_kernels`. Per-view. Default is
            ``"linear"``.
        gamma: Kernel coefficient for RBF, polynomial and sigmoid kernels.
            Per-view. Default is None.
        degree: Polynomial kernel degree. Per-view. Default is 1.
        coef0: Polynomial and sigmoid kernel constant. Per-view. Default is 1.
        kernel_params: Extra kernel keyword arguments. Per-view. Default is
            None.
        eps: Floor added to the within-view matrices. Default is 1e-3.
        random_state: Seed for PARAFAC. Default is None.

    Attributes:
        weights_: Dual coefficients of each view, shape (n_samples,
            n_components).

    References:
        Kim, T.-K., Wong, S.-F., & Cipolla, R. (2007). Tensor canonical
        correlation analysis for action classification. CVPR.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.nonparametric import KTCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((20, 5))
        >>> X2 = rng.standard_normal((20, 5))
        >>> X3 = rng.standard_normal((20, 5))
        >>> model = KTCCA(random_state=0).fit([X1, X2, X3])
    """

    def __init__(
        self,
        n_components: int = 1,
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
        super().__init__(n_components=n_components, center=center)
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.eps = eps
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> KTCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
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
            self.n_components,
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
        """Whitened kernels and each view's inverse square-root matrix."""
        whitened = []
        cov_invsqrt = []
        for i, K in enumerate(kernels):
            cov = (1.0 - c[i]) * K @ K + c[i] * K
            invsqrt = psd_inverse_sqrt(cov, self.eps)
            whitened.append(K @ invsqrt)
            cov_invsqrt.append(invsqrt)
        return whitened, cov_invsqrt
