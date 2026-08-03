"""TCCA — Tensor Canonical Correlation Analysis."""

from __future__ import annotations

from numbers import Integral
from typing import Any, ClassVar

import numpy as np
import tensorly as tl
from numpy.typing import ArrayLike
from scipy.linalg import sqrtm
from sklearn.utils._param_validation import Interval
from tensorly.decomposition import parafac

from cca_zoo._base import BaseModel
from cca_zoo._utils._param_constraints import POSITIVE_EPS, RIDGE_PARAMETER
from cca_zoo._utils._validation import perview_parameter


class TCCA(BaseModel):
    r"""Tensor Canonical Correlation Analysis.

    Extends CCA to more than two views by exploiting higher-order
    cross-view correlations via a tensor product structure.  The method
    constructs the order-M cross-moment tensor:

    $$
    \mathcal{M}_{p_1 p_2 \ldots p_M}
        = \frac{1}{n} \sum_{i=1}^n
            \tilde{x}_{1,i}^{(p_1)}
            \tilde{x}_{2,i}^{(p_2)}
            \cdots
            \tilde{x}_{M,i}^{(p_M)}
    $$

    where $\tilde{X}_j = X_j \Sigma_j^{-1/2}$ are the whitened views,
    and then decomposes $\mathcal{M}$ using PARAFAC to recover the
    canonical directions.

    References:
        Kim, T.-K., Wong, S.-F., & Cipolla, R. (2007). Tensor canonical
        correlation analysis for action classification. *CVPR 2007*. IEEE.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.
        c: Ridge regularisation in ``[0, 1]``.  Default is 0.
        eps: Regularisation floor for within-view covariance matrices.
        random_state: Seed for reproducibility (passed to PARAFAC).

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 5))
        >>> X2 = rng.standard_normal((50, 5))
        >>> X3 = rng.standard_normal((50, 5))
        >>> model = TCCA(latent_dimensions=2, random_state=0).fit([X1, X2, X3])
        >>> scores = model.transform([X1, X2, X3])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "c": RIDGE_PARAMETER,
        "eps": POSITIVE_EPS,
        "random_state": [None, Interval(Integral, 0, None, closed="left")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        c: float | list[float] = 0.0,
        eps: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.c = c
        self.eps = eps
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> TCCA:
        """Fit the TCCA model.

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
        whitened, cov_invsqrt = self._whiten_views(views_, c_)

        # Build cross-moment tensor via sequential outer products
        M: np.ndarray | None = None
        for i, wv in enumerate(whitened):
            if M is None:
                M = wv
            else:
                for _ in range(len(M.shape) - 1):
                    wv = np.expand_dims(wv, 1)
                M = np.expand_dims(M, -1) @ wv
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
            cov_invsqrt[i] @ fac for i, fac in enumerate(parafac_result.factors)
        ]
        return self

    def _whiten_views(
        self,
        views: list[np.ndarray],
        c: list[float],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Whiten each view using its regularised covariance.

        Args:
            views: Centred view arrays.
            c: Per-view regularisation parameters.

        Returns:
            Tuple of (whitened_views, inverse_sqrt_covariances).
        """
        whitened = []
        cov_invsqrt = []
        for i, v in enumerate(views):
            cov = (1.0 - c[i]) * np.cov(v, rowvar=False) + c[i] * np.eye(v.shape[1])
            min_eig = np.linalg.eigvalsh(cov).min()
            if min_eig < self.eps:
                cov += (self.eps - min_eig) * np.eye(cov.shape[0])
            invsqrt = np.linalg.inv(sqrtm(cov).real)
            whitened.append(v @ invsqrt)
            cov_invsqrt.append(invsqrt)
        return whitened, cov_invsqrt
