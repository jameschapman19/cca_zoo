"""Tensor CCA."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import StrOptions

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import cross_moment_tensor, psd_inverse_sqrt
from cca_zoo._utils._param_constraints import RANDOM_STATE, RIDGE_PARAMETER
from cca_zoo._utils._validation import perview_parameter


class TCCA(BaseModel):
    r"""Tensor CCA: higher-order correlation of three or more views.

    Decomposes the cross-moment tensor of the whitened views
    $\tilde X_i = X_i \Sigma_i^{-1/2}$,

    $$
    \mathcal{M} = \frac{1}{n} \sum_s \tilde x_{1s} \otimes \cdots \otimes \tilde x_{Ms},
    $$

    by PARAFAC; the factors give the canonical directions.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's covariance towards the identity,
            in ``[0, 1]``: 0 is CCA and 1 is PLS. Per-view. Default is 0.
        init: PARAFAC's initialisation: ``"svd"``, deterministic, or
            ``"random"``, seeded by ``random_state``; random starts show
            whether the SVD start found the best of the tensor's local optima.
            Default is ``"svd"``.
        random_state: Seed for ``init="random"``, as sklearn's NMF uses it.
            Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Luo, Y., Tao, D., Ramamohanarao, K., Xu, C., & Wen, Y. (2015). Tensor
        canonical correlation analysis for multi-view dimension reduction.
        IEEE Transactions on Knowledge and Data Engineering, 27(11), 3111-3124.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import TCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 5))
        >>> X2 = rng.standard_normal((50, 5))
        >>> X3 = rng.standard_normal((50, 5))
        >>> model = TCCA(n_components=2, random_state=0).fit([X1, X2, X3])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "shrinkage": RIDGE_PARAMETER,
        "init": [StrOptions({"svd", "random"})],
        "random_state": RANDOM_STATE,
    }

    _EPS: ClassVar[float] = 1e-6

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float | list[float] = 0.0,
        init: str = "svd",
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.shrinkage = shrinkage
        self.init = init
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> TCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        c_ = perview_parameter("shrinkage", self.shrinkage, 0.0, self.n_views_)
        whitened, cov_invsqrt = self._whiten_views(views_, c_)

        M = cross_moment_tensor(whitened)

        # Imported here: importing tensorly prints SyntaxWarnings on Python
        # 3.12, which every user of cca_zoo would otherwise see.
        import tensorly as tl
        from tensorly.decomposition import parafac

        with tl.backend_context("numpy"):
            parafac_result = parafac(
                M,
                self.n_components,
                init=self.init,
                random_state=self.random_state,
                verbose=False,
            )
        self.weights_: list[np.ndarray] = [
            cov_invsqrt[i] @ fac for i, fac in enumerate(parafac_result.factors)
        ]
        self._fit_maps_and_importances(views_)
        return self

    def _whiten_views(
        self,
        views: list[np.ndarray],
        c: list[float],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Whitened views and each view's inverse square-root covariance."""
        whitened = []
        cov_invsqrt = []
        for i, v in enumerate(views):
            cov = (1.0 - c[i]) * np.cov(v, rowvar=False) + c[i] * np.eye(v.shape[1])
            invsqrt = psd_inverse_sqrt(cov, self._EPS)
            whitened.append(v @ invsqrt)
            cov_invsqrt.append(invsqrt)
        return whitened, cov_invsqrt
