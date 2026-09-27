"""Kernel generalized CCA."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import pairwise_kernels

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._param_constraints import KERNEL_PARAMETERS
from cca_zoo._utils._validation import perview_parameter


class KGCCA(BaseModel):
    r"""Kernel generalized CCA.

    :class:`~cca_zoo.linear.GCCA` in the dual: $T$ holds the top
    eigenvectors of
    $\sum_i \mu_i K_i (c_i K_i + (1 - c_i) K_i^2)^{-1} K_i$ and
    $\alpha_i = K_i^+ T$.

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
        view_weights: Weight of each view; None weights them equally.
            Default is None.
        eps: Floor added to the within-view matrices. Default is 1e-6.

    Attributes:
        weights_: Dual coefficients of each view, shape (n_samples,
            n_components).

    References:
        Tenenhaus, A., Philippe, C., & Frouin, V. (2015). Kernel generalized
        canonical correlation analysis. Computational Statistics & Data
        Analysis, 90, 114-131.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.nonparametric import KGCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((30, 5))
        >>> X2 = rng.standard_normal((30, 5))
        >>> X3 = rng.standard_normal((30, 5))
        >>> model = KGCCA(n_components=2, kernel="rbf").fit([X1, X2, X3])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        **KERNEL_PARAMETERS,
        "view_weights": [None, "array-like"],
    }

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
        view_weights: list[float] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.view_weights = view_weights
        self.eps = eps

    def fit(self, views: list[ArrayLike], y: None = None) -> KGCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
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

        _, eigvecs = gevp(Q, None, self.n_components)
        T = eigvecs[:, : self.n_components]
        self.weights_: list[np.ndarray] = [np.linalg.pinv(K) @ T for K in kernels]
        # Store kernel parameters for transform
        self._kernel = kernel_
        self._gamma = gamma_
        self._degree = degree_
        self._coef0 = coef0_
        self._kp = kp_
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
