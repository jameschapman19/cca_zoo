"""Kernel CCA."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag
from sklearn.metrics import pairwise_kernels

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._param_constraints import KERNEL_PARAMETERS
from cca_zoo._utils._validation import perview_parameter


class KCCA(BaseModel):
    r"""Kernel CCA for two or more views.

    Solves MCCA's eigenproblem in the dual, $A \alpha = \lambda B \alpha$,
    with $A$ the between-view blocks $K_i K_j$ and
    $B = \operatorname{blockdiag}(c_i K_i + (1 - c_i) K_i^2)$.

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
        eps: Floor added to ``B``. Default is 1e-3.

    Attributes:
        weights_: Dual coefficients of each view, shape (n_samples,
            n_components).

    References:
        Hardoon, D. R., Szedmak, S., & Shawe-Taylor, J. (2004). Canonical
        correlation analysis: An overview with application to learning
        methods. Neural Computation, 16(12), 2639-2664.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.nonparametric import KCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((30, 5))
        >>> X2 = rng.standard_normal((30, 5))
        >>> model = KCCA(n_components=2, kernel="rbf", c=0.1).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        **KERNEL_PARAMETERS,
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
        eps: float = 1e-3,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.c = c
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.eps = eps

    def fit(self, views: list[ArrayLike], y: None = None) -> KCCA:
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
        kernels = self._compute_kernels(views_, kernel_, gamma_, degree_, coef0_, kp_)
        A = self._build_A(kernels)
        B = self._build_B(kernels, c_)
        splits = np.cumsum([k.shape[1] for k in kernels])
        _, eigvecs = gevp(A, B, self.n_components)
        self.weights_: list[np.ndarray] = list(np.split(eigvecs, splits[:-1], axis=0))
        # Store kernel parameters for transform
        self._kernel: list[str] = kernel_
        self._gamma: list[float | None] = gamma_
        self._degree: list[float] = degree_
        self._coef0: list[float] = coef0_
        self._kp: list[dict[str, object]] = kp_
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

    def _compute_kernels(
        self,
        views: list[np.ndarray],
        kernel: list[str],
        gamma: list[float | None],
        degree: list[float],
        coef0: list[float],
        kp: list[dict[str, object]],
    ) -> list[np.ndarray]:
        """Training kernel matrix of each view, shape (n_samples, n_samples)."""
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
        """Between-view kernel block matrix."""
        all_k = np.hstack(kernels)
        A = np.cov(all_k, rowvar=False)
        A -= block_diag(*[np.cov(k, rowvar=False) for k in kernels])
        return A / len(kernels)

    def _build_B(self, kernels: list[np.ndarray], c: list[float]) -> np.ndarray:
        """Block-diagonal regularised within-view kernel matrix."""
        blocks = [
            c[i] * kernels[i] + (1.0 - c[i]) * kernels[i] @ kernels[i]
            for i in range(len(kernels))
        ]
        B: np.ndarray = np.asarray(block_diag(*blocks))
        min_eig = np.linalg.eigvalsh(B).min()
        if min_eig < self.eps:
            B += (self.eps - min_eig) * np.eye(B.shape[0])
        return np.asarray(B / len(kernels))
