"""Kernel CCA."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo.linear import MCCA
from cca_zoo.nonparametric._kernel import _BaseKernelModel


class KCCA(_BaseKernelModel):
    r"""Kernel CCA for two or more views.

    :class:`~cca_zoo.linear.MCCA` in each view's kernel feature space. With
    centred kernels $K_i$ the dual problem is $A \alpha = \lambda B \alpha$,
    with between-view blocks $K_i K_j / (n - 1)$ and within-view blocks
    $(1 - c_i) K_i^2 / (n - 1) + c_i K_i$, the covariance and the squared
    norm of the feature-space direction.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's covariance in the kernel feature
            space towards the identity, in ``[0, 1]``: 0 is kernel CCA and 1
            is kernel PLS. Per-view. Default is 0.1: an RBF kernel's feature
            space has a dimension per sample, where unshrunk kernel CCA
            correlates any two views perfectly.
        kernel: Kernel name or callable for
            :func:`~sklearn.metrics.pairwise_kernels`. Per-view. Default is
            ``"linear"``.
        gamma: Kernel coefficient for RBF, polynomial and sigmoid kernels.
            Per-view. Default is None.
        degree: Polynomial kernel degree. Per-view. Default is 3.
        coef0: Polynomial and sigmoid kernel constant. Per-view. Default is 1.
        kernel_params: Extra kernel keyword arguments. Per-view. Default is
            None.

    Attributes:
        weights_: Dual coefficients of each view, shape (n_samples,
            n_components).
        views_fit_: The centred training views, against which the kernel
            of a new view is evaluated and centred.

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
        >>> model = KCCA(n_components=2, kernel="rbf", shrinkage=0.1).fit([X1, X2])
    """

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float | list[float] = 0.1,
        kernel: str | list[str] = "linear",
        gamma: float | list[float | None] | None = None,
        degree: float | list[float] = 3,
        coef0: float | list[float] = 1.0,
        kernel_params: dict[str, object] | list[dict[str, object]] | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.shrinkage = shrinkage
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def fit(self, views: list[ArrayLike], y: None = None) -> KCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        linear = MCCA(self.n_components, center=False, shrinkage=self.shrinkage)
        features, projections = self._feature_maps(views_)
        self._set_weights(projections, linear.fit(features).weights_)
        self._fit_maps_and_importances(views_)
        return self
