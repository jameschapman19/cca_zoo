"""Kernel generalized CCA."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo.linear import GCCA
from cca_zoo.nonparametric._kernel import _BaseKernelModel


class KGCCA(_BaseKernelModel):
    r"""Kernel generalized CCA.

    :class:`~cca_zoo.linear.GCCA` in each view's kernel feature space. With
    centred kernels $K_i$, $T$ holds the top eigenvectors of
    $\sum_i \mu_i K_i ((1 - c_i) K_i^2 / (n - 1) + c_i K_i)^{-1} K_i$.
    This is Carroll's MAXVAR criterion in kernel form, one of the methods
    Tenenhaus, Philippe and Frouin's kernel generalized CCA covers.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's covariance in the kernel feature
            space towards the identity, in ``[0, 1]``: 0 is kernel CCA and 1
            is kernel PLS. Per-view. Default is 0.1, as for
            :class:`~cca_zoo.nonparametric.KCCA`.
        kernel: Kernel name or callable for
            :func:`~sklearn.metrics.pairwise_kernels`. Per-view. Default is
            ``"linear"``.
        gamma: Kernel coefficient for RBF, polynomial and sigmoid kernels.
            Per-view. Default is None.
        degree: Polynomial kernel degree. Per-view. Default is 3.
        coef0: Polynomial and sigmoid kernel constant. Per-view. Default is 1.
        kernel_params: Extra kernel keyword arguments. Per-view. Default is
            None.
        view_weights: Weight of each view; None weights them equally.
            Default is None.

    Attributes:
        weights_: Dual coefficients of each view, shape (n_samples,
            n_components).
        views_fit_: The centred training views, against which the kernel
            of a new view is evaluated and centred.

    References:
        Tenenhaus, A., Philippe, C., & Frouin, V. (2015). Kernel generalized
        canonical correlation analysis. Computational Statistics & Data
        Analysis, 90, 114-131.
        Carroll, J. D. (1968). Generalization of canonical correlation analysis
        to three or more sets of variables. Proceedings of the 76th Annual
        Convention of the American Psychological Association, 3, 227-228.

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
        **_BaseKernelModel._parameter_constraints,
        "view_weights": [None, "array-like"],
    }

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
        view_weights: list[float] | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.shrinkage = shrinkage
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.view_weights = view_weights

    def fit(self, views: list[ArrayLike], y: None = None) -> KGCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        linear = GCCA(
            self.n_components,
            center=False,
            shrinkage=self.shrinkage,
            view_weights=self.view_weights,
        )
        features, projections = self._feature_maps(views_)
        self._set_weights(projections, linear.fit(features).weights_)
        self._fit_maps_and_importances(views_)
        return self
