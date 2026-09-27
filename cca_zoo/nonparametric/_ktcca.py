"""Kernel tensor CCA."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._utils._param_constraints import RANDOM_STATE
from cca_zoo.linear import TCCA
from cca_zoo.nonparametric._kernel import _BaseKernelModel


class KTCCA(_BaseKernelModel):
    """Kernel tensor CCA.

    :class:`~cca_zoo.linear.TCCA` in each view's kernel feature space:
    PARAFAC of the cross-moment tensor of the whitened feature maps.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's covariance in the kernel feature
            space towards the identity, in ``[0, 1]``: 0 is kernel CCA and 1
            is kernel PLS. Per-view. Default is 0.1.
        kernel: Kernel name or callable for
            :func:`~sklearn.metrics.pairwise_kernels`. Per-view. Default is
            ``"linear"``.
        gamma: Kernel coefficient for RBF, polynomial and sigmoid kernels.
            Per-view. Default is None.
        degree: Polynomial kernel degree. Per-view. Default is 3.
        coef0: Polynomial and sigmoid kernel constant. Per-view. Default is 1.
        kernel_params: Extra kernel keyword arguments. Per-view. Default is
            None.
        random_state: Seed for PARAFAC. Default is None.

    Attributes:
        weights_: Dual coefficients of each view, shape (n_samples,
            n_components).
        train_views_: The centred training views, against which the kernel
            of a new view is evaluated and centred.

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

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **_BaseKernelModel._parameter_constraints,
        "random_state": RANDOM_STATE,
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
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.shrinkage = shrinkage
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
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
        linear = TCCA(
            self.n_components,
            center=False,
            shrinkage=self.shrinkage,
            random_state=self.random_state,
        )
        self._set_weights(linear.fit(self._feature_maps(views_)).weights_)
        return self._finish_fit(views_)
