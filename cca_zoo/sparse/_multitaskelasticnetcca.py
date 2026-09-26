"""Row-sparse CCA by coordinate descent on the EY loss."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import group_coordinate_descent_ey
from cca_zoo._utils._validation import perview_parameter


class MultiTaskElasticNetCCA(BaseModel):
    r"""Row-sparse multiview CCA with a multi-task elastic-net penalty on the EY loss.

    As :class:`~cca_zoo.sparse.ElasticNetCCA` with
    :class:`~sklearn.linear_model.MultiTaskElasticNet`'s penalty,

    $$
    \mathcal{L}_{EY}(Z_1, \dots, Z_M)
        + \sum_i \left( \alpha_i \rho_i \|W_i\|_{2,1}
        + \tfrac{1}{2} \alpha_i (1-\rho_i) \|W_i\|_F^2 \right),
    $$

    so each feature is used by every component or by none. Each row is
    updated by a proximal-gradient step with backtracking.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        alpha: Penalty strength. Per-view. Default is 1.0.
        l1_ratio: Row-group share of the penalty in ``[0, 1]``. Per-view.
            Default is 0.5.
        max_iter: Maximum coordinate-descent sweeps. Default is 100.
        tol: Tolerance on the change in the objective. Default is 1e-6.
        random_state: Seed for the initial weights. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.sparse import MultiTaskElasticNetCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 20))
        >>> X2 = rng.standard_normal((200, 15))
        >>> model = MultiTaskElasticNetCCA(n_components=2, alpha=0.1).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "l1_ratio": [Interval(Real, 0, 1, closed="both"), "array-like"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        alpha: float | list[float] = 1.0,
        l1_ratio: float | list[float] = 0.5,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> MultiTaskElasticNetCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        alpha_ = perview_parameter("alpha", self.alpha, 1.0, self.n_views_)
        l1_ratio_ = perview_parameter("l1_ratio", self.l1_ratio, 0.5, self.n_views_)
        rng = np.random.default_rng(self.random_state)
        weights, _ = group_coordinate_descent_ey(
            bases=views_,
            k=self.n_components,
            alpha=alpha_,
            l1_ratio=l1_ratio_,
            max_iter=self.max_iter,
            tol=self.tol,
            rng=rng,
        )
        self.weights_: list[np.ndarray] = weights
        return self
