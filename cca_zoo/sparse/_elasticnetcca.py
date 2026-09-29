"""Sparse CCA by coordinate descent on the elastic-net-penalised EY loss."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._ey import coordinate_descent_ey, ey_cross_covariance
from cca_zoo._utils._param_constraints import RANDOM_STATE
from cca_zoo._utils._validation import perview_parameter


class ElasticNetCCA(BaseModel):
    r"""Sparse multiview CCA by coordinate descent on the elastic-net EY loss.

    Minimises, over  = X_i W_i$,

    $$
    \mathcal{L}_{EY}(Z_1, \dots, Z_M)
        + \sum_i \left( \alpha_i \rho_i \|W_i\|_1
        + \tfrac{1}{2} \alpha_i (1-\rho_i) \|W_i\|_F^2 \right),
    $$

    with $\rho$ = ``l1_ratio``, by cyclic coordinate descent as in
    :class:`~sklearn.linear_model.ElasticNet`; each coordinate's quartic
    restriction is minimised exactly. The loss is not jointly convex, so the
    result can depend on ``random_state``.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        alpha: Penalty strength. Per-view. Default is 1.0.
        l1_ratio: L1 share of the penalty in ``[0, 1]``. Per-view. Default
            is 0.5.
        max_iter: Maximum coordinate-descent sweeps at each stage of the
            penalty path. Default is 1000, as sklearn's ElasticNet.
        tol: Tolerance on the change in the objective. Default is 1e-6.
        random_state: Seed for the initial weights. Default is None.
        positive: Whether to constrain the weights to be non-negative.
            Default is False.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Coordinate-descent sweeps at the full penalty.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.sparse import ElasticNetCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 20))
        >>> X2 = rng.standard_normal((200, 15))
        >>> model = ElasticNetCCA(n_components=2, alpha=[0.1, 0.5]).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "l1_ratio": [Interval(Real, 0, 1, closed="both"), "array-like"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "positive": ["boolean"],
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        alpha: float | list[float] = 1.0,
        l1_ratio: float | list[float] = 0.5,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: int | None = None,
        positive: bool = False,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.positive = positive

    def fit(self, views: list[ArrayLike], y: None = None) -> ElasticNetCCA:
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
        weights, self.n_iter_, converged = coordinate_descent_ey(
            bases=views_,
            k=self.n_components,
            alpha=alpha_,
            l1_ratio=l1_ratio_,
            max_iter=self.max_iter,
            tol=self.tol,
            rng=rng,
            positive=self.positive,
        )
        warn_if_not_converged(self, converged)
        # The elementwise penalty rules out canonical_rotation, but reordering
        # the components by their reward changes neither loss nor penalty.
        reward, _ = ey_cross_covariance([v @ w for v, w in zip(views_, weights)])
        order = np.argsort(-np.diag(reward), kind="stable")
        self.weights_: list[np.ndarray] = [w[:, order] for w in weights]
        return self._finish_fit(views_)
