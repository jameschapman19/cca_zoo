"""Sparse CCA by iterative penalised least squares (Mai and Zhang, 2019)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._param_constraints import (
    NONNEGATIVE_PER_VIEW,
    POSITIVE_EPS,
    POSITIVE_INT,
    RANDOM_STATE,
    RIDGE_PARAMETER,
)
from cca_zoo._utils._validation import perview_parameter
from cca_zoo.sparse._deflation import (
    Deflation,
    elastic_net,
    others_score,
    pls_direction,
)


class IPLSCCA(BaseModel):
    r"""Sparse CCA by iterative penalised least squares.

    Each view's weights solve an elastic-net regression onto the other views'
    summed score, then are rescaled to a unit-variance score, as in
    Waaijenborg et al.'s elastic-net CCA and Mai and Zhang's lasso version:

    $$
    \hat{\mathbf{w}}_i = \arg\min_{\mathbf{w}}
        \frac{1}{2n} \|X_i \mathbf{w} - \bar{\mathbf{s}}_{\neg i}\|_2^2
        + \alpha_i \Bigl(r \|\mathbf{w}\|_1 + \tfrac{1-r}{2} \|\mathbf{w}\|_2^2\Bigr).
    $$

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        alpha: Penalty strength. Per-view. Default is 0.
        l1_ratio: Share of the penalty that is L1: 1 is Mai and Zhang's lasso,
            and Waaijenborg et al. use an elastic net. Per-view. Default is 1.
        max_iter: Maximum iterations per latent dimension. Default is 500.
        tol: Convergence tolerance on the change in weights. Default is 1e-6.
        random_state: Seed for the start and the lasso's coordinate order.
            Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Most iterations run by any component.

    References:
        Waaijenborg, S., de Witt Hamer, P. C. V., & Zwinderman, A. H. (2008).
        Quantifying the association between gene expressions and DNA-markers by
        penalized canonical correlation analysis. Statistical Applications in
        Genetics and Molecular Biology, 7(1).

        Mai, Q., & Zhang, X. (2019). An iterative penalized least squares
        approach to sparse canonical correlation analysis. Biometrics, 75(3),
        734-744.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.sparse import IPLSCCA
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 10)), rng.standard_normal((50, 8))
        >>> model = IPLSCCA(alpha=0.1, random_state=0).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": NONNEGATIVE_PER_VIEW,
        "l1_ratio": RIDGE_PARAMETER,
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        alpha: float | list[float] = 0.0,
        l1_ratio: float | list[float] = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> IPLSCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        regressions = [
            elastic_net(a, r, self.tol, self.random_state)
            for a, r in zip(
                perview_parameter("alpha", self.alpha, 0.0, self.n_views_),
                perview_parameter("l1_ratio", self.l1_ratio, 1.0, self.n_views_),
            )
        ]
        self.n_iter_: int = 0
        converged = True
        deflation = Deflation(views_, self.n_components)
        for deflated in deflation:
            w = pls_direction(deflated, rng)
            for n_iter in range(1, self.max_iter + 1):
                previous = [wi.copy() for wi in w]
                # Regress each view on the others' score, rescaled to unit variance.
                for i, (view, regression) in enumerate(zip(deflated, regressions)):
                    regression.fit(view, others_score(deflated, w, i))
                    w[i] = regression.coef_ / max(
                        (view @ regression.coef_).std(), 1e-12
                    )
                if max(np.linalg.norm(a - b) for a, b in zip(w, previous)) < self.tol:
                    break
            else:
                converged = False
            self.n_iter_ = max(self.n_iter_, n_iter)
            deflation.record(w)
        warn_if_not_converged(self, converged)
        self.weights_: list[np.ndarray] = deflation.weights()
        self._fit_maps_and_importances(views_)
        return self
