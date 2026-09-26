"""CCA by row-sparse reduced rank regression."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import MultiTaskLasso
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._param_constraints import POSITIVE_EPS, POSITIVE_INT
from cca_zoo.linear._rrr_common import (
    _postprocess_rrr_fit,
    _whiten_response,
)


def _row_sparse_rrr(
    X: np.ndarray, Y_tilde: np.ndarray, alpha: float, max_iter: int, tol: float
) -> np.ndarray:
    """Solve ``min_B ||Y - XB||^2 / n + alpha * sum_j ||B[j]||`` by MultiTaskLasso.

    sklearn scales the loss by ``1 / (2n)``, so its ``alpha`` is half of this
    one.
    """
    model = MultiTaskLasso(
        alpha=alpha / 2.0, fit_intercept=False, max_iter=max_iter, tol=tol
    )
    model.fit(X, Y_tilde)
    return np.asarray(model.coef_.T)


class CCAR3(BaseModel):
    r"""Two-view CCA by row-sparse reduced rank regression.

    Whitens $Y$ by its (optionally Ledoit-Wolf) covariance, regresses
    $\tilde{Y} = Y \Sigma_Y^{-1/2}$ on $X$, and takes the canonical
    directions from the rank-``n_components`` SVD of the coefficients. With
    ``highdim=True`` the regression has a row-group lasso penalty,

    $$
    \hat{B} = \underset{B}{\mathrm{argmin}}\ \frac{1}{n}
        \lVert \tilde{Y} - X B \rVert_F^2
        + \alpha \sum_j \lVert B_{j, :} \rVert_2,
    $$

    which drops whole features of $X$; swap the views to make the other one
    sparse. A port of the R package ccar3.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        alpha: Row-group lasso strength when ``highdim=True``. Default is 0.
        highdim: Whether to use the penalised regression rather than the
            closed-form least-squares one. Default is True.
        ledoit_wolf: Whether to shrink the covariance of ``Y``. Default is
            True.
        max_iter: Maximum MultiTaskLasso iterations. Default is 10000.
        tol: MultiTaskLasso tolerance. Default is 1e-4.
        eps: Ridge added to covariances before inversion. Default is 1e-8.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Donnat, C., & Tuzhilina, E. (2024). Canonical Correlation Analysis
        as Reduced Rank Regression in High Dimensions. arXiv:2405.19539.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.linear import CCAR3
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = CCAR3(n_components=2, alpha=0.1).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="left")],
        "highdim": ["boolean"],
        "ledoit_wolf": ["boolean"],
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
        "eps": POSITIVE_EPS,
    }

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        alpha: float = 0.0,
        highdim: bool = True,
        ledoit_wolf: bool = True,
        max_iter: int = 10_000,
        tol: float = 1e-4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.alpha = alpha
        self.highdim = highdim
        self.ledoit_wolf = ledoit_wolf
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

    def fit(self, views: list[ArrayLike], y: None = None) -> CCAR3:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.

        Raises:
            ValueError: If there are not exactly two views.
        """
        views_ = self._setup_fit(views)
        if self.n_views_ != 2:
            raise ValueError(
                f"CCAR3 requires exactly 2 views, got {self.n_views_}. "
                "Use MCCA for more than 2 views."
            )
        X, Y = views_
        n = X.shape[0]

        Y_tilde, sqrt_inv_Sy = _whiten_response(Y, self.ledoit_wolf)

        if self.highdim:
            B = _row_sparse_rrr(
                X, Y_tilde, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol
            )
        else:
            Sx = X.T @ X / n + self.eps * np.eye(X.shape[1])
            B = np.linalg.solve(Sx, X.T @ Y_tilde / n)

        U, V = _postprocess_rrr_fit(
            B, X, Y, sqrt_inv_Sy, self.n_components, ridge=self.eps
        )
        self.weights_: list[np.ndarray] = [U, V]
        return self
