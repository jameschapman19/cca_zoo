"""CCA by entrywise-sparse reduced rank regression."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import Lasso
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._param_constraints import POSITIVE_EPS, POSITIVE_INT
from cca_zoo.linear._rrr_common import _postprocess_rrr_fit, _whiten_response


def _entrywise_sparse_rrr(
    X: np.ndarray, Y: np.ndarray, alpha: float, max_iter: int, tol: float
) -> tuple[np.ndarray, int]:
    """Solve ``min_B ||Y - XB||^2 / n + alpha * sum |B|`` by one Lasso per column.

    sklearn's Lasso scales the loss by ``1 / (2n)``, so its ``alpha`` is half
    of this one. ``alpha=0`` is solved by least squares.

    Returns:
        ``B``, and the most iterations any column's Lasso ran (0 for least
        squares).
    """
    if alpha == 0.0:
        B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return np.asarray(B), 0
    q = Y.shape[1]
    B = np.zeros((X.shape[1], q))
    n_iter = 0
    for k in range(q):
        model = Lasso(
            alpha=alpha / 2.0, fit_intercept=False, max_iter=max_iter, tol=tol
        )
        model.fit(X, Y[:, k])
        B[:, k] = model.coef_
        n_iter = max(n_iter, model.n_iter_)
    return B, n_iter


class ECCA(BaseModel):
    r"""Two-view CCA by entrywise-sparse reduced rank regression.

    Regresses the whitened $\tilde{Y} = Y \Sigma_Y^{-1/2}$ on $X$ with an
    entrywise lasso penalty,

    $$
    \hat{B} = \underset{B}{\mathrm{argmin}}\ \frac{1}{n}
        \lVert \tilde{Y} - X B \rVert_F^2 + \alpha \sum_{j,k} \lvert B_{jk} \rvert,
    $$

    and takes the canonical directions from the rank-``n_components`` SVD of
    the fitted values $X \hat{B}$. The penalty zeroes entries of $\hat{B}$,
    where :class:`~cca_zoo.linear.CCAR3`'s zeroes whole rows; the canonical
    directions combine its columns, so a feature leaves them only when its
    whole row is zero. A port of ``ecca()`` from the R package ccar3.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        alpha: Entrywise lasso strength. Default is 0.
        max_iter: Maximum Lasso iterations. Default is 10000.
        tol: Lasso tolerance. Default is 1e-4.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Most iterations of any column's Lasso; 0 when ``alpha=0``
            is solved by least squares.

    References:
        Donnat, C., & Tuzhilina, E. (2024). Canonical Correlation Analysis
        as Reduced Rank Regression in High Dimensions. arXiv:2405.19539.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import ECCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = ECCA(n_components=2, alpha=0.1).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="left")],
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
    }

    _EPS: ClassVar[float] = 1e-8

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        alpha: float = 0.0,
        max_iter: int = 10_000,
        tol: float = 1e-4,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, views: list[ArrayLike], y: None = None) -> ECCA:
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
                f"ECCA requires exactly 2 views, got {self.n_views_}. "
                "Use MCCA for more than 2 views."
            )
        X, Y = views_

        Y_tilde, sqrt_inv_Sy = _whiten_response(Y, ledoit_wolf=False)
        B, self.n_iter_ = _entrywise_sparse_rrr(
            X, Y_tilde, alpha=self.alpha, max_iter=self.max_iter, tol=self.tol
        )
        U, V = _postprocess_rrr_fit(
            B, X, Y, sqrt_inv_Sy, self.n_components, ridge=self._EPS
        )
        self.weights_: list[np.ndarray] = [U, V]
        self._fit_maps_and_importances(views_)
        return self
