"""CCA by entrywise-sparse reduced rank regression."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso

from cca_zoo.linear._ccar3 import CCAR3


class ECCA(CCAR3):
    r"""Two-view CCA by entrywise-sparse reduced rank regression.

    :class:`~cca_zoo.linear.CCAR3` with an entrywise lasso penalty on the
    regression of the whitened $\tilde{Y} = Y \Sigma_Y^{-1/2}$ on $X$,

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
        ledoit_wolf: Whether to shrink the covariance of ``Y``. Default is
            False, as the R package's ``ecca()``.
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

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        alpha: float = 0.0,
        ledoit_wolf: bool = False,
        max_iter: int = 10_000,
        tol: float = 1e-4,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            alpha=alpha,
            ledoit_wolf=ledoit_wolf,
            max_iter=max_iter,
            tol=tol,
        )

    def _regression(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, int]:
        """``B`` minimising ``||Y - X B||^2 / n + alpha sum_jk |B_jk|``.

        One sklearn Lasso per column of ``Y``, whose loss is scaled by
        ``1 / (2n)``, so its ``alpha`` is half of this one.

        Returns:
            ``B``, and the most iterations any column's Lasso ran.
        """
        B = np.zeros((X.shape[1], Y.shape[1]))
        n_iter = 0
        for k, column in enumerate(Y.T):
            model = Lasso(
                alpha=self.alpha / 2.0,
                fit_intercept=False,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            model.fit(X, column)
            B[:, k] = model.coef_
            n_iter = max(n_iter, int(model.n_iter_))
        return B, n_iter
