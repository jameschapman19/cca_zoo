"""CCA by row-sparse reduced rank regression."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import MultiTaskLasso
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._param_constraints import POSITIVE_EPS, POSITIVE_INT


class CCAR3(BaseModel):
    r"""Two-view CCA by row-sparse reduced rank regression.

    Whitens $Y$ by its (optionally Ledoit-Wolf) covariance, regresses
    $\tilde{Y} = Y \Sigma_Y^{-1/2}$ on $X$, and takes the canonical
    directions from the rank-``n_components`` SVD of the fitted values. The
    regression has a row-group lasso penalty,

    $$
    \hat{B} = \underset{B}{\mathrm{argmin}}\ \frac{1}{n}
        \lVert \tilde{Y} - X B \rVert_F^2
        + \alpha \sum_j \lVert B_{j, :} \rVert_2,
    $$

    which drops whole features of $X$; swap the views to make the other one
    sparse. ``alpha=0`` is solved by least squares. A port of the R package
    ccar3.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        alpha: Row-group lasso strength. Default is 0.
        ledoit_wolf: Whether to shrink the covariance of ``Y``. Default is
            True.
        max_iter: Maximum MultiTaskLasso iterations. Default is 10000.
        tol: MultiTaskLasso tolerance. Default is 1e-4.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        n_iter_: Iterations of the MultiTaskLasso; 0 when ``alpha=0`` is
            solved by least squares.

    References:
        Donnat, C., & Tuzhilina, E. (2024). Canonical Correlation Analysis
        as Reduced Rank Regression in High Dimensions. arXiv:2405.19539.

    Examples:
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
        "ledoit_wolf": ["boolean"],
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        alpha: float = 0.0,
        ledoit_wolf: bool = True,
        max_iter: int = 10_000,
        tol: float = 1e-4,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.alpha = alpha
        self.ledoit_wolf = ledoit_wolf
        self.max_iter = max_iter
        self.tol = tol

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
                f"{type(self).__name__} requires exactly 2 views, got "
                f"{self.n_views_}. Use MCCA for more than 2 views."
            )
        X, Y = views_
        n = len(X)
        # Regress the whitened response on X.
        cov_y = LedoitWolf().fit(Y).covariance_ if self.ledoit_wolf else Y.T @ Y / n
        whitener = _inverse_sqrt(cov_y)
        if self.alpha == 0.0:
            B = np.linalg.lstsq(X, Y @ whitener, rcond=None)[0]
            self.n_iter_: int = 0
        else:
            B, self.n_iter_ = self._regression(X, Y @ whitener)
        # The canonical directions are the right singular vectors of the
        # fitted values X B, not of B, whose singular vectors ignore the
        # covariance of X.
        k = min(self.n_components, *B.shape)
        q = np.linalg.svd(X @ B, full_matrices=False)[2][:k].T
        u, v = B @ q, whitener @ q
        # Whitening each side's scores leaves the pairs mixed; the SVD of the
        # whitened cross-covariance gives the canonical pairs, in order.
        u = u @ _whitener_of_scores(X @ u)
        v = v @ _whitener_of_scores(Y @ v)
        left, _, right_t = np.linalg.svd((X @ u).T @ (Y @ v) / n)
        self.weights_: list[np.ndarray] = [
            np.zeros((X.shape[1], self.n_components)),
            np.zeros((Y.shape[1], self.n_components)),
        ]
        if np.any(B):
            self.weights_[0][:, :k] = u @ left
            self.weights_[1][:, :k] = v @ right_t.T
        self._fit_maps_and_importances(views_)
        return self

    def _regression(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, int]:
        """``B`` minimising ``||Y - X B||^2 / n + alpha sum_j ||B_j||``.

        By sklearn's MultiTaskLasso, whose loss is scaled by ``1 / (2n)``, so
        its ``alpha`` is half of this one.

        Returns:
            ``B``, and the iterations run.
        """
        model = MultiTaskLasso(
            alpha=self.alpha / 2.0,
            fit_intercept=False,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        model.fit(X, Y)
        return np.asarray(model.coef_.T), int(model.n_iter_)


def _inverse_sqrt(S: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Symmetric inverse square root of a PSD matrix.

    Eigenvalues below ``threshold`` times the largest are zeroed, so the
    result does not depend on the units of the data.
    """
    vals, vecs = np.linalg.eigh(S)
    keep = vals > threshold * vals.max()
    return np.asarray(
        (vecs * np.where(keep, 1.0 / np.sqrt(np.abs(vals)), 0.0)) @ vecs.T
    )


def _whitener_of_scores(scores: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """``W`` giving ``scores @ W`` unit-variance, uncorrelated columns."""
    gram = scores.T @ scores / (len(scores) - 1) + ridge * np.eye(scores.shape[1])
    vals, vecs = np.linalg.eigh(gram)
    return np.asarray((vecs / np.sqrt(np.maximum(vals, ridge))) @ vecs.T)
