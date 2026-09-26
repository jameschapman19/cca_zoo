"""MCCA with graphical-lasso within-view covariances."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from scipy.linalg import block_diag
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.utils._param_validation import Interval, StrOptions

from cca_zoo._base import BaseModel
from cca_zoo._utils._param_constraints import (
    POSITIVE_EPS,
    POSITIVE_INT,
    RIDGE_PARAMETER,
)
from cca_zoo._utils._validation import perview_parameter
from cca_zoo.linear._mcca import MCCA


class GraphicalLassoCCA(MCCA):
    """MCCA with each within-view covariance estimated by the graphical lasso.

    Replaces each view's block of :class:`~cca_zoo.linear.MCCA`'s $B$ with
    the covariance implied by :class:`~sklearn.covariance.GraphicalLasso`'s
    sparse precision estimate, an L1 penalty on partial correlations rather
    than shrinkage of the covariance. Solved in the original feature space.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        c: Ridge blend applied to the estimated covariance, as in MCCA.
            Default is 0.
        alpha: Graphical-lasso penalty; None selects it by
            :class:`~sklearn.covariance.GraphicalLassoCV`. Per-view. Default
            is 0.01.
        mode: Graphical-lasso solver, ``"cd"`` or ``"lars"``. Default is
            ``"cd"``.
        max_iter: Maximum graphical-lasso iterations. Default is 100.
        eps: Floor added to the eigenvalues of ``B``. Default is 1e-6.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        covariance_: Estimated covariance of each view.
        precision_: Estimated sparse precision of each view.

    References:
        Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse
        covariance estimation with the graphical lasso. Biostatistics,
        9(3), 432-441.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.linear import GraphicalLassoCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 12))
        >>> X2 = rng.standard_normal((100, 9))
        >>> model = GraphicalLassoCCA(alpha=0.1).fit([X1, X2])
        >>> model.precision_[0].shape
        (12, 12)
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "c": RIDGE_PARAMETER,
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like", None],
        "mode": [StrOptions({"cd", "lars"})],
        "max_iter": POSITIVE_INT,
        "eps": POSITIVE_EPS,
    }

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        c: float | list[float] = 0.0,
        alpha: float | list[float | None] | None = 0.01,
        mode: str = "cd",
        max_iter: int = 100,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            c=c,
            pca=False,
            eps=eps,
        )
        self.alpha = alpha
        self.mode = mode
        self.max_iter = max_iter

    def _build_B(self, views: list[np.ndarray], c: list[float]) -> np.ndarray:
        """Block-diagonal ``B`` from each view's graphical-lasso covariance."""
        alpha_ = perview_parameter("alpha", self.alpha, None, len(views))
        covariances = []
        precisions = []
        for v, a in zip(views, alpha_):
            if a is None:
                estimator = GraphicalLassoCV(
                    mode=self.mode, max_iter=self.max_iter
                ).fit(v)
            else:
                estimator = GraphicalLasso(
                    alpha=a,
                    mode=self.mode,
                    max_iter=self.max_iter,
                    assume_centered=self.center,
                ).fit(v)
            covariances.append(estimator.covariance_)
            precisions.append(estimator.precision_)
        self.covariance_: list[np.ndarray] = covariances
        self.precision_: list[np.ndarray] = precisions

        blocks = [
            (1.0 - c[i]) * cov + c[i] * np.eye(cov.shape[0])
            for i, cov in enumerate(covariances)
        ]
        B: np.ndarray = np.asarray(block_diag(*blocks))
        min_eig = np.linalg.eigvalsh(B).min()
        if min_eig < self.eps:
            B += (self.eps - min_eig) * np.eye(B.shape[0])
        return np.asarray(B / len(views))
