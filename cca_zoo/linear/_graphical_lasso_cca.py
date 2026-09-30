"""MCCA with graphical-lasso within-view covariances."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag
from sklearn.covariance import GraphicalLassoCV
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._param_constraints import (
    POSITIVE_EPS,
    POSITIVE_INT,
    RIDGE_PARAMETER,
)
from cca_zoo._utils._validation import perview_parameter
from cca_zoo.linear._mcca import MCCA


def graphical_lasso(
    correlation: np.ndarray, alpha: float, max_iter: int, tol: float
) -> tuple[np.ndarray, int, bool]:
    r"""The graphical lasso by ADMM, with the step adapted to balance residuals.

    Minimises $\operatorname{tr}(S \Theta) - \log\det \Theta + \alpha
    \lVert \Theta \rVert_1$ over the off-diagonal entries, sklearn's
    problem, splitting $\Theta = Z$ (Boyd et al., 2011, Sections 3.4.1 and
    6.5). Each step is exact, an eigendecomposition or a soft threshold, so
    it converges where coordinate descent does not.

    Args:
        correlation: Correlation matrix, shape (p, p).
        alpha: Penalty on the off-diagonal entries.
        max_iter: Maximum iterations.
        tol: Tolerance on the primal and dual residuals, per entry.

    Returns:
        ``(precision, n_iter, converged)``, the precision sparse.
    """
    p = len(correlation)
    off_diagonal = ~np.eye(p, dtype=bool)
    rho, z, u = 1.0, np.eye(p), np.zeros((p, p))
    for n_iter in range(1, max_iter + 1):
        eigenvalues, vectors = np.linalg.eigh(rho * (z - u) - correlation)
        roots = (eigenvalues + np.sqrt(eigenvalues**2 + 4 * rho)) / (2 * rho)
        theta = (vectors * roots) @ vectors.T
        previous = z
        z = theta + u
        z[off_diagonal] = np.sign(z[off_diagonal]) * np.maximum(
            np.abs(z[off_diagonal]) - alpha / rho, 0.0
        )
        u = u + theta - z
        primal = np.linalg.norm(theta - z)
        dual = rho * np.linalg.norm(z - previous)
        if max(primal, dual) < tol * p:
            return z, n_iter, True
        if primal > 10 * dual:
            rho, u = 2 * rho, u / 2
        elif dual > 10 * primal:
            rho, u = rho / 2, 2 * u
    return z, max_iter, False


class GraphicalLassoCCA(MCCA):
    """MCCA with each within-view covariance estimated by the graphical lasso.

    Replaces each view's block of :class:`~cca_zoo.linear.MCCA`'s $B$ with
    the covariance implied by a sparse precision estimate, an L1 penalty on
    partial correlations rather than shrinkage of the covariance. The lasso
    is fitted to each view's correlation matrix, so ``alpha`` is the same in
    any units, and solved by :func:`graphical_lasso`, ADMM: sklearn's
    coordinate descent fails to converge, or to run, on strongly correlated
    features. Solved in the original feature space.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's estimated covariance towards the
            identity, in ``[0, 1]``, as in MCCA.
            Default is 0.
        alpha: Graphical-lasso penalty on the correlation scale; None selects
            it by :class:`~sklearn.covariance.GraphicalLassoCV`. Per-view.
            With None, the refit at the selected penalty is by ADMM too.
            Default is 0.01.
        max_iter: Maximum ADMM iterations per view. Default is 1000.
        tol: Tolerance on ADMM's residuals, per entry. Default is 1e-8.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        covariance_: Estimated covariance of each view.
        precision_: Estimated sparse precision of each view.
        n_iter_: Most graphical-lasso iterations of any view.

    References:
        Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse
        covariance estimation with the graphical lasso. Biostatistics,
        9(3), 432-441.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import GraphicalLassoCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 12))
        >>> X2 = rng.standard_normal((100, 9))
        >>> model = GraphicalLassoCCA(alpha=0.1).fit([X1, X2])
        >>> model.precision_[0].shape
        (12, 12)
    """

    _supports_array_api: ClassVar[bool] = False
    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "shrinkage": RIDGE_PARAMETER,
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like", None],
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float | list[float] = 0.0,
        alpha: float | list[float | None] | None = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            shrinkage=shrinkage,
            pca=False,
        )
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        sample_weight: ArrayLike | None = None,
    ) -> GraphicalLassoCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.
            sample_weight: Weight of each sample; an integer weight is the
                same as repeating the sample. None weights samples equally.

        Returns:
            self.

        Raises:
            ValueError: If ``sample_weight`` is given with ``alpha=None``:
                cross-validating the penalty splits the rows themselves.
        """
        alpha_ = perview_parameter("alpha", self.alpha, None, len(views))
        if sample_weight is not None and None in alpha_:
            raise ValueError(
                "alpha=None selects the penalty by cross-validating over the "
                "rows, which cannot use sample_weight; pass alpha."
            )
        return cast(GraphicalLassoCCA, super().fit(views, y, sample_weight))

    def _build_B(self, views: list[np.ndarray], c: list[float]) -> np.ndarray:
        """Block-diagonal ``B`` from each view's graphical-lasso covariance."""
        alpha_ = perview_parameter("alpha", self.alpha, None, len(views))
        covariances = []
        precisions = []
        self.n_iter_: int = 0
        converged = True
        for v, a in zip(views, alpha_):
            # The views are centred when center=True, so these are their
            # correlations and scales, weighted when fit had sample_weight.
            scale = np.sqrt(np.sum(v**2, axis=0) / (len(v) - 1))
            standardised = v / scale
            correlation = standardised.T @ standardised / (len(v) - 1)
            penalty = GraphicalLassoCV().fit(standardised).alpha_ if a is None else a
            # The lasso penalises partial correlations: one feature has none.
            precision, n_iter, done = (
                (np.ones((1, 1)), 0, True)
                if v.shape[1] == 1
                else graphical_lasso(correlation, penalty, self.max_iter, self.tol)
            )
            covariances.append(np.linalg.inv(precision) * np.outer(scale, scale))
            precisions.append(precision / np.outer(scale, scale))
            self.n_iter_ = max(self.n_iter_, n_iter)
            converged = converged and done
        warn_if_not_converged(self, converged)
        self.covariance_: list[np.ndarray] = covariances
        self.precision_: list[np.ndarray] = precisions

        blocks = [
            (1.0 - c[i]) * cov + c[i] * np.eye(cov.shape[0])
            for i, cov in enumerate(covariances)
        ]
        B: np.ndarray = np.asarray(block_diag(*blocks))
        min_eig = np.linalg.eigvalsh(B).min()
        if min_eig < self._EPS:
            B += (self._EPS - min_eig) * np.eye(B.shape[0])
        return np.asarray(B / len(views))
