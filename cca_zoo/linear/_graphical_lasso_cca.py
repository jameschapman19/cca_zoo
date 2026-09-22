r"""GraphicalLassoCCA — MCCA with a sparse-precision within-view covariance."""

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
    r"""GraphicalLassoCCA -- MCCA with a sparse-precision within-view covariance.

    Every within-view regularisation already in this library changes how
    the *covariance* :math:`X_i^\top X_i` is estimated before it goes into
    :class:`~cca_zoo.linear.MCCA`'s generalised eigenproblem -- a fixed
    ridge blend toward the identity (``MCCA``'s own ``c``), Ledoit-Wolf
    shrinkage (:class:`~cca_zoo.linear.CCAR3`'s ``ledoit_wolf``), or
    concentration-step trimming (:class:`~cca_zoo.linear.TrimmedCCA`).
    None of them touch the *inverse* covariance directly.
    ``GraphicalLassoCCA`` does: each view's block of
    :class:`~cca_zoo.linear.MCCA`'s within-view matrix :math:`B` is built
    from :class:`sklearn.covariance.GraphicalLasso`'s (or, with
    ``alpha=None``, :class:`sklearn.covariance.GraphicalLassoCV`'s)
    L1-penalised precision estimate's implied covariance, in place of the
    raw sample covariance -- an L1 penalty on each view's *partial*
    correlations (conditional independence structure) rather than an L2
    shrinkage of the covariance itself. The between-view matrix :math:`A`
    is untouched (plain sample cross-covariance, as in ``MCCA``), so only
    the "how confidently does this view's own covariance matrix invert"
    side of the eigenproblem changes.

    Since the point of estimating a sparse precision matrix is usually the
    sparse structure itself, not a dimensionality-reduced approximation of
    it, this always solves the eigenproblem directly in each view's
    original feature space (:class:`~cca_zoo.linear.MCCA`'s
    ``pca=True`` shortcut is not applicable here and isn't exposed).

    Note:
        :class:`sklearn.covariance.GraphicalLasso` (and its CV variant)
        estimate a *sparse precision* matrix under a Gaussian assumption
        and are themselves most useful in the high-dimensional
        (:math:`p \gtrsim n`) regime a plain sample covariance can't
        invert reliably -- exactly where ``MCCA``'s own docs recommend
        ``pca=True`` instead. This is a different way to make that same
        regime tractable: constrain the *inverse* covariance's structure
        rather than truncate the covariance's rank.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default
            True.
        c: Ridge blend applied on top of the graphical-lasso covariance
            estimate, same semantics as :class:`~cca_zoo.linear.MCCA`'s own
            ``c`` (``(1 - c) * cov + c * I``). Default 0 -- the graphical
            lasso's own ``alpha`` is normally regularisation enough on its
            own.
        alpha: Graphical-lasso L1 penalty strength(s). A scalar or per-view
            list of non-negative floats, or ``None`` (per view) to select
            it automatically via :class:`~sklearn.covariance.GraphicalLassoCV`
            (slower -- a 5-fold search per view every fit -- but avoids
            hand-tuning ``alpha``, which lives on the raw covariance scale
            rather than a bounded ``[0, 1]`` ridge parameter and so has no
            single sensible default across arbitrarily scaled data).
            Default 0.01, matching :class:`~sklearn.covariance.GraphicalLasso`'s
            own default.
        mode: Graphical-lasso solver, ``"cd"`` (coordinate descent) or
            ``"lars"`` -- passed straight through to
            :class:`~sklearn.covariance.GraphicalLasso` /
            :class:`~sklearn.covariance.GraphicalLassoCV`. Default ``"cd"``.
        max_iter: Maximum graphical-lasso iterations. Default 100.
        eps: Small constant added to the eigenvalues of ``B`` to ensure
            positive definiteness. Default 1e-6.

    References:
        Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse
        covariance estimation with the graphical lasso. Biostatistics,
        9(3), 432-441.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 12))
        >>> X2 = rng.standard_normal((100, 9))
        >>> model = GraphicalLassoCCA(alpha=0.1).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
        >>> precisions = model.precision_  # sparse per-view precision matrices
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
        latent_dimensions: int = 1,
        center: bool = True,
        c: float | list[float] = 0.0,
        alpha: float | list[float | None] | None = 0.01,
        mode: str = "cd",
        max_iter: int = 100,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            center=center,
            c=c,
            pca=False,
            eps=eps,
        )
        self.alpha = alpha
        self.mode = mode
        self.max_iter = max_iter

    def _build_B(self, views: list[np.ndarray], c: list[float]) -> np.ndarray:
        """Build B from each view's graphical-lasso covariance estimate.

        Args:
            views: Centred view arrays.
            c: Per-view ridge-blend parameters.

        Returns:
            Symmetric positive-definite matrix of shape
            (sum_features, sum_features).
        """
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
