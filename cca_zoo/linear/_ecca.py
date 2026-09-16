"""ECCA — CCA via entrywise-sparse reduced rank regression."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import Lasso
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._param_constraints import POSITIVE_EPS, POSITIVE_INT
from cca_zoo.linear._rrr_common import _postprocess_rrr_fit


def _entrywise_sparse_rrr(
    X: np.ndarray, Y: np.ndarray, lambda_: float, max_iter: int, tol: float
) -> np.ndarray:
    r"""Solve min_B (1/n)||Y - XB||^2 + lambda_ * sum_{j,k} |B[j,k]|.

    Unlike :func:`cca_zoo.linear._ccar3._row_sparse_rrr`'s row-group penalty,
    an entrywise L1 penalty on ``B`` places no coupling between a row's
    entries, so the problem separates exactly into one independent Lasso
    regression per column of ``Y``: column $k$'s objective is
    $\frac1n\lVert y_k - Xb_k \rVert^2 + \lambda \lVert b_k
    \rVert_1$, exactly :class:`~sklearn.linear_model.Lasso`'s own objective
    at ``alpha = lambda_ / 2`` (sklearn's $\frac{1}{2n}$ loss convention,
    not ``cca_zoo``'s $\frac1n$ -- see ``_row_sparse_rrr``'s docstring for
    the same factor-of-2 derivation). The reference R implementation
    (``ecca()`` in the `ccar3 <https://github.com/jameschapman19/ccar3>`_
    package) instead solves this with a single matrix-free ADMM over the
    whole ``B`` at once, needed there for its own memory-efficiency goals;
    since the entrywise penalty makes the columns independent regardless of
    solver, a bank of per-column Lasso fits reaches the same optimum with
    no ADMM machinery at all -- 0.10s vs. 59.0s for R's ``ecca()`` at the
    same n=300, p=300, q=100 problem in a direct benchmark (R's ADMM never
    converges early there at that ``lambda_``, running its full 20,000
    -iteration budget; sklearn's per-column coordinate descent does).

    At ``lambda_ == 0`` this instead solves the unpenalised least-squares
    problem directly (``Lasso(alpha=0)`` is mathematically the same
    problem, but sklearn's own coordinate descent warns it "does not
    converge well" there and recommends exactly this alternative).
    """
    if lambda_ == 0.0:
        B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        return np.asarray(B)
    q = Y.shape[1]
    B = np.zeros((X.shape[1], q))
    for k in range(q):
        model = Lasso(
            alpha=lambda_ / 2.0, fit_intercept=False, max_iter=max_iter, tol=tol
        )
        model.fit(X, Y[:, k])
        B[:, k] = model.coef_
    return B


class ECCA(BaseModel):
    r"""Canonical Correlation Analysis via entrywise-sparse Reduced Rank Regression.

    Like :class:`~cca_zoo.linear.CCAR3`, recasts two-view CCA as a reduced
    -rank regression of ``X`` onto ``Y``, but with an entrywise penalty on
    the coefficient matrix $B$ instead of a row-group one, and -- unlike
    ``CCAR3`` -- fit directly against raw (centred) $Y$, with no
    Ledoit-Wolf pre-whitening step:

    $$
    \hat{B} = \underset{B}{\mathrm{argmin}}\ \frac{1}{n}
        \lVert Y - X B \rVert_F^2
        + \lambda \sum_{j,k} \lvert B_{j,k} \rvert
    $$

    Where :class:`~cca_zoo.linear.CCAR3`'s row-group-lasso penalty zeroes
    whole $X$ features at once (a feature is either used by every canonical
    variate or by none), this entrywise penalty can zero individual
    ``(feature, component)`` entries independently -- a feature can
    contribute to component 1 while being dropped from component 2. This
    exactly mirrors the relationship between
    :class:`~cca_zoo.sparse.ElasticNetCCA` (entrywise) and
    :class:`~cca_zoo.sparse.MultiTaskElasticNetCCA` (row-group) one level
    up, but here for the reduced-rank-regression family rather than the
    Eckart-Young-loss family. The rank-``latent_dimensions`` SVD of
    $\hat{B}$ gives the canonical directions, whitened so the canonical
    variates have unit variance, sign-aligned to positive correlation, and
    sorted in descending order -- the same postprocessing ``CCAR3`` uses,
    just with no Y-covariance un-whitening step (there's no Y-whitening to
    undo here).

    This is a NumPy port of the reference R implementation's ``ecca()``
    function ([ccar3](https://github.com/jameschapman19/ccar3)), solved by
    a bank of independent :class:`~sklearn.linear_model.Lasso` fits (one
    per column of $Y$) rather than the R package's single matrix-free ADMM
    over the whole $B$ at once -- the entrywise penalty makes the columns
    of $B$ independent regardless of solver (see
    :func:`_entrywise_sparse_rrr`'s docstring), so this reaches the same
    optimum with a far simpler, already-well-tested solver. The absence of
    Y-whitening (unlike ``CCAR3``) is deliberate, not an oversight: the R
    reference's own ``Sy``/``LW_Sy`` machinery is present in ``cca_rrr()``
    but explicitly *not* used by ``ecca()`` -- its `ecca_across_lambdas`
    keeps an ``Sy`` parameter only "for compatibility" and ignores it. The
    R package's optional block/graph ``groups`` argument (arbitrary
    ``(x, y)`` index pairs sharing one penalty) is out of scope here; use
    `GridSearchCV` from `cca_zoo.model_selection` to select ``lambda_`` as
    for any other estimator.

    References:
        `ccar3 <https://github.com/jameschapman19/ccar3>`_'s ``ecca()``,
        the entrywise-sparse companion to
        Donnat, C., & Tuzhilina, E. (2024). Canonical Correlation Analysis
        as Reduced Rank Regression in High Dimensions. arXiv:2405.19539.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.
        lambda_: Entrywise lasso regularisation strength. ``0`` disables
            the penalty. Default is 0.
        max_iter: Maximum number of coordinate-descent iterations, passed
            straight through to :class:`~sklearn.linear_model.Lasso`.
            Default 10_000.
        tol: Convergence tolerance, passed straight through to
            :class:`~sklearn.linear_model.Lasso`. Default 1e-4.
        eps: Small constant added to covariance matrices before inversion,
            for numerical stability. Default 1e-8.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = ECCA(latent_dimensions=2, lambda_=0.1).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "lambda_": [Interval(Real, 0, None, closed="left")],
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
        "eps": POSITIVE_EPS,
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        lambda_: float = 0.0,
        max_iter: int = 10_000,
        tol: float = 1e-4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

    def fit(self, views: list[ArrayLike], y: None = None) -> ECCA:
        """Fit the ECCA model.

        Args:
            views: List of exactly two arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If the number of views is not exactly 2.
            ValueError: If views have inconsistent numbers of samples.
        """
        views_ = self._setup_fit(views)
        if self.n_views_ != 2:
            raise ValueError(
                f"ECCA requires exactly 2 views, got {self.n_views_}. "
                "Use MCCA for more than 2 views."
            )
        X, Y = views_

        B = _entrywise_sparse_rrr(
            X, Y, lambda_=self.lambda_, max_iter=self.max_iter, tol=self.tol
        )

        no_whitening = np.eye(Y.shape[1])
        U, V = _postprocess_rrr_fit(
            B, X, Y, no_whitening, self.latent_dimensions, ridge=self.eps
        )
        self.weights_: list[np.ndarray] = [U, V]
        return self
