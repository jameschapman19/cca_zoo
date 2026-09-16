"""CCAR3 — Canonical Correlation Analysis via Reduced Rank Regression."""

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


def _sqrt_inv_psd(S: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Symmetric inverse square root of a PSD matrix, zeroing small eigenvalues."""
    vals, vecs = np.linalg.eigh(S)
    inv_sqrt_vals = np.where(vals > threshold, 1.0 / np.sqrt(np.abs(vals)), 0.0)
    return (vecs * inv_sqrt_vals) @ vecs.T


def _whiten_factor(G: np.ndarray, ridge: float) -> np.ndarray:
    """Return W such that W.T @ G @ W == I, via a (jittered) Cholesky factor."""
    p = G.shape[0]
    G = (G + G.T) / 2 + ridge * np.eye(p)
    try:
        L = np.linalg.cholesky(G)
        return np.asarray(np.linalg.inv(L).T)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(G)
        vals = np.maximum(vals, ridge)
        return np.asarray((vecs * (1.0 / np.sqrt(vals))) @ vecs.T)


def _row_sparse_rrr(
    X: np.ndarray, Y_tilde: np.ndarray, lambda_: float, max_iter: int, tol: float
) -> np.ndarray:
    r"""Solve min_B (1/n)||Y_tilde - XB||^2 + lambda_ * sum_j ||B[j,:]||_2.

    This is exactly :class:`~sklearn.linear_model.MultiTaskLasso`'s row-group
    (L2,1) penalised multi-output regression -- ``B``'s rows are ``X``'s
    features, its columns the whitened targets, and MultiTaskLasso's own
    objective is $\frac{1}{2n}\lVert Y - XB \rVert_F^2 + \alpha \sum_j
    \lVert B_{j,:} \rVert_2$, i.e. this objective at ``alpha = lambda_ / 2``
    (the factor of 2 is sklearn's own $\frac{1}{2n}$ convention, not
    ``cca_zoo``'s $\frac1n$). A previous version of this function solved the
    same problem with a hand-rolled ADMM; verified against an independent
    proximal-gradient solve, that ADMM converged to a *different*, higher
    -objective stationary point on every tested problem, because its
    B-update's linear system was missing this same factor of 2 on the
    smooth term's gradient (silently doubling the effective penalty).
    sklearn's coordinate-descent solver is both correct (provably converges
    to the global optimum of this convex problem) and substantially faster
    than the ADMM it replaces.
    """
    model = MultiTaskLasso(
        alpha=lambda_ / 2.0, fit_intercept=False, max_iter=max_iter, tol=tol
    )
    model.fit(X, Y_tilde)
    return np.asarray(model.coef_.T)


def _postprocess_rrr_fit(
    B: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    sqrt_inv_Sy: np.ndarray,
    r: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a reduced-rank coefficient matrix into whitened canonical directions."""
    n, p = X.shape
    q = Y.shape[1]

    if not np.any(B):
        return np.zeros((p, r)), np.zeros((q, r))

    r_eff = min(r, *B.shape)
    U0, _, Vt0 = np.linalg.svd(B, full_matrices=False)
    U0 = U0[:, :r_eff]
    V0 = sqrt_inv_Sy @ Vt0[:r_eff, :].T

    XU0 = X @ U0
    YV0 = Y @ V0
    GX = XU0.T @ XU0 / n
    GY = YV0.T @ YV0 / n

    U = U0 @ _whiten_factor(GX, ridge)
    V = V0 @ _whiten_factor(GY, ridge)

    XU = X @ U
    YV = Y @ V
    cor = np.diag(XU.T @ YV / n).copy()

    neg = cor < 0
    V[:, neg] *= -1
    cor[neg] *= -1

    order = np.argsort(-cor)
    U, V, cor = U[:, order], V[:, order], cor[order]

    if r_eff < r:
        U = np.hstack([U, np.zeros((p, r - r_eff))])
        V = np.hstack([V, np.zeros((q, r - r_eff))])
    return U, V


class CCAR3(BaseModel):
    r"""Canonical Correlation Analysis via Reduced Rank Regression.

    Recasts two-view CCA as a reduced-rank regression: ``Y`` is first
    whitened by its (optionally Ledoit-Wolf shrunk) covariance,

    $$
    \tilde{Y} = Y \Sigma_Y^{-1/2},
    $$

    and a coefficient matrix $B$ relating $X$ to $\tilde{Y}$ is estimated.
    In the low-dimensional regime (``highdim=False``) this has the closed
    form $B = \Sigma_X^{-1} X^\top \tilde{Y} / n$ (an ordinary reduced-rank
    regression, distinct from the classical CCA eigenproblem — the two
    agree only when $\Sigma_X$ is close to isotropic). In the
    high-dimensional regime (``highdim=True``, the default), $B$ is
    instead estimated by a row-wise group-lasso-penalised regression,

    $$
    \begin{aligned}
    \hat{B} = \underset{B}{\mathrm{argmin}}\ \frac{1}{n}
        \lVert \tilde{Y} - X B \rVert_F^2
        + \lambda \sum_{j=1}^{p} \lVert B_{j, :} \rVert_2
    \end{aligned}
    $$

    which is exactly the problem :class:`~sklearn.linear_model.MultiTaskLasso`
    solves (at ``alpha = lambda_ / 2``, to match sklearn's own $\frac{1}{2n}$
    loss convention), so it's solved by delegating to that estimator's
    coordinate-descent solver rather than a hand-rolled one. This drives
    whole rows of $B$ (whole $X$ features) to zero, giving a
    sparse-in-$X$ solution well-suited to $p \gg n$. The rank-
    ``latent_dimensions`` SVD of $\hat{B}$ gives the canonical directions,
    which are then whitened so that the canonical variates have unit
    variance, sign-aligned to positive correlation, and sorted in
    descending order.

    Because the penalty acts on rows of $B$, sparsity is induced only in
    $X$; $Y$ is handled densely via its inverse-square-root covariance.
    Swap the order of ``views`` to regularise the other view instead.

    This is a NumPy port of the reference R implementation,
    [ccar3](https://github.com/jameschapman19/ccar3), reusing scikit-learn's
    own :class:`~sklearn.linear_model.MultiTaskLasso` in place of the R
    package's CVXR/rrpack solver backends; use `GridSearchCV` from
    `cca_zoo.model_selection` to select ``lambda_`` as for any other
    estimator.

    References:
        Donnat, C., & Tuzhilina, E. (2024). Canonical Correlation Analysis
        as Reduced Rank Regression in High Dimensions. arXiv:2405.19539.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.
        lambda_: Row-group-lasso regularisation strength used when
            ``highdim=True``. ``0`` disables the penalty. Default is 0.
        highdim: Whether to estimate the reduced-rank coefficient with the
            group-lasso penalty (default, needed when ``X`` has more
            features than samples) or with the closed-form low-dimensional
            solution (``False``).
        ledoit_wolf: Whether to shrink the ``Y`` covariance matrix with
            Ledoit-Wolf shrinkage before inverting it. Default True.
        max_iter: Maximum number of coordinate-descent iterations used when
            ``highdim=True`` (passed straight through to
            :class:`~sklearn.linear_model.MultiTaskLasso`). Default 10_000.
        tol: Convergence tolerance used when ``highdim=True`` (passed
            straight through to :class:`~sklearn.linear_model.MultiTaskLasso`).
            Default 1e-4.
        eps: Small constant added to covariance matrices before inversion,
            for numerical stability. Default 1e-8.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = CCAR3(latent_dimensions=2, highdim=False).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "lambda_": [Interval(Real, 0, None, closed="left")],
        "highdim": ["boolean"],
        "ledoit_wolf": ["boolean"],
        "max_iter": POSITIVE_INT,
        "tol": POSITIVE_EPS,
        "eps": POSITIVE_EPS,
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        lambda_: float = 0.0,
        highdim: bool = True,
        ledoit_wolf: bool = True,
        max_iter: int = 10_000,
        tol: float = 1e-4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.lambda_ = lambda_
        self.highdim = highdim
        self.ledoit_wolf = ledoit_wolf
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

    def fit(self, views: list[ArrayLike], y: None = None) -> CCAR3:
        """Fit the CCAR3 model.

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
                f"CCAR3 requires exactly 2 views, got {self.n_views_}. "
                "Use MCCA for more than 2 views."
            )
        X, Y = views_
        n = X.shape[0]

        Sy = LedoitWolf().fit(Y).covariance_ if self.ledoit_wolf else Y.T @ Y / n
        sqrt_inv_Sy = _sqrt_inv_psd(Sy)
        Y_tilde = Y @ sqrt_inv_Sy

        if self.highdim:
            B = _row_sparse_rrr(
                X, Y_tilde, lambda_=self.lambda_, max_iter=self.max_iter, tol=self.tol
            )
        else:
            Sx = X.T @ X / n + self.eps * np.eye(X.shape[1])
            B = np.linalg.solve(Sx, X.T @ Y_tilde / n)

        U, V = _postprocess_rrr_fit(
            B, X, Y, sqrt_inv_Sy, self.latent_dimensions, ridge=self.eps
        )
        self.weights_: list[np.ndarray] = [U, V]
        return self
