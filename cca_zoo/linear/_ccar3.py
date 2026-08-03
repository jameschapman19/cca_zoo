"""CCAR3 — Canonical Correlation Analysis via Reduced Rank Regression."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.covariance import LedoitWolf
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


def _admm_row_sparse_rrr(
    X: np.ndarray,
    Y_tilde: np.ndarray,
    lambda_: float,
    rho: float,
    max_iter: int,
    tol: float,
    ridge: float,
) -> np.ndarray:
    """Solve min_B (1/n)||Y_tilde - XB||^2 + lambda_ * sum_j ||B[j,:]||_2 via ADMM."""
    n, p = X.shape
    Sx = X.T @ X / n
    L = np.linalg.cholesky(Sx + (rho + ridge) * np.eye(p))
    prod_xy = X.T @ Y_tilde / n

    U = np.zeros_like(prod_xy)
    Z = np.zeros_like(prod_xy)
    B = Z
    for _ in range(max_iter):
        rhs = prod_xy + rho * (Z - U)
        B = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        Z_old = Z
        Z = B + U
        row_norms = np.linalg.norm(Z, axis=1)
        shrinkage = np.zeros_like(row_norms)
        nonzero = row_norms > 0
        shrinkage[nonzero] = np.maximum(0.0, 1.0 - (lambda_ / rho) / row_norms[nonzero])
        Z = Z * shrinkage[:, None]
        U = U + B - Z

        primal = np.linalg.norm(Z - B) / np.sqrt(p)
        dual = np.linalg.norm(Z_old - Z) / np.sqrt(p)
        if max(primal, dual) < tol:
            break
    return np.asarray(B)


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
    solved by ADMM:

    $$
    \begin{aligned}
    \hat{B} = \underset{B}{\mathrm{argmin}}\ \frac{1}{n}
        \lVert \tilde{Y} - X B \rVert_F^2
        + \lambda \sum_{j=1}^{p} \lVert B_{j, :} \rVert_2
    \end{aligned}
    $$

    which drives whole rows of $B$ (whole $X$ features) to zero, giving a
    sparse-in-$X$ solution well-suited to $p \gg n$. The rank-
    ``latent_dimensions`` SVD of $\hat{B}$ gives the canonical directions,
    which are then whitened so that the canonical variates have unit
    variance, sign-aligned to positive correlation, and sorted in
    descending order.

    Because the penalty acts on rows of $B$, sparsity is induced only in
    $X$; $Y$ is handled densely via its inverse-square-root covariance.
    Swap the order of ``views`` to regularise the other view instead.

    This is a from-scratch NumPy port of the reference R implementation,
    [ccar3](https://github.com/jameschapman19/ccar3); cross-validation
    utilities and the CVXR/rrpack solver backends are out of scope here —
    use `GridSearchCV` from `cca_zoo.model_selection` to select ``lambda_``
    as for any other estimator.

    References:
        Donnat, C., & Tuzhilina, E. (2024). Canonical Correlation Analysis
        as Reduced Rank Regression in High Dimensions. arXiv:2405.19539.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.
        lambda_: Row-group-lasso regularisation strength used when
            ``highdim=True``. ``0`` disables the penalty. Default is 0.
        highdim: Whether to estimate the reduced-rank coefficient with the
            ADMM-solved group-lasso penalty (default, needed when ``X`` has
            more features than samples) or with the closed-form
            low-dimensional solution (``False``).
        ledoit_wolf: Whether to shrink the ``Y`` covariance matrix with
            Ledoit-Wolf shrinkage before inverting it. Default True.
        rho: ADMM step-size parameter. Default 1.0.
        max_iter: Maximum number of ADMM iterations. Default 10_000.
        tol: ADMM convergence tolerance on the primal/dual residuals.
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
        "rho": POSITIVE_EPS,
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
        rho: float = 1.0,
        max_iter: int = 10_000,
        tol: float = 1e-4,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.lambda_ = lambda_
        self.highdim = highdim
        self.ledoit_wolf = ledoit_wolf
        self.rho = rho
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
            B = _admm_row_sparse_rrr(
                X,
                Y_tilde,
                lambda_=self.lambda_,
                rho=self.rho,
                max_iter=self.max_iter,
                tol=self.tol,
                ridge=self.eps,
            )
        else:
            Sx = X.T @ X / n + self.eps * np.eye(X.shape[1])
            B = np.linalg.solve(Sx, X.T @ Y_tilde / n)

        U, V = _postprocess_rrr_fit(
            B, X, Y, sqrt_inv_Sy, self.latent_dimensions, ridge=self.eps
        )
        self.weights_: list[np.ndarray] = [U, V]
        return self
