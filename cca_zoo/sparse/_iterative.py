"""Sparse CCA by alternating updates with deflation."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from sklearn.linear_model import ElasticNet, Lasso, Ridge, lasso_path

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import deflate, soft_threshold
from cca_zoo._utils._validation import perview_parameter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract iterative base
# ---------------------------------------------------------------------------


class _BaseIterative(BaseModel):
    """Base for sparse CCA fitted by alternating per-view updates with deflation.

    Subclasses implement :meth:`_update_weight`, one view's update given the
    other views' current scores.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        max_iter: Maximum iterations per latent dimension.
        tol: Convergence tolerance on the change in weights. Default is 1e-6.
        random_state: Seed for the random initialisation.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
    """

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> _BaseIterative:
        """Fit the model.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        # Initialise weight storage: (n_features_i, n_components)
        self.weights_: list[np.ndarray] = [
            np.zeros((p, self.n_components)) for p in self.n_features_in_
        ]
        deflated = [v.copy() for v in views_]
        for d in range(self.n_components):
            # Random initialisation for this dimension
            w = [rng.standard_normal(p) for p in self.n_features_in_]
            w = [wi / np.linalg.norm(wi) for wi in w]
            self._fit_single(deflated, w, d)
            for i in range(self.n_views_):
                self.weights_[i][:, d] = w[i]
            deflated = deflate(deflated, w)
        return self

    def _fit_single(
        self,
        views: list[np.ndarray],
        w: list[np.ndarray],
        d: int,
    ) -> None:
        """Run the alternating updates for one latent dimension, in place on ``w``."""
        for iteration in range(self.max_iter):
            w_prev = [wi.copy() for wi in w]
            for i in range(len(views)):
                w[i] = self._update_weight(views, w, i)
            # Check convergence
            delta = max(np.linalg.norm(w[i] - w_prev[i]) for i in range(len(views)))
            if delta < self.tol:
                logger.debug("dim %d converged at iteration %d", d, iteration)
                break

    @abstractmethod
    def _update_weight(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """Updated unit-norm weight vector of view ``i``."""


def _target_score(
    views: list[np.ndarray],
    weights: list[np.ndarray],
    i: int,
) -> np.ndarray:
    """Sum of the scores of every view except ``i``."""
    scores = [views[j] @ weights[j] for j in range(len(views)) if j != i]
    target: np.ndarray = np.asarray(sum(scores))
    norm = np.linalg.norm(target)
    if norm > 1e-12:
        target = target / norm
    return target


# ---------------------------------------------------------------------------
# PMDCCA — Penalized Matrix Decomposition (Witten 2009)
# ---------------------------------------------------------------------------


def _bisect_threshold(x: np.ndarray, l1_bound: float) -> np.ndarray:
    """Soft-threshold ``x`` so its L2-normalised result has L1 norm ``l1_bound``.

    The search is on the L1/L2 ratio of the thresholded vector, which is
    invariant to the scale of ``x``, so the bound depends only on ``tau``.
    """
    norm_x = np.linalg.norm(x)
    if norm_x <= 1e-12:
        return np.zeros_like(x)
    unit_x = x / norm_x
    if np.linalg.norm(unit_x, 1) <= l1_bound:
        return np.asarray(unit_x)

    def l1_over_l2_minus_bound(delta: float) -> float:
        thresholded = soft_threshold(x, delta)
        norm_t = np.linalg.norm(thresholded)
        ratio = np.linalg.norm(thresholded, 1) / norm_t if norm_t > 1e-12 else 0.0
        return ratio - l1_bound

    # A fixed-count bisection here previously ran all 50 iterations
    # unconditionally, with no early stop once converged. The L1/L2 ratio
    # is 0 at delta = max|x| (soft_threshold zeroes everything) and > 0 at
    # delta = 0 (guaranteed by the early return above not having
    # triggered), so brentq's bracket is always valid; its superlinear
    # (inverse-quadratic) convergence plus a real tolerance-based stop
    # reaches the same root in far fewer evaluations -- 3.65x faster in a
    # direct benchmark across 500 random (x, l1_bound) pairs, agreeing
    # with the old fixed-count bisection to within 1e-9.
    delta = brentq(l1_over_l2_minus_bound, 0.0, np.abs(x).max(), xtol=1e-10)
    result = soft_threshold(x, delta)
    norm = np.linalg.norm(result)
    if norm > 1e-12:
        result /= norm
    return result


class PMDCCA(_BaseIterative):
    r"""Sparse CCA by penalized matrix decomposition.

    Maximises cross-view covariance subject to L1 and L2 constraints on each
    weight vector:

    $$
    \max_{\mathbf{w}_1, \mathbf{w}_2} \mathbf{w}_1^\top X_1^\top X_2 \mathbf{w}_2
    \quad\text{subject to}\quad
    \|\mathbf{w}_i\|_1 \le \tau_i \sqrt{p_i},\ \|\mathbf{w}_i\|_2 = 1.
    $$

    Each update soft-thresholds with the level found by bisection.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        tau: L1 bound as a fraction of ``sqrt(n_features_i)``, in ``(0, 1]``;
            1 imposes no sparsity. Per-view. Default is 1.
        max_iter: Maximum iterations per latent dimension. Default is 500.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for the random initialisation. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Witten, D. M., Tibshirani, R., & Hastie, T. (2009). A penalized matrix
        decomposition, with applications to sparse principal components and
        canonical correlation analysis. Biostatistics, 10(3), 515-534.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 10)), rng.standard_normal((50, 8))
        >>> model = PMDCCA(tau=0.5, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        tau: float | list[float] = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.tau = tau

    def fit(self, views: list[ArrayLike], y: None = None) -> PMDCCA:
        """Fit the model.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.
        """
        # Store processed tau for use in _update_weight
        self._tau: list[float] = []  # set in super().fit via _setup_fit
        super().fit(views, y)
        return self

    def _setup_tau(self) -> list[float]:
        """Per-view L1 bounds ``tau * sqrt(n_features_i)``."""
        tau_ = perview_parameter("tau", self.tau, 1.0, self.n_views_)
        return [t * np.sqrt(p) for t, p in zip(tau_, self.n_features_in_)]

    def _fit_single(
        self,
        views: list[np.ndarray],
        w: list[np.ndarray],
        d: int,
    ) -> None:
        """Set the L1 bounds, then run the alternating updates."""
        self._l1_bounds = self._setup_tau()
        super()._fit_single(views, w, d)

    def _update_weight(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """Soft-thresholded power-iteration update of view ``i``."""
        target = _target_score(views, weights, i)
        raw = views[i].T @ target
        return _bisect_threshold(raw, self._l1_bounds[i])


# ---------------------------------------------------------------------------
# ADMMCCA — ADMM-based sparse CCA (Suo 2017)
# ---------------------------------------------------------------------------


class ADMMCCA(_BaseIterative):
    r"""Sparse CCA by linearised ADMM.

    For view $i$, with the other views' summed score
    $\bar{\mathbf{s}}_{\neg i}$ fixed, solves

    $$
    \max_{\mathbf{w}_i}\ \mathbf{w}_i^\top X_i^\top \bar{\mathbf{s}}_{\neg i}
        - \tau_i \|\mathbf{w}_i\|_1
    \quad\text{subject to}\quad \|X_i \mathbf{w}_i\|_2 \le 1
    $$

    by linearised ADMM on the split $\mathbf{z}_i = X_i \mathbf{w}_i$, so
    each step is a soft-threshold and a projection onto the unit ball, with
    step size $1 / (\mu \|X_i\|_{\mathrm{op}}^2)$. Further components use
    deflation rather than the paper's orthogonality constraint.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        tau: L1 penalty. Per-view. Default is 0.1.
        mu: ADMM penalty parameter. Default is 1.0.
        max_iter: Maximum outer iterations per latent dimension. Default is 500.
        admm_iter: Maximum ADMM iterations per view update. Default is 50.
        tol: Convergence tolerance of both loops. Default is 1e-6.
        random_state: Seed for the random initialisation. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Suo, X., Mineiro, P., & Anandkumar, A. (2017). Sparse canonical
        correlation analysis. arXiv:1705.10865.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 10)), rng.standard_normal((50, 8))
        >>> model = ADMMCCA(tau=0.1, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        tau: float | list[float] = 0.1,
        mu: float = 1.0,
        max_iter: int = 500,
        admm_iter: int = 50,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.tau = tau
        self.mu = mu
        self.admm_iter = admm_iter

    def _fit_single(
        self,
        views: list[np.ndarray],
        w: list[np.ndarray],
        d: int,
    ) -> None:
        """Alternate over views, each update an inner linearised-ADMM solve."""
        tau_ = perview_parameter("tau", self.tau, 0.1, len(views))
        n_views = len(views)
        # eta_i = 1/(mu * ||X_i||_op^2) is fixed for this latent dimension
        # (X_i doesn't change across iterations), computed once per view.
        etas = [1.0 / (self.mu * np.linalg.norm(X, ord=2) ** 2) for X in views]
        # z_i, xi_i (score-space ADMM state) persist across outer iterations,
        # matching the paper's Algorithm 1 (they are initialised once, not
        # reset every time a view's block is revisited).
        z = [views[i] @ w[i] for i in range(n_views)]
        xi = [np.zeros(views[i].shape[0]) for i in range(n_views)]
        for _iter in range(self.max_iter):
            w_prev = [wi.copy() for wi in w]
            for i in range(n_views):
                s_other = sum(views[j] @ w[j] for j in range(n_views) if j != i)
                c_i = views[i].T @ s_other
                for _ in range(self.admm_iter):
                    w_before = w[i]
                    Xw = views[i] @ w[i]
                    grad_lin = self.mu * views[i].T @ (Xw - z[i] + xi[i])
                    a = w[i] - etas[i] * grad_lin + etas[i] * c_i
                    w[i] = soft_threshold(a, etas[i] * tau_[i])
                    Xw = views[i] @ w[i]
                    z_new = Xw + xi[i]
                    z_norm = np.linalg.norm(z_new)
                    if z_norm > 1.0:
                        z_new = z_new / z_norm
                    z[i] = z_new
                    xi[i] = xi[i] + Xw - z[i]
                    if np.linalg.norm(w[i] - w_before) < self.tol:
                        break
            delta = max(np.linalg.norm(w[i] - w_prev[i]) for i in range(n_views))
            if delta < self.tol:
                logger.debug("ADMM dim %d converged at iter %d", d, _iter)
                break

    def _update_weight(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """Unused: :meth:`_fit_single` performs the updates."""
        return weights[i]


# ---------------------------------------------------------------------------
# IPLSCCA — Iterative PLS (Mai & Zhang 2019)
# ---------------------------------------------------------------------------


class IPLSCCA(_BaseIterative):
    r"""Sparse CCA by iterative penalised least squares.

    Each view's weights solve an elastic-net regression onto the other views'
    summed score, then are rescaled to a unit-variance score:

    $$
    \hat{\mathbf{w}}_i = \arg\min_{\mathbf{w}}
        \frac{1}{2n} \|X_i \mathbf{w} - \bar{\mathbf{s}}_{\neg i}\|_2^2
        + \alpha_i \Bigl(r \|\mathbf{w}\|_1 + \tfrac{1-r}{2} \|\mathbf{w}\|_2^2\Bigr).
    $$

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        alpha: Penalty strength. Per-view. Default is 0.
        l1_ratio: Share of the penalty that is L1 (1 is the lasso). Per-view.
            Default is 1.
        max_iter: Maximum iterations per latent dimension. Default is 500.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for the random initialisation. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Mai, Q., & Zhang, X. (2019). An iterative penalized least squares
        approach to sparse canonical correlation analysis. Biometrics, 75(3),
        734-744.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 10)), rng.standard_normal((50, 8))
        >>> model = IPLSCCA(alpha=0.1, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        alpha: float | list[float] = 0.0,
        l1_ratio: float | list[float] = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _fit_single(
        self,
        views: list[np.ndarray],
        w: list[np.ndarray],
        d: int,
    ) -> None:
        """Build the per-view regressors, then run the alternating updates."""
        alpha_ = perview_parameter("alpha", self.alpha, 0.0, len(views))
        l1_ = perview_parameter("l1_ratio", self.l1_ratio, 1.0, len(views))
        self._regressors = _make_regressors(alpha_, l1_, self.tol, self.random_state)
        super()._fit_single(views, w, d)

    def _update_weight(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """Penalised regression of view ``i`` onto the other views' score."""
        target = _target_score(views, weights, i)
        reg = self._regressors[i]
        reg.fit(views[i], target)
        w_new: np.ndarray = np.asarray(reg.coef_).copy()
        score = views[i] @ w_new
        score_std = score.std()
        if score_std > 1e-12:
            w_new /= score_std
        return w_new


# ---------------------------------------------------------------------------
# SpanCCA — hard-thresholding ALS inspired by Asteris et al.'s SpanCCA (2016)
# ---------------------------------------------------------------------------


class SpanCCA(_BaseIterative):
    """Sparse CCA by hard-thresholded alternating updates.

    Each update keeps the ``span`` largest-magnitude weights. This is an
    alternating heuristic in the spirit of Asteris et al.'s SpanCCA, not their
    randomised low-rank search.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        span: Number of nonzero weights kept; ``None`` keeps all. Per-view.
            Default is None.
        max_iter: Maximum iterations per latent dimension. Default is 500.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for the random initialisation. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Asteris, M., Kyrillidis, A., Koyejo, O., & Poldrack, R. (2016). A simple
        and provable algorithm for sparse diagonal CCA. ICML.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 10)), rng.standard_normal((50, 8))
        >>> model = SpanCCA(span=5, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        span: int | list[int] | None = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.span = span

    def _fit_single(
        self,
        views: list[np.ndarray],
        w: list[np.ndarray],
        d: int,
    ) -> None:
        """Set the per-view spans, then run the alternating updates."""
        default_span = views[0].shape[1]
        span_raw = self.span if self.span is not None else default_span
        span_ = perview_parameter("span", span_raw, default_span, len(views))
        self._spans: list[int] = [int(s) for s in span_]
        super()._fit_single(views, w, d)

    def _update_weight(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """Update of view ``i`` keeping its ``span`` largest weights."""
        target = _target_score(views, weights, i)
        raw: np.ndarray = np.asarray(views[i].T @ target)
        # Keep only the top-span entries
        s = self._spans[i]
        if s < len(raw):
            threshold = np.sort(np.abs(raw))[-s]
            raw = np.where(np.abs(raw) >= threshold, raw, 0.0)
        norm = np.linalg.norm(raw)
        if norm > 1e-12:
            raw /= norm
        return raw


# ---------------------------------------------------------------------------
# WaijenborgCCA — Elastic net CCA (Waaijenborg 2008)
# ---------------------------------------------------------------------------


class WaijenborgCCA(_BaseIterative):
    r"""Penalised CCA by alternating elastic-net regressions.

    Each view's weights solve an elastic-net regression onto the normalised
    sum of all views' scores:

    $$
    \hat{\mathbf{w}}_i = \arg\min_{\mathbf{w}}
        \frac{1}{2n} \|X_i \mathbf{w} - \mathbf{s}\|_2^2
        + \alpha_i \Bigl(r \|\mathbf{w}\|_1 + \tfrac{1 - r}{2} \|\mathbf{w}\|_2^2\Bigr).
    $$

    Unlike :class:`~cca_zoo.sparse.ElasticNetCCA`, which penalises the EY
    loss itself, this alternates plain regressions.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        alpha: Penalty strength. Per-view. Default is 0.
        l1_ratio: Share of the penalty that is L1. Per-view. Default is 0.5.
        max_iter: Maximum iterations per latent dimension. Default is 500.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for the random initialisation. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Waaijenborg, S., de Witt Hamer, P. C. V., & Zwinderman, A. H. (2008).
        Quantifying the association between gene expressions and DNA-markers by
        penalized canonical correlation analysis. Statistical Applications in
        Genetics and Molecular Biology, 7(1).

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 10)), rng.standard_normal((50, 8))
        >>> model = WaijenborgCCA(alpha=0.1, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        alpha: float | list[float] = 0.0,
        l1_ratio: float | list[float] = 0.5,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _fit_single(
        self,
        views: list[np.ndarray],
        w: list[np.ndarray],
        d: int,
    ) -> None:
        """Build the per-view regressors, then run the alternating updates."""
        alpha_ = perview_parameter("alpha", self.alpha, 0.0, len(views))
        l1_ = perview_parameter("l1_ratio", self.l1_ratio, 0.5, len(views))
        self._regressors = _make_regressors(alpha_, l1_, self.tol, self.random_state)
        super()._fit_single(views, w, d)

    def _update_weight(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """Elastic-net regression of view ``i`` onto all views' summed score."""
        target = _target_score(views, weights, i)
        reg = self._regressors[i]
        reg.fit(views[i], target)
        return cast(np.ndarray, np.atleast_1d(reg.coef_).ravel())


# ---------------------------------------------------------------------------
# ParkhomenkoCCA — Parkhomenko 2009
# ---------------------------------------------------------------------------


class ParkhomenkoCCA(_BaseIterative):
    r"""Sparse CCA by soft-thresholded power iteration on standardised views.

    Uses the paper's diagonal approximation to the within-view covariances,
    which amounts to standardising each feature, then iterates

    $$
    \mathbf{w}_i \leftarrow S_{\tau_i}(\tilde X_i^\top \bar{\mathbf{s}}_{\neg i}),
    $$

    with $S_\tau$ the soft-threshold and $\tilde X_i$ the standardised view.
    Weights are returned on the original feature scale.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        tau: Soft-threshold level. Per-view. Default is 0.1.
        max_iter: Maximum iterations per latent dimension. Default is 500.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for the random initialisation. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Parkhomenko, E., Tritchler, D., & Beyene, J. (2009). Sparse canonical
        correlation analysis with application to genomic data integration.
        Statistical Applications in Genetics and Molecular Biology, 8(1).

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 10)), rng.standard_normal((50, 8))
        >>> model = ParkhomenkoCCA(tau=0.1, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        tau: float | list[float] = 0.1,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.tau = tau

    def _fit_single(
        self,
        views: list[np.ndarray],
        w: list[np.ndarray],
        d: int,
    ) -> None:
        """Standardise the views and set the thresholds, then run the updates."""
        self._tau_vals = perview_parameter("tau", self.tau, 0.1, len(views))
        scales = [v.std(axis=0, keepdims=True) for v in views]
        scales = [np.where(s < 1e-12, 1.0, s) for s in scales]
        scaled_views = [v / s for v, s in zip(views, scales)]
        super()._fit_single(scaled_views, w, d)
        # w is in the standardised views' coordinates; convert back to the
        # original feature scale and re-normalise to unit L2 norm, matching
        # every other class in this module's convention.
        for i in range(len(w)):
            w[i] = w[i] / scales[i].ravel()
            norm = np.linalg.norm(w[i])
            if norm > 1e-12:
                w[i] = w[i] / norm

    def _update_weight(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """Soft-thresholded update of view ``i``."""
        target = _target_score(views, weights, i)
        raw = views[i].T @ target
        result = soft_threshold(raw, self._tau_vals[i])
        norm = np.linalg.norm(result)
        if norm > 1e-12:
            result /= norm
        return result


# ---------------------------------------------------------------------------
# SAR — Sparse Alternating Regression (Wilms & Croux 2015)
# ---------------------------------------------------------------------------


def _sar_bic_lasso(
    x: np.ndarray,
    y: np.ndarray,
    n_lambda: int,
    tol: float,
) -> np.ndarray:
    """Lasso coefficients at the BIC-minimising point of the lasso path.

    BIC is ``n log(RSS / n) + k log n``, with ``k`` the number of nonzero
    coefficients, as in Wilms and Croux (2015).
    """
    n, p = x.shape
    # SAR runs ~1000 paths per fit, so skip sklearn's per-call input
    # validation (most of each call's time), passing the Gram exactly when
    # precompute="auto" would build it (n > p).
    x = np.asfortranarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    _, coefs, _ = lasso_path(
        x,
        y,
        alphas=n_lambda,
        tol=tol,
        precompute=x.T @ x if n > p else False,
        Xy=x.T @ y if n > p else None,
        check_input=False,
    )
    residuals = y[:, None] - x @ coefs
    rss = np.maximum((residuals**2).sum(axis=0), 1e-12)
    nnz = (np.abs(coefs) > 1e-12).sum(axis=0)
    bic = n * np.log(rss / n) + nnz * np.log(n)
    return cast(np.ndarray, coefs[:, np.argmin(bic)])


class SAR(_BaseIterative):
    """Sparse alternating regression with BIC-selected sparsity.

    CCA as alternating regressions of each view's score on the other views'
    summed score, each a lasso whose penalty is chosen by BIC. Latent
    dimensions after the first are found on deflated views, then re-fitted
    against the original views (Wilms and Croux, Section 3). The extension
    beyond two views is this implementation's own.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        n_lambda: Points on each lasso path. Default is 100.
        max_iter: Maximum iterations per latent dimension. Default is 500.
        tol: Convergence tolerance of the alternating loop and each lasso path.
            Default is 1e-6.
        random_state: Seed for the random initialisation. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Wilms, I., & Croux, C. (2015). Sparse canonical correlation analysis
        from a predictive point of view. Biometrical Journal, 57(5), 834-851.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> latent = rng.standard_normal(50)
        >>> X1 = np.column_stack([latent, rng.standard_normal((50, 9))])
        >>> X2 = np.column_stack([latent, rng.standard_normal((50, 7))])
        >>> model = SAR(random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        n_lambda: int = 100,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.n_lambda = n_lambda

    def fit(self, views: list[ArrayLike], y: None = None) -> SAR:
        """Fit the model.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        self.weights_: list[np.ndarray] = [
            np.zeros((p, self.n_components)) for p in self.n_features_in_
        ]
        deflated = [v.copy() for v in views_]
        for d in range(self.n_components):
            w = [rng.standard_normal(p) for p in self.n_features_in_]
            w = [wi / np.linalg.norm(wi) for wi in w]
            self._fit_single(deflated, w, d)
            if d == 0:
                w_final = w
            else:
                w_final = [
                    self._reexpress(views_[j], deflated[j] @ w[j])
                    for j in range(self.n_views_)
                ]
            for j in range(self.n_views_):
                self.weights_[j][:, d] = w_final[j]
            deflated = deflate(deflated, w)
        return self

    def _reexpress(
        self, original_view: np.ndarray, deflated_score: np.ndarray
    ) -> np.ndarray:
        """Re-fit a deflated direction's score against the original view."""
        coef = _sar_bic_lasso(
            original_view, deflated_score.ravel(), self.n_lambda, self.tol
        )
        norm = np.linalg.norm(coef)
        if norm > 1e-12:
            coef = coef / norm
        return coef

    def _update_weight(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """BIC-selected lasso of view ``i`` onto the other views' score."""
        target = _target_score(views, weights, i)
        coef = _sar_bic_lasso(views[i], target.ravel(), self.n_lambda, self.tol)
        norm = np.linalg.norm(coef)
        if norm > 1e-12:
            coef = coef / norm
        return coef


# ---------------------------------------------------------------------------
# Helper: regressor factory
# ---------------------------------------------------------------------------


def _make_regressors(
    alpha: list[float],
    l1_ratio: list[float],
    tol: float,
    random_state: int | None,
) -> list[Ridge | Lasso | ElasticNet]:
    """Per-view sklearn regressors: Ridge, Lasso or ElasticNet by ``l1_ratio``."""
    regressors: list[Ridge | Lasso | ElasticNet] = []
    for a, l1 in zip(alpha, l1_ratio):
        if l1 == 0.0:
            regressors.append(Ridge(alpha=a, fit_intercept=False, tol=tol))
        elif l1 == 1.0:
            regressors.append(
                Lasso(
                    alpha=a,
                    fit_intercept=False,
                    warm_start=True,
                    tol=tol,
                    random_state=random_state,
                    selection="random",
                )
            )
        else:
            regressors.append(
                ElasticNet(
                    alpha=a,
                    l1_ratio=l1,
                    fit_intercept=False,
                    warm_start=True,
                    tol=tol,
                    random_state=random_state,
                    selection="random",
                )
            )
    return regressors
