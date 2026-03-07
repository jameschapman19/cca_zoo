"""ALS-based sparse and regularised CCA variants.

All classes in this module use an Alternating Least Squares (ALS) loop with
optional deflation to extract multiple canonical directions.

Classes:
    PLS_ALS: ALS variant of PLS (simple power iteration).
    SCCA_PMD: Sparse CCA via Penalized Matrix Decomposition (Witten 2009).
    SCCA_ADMM: Sparse CCA via ADMM (Suo 2017).
    SCCA_IPLS: Iterative PLS with lasso penalty (Mai & Zhang 2019).
    SCCA_Span: SpanCCA (Asteris 2016).
    ElasticCCA: Elastic net regularised CCA (Waaijenborg 2008).
    ParkhomenkoCCA: Sparse CCA via soft-thresholding (Parkhomenko 2009).
"""

from __future__ import annotations

import logging
from abc import abstractmethod

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import ElasticNet, Lasso, Ridge

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import deflate, soft_threshold
from cca_zoo._utils._validation import perview_parameter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract iterative base
# ---------------------------------------------------------------------------


class _BaseIterative(BaseModel):
    """Abstract base for ALS-based iterative CCA methods with deflation.

    Subclasses implement :meth:`_update_weight` which updates the weight
    vector for a single view given the current scores of all other views.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        max_iter: Maximum number of ALS iterations per latent dimension.
        tol: Convergence tolerance (weight change L2 norm). Default 1e-6.
        random_state: Seed for reproducible random initialisation.
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> _BaseIterative:
        """Fit the model by ALS with deflation for each latent dimension.

        Args:
            views: List of arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        views = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        # Initialise weight storage: (n_features_i, latent_dimensions)
        self.weights_: list[np.ndarray] = [
            np.zeros((p, self.latent_dimensions)) for p in self.n_features_in_
        ]
        deflated = [v.copy() for v in views]
        for d in range(self.latent_dimensions):
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
        """Run the ALS loop for a single latent dimension in-place on w.

        Args:
            views: Deflated view arrays for the current dimension.
            w: Weight vectors (updated in-place) for each view.
            d: Current latent dimension index (for logging).
        """
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
        """Compute the updated weight vector for view i.

        Args:
            views: Current (deflated) view arrays.
            weights: Current weight vectors for all views.
            i: Index of the view to update.

        Returns:
            Updated normalised weight vector for view i, shape (n_features_i,).
        """


def _target_score(
    views: list[np.ndarray],
    weights: list[np.ndarray],
    i: int,
) -> np.ndarray:
    """Sum of projected scores from all views except i.

    Args:
        views: View arrays.
        weights: Weight vectors.
        i: Index of the view to exclude.

    Returns:
        Summed score array of shape (n_samples,) or (n_samples, 1).
    """
    scores = [views[j] @ weights[j] for j in range(len(views)) if j != i]
    target = sum(scores)
    norm = np.linalg.norm(target)
    if norm > 1e-12:
        target = target / norm
    return target


# ---------------------------------------------------------------------------
# PLS_ALS — ALS variant of PLS
# ---------------------------------------------------------------------------


class PLS_ALS(_BaseIterative):
    r"""Alternating Least Squares variant of Partial Least Squares.

    Maximises the sum of cross-view covariances using simple power-iteration
    updates, without regularisation:

    .. math::

        \mathbf{w}_i \leftarrow
            \frac{X_i^\top \bar{\mathbf{s}}_{\neg i}}
                 {\|X_i^\top \bar{\mathbf{s}}_{\neg i}\|_2}

    where :math:`\bar{\mathbf{s}}_{\neg i}` is the normalised sum of
    projected scores from all views except :math:`i`.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        max_iter: Maximum ALS iterations per dimension. Default is 500.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = PLS_ALS(latent_dimensions=2, random_state=0).fit([X1, X2])
    """

    def _update_weight(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """Update weight for view i via unnormalised power step.

        Args:
            views: Current view arrays.
            weights: Current weight vectors.
            i: View index to update.

        Returns:
            Normalised weight vector for view i.
        """
        target = _target_score(views, weights, i)
        new_w = views[i].T @ target
        norm = np.linalg.norm(new_w)
        if norm > 1e-12:
            new_w /= norm
        return new_w


# ---------------------------------------------------------------------------
# SCCA_PMD — Penalized Matrix Decomposition (Witten 2009)
# ---------------------------------------------------------------------------


def _bisect_threshold(x: np.ndarray, l1_bound: float) -> np.ndarray:
    """Find the soft threshold that achieves ||soft_threshold(x, delta)||_1 = l1_bound.

    Uses bisection.  If ||x||_1 <= l1_bound no thresholding is applied.

    Args:
        x: Input vector.
        l1_bound: Target L1 norm.

    Returns:
        Soft-thresholded vector with L2-normalised result.
    """
    if np.linalg.norm(x, 1) <= l1_bound:
        return x / np.linalg.norm(x)
    lo, hi = 0.0, np.abs(x).max()
    for _ in range(50):
        mid = (lo + hi) / 2.0
        thresholded = soft_threshold(x, mid)
        l1 = np.linalg.norm(thresholded, 1)
        if l1 > l1_bound:
            lo = mid
        else:
            hi = mid
    result = soft_threshold(x, (lo + hi) / 2.0)
    norm = np.linalg.norm(result)
    if norm > 1e-12:
        result /= norm
    return result


class SCCA_PMD(_BaseIterative):
    r"""Sparse CCA via Penalized Matrix Decomposition.

    Maximises the cross-view covariance subject to L1 norm constraints on
    each weight vector:

    .. math::

        \max_{\mathbf{w}_1, \mathbf{w}_2}
            \mathbf{w}_1^\top X_1^\top X_2 \mathbf{w}_2

        \text{subject to }
        \|\mathbf{w}_i\|_1 \leq \tau_i \sqrt{p_i},\quad
        \|\mathbf{w}_i\|_2 = 1

    The update for each view uses bisection to find the soft-threshold that
    satisfies the L1 constraint exactly.

    References:
        Witten, D. M., Tibshirani, R., & Hastie, T. (2009). A penalized
        matrix decomposition, with applications to sparse principal
        components and canonical correlation analysis. *Biostatistics*,
        10(3), 515–534.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        tau: L1 bound scaling factor(s) in ``(0, 1]``.  The actual L1 bound
            is ``tau * sqrt(n_features_i)``.  Default is 1 (no sparsity).
        max_iter: Maximum ALS iterations. Default is 500.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = SCCA_PMD(tau=0.5, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        tau: float | list[float] = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            center=center,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.tau = tau

    def fit(self, views: list[ArrayLike], y: None = None) -> SCCA_PMD:
        """Fit the SCCA_PMD model.

        Args:
            views: List of arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
        """
        # Store processed tau for use in _update_weight
        self._tau: list[float] = []  # set in super().fit via _setup_fit
        return super().fit(views, y)  # type: ignore[return-value]

    def _setup_tau(self) -> list[float]:
        """Compute per-view L1 bounds from tau and feature dimensions.

        Returns:
            List of L1 bound values, one per view.
        """
        tau_ = perview_parameter("tau", self.tau, 1.0, self.n_views_)
        return [t * np.sqrt(p) for t, p in zip(tau_, self.n_features_in_)]

    def _fit_single(
        self,
        views: list[np.ndarray],
        w: list[np.ndarray],
        d: int,
    ) -> None:
        """Run the ALS loop; set per-view L1 bounds first.

        Args:
            views: Deflated view arrays.
            w: Weight vectors (updated in-place).
            d: Current latent dimension index.
        """
        self._l1_bounds = self._setup_tau()
        super()._fit_single(views, w, d)

    def _update_weight(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """Update weight for view i with L1 bisection.

        Args:
            views: Current view arrays.
            weights: Current weight vectors.
            i: View index to update.

        Returns:
            Sparse normalised weight vector for view i.
        """
        target = _target_score(views, weights, i)
        raw = views[i].T @ target
        return _bisect_threshold(raw, self._l1_bounds[i])


# ---------------------------------------------------------------------------
# SCCA_ADMM — ADMM-based sparse CCA (Suo 2017)
# ---------------------------------------------------------------------------


class SCCA_ADMM(_BaseIterative):
    r"""Sparse CCA via Alternating Direction Method of Multipliers.

    Solves the sparse CCA problem using ADMM to enforce both the L1 sparsity
    constraint on weight vectors and the unit-norm constraint on the projected
    scores simultaneously.

    For view :math:`i` the ADMM sub-problems are:

    * :math:`\mathbf{w}_i` update — proximal gradient step w.r.t. the data
      fidelity term.
    * Auxiliary variable :math:`\mathbf{z}_i` update — soft thresholding.
    * Dual variable update.

    References:
        Suo, X., Mineiro, P., & Anandkumar, A. (2017). Sparse canonical
        correlation analysis. *arXiv:1705.10865*.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        tau: L1 regularisation weight(s). Default is 0.1.
        mu: ADMM penalty parameter (step size). Default is 1.0.
        max_iter: Maximum outer iterations. Default is 500.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = SCCA_ADMM(tau=0.1, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        tau: float | list[float] = 0.1,
        mu: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            center=center,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.tau = tau
        self.mu = mu

    def _fit_single(
        self,
        views: list[np.ndarray],
        w: list[np.ndarray],
        d: int,
    ) -> None:
        """Run the ADMM loop for a single latent dimension.

        Args:
            views: Deflated view arrays.
            w: Weight vectors (updated in-place).
            d: Current latent dimension index.
        """
        tau_ = perview_parameter("tau", self.tau, 0.1, len(views))
        n = views[0].shape[0]
        z = [wi.copy() for wi in w]
        eta = [np.zeros_like(wi) for wi in w]
        for _iter in range(self.max_iter):
            w_prev = [wi.copy() for wi in w]
            # Compute gradient targets
            targets = [_target_score(views, w, i) for i in range(len(views))]
            for i in range(len(views)):
                XtX = views[i].T @ views[i]
                Xtarget = views[i].T @ targets[i]
                # w-update: proximal gradient
                gradient = XtX @ w[i] - Xtarget + self.mu * (w[i] - z[i] + eta[i])
                w[i] = w[i] - (gradient / (np.linalg.norm(XtX) / n + self.mu))
                # z-update: soft thresholding
                z[i] = soft_threshold(w[i] + eta[i], tau_[i] / self.mu)
                # Project z to unit ball if needed
                z_norm = np.linalg.norm(z[i])
                if z_norm > 1.0:
                    z[i] /= z_norm
                # dual update
                eta[i] = eta[i] + w[i] - z[i]
            # copy z to w and check convergence
            for i in range(len(views)):
                w[i] = z[i].copy()
            delta = max(np.linalg.norm(w[i] - w_prev[i]) for i in range(len(views)))
            if delta < self.tol:
                logger.debug("ADMM dim %d converged at iter %d", d, _iter)
                break

    def _update_weight(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """Not used — SCCA_ADMM overrides _fit_single directly.

        Args:
            views: View arrays (unused).
            weights: Weight vectors (unused).
            i: View index (unused).

        Returns:
            Current weight for view i (unchanged).
        """
        return weights[i]


# ---------------------------------------------------------------------------
# SCCA_IPLS — Iterative PLS (Mai & Zhang 2019)
# ---------------------------------------------------------------------------


class SCCA_IPLS(_BaseIterative):
    r"""Iterative PLS with elastic net penalty on weight vectors.

    Alternates between penalised regression sub-problems.  For view :math:`i`:

    .. math::

        \hat{\mathbf{w}}_i = \arg\min_{\mathbf{w}}
            \frac{1}{2n} \|X_i \mathbf{w} - \bar{\mathbf{s}}_{\neg i}\|_2^2
            + \alpha_i \Bigl(
                l_1 \|\mathbf{w}\|_1
                + \tfrac{1-l_1}{2} \|\mathbf{w}\|_2^2
            \Bigr)

    followed by a normalisation step to enforce unit variance of the score.

    References:
        Mai, Q., & Zhang, X. (2019). An iterative penalized least squares
        approach to sparse canonical correlation analysis. *Biometrics*,
        75(3), 734–744.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        alpha: Elastic net penalty strength(s). Default is 0.
        l1_ratio: Ratio of L1 to total penalty. 1 = lasso, 0 = ridge.
            Default is 1.
        max_iter: Maximum ALS iterations. Default is 500.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = SCCA_IPLS(alpha=0.1, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        alpha: float | list[float] = 0.0,
        l1_ratio: float | list[float] = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
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
        """Initialise per-view regressors and run ALS.

        Args:
            views: Deflated view arrays.
            w: Weight vectors (updated in-place).
            d: Current latent dimension index.
        """
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
        """Penalised regression update with score normalisation.

        Args:
            views: Current view arrays.
            weights: Current weight vectors.
            i: View index to update.

        Returns:
            Updated weight vector for view i.
        """
        target = _target_score(views, weights, i)
        reg = self._regressors[i]
        reg.fit(views[i], target)
        w_new = reg.coef_.copy()
        score = views[i] @ w_new
        score_std = score.std()
        if score_std > 1e-12:
            w_new /= score_std
        return w_new


# ---------------------------------------------------------------------------
# SCCA_Span — SpanCCA (Asteris 2016)
# ---------------------------------------------------------------------------


class SCCA_Span(_BaseIterative):
    r"""SpanCCA — sparse CCA via truncated power iteration.

    Solves sparse CCA by a sparse power iteration where each weight update
    retains only the ``span`` entries with the largest absolute values.

    References:
        Asteris, M., Khanna, R., Kyrillidis, A., & Dimakis, A. G. (2016).
        Bilinear approaches for online learning over large feature spaces.
        *NeurIPS 2016*. (SpanCCA algorithm).

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        span: Number of non-zero entries to retain per view.  Either a single
            int or a list.  Default is None (keep all — no sparsity).
        max_iter: Maximum ALS iterations. Default is 500.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = SCCA_Span(span=5, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        span: int | list[int] | None = None,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
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
        """Set per-view span values and run ALS.

        Args:
            views: Deflated view arrays.
            w: Weight vectors (updated in-place).
            d: Current latent dimension index.
        """
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
        """Hard-threshold update keeping top-span entries.

        Args:
            views: Current view arrays.
            weights: Current weight vectors.
            i: View index to update.

        Returns:
            Sparse normalised weight vector for view i.
        """
        target = _target_score(views, weights, i)
        raw = views[i].T @ target
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
# ElasticCCA — Elastic net CCA (Waaijenborg 2008)
# ---------------------------------------------------------------------------


class ElasticCCA(_BaseIterative):
    r"""Elastic net regularised CCA.

    Alternates between elastic net regression sub-problems, regressing each
    view's score against the sum of all other views' scores:

    .. math::

        \hat{\mathbf{w}}_i = \arg\min_{\mathbf{w}}
            \frac{1}{2n} \|X_i \mathbf{w} - \mathbf{s}_{\text{all}}\|_2^2
            + \alpha_i \Bigl(
                l_1 \|\mathbf{w}\|_1
                + \tfrac{1 - l_1}{2} \|\mathbf{w}\|_2^2
            \Bigr)

    where :math:`\mathbf{s}_{\text{all}} = \sum_j X_j \mathbf{w}_j / \|\cdot\|`.

    References:
        Waaijenborg, S., de Witt Hamer, P. C. V., & Zwinderman, A. H.
        (2008). Quantifying the association between gene expressions and
        DNA-markers by penalized canonical correlation analysis.
        *Statistical Applications in Genetics and Molecular Biology*, 7(1).

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        alpha: Elastic net regularisation strength. Default is 0.
        l1_ratio: L1 / total penalty ratio. Default is 0.5.
        max_iter: Maximum ALS iterations. Default is 500.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = ElasticCCA(alpha=0.1, l1_ratio=0.5, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        alpha: float | list[float] = 0.0,
        l1_ratio: float | list[float] = 0.5,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
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
        """Initialise per-view regressors and run ALS.

        Args:
            views: Deflated view arrays.
            w: Weight vectors (updated in-place).
            d: Current latent dimension index.
        """
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
        """Elastic net regression against sum-of-all-scores target.

        Args:
            views: Current view arrays.
            weights: Current weight vectors.
            i: View index to update.

        Returns:
            Updated weight vector for view i.
        """
        scores = np.stack([views[j] @ weights[j] for j in range(len(views))], axis=0)
        target = scores.sum(axis=0)
        norm = np.linalg.norm(target)
        if norm > 1e-12:
            target /= norm
        reg = self._regressors[i]
        reg.fit(views[i], target)
        return np.atleast_1d(reg.coef_).ravel()


# ---------------------------------------------------------------------------
# ParkhomenkoCCA — Parkhomenko 2009
# ---------------------------------------------------------------------------


class ParkhomenkoCCA(_BaseIterative):
    r"""Sparse CCA via soft-thresholding power iteration (Parkhomenko 2009).

    Uses a fixed soft-threshold :math:`\tau_i` rather than the adaptive
    bisection search of :class:`SCCA_PMD`:

    .. math::

        \mathbf{w}_i \leftarrow
            S_{\tau_i}(X_i^\top \bar{\mathbf{s}}_{\neg i})

    where :math:`S_\tau` is the element-wise soft-threshold operator.

    References:
        Parkhomenko, E., Tritchler, D., & Beyene, J. (2009). Sparse
        canonical correlation analysis with application to genomic data
        integration. *Statistical Applications in Genetics and Molecular
        Biology*, 8(1).

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        tau: Soft-threshold parameter(s). Default is 0.1.
        max_iter: Maximum ALS iterations. Default is 500.
        tol: Convergence tolerance. Default is 1e-6.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> model = ParkhomenkoCCA(tau=0.1, random_state=0).fit([X1, X2])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        tau: float | list[float] = 0.1,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
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
        """Set per-view tau values and run ALS.

        Args:
            views: Deflated view arrays.
            w: Weight vectors (updated in-place).
            d: Current latent dimension index.
        """
        self._tau_vals = perview_parameter("tau", self.tau, 0.1, len(views))
        super()._fit_single(views, w, d)

    def _update_weight(
        self,
        views: list[np.ndarray],
        weights: list[np.ndarray],
        i: int,
    ) -> np.ndarray:
        """Soft-threshold update with normalisation.

        Args:
            views: Current view arrays.
            weights: Current weight vectors.
            i: View index to update.

        Returns:
            Sparse normalised weight vector for view i.
        """
        target = _target_score(views, weights, i)
        raw = views[i].T @ target
        result = soft_threshold(raw, self._tau_vals[i])
        norm = np.linalg.norm(result)
        if norm > 1e-12:
            result /= norm
        return result


# ---------------------------------------------------------------------------
# Helper: regressor factory
# ---------------------------------------------------------------------------


def _make_regressors(
    alpha: list[float],
    l1_ratio: list[float],
    tol: float,
    random_state: int | None,
) -> list[Ridge | Lasso | ElasticNet]:
    """Build per-view sklearn regressors.

    Args:
        alpha: Per-view regularisation strengths.
        l1_ratio: Per-view L1/total penalty ratios.
        tol: Solver tolerance.
        random_state: Seed for reproducibility.

    Returns:
        List of fitted sklearn regressor objects (one per view).
    """
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
