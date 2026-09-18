"""HuberCCA -- bounded-influence Eckart-Young CCA."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._utils._ey import cheap_orthonormal_projection_weights
from cca_zoo.linear.gradient._base import BaseFullBatchEYModel


def _huber_sample_weight(representations: list[np.ndarray], delta: float) -> np.ndarray:
    r"""Per-sample Huber-style weight capping each sample's leverage.

    Each view's embedding is standardised to unit variance per component (so
    views/components with different natural scales contribute comparably),
    then a sample's leverage is its combined norm across all (view,
    component) pairs. The cutoff is ``delta`` *times the dataset's own
    median leverage*, not an absolute Z-score radius: an absolute radius
    would need re-tuning for every ``(n_views, latent_dimensions)``
    combination -- the same lesson already applied to
    :func:`cca_zoo._utils._ey.ey_diag_hessian`'s percentile floor. Scaling
    by the dataset's own median instead makes the cutoff self-calibrating
    and guarantees at least half the data keeps weight 1 whenever
    ``delta >= 1``.

    Samples within the cutoff keep weight 1; samples beyond it are
    downweighted in inverse proportion to their leverage, capping (never
    zeroing) their contribution -- like :class:`~sklearn.linear_model.HuberRegressor`,
    a *smooth* downweighting, not the exact zeroing a hinge or
    epsilon-insensitive loss would give points already meeting its target.

    Args:
        representations: List of M arrays, each of shape (n_samples, k).
        delta: Cutoff as a multiple of the dataset's median leverage.

    Returns:
        Array of shape (n_samples,), values in ``(0, 1]``.
    """
    standardised = [
        (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-12) for z in representations
    ]
    leverage = np.sqrt(sum((s**2).sum(axis=1) for s in standardised))
    cutoff = delta * np.median(leverage) + 1e-12
    result: np.ndarray = np.minimum(1.0, cutoff / (leverage + 1e-12))
    return result


def _weighted_ey(
    representations: list[np.ndarray], sample_weight: np.ndarray
) -> tuple[float, list[np.ndarray]]:
    r"""Sample-weighted EY objective and its gradient w.r.t. each ``Z_i``.

    Identical in structure to :func:`cca_zoo._utils._ey.ey_loss` and
    :func:`cca_zoo._utils._ey.ey_grad_z`, but every sample's contribution to
    the (centring, cross-covariance, auto-covariance) statistics is scaled
    by ``sample_weight``, with ``n - 1`` replaced by the effective sample
    size ``W - 1`` (``W = sum(sample_weight)``). ``sample_weight`` is treated
    as fixed here (not differentiated through) -- the same stop-gradient
    treatment IRLS gives its weights, recomputed from the previous step's
    representations by the caller.

    Args:
        representations: List of M arrays, each of shape (n_samples, k).
        sample_weight: Array of shape (n_samples,), fixed per-sample weights.

    Returns:
        Tuple ``(objective, grad_z)``: scalar objective value, and a list of
        M gradient arrays each of shape (n_samples, k).
    """
    m = len(representations)
    w = sample_weight
    w_sum = w.sum()
    means = [np.average(z, axis=0, weights=w) for z in representations]
    centred = [z - mu for z, mu in zip(representations, means)]
    total = sum(centred)

    k = centred[0].shape[1]
    C = np.zeros((k, k))
    V = np.zeros((k, k))
    for zi in centred:
        wzi = w[:, None] * zi
        V += wzi.T @ zi / (w_sum - 1)
        for zj in centred:
            C += wzi.T @ zj / (w_sum - 1)
    C /= m
    V /= m
    objective = float(-2.0 * np.trace(C) + np.trace(V @ V))

    scale = 4.0 / (m * (w_sum - 1))
    grad_z = [scale * w[:, None] * (zc @ V - total) for zc in centred]
    return objective, grad_z


class HuberCCA(BaseFullBatchEYModel):
    r"""Huber CCA: bounded-influence Eckart-Young CCA.

    Standard :class:`~cca_zoo.linear.gradient.CCAEY` weights every sample
    equally, so its cross- and auto-covariance estimates (and hence its
    gradient) can be dominated by a handful of high-leverage points -- their
    contribution to a quadratic statistic grows with the *square* of their
    magnitude, unbounded. This is the same weakness ordinary least squares
    has relative to :class:`~sklearn.linear_model.HuberRegressor`: quadratic
    loss growth lets outliers dominate, so Huber loss caps it to linear
    growth instead.

    ``HuberCCA`` reweights each sample by a Huber-style factor of its own
    leverage (see :func:`_huber_sample_weight`) before forming the EY
    cross- and auto-covariance statistics (see :func:`_weighted_ey`):
    samples within ``delta`` times the dataset's own median leverage keep
    weight 1, samples beyond it are downweighted so their contribution is
    capped rather than unbounded -- every sample still contributes something,
    just never an unbounded amount. Fit by full-batch L-BFGS-B
    (:meth:`~cca_zoo.linear.gradient._base.BaseFullBatchEYModel._fit_lbfgsb`),
    exactly as :class:`~cca_zoo.linear.gradient.CCAEY`; the leverage weight
    is recomputed from the current representations at every evaluation,
    the same stop-gradient (IRLS-style) treatment described in
    :func:`_weighted_ey`.

    Note:
        Like plain ``CCAEY`` at its unregularised ``c=0`` (this estimator
        has no ridge-blend ``c`` of its own), optimising this objective can
        be poorly conditioned when the number of samples doesn't outnumber
        the number of features by a healthy margin. If you see ``nan`` or
        diverging weights, this is the same caveat ``CCAEY`` documents, not
        specific to the Huber reweighting.

        Reweighting a CCA fit by each sample's own leverage is not a new
        idea: Filzmoser, Dehon & Croux (2000) and Branco, Croux, Filzmoser &
        Oliveira (2005) already reweight Wold's (1966) alternating-regression
        CCA fit by robust-distance-based weights on each side in turn
        ("Robust Alternating Regression"), and Croux & Dehon (2002) instead
        plug a minimum covariance determinant estimator directly into the
        classical covariance-matrix eigenproblem. ``HuberCCA``'s difference
        is mechanism, not the underlying bounded-influence idea: it
        reweights the *unconstrained EY loss's own* sample statistics
        directly (see :func:`_huber_sample_weight`, :func:`_weighted_ey`)
        rather than an alternating regression or a general-purpose
        covariance estimator, with a cutoff self-calibrated to the batch's
        own median leverage rather than an MVE- or M-scale-based threshold.

    References:
        Filzmoser, P., Dehon, C., & Croux, C. (2000). Outlier resistant
        estimators for canonical correlation analysis. In COMPSTAT:
        Proceedings in Computational Statistics 2000 (pp. 301-306).
        Physica-Verlag.

        Branco, J. A., Croux, C., Filzmoser, P., & Oliveira, M. R. (2005).
        Robust canonical correlations: A comparative study. Computational
        Statistics, 20(2), 203-229.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        delta: Huber cutoff, as a multiple of the dataset's median sample
            leverage; samples beyond it are downweighted. Values below 1
            downweight the majority of the data and are not recommended.
            Smaller values are more robust but discard more of the data's
            genuine signal. Default is 4.0.
        max_iter: Maximum number of L-BFGS-B iterations. Default is 1000.
        tol: Convergence tolerance, passed to L-BFGS-B as ``ftol``. Default
            is 1e-8 (see :class:`~cca_zoo.linear.gradient.CCAEY`'s docstring
            for why a loose ``ftol`` risks silent premature convergence).
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((1000, 20))
        >>> X2 = rng.standard_normal((1000, 15))
        >>> model = HuberCCA(latent_dimensions=4, random_state=0)
        >>> model = model.fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseFullBatchEYModel._parameter_constraints,
        "delta": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        delta: float = 4.0,
        max_iter: int = 1000,
        tol: float = 1e-8,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            center=center,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.delta = delta

    def fit(self, views: list[ArrayLike], y: None = None) -> HuberCCA:
        """Fit HuberCCA by full-batch L-BFGS-B on the Huber-weighted EY loss.

        Args:
            views: List of arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        self.weights_ = self._fit_lbfgsb(views_, rng)
        return self

    def _initial_weights(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Cheap, data-informed initial weights (see :class:`CCAEY`'s note).

        Gives exactly unit-variance, uncorrelated projections on the full
        dataset -- the natural starting point for this loss too, since
        clean, well-conditioned data has uniform sample weight everywhere
        (see :func:`_huber_sample_weight`).
        """
        return cheap_orthonormal_projection_weights(
            views, self.latent_dimensions, None, rng
        )

    def _derivative(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> list[np.ndarray]:
        r"""Analytic gradient of the bounded-influence EY loss w.r.t. each $W_k$.

        Chain rule through $Z_k = X_k W_k$: since the weighted mean of
        $Z_k$ equals the weighted mean of $X_k$ times $W_k$, the
        weighted-mean-centred view plays the same role
        :func:`~cca_zoo._utils._ey.ey_grad_z`'s plain centred view plays for
        ``CCAEY`` -- weighted-mean-centre the view, then contract with the
        (fixed-weight) gradient w.r.t. $Z_k$ from :func:`_weighted_ey`.
        """
        del weights
        sample_weight = _huber_sample_weight(representations, self.delta)
        _, grad_z = _weighted_ey(representations, sample_weight)
        grads = []
        for view, gz in zip(views, grad_z):
            view_mean = np.average(view, axis=0, weights=sample_weight)
            view_cw = view - view_mean
            grads.append(view_cw.T @ gz)
        return grads

    def _objective(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> float:
        r"""Scalar bounded-influence EY loss, used for the ``tol`` check."""
        del views, weights
        sample_weight = _huber_sample_weight(representations, self.delta)
        objective, _ = _weighted_ey(representations, sample_weight)
        return objective
