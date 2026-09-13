"""SupportVectorCCA -- hinge-capped-reward Eckart-Young CCA."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo.linear.gradient._base import BaseGradientModel


def _hinge_ey(
    representations: list[np.ndarray],
    means: list[np.ndarray],
    tau: float,
) -> tuple[float, list[np.ndarray]]:
    r"""EY loss with its reward capped per sample, and its gradient w.r.t. ``Z_i``.

    The plain EY loss (see :mod:`cca_zoo._utils._ey`) is
    $\mathcal{L}_{EY} = -2\operatorname{tr}(C) + \operatorname{tr}(VV)$, and
    $\operatorname{tr}(C)$ decomposes exactly as a sum of per-sample terms:

    $$
    \operatorname{tr}(C) = \frac{1}{M(n-1)}\sum_n R[n], \qquad
    R[n] = \Big\|\sum_i \tilde Z_i[n,:]\Big\|^2
    $$

    ($\tilde Z_i$ centred, $R[n]$ the same per-sample cross-view sum already
    computed as ``total`` inside :func:`cca_zoo._utils._ey.ey_grad_z`). This
    is EY's own reward, unbounded and linear in $R[n]$: more matched-sample
    alignment is always rewarded, with nothing to stop it growing arbitrarily
    on its own (only the *separate* penalty term keeps the overall objective
    well-behaved). Capping it at a per-sample target ``tau`` --
    $-2\operatorname{tr}(C) \to -\frac{2}{M(n-1)}\sum_n\min(R[n],\tau)$ --
    turns that unbounded linear credit into scikit-learn SVM's own hinge
    shape (up to an additive constant, since
    $\min(r,\tau)=\tau-\max(\tau-r,0)$): samples already at or above the
    target contribute *exactly* zero to the reward's gradient, the rest keep
    pushing as before. Chosen over an SVR-style symmetric tube because this
    quantity has no natural "too much" direction to also penalise -- more
    correlation is never bad, so a one-sided margin is the right shape, not a
    two-sided tube (see :class:`SupportVectorCCA`'s docstring).

    That zeroing is specific to the reward, not the full gradient below: the
    penalty term $\operatorname{tr}(VV)$ still contributes $z_kV$ for every
    sample regardless of capping, since (unlike scikit-learn SVM's $\|w\|^2$,
    which involves no data at all) $V$ is itself built from every sample's
    embedding. A capped sample stops being pulled towards the current
    consensus direction, but still helps anchor the fit's overall scale --
    it is not literally inert the way a non-support point in a real SVM's
    dual is.

    The penalty $\operatorname{tr}(VV)$ -- EY's own regularisation, already
    doing the job scikit-learn SVM's $\|w\|^2$ term does -- is left
    completely untouched, exactly as scikit-learn SVR/SVM only ever modify
    the *fit* term of ridge regression, never its regulariser.

    ``means`` and ``tau`` must both be genuinely fixed (not recomputed from
    a perturbed ``representations`` when verifying this gradient): unlike
    the plain EY loss, whose per-sample derivative is linear in the centred
    residual (so the correction from differentiating through the mean
    exactly cancels via translation invariance -- summing to zero is a
    property of a *linear* derivative), this loss's derivative is a
    (nonlinear) step function of $R[n]$, which has no such cancellation
    guarantee. Both are supplied externally and held fixed here, the same
    stop-gradient convention already used for ``HuberCCA``'s sample weights
    and this module's own ``tau`` self-calibration below.

    Args:
        representations: List of M arrays, each of shape (n_samples, k).
        means: Fixed per-view means, each of shape (k,).
        tau: Fixed per-sample reward cap.

    Returns:
        Tuple ``(objective, grad_z)``: scalar objective value, and a list of
        M gradient arrays each of shape (n_samples, k).
    """
    m = len(representations)
    n = representations[0].shape[0]
    centred = [z - mu for z, mu in zip(representations, means)]
    total = sum(centred)
    R = (total**2).sum(axis=1)
    active = (R < tau).astype(float)

    k = centred[0].shape[1]
    V = np.zeros((k, k))
    for zi in centred:
        V += zi.T @ zi / (n - 1)
    V /= m

    objective = float(
        np.trace(V @ V) - (2.0 / (m * (n - 1))) * np.minimum(R, tau).sum()
    )
    scale = 4.0 / (m * (n - 1))
    grad_z = [scale * (zc @ V - active[:, None] * total) for zc in centred]
    return objective, grad_z


def _self_calibrated_tau(
    representations: list[np.ndarray], means: list[np.ndarray], tau_mult: float
) -> float:
    r"""Per-batch reward cap, as a multiple of the batch's own mean ``R``.

    A fixed absolute cap would need re-tuning for every
    ``(n_views, latent_dimensions, batch_size)`` combination -- the same
    lesson already applied to ``HuberCCA``'s leverage cutoff and
    :func:`cca_zoo._utils._ey.ey_diag_hessian`'s percentile floor. Scaling by
    the batch's own mean ``R`` makes ``tau_mult`` a self-calibrating,
    portable knob instead: ``tau_mult=1`` caps roughly the better-than-
    average half of the batch every step, regardless of scale.

    Args:
        representations: List of M arrays, each of shape (n_samples, k).
        means: Fixed per-view means, each of shape (k,).
        tau_mult: Cap as a multiple of the batch's mean ``R``.

    Returns:
        The scalar cap ``tau``.
    """
    total = sum(z - mu for z, mu in zip(representations, means))
    R = (total**2).sum(axis=1)
    return float(tau_mult * R.mean())


class SupportVectorCCA(BaseGradientModel):
    r"""Support Vector CCA: hinge-capped-reward Eckart-Young CCA.

    The EY loss's reward term rewards matched-sample cross-view alignment
    without limit -- see :func:`_hinge_ey` for the exact per-sample
    decomposition of $\operatorname{tr}(C)$ this relies on. This estimator
    caps that reward per sample at a self-calibrated target ``tau`` (see
    :func:`_self_calibrated_tau`), which is algebraically a one-sided hinge:
    samples whose current cross-view alignment already meets the target
    contribute *exactly* zero to the reward's gradient -- genuine
    support-vector sparsity in that term, not a smooth downweighting like
    :class:`~cca_zoo.linear.gradient.HuberCCA`. The penalty term
    $\operatorname{tr}(VV)$ -- EY's own regulariser -- is left untouched, the
    same way scikit-learn's :class:`~sklearn.svm.SVR`/:class:`~sklearn.svm.SVC`
    only ever modify a regression/classification loss's *fit* term, never
    its $\|w\|^2$ regulariser. Unlike a real SVM's $\|w\|^2$, though,
    $V$ is itself built from every sample, so a capped sample still
    contributes through the penalty -- see :func:`_hinge_ey`'s docstring for
    why this estimator's sparsity is real but partial, not full inertness.

    Note:
        This uses a one-sided hinge (like :class:`~sklearn.svm.SVC`'s margin),
        not :class:`~sklearn.svm.SVR`'s symmetric epsilon-insensitive tube,
        because matched-sample alignment has no natural "too much" direction
        to also penalise -- unlike a regression residual, more correlation is
        never bad, so only a floor makes sense, not a two-sided band.

    Note:
        Because $\operatorname{tr}(C)$'s exact rewrite is in terms of the
        $(n,n)$ cross-Gram matrices $Z_i\tilde Z_j^\top$ (their diagonal, to
        be precise -- see :func:`_hinge_ey`), this loss is already expressed
        in fully kernel-native terms: substituting $Z_i = K_i\alpha_i$ for a
        data kernel matrix $K_i$ needs no further reformulation. Not yet
        implemented here (this is the primal, unkernelised estimator).

    Note:
        Once every sample meets the target the gradient is exactly zero and
        training stops -- by design, not a bug: this estimator seeks a
        direction that is "good enough" for (a self-calibrated notion of)
        most samples, not the single sharpest-correlation direction
        :class:`CCAEY` would keep refining towards.

    Note:
        Like plain ``CCAEY`` at its unregularised ``c=0`` (the penalty here
        is identical, untouched), gradient descent on this objective can
        diverge to ``nan`` when a mini-batch's samples don't outnumber the
        number of features by a healthy margin -- capping the reward does
        not fix this, since it is the *penalty* term's own conditioning at
        fault, not the reward. If you see ``nan`` weights, increase
        ``batch_size``.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        tau_mult: Reward cap as a multiple of the current batch's own mean
            per-sample reward (self-calibrating; see
            :func:`_self_calibrated_tau`). Smaller values cap more
            aggressively (more samples become non-support, more sparsity,
            less refinement); larger values approach plain ``CCAEY``.
            Default is 1.0.
        learning_rate: Gradient step size. Default is 1e-2.
        max_iter: Number of gradient steps. Default is 1000.
        batch_size: Mini-batch size. ``None`` uses the full dataset.
        tol: Convergence tolerance. Default is 1e-6.
        momentum: Momentum coefficient in ``[0, 1)``. Default is 0.9.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((5000, 200))
        >>> X2 = rng.standard_normal((5000, 150))
        >>> model = SupportVectorCCA(
        ...     latent_dimensions=4, batch_size=128, random_state=0
        ... )
        >>> model = model.fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseGradientModel._parameter_constraints,
        "tau_mult": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        tau_mult: float = 1.0,
        learning_rate: float = 1e-2,
        max_iter: int = 1000,
        batch_size: int | None = None,
        tol: float = 1e-6,
        momentum: float = 0.9,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            latent_dimensions=latent_dimensions,
            center=center,
            learning_rate=learning_rate,
            max_iter=max_iter,
            batch_size=batch_size,
            tol=tol,
            momentum=momentum,
            random_state=random_state,
        )
        self.tau_mult = tau_mult

    def fit(self, views: list[ArrayLike], y: None = None) -> SupportVectorCCA:
        """Fit SupportVectorCCA by mini-batch momentum gradient descent.

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
        self.weights_ = self._gradient_descent(views_, rng)
        return self

    def _batch_stats(
        self, representations: list[np.ndarray]
    ) -> tuple[list[np.ndarray], float]:
        means = [z.mean(axis=0) for z in representations]
        tau = _self_calibrated_tau(representations, means, self.tau_mult)
        return means, tau

    def _derivative(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> list[np.ndarray]:
        r"""Analytic gradient of the hinge-capped EY loss w.r.t. each $W_k$.

        Chain rule through $Z_k = X_k W_k$: identical in structure to
        :func:`cca_zoo._utils._ey.ey_grad_z`'s own weight-space contraction
        -- the plain centred view against the (fixed-statistics) gradient
        w.r.t. $Z_k$ from :func:`_hinge_ey`.
        """
        del weights
        means, tau = self._batch_stats(representations)
        _, grad_z = _hinge_ey(representations, means, tau)
        grads = []
        for view, gz in zip(views, grad_z):
            view_c = view - view.mean(axis=0)
            grads.append(view_c.T @ gz)
        return grads

    def _objective(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> float:
        r"""Scalar hinge-capped EY loss, used for the ``tol`` check."""
        del views, weights
        means, tau = self._batch_stats(representations)
        objective, _ = _hinge_ey(representations, means, tau)
        return objective
