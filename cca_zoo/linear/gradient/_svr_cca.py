"""SupportVectorCCA -- epsilon-insensitive Eckart-Young CCA."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo.linear.gradient._base import BaseGradientModel


def _epsilon_insensitive_ey(
    representations: list[np.ndarray],
    means: list[np.ndarray],
    stds: list[np.ndarray],
    epsilon: float,
) -> tuple[float, list[np.ndarray]]:
    r"""Epsilon-insensitive multiview consensus loss and its gradient w.r.t. ``Z_i``.

    Each view is standardised (per component, using the *fixed* ``means``/
    ``stds`` supplied by the caller -- not differentiated through, the same
    stop-gradient convention :func:`cca_zoo.linear.gradient._huber_cca._weighted_ey`
    gives its sample weights) so all views are on comparable footing, then
    compared against the cross-view consensus (the mean of all M
    standardised views) via scikit-learn SVR's own epsilon-insensitive loss:

    $$
    S_i = \frac{Z_i - \mu_i}{\sigma_i}, \qquad
    \bar{S} = \frac{1}{M}\sum_i S_i, \qquad
    R_i = S_i - \bar{S}
    $$
    $$
    \mathcal{L} = \frac{1}{nM}\sum_i \sum_{n,k} \max(|R_i[n,k]| - \varepsilon, 0)
    $$

    A sample-component already within ``epsilon`` of the cross-view
    consensus sits in the loss's flat zone and gets *exactly* zero gradient
    -- unlike :func:`~cca_zoo.linear.gradient._huber_cca._weighted_ey`'s smooth
    downweighting, this is the genuine sparsity mechanism scikit-learn's
    :class:`~sklearn.svm.SVR` gives points already inside its epsilon-tube.

    Args:
        representations: List of M arrays, each of shape (n_samples, k).
        means: Fixed per-view, per-component means, each of shape (k,).
        stds: Fixed per-view, per-component standard deviations, each (k,).
        epsilon: Tolerance below which a residual contributes nothing.

    Returns:
        Tuple ``(objective, grad_z)``: scalar objective value, and a list of
        M gradient arrays each of shape (n_samples, k).
    """
    m = len(representations)
    n = representations[0].shape[0]
    standardised = [(z - mu) / s for z, mu, s in zip(representations, means, stds)]
    consensus = sum(standardised) / m
    residual = [s - consensus for s in standardised]
    subgrad = [np.sign(r) * (np.abs(r) > epsilon) for r in residual]
    mean_subgrad = sum(subgrad) / m

    objective = float(
        sum(np.maximum(np.abs(r) - epsilon, 0.0).sum() for r in residual) / (n * m)
    )
    grad_z = [(g - mean_subgrad) / (n * m) / s for g, s in zip(subgrad, stds)]
    return objective, grad_z


class SupportVectorCCA(BaseGradientModel):
    r"""Support Vector CCA: epsilon-insensitive Eckart-Young CCA.

    Where :class:`~cca_zoo.linear.gradient.HuberCCA` downweights every
    high-leverage sample smoothly (bounded but never zero), this estimator
    borrows scikit-learn :class:`~sklearn.svm.SVR`'s epsilon-insensitive loss
    directly: each view's embedding is standardised and compared against the
    cross-view consensus (the average of all views' standardised
    embeddings), and any sample-component already within ``epsilon`` of that
    consensus gets *exactly* zero gradient (see
    :func:`_epsilon_insensitive_ey`). Only the samples the current fit
    doesn't already explain within tolerance keep pushing the weights --
    the genuine "only a subset of the data matters" sparsity a support
    vector machine gives you, applied to the EY loss's own multiview
    consensus rather than to a single regression target.

    Note:
        This is a **primal**, unkernelised estimator -- the counterpart to
        scikit-learn's :class:`~sklearn.svm.LinearSVR`, not its kernelised
        :class:`~sklearn.svm.SVR`. A kernelised dual formulation (embeddings
        as ``K_i @ alpha_i``, genuine "support vectors" as training points
        with nonzero dual coefficients) is a natural extension of this
        estimator, not yet implemented.

    Note:
        Once every sample-component is within the epsilon-tube the gradient
        is exactly zero and training stops -- by design, not a bug: like
        ``SVR``, this estimator seeks *a* fit within tolerance, not the
        single best-fitting direction, so a looser ``epsilon`` can converge
        to a less sharply optimal direction than :class:`CCAEY` would.
        ``epsilon`` is in standardised (unit-variance) units, so scikit-learn
        ``SVR``'s default of ``0.1`` is a reasonable starting point here too.

    Note:
        Like plain ``CCAEY`` at its unregularised ``c=0``, gradient descent
        on this objective can be unstable when a mini-batch's samples don't
        outnumber the number of features by a healthy margin. If you see
        ``nan`` weights, increase ``batch_size``.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        epsilon: Tolerance (in standardised units) below which a
            sample-component's deviation from the cross-view consensus is
            ignored entirely. Default is 0.1.
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
        "epsilon": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        epsilon: float = 0.1,
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
        self.epsilon = epsilon

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
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        means = [z.mean(axis=0) for z in representations]
        stds = [z.std(axis=0) + 1e-9 for z in representations]
        return means, stds

    def _derivative(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> list[np.ndarray]:
        r"""Analytic gradient of the epsilon-insensitive EY loss w.r.t. each $W_k$.

        Chain rule through $Z_k = X_k W_k$: the loss is translation-invariant
        in each $Z_k$ (built entirely from centred, standardised
        quantities), so -- exactly as for :func:`cca_zoo._utils._ey.ey_grad_z`
        -- the weight-space gradient is the plain centred view contracted
        with the (fixed-statistics) gradient w.r.t. $Z_k$ from
        :func:`_epsilon_insensitive_ey`.
        """
        del weights
        means, stds = self._batch_stats(representations)
        _, grad_z = _epsilon_insensitive_ey(representations, means, stds, self.epsilon)
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
        r"""Scalar epsilon-insensitive EY loss, used for the ``tol`` check."""
        del views, weights
        means, stds = self._batch_stats(representations)
        objective, _ = _epsilon_insensitive_ey(
            representations, means, stds, self.epsilon
        )
        return objective
