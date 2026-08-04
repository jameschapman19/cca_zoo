"""CCA_EY — Eckart-Young CCA, continuously blended with PLS_EY via a ridge parameter."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._utils._ey import ey_cross_covariance, weight_gram_mean
from cca_zoo.linear.gradient._base import BaseGradientModel


class CCA_EY(BaseGradientModel):
    r"""Eckart-Young CCA for large-scale data, ridge-blended with PLS_EY.

    Optimises the unconstrained Eckart-Young (EY) objective by mini-batch
    momentum gradient descent directly on the raw (centred) views, with no
    manifold projection step and no upfront whitening: unlike classical CCA,
    which whitens each view before finding the correlated directions, the EY
    reformulation folds the orthonormalising pressure into the loss itself,
    so a full-batch preprocessing pass over the data is never needed. This
    matches how the same underlying loss is used, unwhitened, by
    :class:`~cca_zoo.linear.gradient.PLS_EY`, :class:`~cca_zoo.tree.TreeCCA`,
    and :class:`~cca_zoo.deep.DCCA_EY`.

    For embeddings $Z_i = X_i W_i$, let $C$ and $V$ be the mean
    pairwise cross-covariance and mean auto-covariance across views (see
    :func:`cca_zoo._utils._ey.ey_cross_covariance`), and
    $B = \frac{1}{M}\sum_i W_i^\top W_i$ the mean weight Gram matrix
    (see :func:`cca_zoo._utils._ey.weight_gram_mean`). ``c`` blends the
    *within-view normalisation* between the data's own auto-covariance and
    the identity (in weight space, $W_i^\top I W_i = W_i^\top W_i$) —
    exactly the canonical-ridge blend $(1-c)X^\top X + cI$ already used by
    :class:`~cca_zoo.linear.rCCA`, translated into this unconstrained,
    stochastic setting:

    $$
    V_c = (1 - c) V + c B, \qquad
    \mathcal{L}_{EY}(c) = -2 \operatorname{tr}(C - c V) + \operatorname{tr}(V_c V_c)
    $$

    ``c=0`` recovers plain (unregularised) ``CCA_EY`` exactly; ``c=1``
    recovers :class:`~cca_zoo.linear.gradient.PLS_EY`'s loss exactly (its
    reward excludes the $i=j$ terms that $\mathcal{L}_{EY}(0)$
    includes, and its penalty is purely $\operatorname{tr}(BB)$) —
    both endpoints, and the gradient at intermediate $c$, are verified
    against finite differences and against ``PLS_EY``'s own independently
    verified gradient. This objective has the canonical directions as a
    stationary point without requiring an explicit orthonormality
    constraint, unlike a plain squared-projection-distance loss.

    Note:
        Unlike the exact, closed-form :class:`~cca_zoo.linear.rCCA` (where
        ``c=0`` is always numerically safe), gradient descent on the raw,
        *unregularised* ($c=0$) objective can diverge to ``nan`` when a
        mini-batch's samples don't outnumber the number of features by a
        healthy margin — e.g. ``n_features`` approaching or exceeding
        ``batch_size`` — since nothing then bounds the weights in the
        data's near-null directions. If you see ``nan`` weights, increase
        ``c`` (a small value like 0.1-0.3 is usually enough) or
        ``batch_size`` rather than assuming the model doesn't apply to your
        data.

    References:
        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        c: Ridge blend in ``[0, 1]`` between ``CCA_EY`` (0) and ``PLS_EY``
            (1). Default is 0 (standard, unregularised CCA_EY); see the
            note above on numerical stability for high-dimensional data.
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
        >>> model = CCA_EY(latent_dimensions=4, batch_size=128, random_state=0)
        >>> model = model.fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseGradientModel._parameter_constraints,
        "c": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        c: float = 0.0,
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
        self.c = c

    def fit(self, views: list[ArrayLike], y: None = None) -> CCA_EY:
        """Fit CCA_EY by mini-batch momentum gradient descent.

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

    def _derivative(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> list[np.ndarray]:
        r"""Analytic gradient of $\mathcal{L}_{EY}(c)$ w.r.t. each $W_k$.

        Combines the chain-rule gradient through the embeddings (as for
        plain ``CCA_EY``, scaled by ``(1 - c)`` plus a direct
        ``c``-scaled reward correction) with a *direct* weight-space
        gradient contribution from ``B``'s dependence on $W_k$ (as for
        ``PLS_EY``, scaled by ``c``). Verified against finite differences
        for ``c`` in ``{0, 0.3, 0.5, 0.7, 1}``, and, at ``c=0``/``c=1``,
        against the unregularised ``CCA_EY`` gradient and ``PLS_EY``'s own
        gradient respectively (both matches exact).

        Args:
            views: Mini-batch of view arrays.
            representations: Current embeddings.
            weights: Current weight matrices.

        Returns:
            List of gradient matrices, one per view.
        """
        m = len(views)
        n = views[0].shape[0]
        c = self.c
        centred_reps = [z - z.mean(axis=0) for z in representations]
        total = sum(centred_reps)
        _, v_data = ey_cross_covariance(representations)
        b = weight_gram_mean(weights)
        v_blend = (1 - c) * v_data + c * b
        scale = 4.0 / (m * (n - 1))
        grads = []
        for k, (view, zk) in enumerate(zip(views, centred_reps)):
            view_c = view - view.mean(axis=0)
            z_term = scale * (c * zk + (1 - c) * (zk @ v_blend) - total)
            grad = view_c.T @ z_term + (4.0 * c / m) * (weights[k] @ v_blend)
            grads.append(grad)
        return grads

    def _objective(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> float:
        r"""Scalar $\mathcal{L}_{EY}(c)$, used for the ``tol`` convergence check."""
        del views
        c = self.c
        C, v_data = ey_cross_covariance(representations)
        b = weight_gram_mean(weights)
        v_blend = (1 - c) * v_data + c * b
        reward = C - c * v_data
        return float(-2.0 * np.trace(reward) + np.trace(v_blend @ v_blend))
