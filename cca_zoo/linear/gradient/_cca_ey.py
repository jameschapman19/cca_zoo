"""CCAEY — Eckart-Young CCA, continuously blended with PLSEY via a ridge parameter."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import deprecated
from sklearn.utils._param_validation import Interval

from cca_zoo._utils._ey import (
    cheap_orthonormal_projection_weights,
    ey_cross_covariance,
    order_components,
    weight_gram_mean,
)
from cca_zoo.linear.gradient._base import BaseFullBatchEYModel


class CCAEY(BaseFullBatchEYModel):
    r"""Eckart-Young CCA for 2 or more views, ridge-blended with PLSEY.

    Minimises the unconstrained Eckart-Young (EY) objective directly on the
    raw (centred) views, with no manifold projection step and no upfront
    whitening: unlike classical CCA, which whitens each view before finding
    the correlated directions, the EY reformulation folds the
    orthonormalising pressure into the loss itself, so a full-batch
    preprocessing pass over the data covariance is never needed. This
    matches how the same underlying loss is used, unwhitened, by
    :class:`~cca_zoo.linear.gradient.PLSEY`, :class:`~cca_zoo.tree.TreeCCA`,
    and :class:`~cca_zoo.deep.DCCAEY`.

    For embeddings $Z_i = X_i W_i$ ($i = 1, \dots, M$, $M \ge 2$), let $C$
    and $V$ be the mean pairwise cross-covariance and mean auto-covariance
    across views (see :func:`cca_zoo._utils._ey.ey_cross_covariance`), and
    $B = \frac{1}{M}\sum_i W_i^\top W_i$ the mean weight Gram matrix
    (see :func:`cca_zoo._utils._ey.weight_gram_mean`). ``c`` blends the
    *within-view normalisation* between the data's own auto-covariance and
    the identity (in weight space, $W_i^\top I W_i = W_i^\top W_i$) —
    exactly the canonical-ridge blend $(1-c)X^\top X + cI$ already used by
    :class:`~cca_zoo.linear.rCCA`, translated into this unconstrained
    setting:

    $$
    V_c = (1 - c) V + c B, \qquad
    \mathcal{L}_{EY}(c) = -2 \operatorname{tr}(C - c V) + \operatorname{tr}(V_c V_c)
    $$

    ``c=0`` recovers plain (unregularised) ``CCAEY`` exactly; ``c=1``
    recovers :class:`~cca_zoo.linear.gradient.PLSEY`'s loss exactly (its
    reward excludes the $i=j$ terms that $\mathcal{L}_{EY}(0)$
    includes, and its penalty is purely $\operatorname{tr}(BB)$) —
    both endpoints, and the gradient at intermediate $c$, are verified
    against finite differences and against ``PLSEY``'s own independently
    verified gradient. This objective has the canonical directions as a
    stationary point without requiring an explicit orthonormality
    constraint, unlike a plain squared-projection-distance loss.

    Fit by full-batch L-BFGS-B
    (:meth:`~cca_zoo.linear.gradient._base.BaseFullBatchEYModel._fit_lbfgsb`)
    using the loss's exact analytic gradient. For mini-batch training on
    datasets too large for a full-batch gradient evaluation, see
    :class:`~cca_zoo.linear.gradient.StochasticCCAEY`.

    The loss is invariant to rotating every view's fitted embedding by a
    common orthogonal matrix, so the raw L-BFGS-B solution recovers the
    right canonical *subspace* but not individually ordered, canonically
    meaningful components. After fitting, weights are rotated into
    descending-correlation order by a cheap post-hoc
    :func:`~cca_zoo._utils._ey.order_components` step (a small $k \times k$
    eigendecomposition), matching the convention of exact
    eigendecomposition-based solvers like :class:`~cca_zoo.linear.MCCA`.

    Note:
        Unlike the exact, closed-form :class:`~cca_zoo.linear.rCCA` (where
        ``c=0`` is always numerically safe), optimising the raw,
        *unregularised* ($c=0$) objective can be poorly conditioned when the
        number of samples doesn't outnumber the number of features by a
        healthy margin, since nothing then bounds the weights in the data's
        near-null directions. If you see ``nan`` or diverging weights,
        increase ``c`` (a small value like 0.1-0.3 is usually enough).

    References:
        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default True.
        c: Ridge blend in ``[0, 1]`` between ``CCAEY`` (0) and ``PLSEY``
            (1). Default is 0 (standard, unregularised CCAEY); see the
            note above on numerical stability for high-dimensional data.
        max_iter: Maximum number of L-BFGS-B iterations. Default is 1000.
        tol: Convergence tolerance, passed to L-BFGS-B as ``ftol``. Default
            is 1e-6.
        random_state: Seed for reproducibility.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((1000, 20))
        >>> X2 = rng.standard_normal((1000, 15))
        >>> model = CCAEY(latent_dimensions=4, random_state=0)
        >>> model = model.fit([X1, X2])

        More than two views are supported directly:

        >>> X3 = rng.standard_normal((1000, 10))
        >>> model = CCAEY(latent_dimensions=4, random_state=0).fit([X1, X2, X3])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseFullBatchEYModel._parameter_constraints,
        "c": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        c: float = 0.0,
        max_iter: int = 1000,
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
        self.c = c

    def fit(self, views: list[ArrayLike], y: None = None) -> CCAEY:
        """Fit CCAEY by full-batch L-BFGS-B on the EY loss.

        Args:
            views: List of 2 or more arrays, each (n_samples, n_features_i).
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
        representations = [v @ w for v, w in zip(views_, self.weights_)]
        self.weights_ = order_components(self.weights_, representations, self.c)
        return self

    def _initial_weights(
        self, views: list[np.ndarray], rng: np.random.Generator
    ) -> list[np.ndarray]:
        """Cheap, data-informed initial weights (see class docstring note).

        Overrides :meth:`BaseFullBatchEYModel._initial_weights`'s plain
        weight-orthonormal default: gives exactly unit-variance,
        uncorrelated projections on the full dataset instead, matching this
        loss's own reward term at its fixed point (see
        :func:`cca_zoo._utils._ey.cheap_orthonormal_projection_weights`).
        """
        return cheap_orthonormal_projection_weights(
            views, self.latent_dimensions, None, rng
        )

    def _penalty_matrix(self, v_blend: np.ndarray) -> np.ndarray:
        r"""Cross-component matrix used by the gradient's decorrelation penalty.

        Identity hook: returns ``v_blend`` unchanged, so every component's
        penalty gradient sees every other component symmetrically -- the
        loss is then invariant to jointly rotating every view's embedding
        by any common orthogonal matrix (see
        :func:`~cca_zoo._utils._ey.order_components`), so a converged fit
        recovers the right subspace but not individually ordered
        components.

        Overridden by :class:`~cca_zoo.linear.gradient.StochasticCCAEY`
        (``ordered=True``) to mask ``v_blend`` to its upper triangle
        instead, so component $d$'s penalty only sees components $\le d$.
        That one-line change is a generalised-eigenproblem analogue of
        Sanger's rule (the Generalized Hebbian Algorithm): breaking the
        symmetry this way forces the *training dynamics themselves* to
        converge directly to ordered, individually meaningful components,
        rather than fixing up an arbitrary rotation after the fact.

        Args:
            v_blend: The blended penalty matrix ``(1 - c) * v_data + c * b``
                computed by :meth:`_derivative`.

        Returns:
            The (possibly masked) matrix to use in the penalty gradient.
        """
        return v_blend

    def _derivative(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> list[np.ndarray]:
        r"""Analytic gradient of $\mathcal{L}_{EY}(c)$ w.r.t. each $W_k$.

        Combines the chain-rule gradient through the embeddings (as for
        plain ``CCAEY``, scaled by ``(1 - c)`` plus a direct
        ``c``-scaled reward correction) with a *direct* weight-space
        gradient contribution from ``B``'s dependence on $W_k$ (as for
        ``PLSEY``, scaled by ``c``). Verified against finite differences
        for ``c`` in ``{0, 0.3, 0.5, 0.7, 1}``, and, at ``c=0``/``c=1``,
        against the unregularised ``CCAEY`` gradient and ``PLSEY``'s own
        gradient respectively (both matches exact).

        Every use of the blended penalty matrix ``v_blend`` is routed
        through :meth:`_penalty_matrix`, a no-op hook by default (see its
        own docstring for the ordered-training variant it enables).

        Args:
            views: Per-view arrays.
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
        penalty = self._penalty_matrix(v_blend)
        scale = 4.0 / (m * (n - 1))
        grads = []
        for k, (view, zk) in enumerate(zip(views, centred_reps)):
            view_c = view - view.mean(axis=0)
            z_term = scale * (c * zk + (1 - c) * (zk @ penalty) - total)
            grad = view_c.T @ z_term + (4.0 * c / m) * (weights[k] @ penalty)
            grads.append(grad)
        return grads

    def _objective(
        self,
        views: list[np.ndarray],
        representations: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> float:
        r"""Scalar $\mathcal{L}_{EY}(c)$."""
        del views
        c = self.c
        C, v_data = ey_cross_covariance(representations)
        b = weight_gram_mean(weights)
        v_blend = (1 - c) * v_data + c * b
        reward = C - c * v_data
        return float(-2.0 * np.trace(reward) + np.trace(v_blend @ v_blend))


@deprecated("Renamed to CCAEY for sklearn-style naming; use CCAEY instead.")
class CCA_EY(CCAEY):
    pass


@deprecated("CCAEY now supports 2 or more views directly; use CCAEY instead of MCCAEY.")
class MCCAEY(CCAEY):
    pass


@deprecated(
    "CCAEY now supports 2 or more views directly; use CCAEY instead of MCCA_EY."
)
class MCCA_EY(CCAEY):
    pass
