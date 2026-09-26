"""MARSCCA — multivariate-adaptive-regression-spline Canonical Correlation Analysis."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    _jacobi_scaled,
    cheap_orthonormal_projection_weights,
    ey_grad_z,
    ridge_basis_ey_closed_form,
    ridge_basis_ey_gep,
)
from cca_zoo._utils._validation import perview_parameter, validate_views

# A hinge factor (feature, knot, sign) is max(0, sign * (x[feature] - knot)).
_Factor = tuple[int, float, int]
_Term = tuple[_Factor, ...]

# Relative tolerance below which a candidate column counts as degenerate:
# identically zero on the training data (relative to the scale of the terms
# its norm is computed from), or already in the span of the current basis
# (relative to its own norm).
_DEGENERATE_TOL = 1e-8
# Relative gap below which two single-hinge scores count as tied.
_TIE_RTOL = 1e-9
# Basis columns processed at once when computing a new parent's projection
# statistics, bounding that step's transient memory.
_Q_CHUNK = 16


def _evaluate_terms(X: np.ndarray, terms: list[_Term]) -> np.ndarray:
    """Raw (uncentred) MARS basis: one column per product-of-hinges term."""
    basis = np.ones((X.shape[0], len(terms)))
    for col, term in enumerate(terms):
        for feature, knot, sign in term:
            basis[:, col] *= np.maximum(0.0, sign * (X[:, feature] - knot))
    return basis


class _HingeScorer:
    r"""Scores every candidate hinge pair of one view without forming any.

    For a parent term $u$ (ones for the constant) and a hinge
    $h_t = u\,(x_j - t)_+$, every quantity the forward-pass score needs is
    an inner product $\langle h_t, w\rangle$ with a vector $w$ (a constant,
    a column of the current basis, a column of the EY gradient, or $h_t$
    itself), and

    $$
    \langle h_t, w\rangle
        = \sum_{x_{ij} \ge t} u_i w_i x_{ij} - t \sum_{x_{ij} \ge t} u_i w_i,
    $$

    two *suffix sums* over feature $j$'s sort order — Friedman's (1991,
    §3.9) fast update. Three engineering choices keep each forward step
    cheap, all exact:

    - Which samples fall between consecutive candidate knots never changes
      during a fit, so it is built once as a sparse block-membership matrix
      of shape ``(n_knots * n_features, n_samples)`` (``n_samples *
      n_features`` nonzeros), plus copies weighted by $x$ and $x^2$. Every
      parent, feature and knot is then scored at once by sparse-times-dense
      products and a cumulative sum over the short knot axis — no Python
      loop, no ``(n_samples, n_features, ...)`` temporary.
    - The current basis is kept as an orthonormal ``q`` grown by
      Gram-Schmidt (applied twice, for stability), so existing columns never
      change. The score needs a candidate's products with ``q`` only through
      $\lVert q^\top h_a\rVert^2$, $\lVert q^\top h_b\rVert^2$ and
      $(q^\top h_a)\cdot(q^\top h_b)$ — sums over ``q``'s columns — so those
      scalars are cached per candidate and grown as columns arrive, while
      the gradient side uses $\langle G, h_\perp\rangle = \langle G_\perp,
      h\rangle$ with $G_\perp$ projected off ``q`` once per step. Memory is
      therefore linear in the number of parents, not in parents times basis
      columns, and a step computes only the new ``q`` columns against
      existing parents, every column against new parents (in chunks), and
      the projected gradient against every parent.
    - Each parent's hinge norms and sums depend on nothing else, so they
      are computed once per parent.

    State grows only through :meth:`add_columns`, called with each accepted
    basis column.
    """

    def __init__(self, X: np.ndarray, n_candidate_knots: int) -> None:
        n, p = X.shape
        rows = np.unique(
            np.linspace(0, n - 1, n_candidate_knots + 2)[1:-1].round().astype(int)
        )
        order = np.argsort(X, axis=0)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(n)[:, None], axis=0)
        block = np.searchsorted(rows, rank, side="right") - 1  # -1: below all knots
        inside = block >= 0
        sample, feature = np.nonzero(inside)
        row = block[inside] * p + feature
        x = X[inside]

        self.X = X
        self.knots: np.ndarray = np.take_along_axis(X, order[rows], axis=0)
        self._blocks = [
            sparse.csr_array((v, (row, sample)), shape=(len(rows) * p, n))
            for v in (np.ones_like(x), x, x**2)
        ]
        self.q = np.zeros((n, 0))
        self._parents = np.zeros((n, 0))
        self._allowed = np.zeros((p, 0), dtype=bool)
        # Per (knot, feature, parent): ||h_a||^2, ||h_b||^2, sum h_a, sum h_b,
        # the scale of the terms cancelling in those norms, ||q^T h_a||^2,
        # ||q^T h_b||^2, (q^T h_a) . (q^T h_b).
        self._stats = np.zeros((len(rows), p, 0, 8))
        self._add_parents(np.ones((n, 1)), np.ones((p, 1), dtype=bool))

    def _hinge_inner(
        self, parents: np.ndarray, w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """``<h_a, w>`` and ``<h_b, w>`` for every knot, feature, parent, column.

        Returns two arrays of shape (n_knots, n_features, n_parents, n_columns).
        """
        n, p = self.X.shape
        uw = (parents[:, :, None] * w[:, None, :]).reshape(n, -1)
        shape = (-1, p, parents.shape[1], w.shape[1])
        suffix0, suffix1 = (
            np.cumsum((b @ uw).reshape(shape)[::-1], axis=0)[::-1]
            for b in self._blocks[:2]
        )
        t = self.knots[:, :, None, None]
        inner_a = suffix1 - t * suffix0
        total = (self.X.T @ uw).reshape(shape[1:]) - t * uw.sum(axis=0).reshape(
            shape[2:]
        )
        return inner_a, inner_a - total

    def _projection_stats(self, parents: np.ndarray, q: np.ndarray) -> np.ndarray:
        """``||q^T h_a||^2``, ``||q^T h_b||^2``, ``(q^T h_a).(q^T h_b)``, stacked last.

        ``q`` is consumed in chunks of :data:`_Q_CHUNK` columns, so the
        transient per-column products never exceed that many columns.
        """
        stats = np.zeros((*self.knots.shape, parents.shape[1], 3))
        for start in range(0, q.shape[1], _Q_CHUNK):
            qh_a, qh_b = self._hinge_inner(parents, q[:, start : start + _Q_CHUNK])
            stats[..., 0] += np.sum(qh_a**2, axis=-1)
            stats[..., 1] += np.sum(qh_b**2, axis=-1)
            stats[..., 2] += np.sum(qh_a * qh_b, axis=-1)
        return stats

    def _add_parents(self, parents: np.ndarray, allowed: np.ndarray) -> None:
        """Register new parent terms, computing their per-candidate statistics."""
        X = self.X
        u2 = parents**2
        suffix = [
            np.cumsum((b @ u2).reshape(-1, X.shape[1], u2.shape[1])[::-1], axis=0)[::-1]
            for b in self._blocks
        ]
        t = self.knots[:, :, None]
        sq_a = suffix[2] - 2 * t * suffix[1] + t**2 * suffix[0]
        sq_all = (X**2).T @ u2 - 2 * t * (X.T @ u2) + t**2 * u2.sum(axis=0)
        # Both norms come out of this expansion by cancellation, so rounding
        # error scales with the magnitude of its terms, not with the result:
        # a hinge that is identically zero leaves noise of order eps * scale.
        scale = (X**2).T @ u2 + 2 * np.abs(t * (X.T @ u2)) + t**2 * u2.sum(axis=0)
        sum_a, sum_b = (
            inner[..., 0]
            for inner in self._hinge_inner(parents, np.ones((X.shape[0], 1)))
        )
        stats = np.concatenate(
            [
                np.stack([sq_a, sq_all - sq_a, sum_a, sum_b, scale], axis=-1),
                self._projection_stats(parents, self.q),
            ],
            axis=-1,
        )
        self._parents = np.column_stack([self._parents, parents])
        self._allowed = np.column_stack([self._allowed, allowed])
        self._stats = np.concatenate([self._stats, stats], axis=2)

    def add_columns(
        self, columns: np.ndarray, parent_allowed: list[np.ndarray | None]
    ) -> None:
        """Extend the basis with accepted raw columns; some also become parents.

        Args:
            columns: New raw (uncentred) basis columns, shape (n_samples, c).
            parent_allowed: One entry per column: None if it cannot parent
                further terms (``max_degree`` reached), else the features it
                may be multiplied by, shape (n_features,).
        """
        new_q: list[np.ndarray] = []
        for col in (columns - columns.mean(axis=0)).T:
            basis = np.column_stack([self.q, *new_q])
            for _ in range(2):
                col = col - basis @ (basis.T @ col)
            new_q.append(col / np.linalg.norm(col))
        new_q_arr = np.column_stack(new_q)
        self._stats[..., 5:] += self._projection_stats(self._parents, new_q_arr)
        self.q = np.column_stack([self.q, new_q_arr])

        parent_idx = [
            c for c, allowed in enumerate(parent_allowed) if allowed is not None
        ]
        if parent_idx:
            self._add_parents(
                columns[:, parent_idx],
                np.column_stack([parent_allowed[c] for c in parent_idx]),
            )

    def gram(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Every candidate's orthogonalised 2x2 Gram matrix, and which are usable.

        Independent of the gradient: a function of the basis and parents
        only.

        Returns:
            ``(ok_a, ok_b, ok_pair, aa, bb, ab)``, each of shape (n_knots,
            n_features, n_parents): whether the positive hinge, the negative
            hinge, and the pair are non-degenerate (and allowed), and the
            entries of the Gram matrix of the two centred hinges after
            projecting out the current basis.
        """
        n = self.X.shape[0]
        sq_a, sq_b, sum_a, sum_b, scale, qq_a, qq_b, qq_ab = np.moveaxis(
            self._stats, -1, 0
        )
        raw_aa = sq_a - sum_a**2 / n
        raw_bb = sq_b - sum_b**2 / n
        aa = raw_aa - qq_a
        bb = raw_bb - qq_b
        ab = -sum_a * sum_b / n - qq_ab
        ok_a = (
            self._allowed
            & (raw_aa > _DEGENERATE_TOL * scale)
            & (aa > _DEGENERATE_TOL * raw_aa)
        )
        ok_b = (
            self._allowed
            & (raw_bb > _DEGENERATE_TOL * scale)
            & (bb > _DEGENERATE_TOL * raw_bb)
        )
        # The pair must be non-degenerate as a whole, to the same standard as a
        # single hinge: its orthogonalised Gram matrix's smallest eigenvalue,
        # det / trace to first order, relative to the raw hinge norms. Judging
        # the two residuals only against each other lets two nearly-in-span
        # hinges that are also nearly parallel compound into a singular basis.
        det = aa * bb - ab**2
        ok_pair = (
            ok_a
            & ok_b
            & (det > _DEGENERATE_TOL * (aa + bb) * np.maximum(raw_aa, raw_bb))
        )
        return ok_a, ok_b, ok_pair, aa, bb, ab

    def best_pair(
        self, grad: np.ndarray
    ) -> tuple[float, int, int, float, tuple[bool, bool]]:
        r"""Best (parent, feature, knot) reflected hinge pair for the current gradient.

        Scores every candidate by how much of the current EY gradient $G$
        the pair $u\,(x_j - t)_+$, $u\,(t - x_j)_+$ can absorb once
        orthogonalised against the current basis ``q``:
        $\operatorname{tr}(G^\top P_H G)$, with $P_H$ the projection onto the
        orthogonalised pair. This is exactly classical MARS's forward-pass
        criterion — the reduction in residual sum of squares from adding
        the pair — with the least-squares residual replaced by the EY loss's
        negative gradient, the same functional-gradient reading
        :class:`~cca_zoo.tree.TreeCCA` uses to grow trees. The reflected
        hinge is $u\,(t - x_j)_+ = h_t - u\,(x_j - t)$, so its inner
        products follow from the same suffix sums and the full-sample
        totals, and the two hinges have disjoint support, so their raw inner
        product is zero.

        When one hinge of the pair is degenerate (identically zero on the
        training data because the parent term vanishes on that side of the
        knot, or already in the basis span), the other is scored alone, the
        same way ``earth`` drops the unused half of a pair.

        Args:
            grad: Current EY gradient for this view, shape (n_samples, k).

        Returns:
            Tuple ``(score, parent, feature, knot, keep)`` for the best
            candidate (``parent`` indexing parents in registration order,
            the constant first), ``keep`` flagging which of the (positive,
            negative) hinges to add; ``score`` is ``-inf`` when every
            candidate is degenerate.
        """
        ok_a, ok_b, ok_pair, aa, bb, ab = self.gram()
        # <G, h_perp> = <G_perp, h>: projecting the gradient off the basis once
        # replaces projecting every candidate. G is not already orthogonal to
        # the basis (the ridge keeps the refit's gradient off zero).
        na, nb = self._hinge_inner(self._parents, grad - self.q @ (self.q.T @ grad))
        na_sq, nb_sq = np.sum(na**2, axis=-1), np.sum(nb**2, axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            pair = (bb * na_sq - 2 * ab * np.sum(na * nb, axis=-1) + aa * nb_sq) / (
                aa * bb - ab**2
            )
            single_a = np.where(ok_a, na_sq / aa, -np.inf)
            single_b = np.where(ok_b, nb_sq / bb, -np.inf)
        scores = np.where(ok_pair, pair, np.maximum(single_a, single_b))
        best = np.unravel_index(np.argmax(scores), scores.shape)
        if ok_pair[best]:
            keep = (True, True)
        else:
            # Once x_j is linear in the basis, the two hinges differ by a
            # vector in its span and tie exactly; prefer the positive one
            # rather than let rounding decide.
            negative = single_b[best] > single_a[best] * (1 + _TIE_RTOL)
            keep = (not negative, negative)
        knot_idx, feature, parent = (int(i) for i in best)
        return (
            float(scores[best]),
            parent,
            feature,
            float(self.knots[knot_idx, feature]),
            keep,
        )


def _backward_eliminate(
    bases: list[np.ndarray], k: int, ridge: list[float], n_terms: int
) -> list[np.ndarray]:
    r"""MARS backward pass on the EY loss: which basis columns survive to ``n_terms``.

    Repeatedly deletes the column, from whichever view, whose removal leaves
    the lowest refit ridge-EY training loss, never a view's last column.
    The refit loss of deleting column $c$ is $-\sum \mu^2$ over the $k$
    largest positive eigenvalues of :func:`~cca_zoo._utils._ey.ridge_basis_ey_gep`
    restricted to the remaining columns, so every candidate is scored at
    once: the candidates' reduced matrices are stacked, reduced to standard
    form through a batched Cholesky factor of the right-hand side, and
    handed to one batched ``eigvalsh``.

    Returns:
        One boolean mask per view over its columns, True for survivors.
    """
    lhs, rhs, view = ridge_basis_ey_gep(*_jacobi_scaled(bases, ridge))
    active = np.ones(len(view), dtype=bool)
    while active.sum() > n_terms:
        idx = np.flatnonzero(active)
        counts = np.bincount(view[idx], minlength=len(bases))
        removable = np.flatnonzero(counts[view[idx]] > 1)
        remaining = np.array([np.delete(idx, c) for c in removable])
        rows, cols = remaining[:, :, None], remaining[:, None, :]
        chol = np.linalg.cholesky(rhs[rows, cols])
        half = np.linalg.solve(chol, lhs[rows, cols])
        standard = np.linalg.solve(chol, np.swapaxes(half, 1, 2))
        mu = np.linalg.eigvalsh(standard)[:, -k:]
        loss = -np.sum(np.maximum(mu, 0.0) ** 2, axis=1)
        active[idx[removable[np.argmin(loss)]]] = False
    return [active[view == i] for i in range(len(bases))]


class _MarsEncoder:
    """Per-view MARS encoder: a centred basis of products of hinge functions.

    Holds the terms selected by :class:`MARSCCA`'s forward pass, the
    training means of their raw columns, and their fitted coefficients.
    """

    def __init__(
        self,
        terms: list[_Term],
        basis_mean: np.ndarray,
        coef: np.ndarray,
        train_pred: np.ndarray,
    ) -> None:
        self.terms_ = terms
        self.basis_mean_ = basis_mean
        self.coef_ = coef
        self.k = coef.shape[1]
        self._train_pred = train_pred

    def predict(self) -> np.ndarray:
        """Encoder output on the training data, shape (n_samples, k)."""
        return self._train_pred

    def predict_new(self, X: np.ndarray) -> np.ndarray:
        """Encoder output for arbitrary (e.g. test) data, shape (n, k)."""
        result: np.ndarray = (
            _evaluate_terms(X, self.terms_) - self.basis_mean_
        ) @ self.coef_
        return result


class MARSCCA(BaseModel):
    r"""MARSCCA — nonlinear multiview CCA with MARS (adaptive hinge-spline) encoders.

    Learns one nonlinear encoder per view — a multivariate adaptive
    regression spline (MARS; Friedman, 1991), $f_i(x) = \sum_m b_{im}(x)
    B_{im}$, a linear combination of basis functions each of which is a
    product of up to ``max_degree`` hinges $\max(0, \pm(x_j - t))$ — that
    jointly minimise the ridge-penalised Eckart-Young (EY) objective (see
    :mod:`cca_zoo._utils._ey`, shared with :class:`~cca_zoo.gam.GAMCCA`,
    :class:`~cca_zoo.tree.TreeCCA`, and the linear ``*EY`` models).

    Where :class:`~cca_zoo.gam.GAMCCA` fixes its spline basis up front (a
    B-spline per feature at quantile knots), MARSCCA *grows* each view's
    basis greedily, as classical MARS does: every forward step scores every
    candidate reflected hinge pair $b(x)\max(0, x_j - t)$,
    $b(x)\max(0, t - x_j)$ — every existing term $b$ (or the constant) as
    parent, every feature $j$ not already in that parent, every candidate
    knot $t$ — by how much of the current EY gradient it can absorb
    (:meth:`_HingeScorer.best_pair`; classical MARS's residual-sum-of-squares
    reduction, with the residual replaced by the EY loss's negative
    gradient), adds the best pair to each view in turn, then refits every
    view's coefficients jointly on the enlarged bases. On a fixed basis the
    ridge-EY fit is a generalized eigenproblem
    (:func:`~cca_zoo._utils._ey.ridge_basis_ey_gep`), so each refit is its
    exact global optimum in closed form, not an iterative solve. Knots are
    therefore placed only where the cross-view signal needs them, and with
    ``max_degree >= 2`` a term can represent a genuine within-view
    interaction (e.g. $x_1 x_2$) that GAMCCA's additive structure cannot.

    The EY loss's all-zero embedding is a stationary point, so there is no
    gradient to select the very first terms against; every view is
    therefore warm-started with a random linear projection
    (:func:`~cca_zoo._utils._ey.cheap_orthonormal_projection_weights`),
    which the first refit replaces entirely.

    The forward pass deliberately overshoots, so, as in classical MARS, a
    backward pass prunes it: starting from every term the forward pass
    added, it repeatedly deletes the term (from whichever view) whose
    removal raises the refit training EY loss least, down to ``n_terms``
    terms in total. Because every refit is a closed-form eigenproblem, each
    deletion is exact — every candidate's refit loss is computed, all at
    once, from one batched eigenvalue decomposition. Unlike the forward
    sequence, the backward sequence can drop a stepping-stone term (say a
    lone hinge in $x_1$) once the interaction it led to has taken over its
    job.

    ``earth`` then picks the size by generalised cross-validation, a
    squared-error criterion with no EY-loss counterpart; its alternative,
    choosing the size along the backward sequence by cross-validation
    (``pmethod="cv"``), carries over exactly as a search over ``n_terms``.
    Pair it with :func:`~cca_zoo.model_selection.one_standard_error` to
    take the smallest model within one standard error of the best rather
    than the noisy maximum::

        GridSearchCV(
            MARSCCA(max_degree=2, max_terms=40),
            {"n_terms": [2, 4, 8, 12, 16, 24, 32, 48, 80]},
            refit=one_standard_error("n_terms"),
        )

    References:
        Friedman, J. H. (1991). Multivariate Adaptive Regression Splines.
        The Annals of Statistics, 19(1), 1-67.

        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Args:
        latent_dimensions: Number of latent components. Must not exceed the
            number of features in any view. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            Default is True.
        max_terms: Maximum number of basis functions per view (each forward
            step adds at most two). Either a single value applied to every
            view or a list of per-view values. Default is 20.
        max_degree: Maximum number of hinge factors in a single basis
            function — 1 gives an additive model, 2 allows pairwise
            interactions, and so on. Either a single value or a list of
            per-view values. Default is 1.
        n_candidate_knots: Number of candidate knots per feature: the
            training values at evenly spaced interior ranks. Per-step cost
            grows linearly in this; every interior data point
            (``n_samples - 2``, as in classical MARS) works, at roughly an
            order of magnitude more fit time than the default. Default is
            20.
        alpha: Ridge penalty strength applied to every basis coefficient.
            Either a single float or a list of per-view floats. Default is
            0.1.
        n_terms: Total number of terms, across all views, to keep after
            the backward pass (every view keeps at least one). None keeps
            every term the forward pass adds. Default is None.
        random_state: Seed for the initial linear warm start.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 5))
        >>> X2 = rng.standard_normal((200, 5))
        >>> model = MARSCCA(latent_dimensions=2).fit([X1, X2])
        >>> scores = model.transform([X1, X2])

        Pairwise interactions, with a larger basis for the second view:

        >>> model = MARSCCA(max_degree=2, max_terms=[10, 20]).fit([X1, X2])
        >>> len(model.basis_functions(0)) <= 10
        True
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "max_terms": [Interval(Integral, 1, None, closed="left"), "array-like"],
        "max_degree": [Interval(Integral, 1, None, closed="left"), "array-like"],
        "n_candidate_knots": [Interval(Integral, 1, None, closed="left")],
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "n_terms": [Interval(Integral, 1, None, closed="left"), None],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        max_terms: int | list[int] = 20,
        max_degree: int | list[int] = 1,
        n_candidate_knots: int = 20,
        alpha: float | list[float] = 0.1,
        n_terms: int | None = None,
        random_state: int = 0,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.n_candidate_knots = n_candidate_knots
        self.alpha = alpha
        self.n_terms = n_terms
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> MARSCCA:
        """Fit the MARSCCA model: greedy forward pass with joint refits.

        Args:
            views: List of 2 or more arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        views_ = self._setup_fit(views)
        k = self.latent_dimensions
        m = self.n_views_
        max_terms_ = perview_parameter("max_terms", self.max_terms, 20, m)
        max_degree_ = perview_parameter("max_degree", self.max_degree, 1, m)
        alpha_ = perview_parameter("alpha", self.alpha, 0.1, m)
        scorers = [_HingeScorer(X, self.n_candidate_knots) for X in views_]

        rng = np.random.default_rng(self.random_state)
        warm_start = cheap_orthonormal_projection_weights(views_, k, None, rng)
        representations = [X @ w for X, w in zip(views_, warm_start)]
        terms: list[list[_Term]] = [[] for _ in views_]
        raw_bases = [np.zeros((X.shape[0], 0)) for X in views_]
        parent_terms: list[list[_Term]] = [[()] for _ in views_]
        growing = [True] * m

        while any(growing):
            grads = ey_grad_z(representations)
            for i in range(m):
                if not growing[i]:
                    continue
                added = self._add_best_pair(
                    scorers[i],
                    terms[i],
                    parent_terms[i],
                    grads[i],
                    max_degree_[i],
                    max_terms_[i],
                )
                if added is None:
                    growing[i] = False
                    continue
                raw_bases[i] = np.column_stack([raw_bases[i], added])
                growing[i] = len(terms[i]) < max_terms_[i]

            bases = [raw - raw.mean(axis=0) for raw in raw_bases]
            coefficients = ridge_basis_ey_closed_form(bases, k, alpha_)
            representations = [b @ c for b, c in zip(bases, coefficients)]

        if self.n_terms is not None and self.n_terms < sum(map(len, terms)):
            if self.n_terms < m:
                raise ValueError(
                    f"n_terms={self.n_terms} is fewer than the number of views "
                    f"({m}); every view keeps at least one term."
                )
            keep = _backward_eliminate(bases, k, alpha_, self.n_terms)
            terms = [
                [t for t, kept in zip(ts, mask) if kept]
                for ts, mask in zip(terms, keep)
            ]
            raw_bases = [raw[:, mask] for raw, mask in zip(raw_bases, keep)]
            bases = [raw - raw.mean(axis=0) for raw in raw_bases]
            coefficients = ridge_basis_ey_closed_form(bases, k, alpha_)
            representations = [b @ c for b, c in zip(bases, coefficients)]

        self.encoders_: list[_MarsEncoder] = [
            _MarsEncoder(t, raw.mean(axis=0), coef, rep)
            for t, raw, coef, rep in zip(
                terms, raw_bases, coefficients, representations
            )
        ]
        return self

    @staticmethod
    def _add_best_pair(
        scorer: _HingeScorer,
        terms: list[_Term],
        parent_terms: list[_Term],
        grad: np.ndarray,
        max_degree: int,
        max_terms: int,
    ) -> np.ndarray | None:
        """Append the best-scoring hinge pair to ``terms``.

        ``parent_terms`` lists the terms ``scorer`` holds as parents, in its
        registration order; new parents are appended to both.

        Returns:
            The new raw basis column(s), shape (n_samples, 1 or 2), or None
            if no candidate is non-degenerate.
        """
        score, parent, j, knot, keep = scorer.best_pair(grad)
        if score == -np.inf:
            return None
        if len(terms) + sum(keep) > max_terms:
            keep = (True, False)
        new_terms: list[_Term] = [
            (*parent_terms[parent], (j, knot, sign))
            for sign, kept in zip((1, -1), keep)
            if kept
        ]
        columns = _evaluate_terms(scorer.X, new_terms)

        parent_allowed: list[np.ndarray | None] = []
        for term in new_terms:
            if len(term) < max_degree:
                allowed = np.ones(scorer.X.shape[1], dtype=bool)
                allowed[[feature for feature, _, _ in term]] = False
                parent_allowed.append(allowed)
                parent_terms.append(term)
            else:
                parent_allowed.append(None)
        scorer.add_columns(columns, parent_allowed)
        terms += new_terms
        return columns

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Project views into the latent space using the fitted encoders.

        Args:
            views: List of arrays, each (n_samples, n_features_i), matching
                the number of views passed to ``fit``.

        Returns:
            List of arrays, each (n_samples, latent_dimensions).

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            ValueError: If fewer than 2 views are provided.
        """
        check_is_fitted(self)
        validated = validate_views(views)
        centred = [v - m for v, m in zip(validated, self.means_)]
        return [enc.predict_new(v) for v, enc in zip(centred, self.encoders_)]

    def basis_functions(self, view: int) -> list[str]:
        """Human-readable form of one view's selected basis functions.

        Knots are reported in the raw (un-centred) feature units, e.g.
        ``"h(x3 - 0.52) * h(1.1 - x0)"`` with ``h(u) = max(0, u)``. Entry
        ``m`` corresponds to row ``m`` of ``encoders_[view].coef_``.

        Args:
            view: Index of the view.

        Returns:
            One string per basis function, in the order they were selected.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
        """
        check_is_fitted(self)
        means = self.means_[view]

        def factor(feature: int, knot: float, sign: int) -> str:
            raw_knot = knot + means[feature]
            if sign < 0:
                return f"h({raw_knot:.4g} - x{feature})"
            op = "-" if raw_knot >= 0 else "+"
            return f"h(x{feature} {op} {abs(raw_knot):.4g})"

        return [
            " * ".join(factor(*f) for f in term) for term in self.encoders_[view].terms_
        ]

    @property
    def weights(self) -> list[np.ndarray]:
        """Not implemented for MARSCCA.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            NotImplementedError: MARSCCA encoders are hinge-spline expansions,
                not linear weight matrices. Use :meth:`basis_functions` and
                ``encoders_[view].coef_`` instead.
        """
        check_is_fitted(self)
        raise NotImplementedError(
            "MARSCCA has no linear weight matrices; its encoders are "
            "multivariate adaptive regression splines. Use the "
            "`basis_functions` method (with `encoders_[view].coef_`) instead "
            "to inspect a fitted view's terms directly."
        )
