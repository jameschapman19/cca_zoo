"""Multivariate adaptive regression spline CCA."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike
from scipy import sparse
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    _jacobi_scaled,
    ey_grad_z,
    ey_loss,
    penalised_basis_ey_closed_form,
    penalised_basis_ey_gep,
    penalised_basis_ey_min_loss,
)
from cca_zoo._utils._validation import perview_parameter

# A hinge factor (feature, knot, sign) is max(0, sign * (x[feature] - knot)).
_Factor = tuple[int, float, int]
_Term = tuple[_Factor, ...]

# Relative tolerance below which a candidate column counts as degenerate:
# identically zero on the training data (relative to the scale of the terms
# its norm is computed from), or already in the span of the current basis
# (relative to its own norm).
_DEGENERATE_TOL = 1e-8
# Forward-pass candidates, ranked by how much EY gradient they absorb, whose
# exact refit loss (earth's criterion, which it applies to every candidate)
# decides which is added. Scoring all of them exactly would take one
# eigenproblem each.
_N_RESCORE = 10
# Most knots per feature and parent under the default minspan (see
# _knot_spacing).
_DEFAULT_MAX_KNOTS = 20
# Bisection steps for the backward pass's constrained eigenvalues: each
# halves an interlacing bracket, so 60 reach machine precision.
_BISECTION_STEPS = 60
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


def _default_nk(n_features: int) -> int:
    """``earth``'s default ``nk``, less the intercept a centred view does not have."""
    return min(200, max(20, 2 * n_features))


def _knot_spacing(
    n_support: np.ndarray,
    n_features: int,
    minspan: int | None,
    endspan: int | None,
    interaction: bool,
) -> tuple[np.ndarray, int]:
    r"""Knot step and end exclusion for parents with the given support sizes.

    Friedman's (1991, eqs. 43 and 45) rules, as in ``earth``: knots at least
    $L = \lfloor -\log_2[-\ln(1 - \alpha) / (p N_m)] / 2.5 \rfloor$ support
    points apart and none within $L_e = \lfloor 3 - \log_2(\alpha / p)
    \rfloor$ of either end, with $\alpha = 0.05$, $p$ features and $N_m$
    support points; $L_e$ is doubled for an interaction parent. The default
    ``minspan=None`` widens $L$ to leave at most :data:`_DEFAULT_MAX_KNOTS`
    knots per feature.

    Args:
        n_support: Support sizes, any shape.
        n_features: Number of features in the view.
        minspan: ``None`` (default rule), ``0`` (Friedman's), or a fixed step.
        endspan: ``None`` (Friedman's) or a fixed end exclusion.
        interaction: Whether the parent is not the constant.

    Returns:
        ``(step, end)``: the step per support size, and the end exclusion.
    """
    alpha = 0.05
    support = np.maximum(np.asarray(n_support), 1)
    end = int(3 - np.log2(alpha / n_features)) if endspan is None else endspan
    end *= 2 if interaction else 1
    friedman = np.maximum(
        (-np.log2(-np.log(1 - alpha) / (n_features * support)) / 2.5).astype(int), 1
    )
    if minspan == 0:
        step = friedman
    elif minspan is None:
        spread = -(-np.maximum(support - 2 * end, 0) // _DEFAULT_MAX_KNOTS)
        step = np.maximum(friedman, spread)
    else:
        step = np.full_like(support, minspan)
    return step, end


def _knot_ranks(
    n_support: int,
    n_features: int,
    minspan: int | None,
    endspan: int | None,
    interaction: bool,
) -> np.ndarray:
    """Support ranks, in one feature's sort order, that may carry a knot."""
    step, end = _knot_spacing(
        np.array(n_support), n_features, minspan, endspan, interaction
    )
    ranks: np.ndarray = np.arange(end, n_support - end, int(step))
    return ranks


def _max_knot_slots(
    n_samples: int, n_features: int, minspan: int | None, endspan: int | None
) -> int:
    """Most knots any parent can have; smaller supports can have more."""
    support = np.arange(1, n_samples + 1)
    most = 0
    for interaction in (False, True):
        step, end = _knot_spacing(support, n_features, minspan, endspan, interaction)
        count = np.maximum(-(-(support - 2 * end) // step), 0)
        most = max(most, int(count.max()))
    return most


class _HingeScorer:
    r"""Scores every candidate hinge pair of one view without forming any.

    For a parent $u$ and hinge $h_t = u\,(x_j - t)_+$, every inner product
    the forward-pass score needs is

    $$
    \langle h_t, w\rangle
        = \sum_{x_{ij} \ge t} u_i w_i x_{ij} - t \sum_{x_{ij} \ge t} u_i w_i,
    $$

    two suffix sums over feature $j$'s sort order (Friedman, 1991, §3.9).
    Each parent's between-knot blocks are built once as a sparse matrix, so
    every parent, feature and knot is scored by one sparse product and a
    cumulative sum. The basis is kept orthonormal (``q``) and each
    candidate's projections onto it are cached as three scalars, so memory
    is linear in the number of parents.
    """

    def __init__(
        self,
        X: np.ndarray,
        minspan: int | None = None,
        endspan: int | None = None,
    ) -> None:
        n, p = X.shape
        self.X = X
        self.minspan = minspan
        self.endspan = endspan
        self.n_slots = _max_knot_slots(n, p, minspan, endspan)
        self.q = np.zeros((n, 0))
        self._parents = np.zeros((n, 0))
        # Per (knot, feature, parent): which candidates may be scored, their
        # knot values, and each parent's block matrices (weights u, u * x).
        self._allowed = np.zeros((self.n_slots, p, 0), dtype=bool)
        self.knots = np.zeros((self.n_slots, p, 0))
        self._blocks: list[tuple[sparse.csr_array, sparse.csr_array]] = []
        self._stacked: tuple[sparse.csr_array, sparse.csr_array]
        # Per (knot, feature, parent): ||h_a||^2, ||h_b||^2, sum h_a, sum h_b,
        # the scale of the terms cancelling in those norms, ||q^T h_a||^2,
        # ||q^T h_b||^2, (q^T h_a) . (q^T h_b).
        self._stats = np.zeros((self.n_slots, p, 0, 8))
        self._add_parents(np.ones((n, 1)), np.ones((p, 1), dtype=bool))

    def _parent_blocks(
        self, u: np.ndarray, interaction: bool
    ) -> tuple[np.ndarray, np.ndarray, list[sparse.csr_array]]:
        """One parent's candidate knots and its u-weighted block matrices.

        Returns:
            ``(knots, valid, blocks)``: knot values, shape (n_slots, n_features),
            padded past the parent's own count; which slots are real, shape
            (n_slots,); and the block matrices with data ``u * x**p`` for
            ``p = 0, 1`` then ``u**2 * x**p`` for ``p = 0, 1, 2``.
        """
        X = self.X
        n, p = X.shape
        support = np.flatnonzero(u != 0)
        ranks = _knot_ranks(len(support), p, self.minspan, self.endspan, interaction)
        slots = len(ranks)
        xs = X[support]
        order = np.argsort(xs, axis=0)
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(len(support))[:, None], axis=0)
        block = np.searchsorted(ranks, rank, side="right") - 1  # -1: below all knots
        inside = block >= 0
        sample, feature = np.nonzero(inside)
        row = block[inside] * p + feature
        col = support[sample]
        x = xs[inside]
        weight = u[col]
        shape = (self.n_slots * p, n)
        blocks = [
            sparse.csr_array((data, (row, col)), shape=shape)
            for data in (weight, weight * x, weight**2, weight**2 * x, weight**2 * x**2)
        ]
        knots = np.zeros((self.n_slots, p))
        knots[:slots] = np.take_along_axis(xs, order[ranks], axis=0)
        return knots, np.arange(self.n_slots) < slots, blocks

    def _block_suffix(
        self, block: sparse.csr_array, w: np.ndarray, n_parents: int
    ) -> np.ndarray:
        """Suffix sums over knots of the parents' blocks times ``w``.

        Returns shape (n_knots, n_features, n_parents, n_columns).
        """
        p = self.X.shape[1]
        sums = (block @ w).reshape(n_parents, self.n_slots, p, w.shape[1])
        suffix: np.ndarray = np.cumsum(sums.transpose(1, 2, 0, 3)[::-1], axis=0)[::-1]
        return suffix

    def _hinge_inner(
        self,
        blocks: tuple[sparse.csr_array, sparse.csr_array],
        parents: np.ndarray,
        knots: np.ndarray,
        w: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``<h_a, w>`` and ``<h_b, w>`` for every knot, feature, parent and column.

        Args:
            blocks: Stacked block matrices of the parents.
            parents: Raw parent values, shape (n_samples, n_parents).
            knots: Knot values, shape (n_knots, n_features, n_parents).
            w: Columns, shape (n_samples, c).

        Returns:
            Two arrays of shape (n_knots, n_features, n_parents, c).
        """
        n = self.X.shape[0]
        suffix0, suffix1 = (self._block_suffix(b, w, parents.shape[1]) for b in blocks)
        t = knots[..., None]
        inner_a = suffix1 - t * suffix0
        uw = parents[:, :, None] * w[:, None, :]
        total = (self.X.T @ uw.reshape(n, -1)).reshape(-1, *uw.shape[1:]) - t * uw.sum(
            axis=0
        )
        return inner_a, inner_a - total

    def _projection_stats(
        self,
        blocks: tuple[sparse.csr_array, sparse.csr_array],
        parents: np.ndarray,
        knots: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        """``||q'h_a||^2``, ``||q'h_b||^2`` and ``(q'h_a).(q'h_b)``, stacked last."""
        stats = np.zeros((*knots.shape, 3))
        for start in range(0, q.shape[1], _Q_CHUNK):
            qh_a, qh_b = self._hinge_inner(
                blocks, parents, knots, q[:, start : start + _Q_CHUNK]
            )
            stats[..., 0] += np.sum(qh_a**2, axis=-1)
            stats[..., 1] += np.sum(qh_b**2, axis=-1)
            stats[..., 2] += np.sum(qh_a * qh_b, axis=-1)
        return stats

    def _add_parents(self, parents: np.ndarray, allowed: np.ndarray) -> None:
        """Register new parents; the first is the constant, the rest interactions."""
        X = self.X
        first = self._parents.shape[1] == 0
        per_parent = [self._parent_blocks(u, interaction=not first) for u in parents.T]
        knots = np.stack([k for k, _, _ in per_parent], axis=-1)
        valid = np.stack([v for _, v, _ in per_parent], axis=-1)
        blocks = [
            sparse.vstack([b[i] for _, _, b in per_parent]).tocsr() for i in range(5)
        ]
        inner = (blocks[0], blocks[1])

        u2 = parents**2
        suffix0, suffix1, suffix2 = (
            self._block_suffix(b, np.ones((X.shape[0], 1)), parents.shape[1])[..., 0]
            for b in blocks[2:]
        )
        sq_a = suffix2 - 2 * knots * suffix1 + knots**2 * suffix0
        sq_all = (X**2).T @ u2 - 2 * knots * (X.T @ u2) + knots**2 * u2.sum(axis=0)
        # Both norms come out of this expansion by cancellation, so rounding
        # error scales with the magnitude of its terms, not with the result:
        # a hinge that is identically zero leaves noise of order eps * scale.
        scale = (
            (X**2).T @ u2 + 2 * np.abs(knots * (X.T @ u2)) + knots**2 * u2.sum(axis=0)
        )
        sum_a, sum_b = (
            h[..., 0]
            for h in self._hinge_inner(inner, parents, knots, np.ones((X.shape[0], 1)))
        )
        stats = np.concatenate(
            [
                np.stack([sq_a, sq_all - sq_a, sum_a, sum_b, scale], axis=-1),
                self._projection_stats(inner, parents, knots, self.q),
            ],
            axis=-1,
        )
        self._parents = np.column_stack([self._parents, parents])
        self._allowed = np.concatenate(
            [self._allowed, allowed[None, :, :] & valid[:, None, :]], axis=2
        )
        self.knots = np.concatenate([self.knots, knots], axis=2)
        self._stats = np.concatenate([self._stats, stats], axis=2)
        self._blocks.append(inner)
        self._stacked = (
            sparse.vstack([b[0] for b in self._blocks]).tocsr(),
            sparse.vstack([b[1] for b in self._blocks]).tocsr(),
        )

    def add_columns(
        self, columns: np.ndarray, parent_allowed: list[np.ndarray | None]
    ) -> None:
        """Extend the basis with accepted raw columns; some also become parents.

        Args:
            columns: New raw basis columns, shape (n_samples, c).
            parent_allowed: Per column, None if it cannot parent further terms,
                else a mask of the features it may be multiplied by.
        """
        new_q: list[np.ndarray] = []
        for col in (columns - columns.mean(axis=0)).T:
            basis = np.column_stack([self.q, *new_q])
            # Gram-Schmidt, twice for orthogonality to rounding.
            orthogonal = col
            for _ in range(2):
                orthogonal = orthogonal - basis @ (basis.T @ orthogonal)
            new_q.append(orthogonal / np.linalg.norm(orthogonal))
        new_q_arr = np.column_stack(new_q)
        self._stats[..., 5:] += self._projection_stats(
            self._stacked, self._parents, self.knots, new_q_arr
        )
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

        Returns:
            ``(ok_a, ok_b, ok_pair, aa, bb, ab)``, each of shape (n_knots,
            n_features, n_parents): whether the positive hinge, the negative
            hinge and the pair are non-degenerate, and the Gram entries of the
            two centred hinges after projecting out the basis.
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

    def best_pairs(
        self, grad: np.ndarray, n: int = 1
    ) -> list[tuple[float, int, int, float, tuple[bool, bool]]]:
        r"""Best ``n`` reflected hinge pairs for the gradient.

        A pair is scored by $\operatorname{tr}(G^\top P_H G)$, with $P_H$ the
        projection onto the pair orthogonalised against the basis: MARS's
        residual-sum-of-squares reduction with the residual replaced by the
        negative EY gradient $G$. A pair with one degenerate hinge is scored on
        the other alone.

        Args:
            grad: EY gradient of this view, shape (n_samples, k).
            n: Number of candidates to return.

        Returns:
            Up to ``n`` tuples ``(score, parent, feature, knot, keep)``, best
            first, with ``keep`` flagging which of the (positive, negative) hinges
            to add.
        """
        ok_a, ok_b, ok_pair, aa, bb, ab = self.gram()
        # <G, h_perp> = <G_perp, h>: projecting the gradient off the basis once
        # replaces projecting every candidate. G is not already orthogonal to
        # the basis (the ridge keeps the refit's gradient off zero).
        na, nb = self._hinge_inner(
            self._stacked,
            self._parents,
            self.knots,
            grad - self.q @ (self.q.T @ grad),
        )
        na_sq, nb_sq = np.sum(na**2, axis=-1), np.sum(nb**2, axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            pair = (bb * na_sq - 2 * ab * np.sum(na * nb, axis=-1) + aa * nb_sq) / (
                aa * bb - ab**2
            )
            single_a = np.where(ok_a, na_sq / aa, -np.inf)
            single_b = np.where(ok_b, nb_sq / bb, -np.inf)
        scores = np.where(ok_pair, pair, np.maximum(single_a, single_b))
        flat = scores.ravel()
        top = np.argsort(-flat, kind="stable")[:n]
        candidates = []
        for best in zip(*np.unravel_index(top[np.isfinite(flat[top])], scores.shape)):
            if ok_pair[best]:
                keep = (True, True)
            else:
                # Once x_j is linear in the basis, the two hinges differ by a
                # vector in its span and tie exactly; prefer the positive one
                # rather than let rounding decide.
                negative = single_b[best] > single_a[best] * (1 + _TIE_RTOL)
                keep = (not negative, bool(negative))
            knot_idx, feature, parent = (int(i) for i in best)
            candidates.append(
                (
                    float(scores[best]),
                    parent,
                    feature,
                    float(self.knots[knot_idx, feature, parent]),
                    keep,
                )
            )
        return candidates


def _constrained_top_eigenvalues(lam: np.ndarray, z: np.ndarray, k: int) -> np.ndarray:
    r"""Top ``k`` eigenvalues of a symmetric matrix restricted to ``v``'s complement.

    For $C = U \operatorname{diag}(\lambda) U^\top$ and unit $v$ with
    $z = U^\top v$, the number of compressed eigenvalues above $\mu$ is
    $\#\{\lambda_j > \mu\} - 1 + [g(\mu) < 0]$ with
    $g(\mu) = \sum_j z_j^2 / (\lambda_j - \mu)$. Interlacing brackets each
    eigenvalue, so bisection finds all of them, for every candidate, from one
    eigendecomposition.

    Args:
        lam: Eigenvalues of ``C``, ascending, shape (d,).
        z: ``U' v`` for each candidate, unit rows, shape (n_candidates, d).
        k: Number of top eigenvalues wanted.

    Returns:
        Shape (n_candidates, k), largest first, ``-inf`` beyond ``d - 1``.
    """
    d = lam.shape[0]
    rank = np.arange(1, k + 1)
    exists = rank <= d - 1
    lo = np.where(exists, lam[np.maximum(d - 1 - rank, 0)], 0.0)
    hi = np.where(exists, lam[d - rank], 0.0)
    lo = np.broadcast_to(lo, (z.shape[0], k)).copy()
    hi = np.broadcast_to(hi, (z.shape[0], k)).copy()
    z2 = z[:, None, :] ** 2
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.sum(z2 / (lam - mid[..., None]), axis=-1)
        above = np.sum(lam > mid[..., None], axis=-1) - 1 + (g < 0)
        lower = above >= rank
        lo = np.where(lower, mid, lo)
        hi = np.where(lower, hi, mid)
    return np.where(exists, 0.5 * (lo + hi), -np.inf)


def _backward_path(
    bases: list[np.ndarray], k: int, ridge: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """MARS backward pass on the EY loss, down to one column per view.

    Repeatedly deletes the column whose removal leaves the lowest refit
    ridge-EY loss. Deleting a column restricts the eigenproblem of
    :func:`~cca_zoo._utils._ey.penalised_basis_ey_gep` to a hyperplane, so
    every candidate's loss comes from one eigendecomposition per step
    (:func:`_constrained_top_eigenvalues`).

    Returns:
        ``(removed, loss)``: column indices in deletion order, and the refit
        loss after each number of deletions, from zero (``loss[0]``) on.
    """
    lhs, rhs, view = penalised_basis_ey_gep(*_jacobi_scaled(bases, ridge))
    active = np.ones(len(view), dtype=bool)
    removed: list[int] = []
    losses: list[float] = []
    while True:
        idx = np.flatnonzero(active)
        lam, vecs = scipy.linalg.eigh(lhs[np.ix_(idx, idx)], rhs[np.ix_(idx, idx)])
        if not losses:
            losses.append(-float(np.sum(np.maximum(lam[-k:], 0.0) ** 2)))
        counts = np.bincount(view[idx], minlength=len(bases))
        removable = np.flatnonzero(counts[view[idx]] > 1)
        if len(removable) == 0:
            break
        # The generalized eigenvectors are L^-T U, so row c is U^T L^-1 e_c:
        # the removal direction already in eigen-coordinates.
        z = vecs[removable]
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        mu = _constrained_top_eigenvalues(lam, z, k)
        loss = -np.sum(np.maximum(mu, 0.0) ** 2, axis=1)
        best = int(np.argmin(loss))
        removed.append(int(idx[removable[best]]))
        losses.append(float(loss[best]))
        active[removed[-1]] = False
    return np.array(removed, dtype=int), np.array(losses)


def _exact_loss(
    raw_bases: list[np.ndarray], view: int, k: int, ridge: list[float]
) -> Callable[[np.ndarray], float] | None:
    """Refit ridge-EY loss as a function of the columns ``view`` would gain.

    None while another view has no basis yet, when the loss cannot rank
    candidates.
    """
    if any(raw.shape[1] == 0 for i, raw in enumerate(raw_bases) if i != view):
        return None

    def loss(columns: np.ndarray) -> float:
        bases = [
            np.column_stack([raw, columns]) if i == view else raw
            for i, raw in enumerate(raw_bases)
        ]
        return penalised_basis_ey_min_loss(_standardised(bases), k, ridge)

    return loss


class _MarsEncoder:
    """Per-view MARS encoder: selected terms, their training means and coefficients."""

    def __init__(
        self,
        terms: list[_Term],
        basis_mean: np.ndarray,
        coef: np.ndarray,
    ) -> None:
        self.terms_ = terms
        self.basis_mean_ = basis_mean
        self.coef_ = coef
        self.k = coef.shape[1]

    def predict_new(self, X: np.ndarray) -> np.ndarray:
        """Encoder output for new data, shape (n, k)."""
        result: np.ndarray = (
            _evaluate_terms(X, self.terms_) - self.basis_mean_
        ) @ self.coef_
        return result


def _standardised(raw_bases: list[np.ndarray]) -> list[np.ndarray]:
    """Each view's basis at zero mean and unit variance, for the ridge."""
    return [(raw - raw.mean(axis=0)) / _nonzero(raw.std(axis=0)) for raw in raw_bases]


def _nonzero(scale: np.ndarray) -> np.ndarray:
    """``scale`` with zeros, a constant column's, replaced by one."""
    return np.where(scale > 0, scale, 1.0)


def _pls_scores(views: list[np.ndarray], k: int) -> list[np.ndarray]:
    """Each view's scores on the top ``k`` PLS directions of centred views.

    The leading eigenvectors of the between-view blocks of the stacked
    cross-product, as in multiview PLS.
    """
    splits = np.cumsum([v.shape[1] for v in views])[:-1]
    stacked = np.hstack(views)
    cross = stacked.T @ stacked
    for block in np.split(np.arange(stacked.shape[1]), splits):
        cross[np.ix_(block, block)] = 0.0
    size = cross.shape[0]
    _, vectors = scipy.linalg.eigh(cross, subset_by_index=(size - k, size - 1))
    return [v @ w for v, w in zip(views, np.split(vectors, splits))]


class MARSCCA(BaseModel):
    r"""Nonlinear CCA with multivariate adaptive regression spline encoders.

    Each view's encoder is a MARS model (Friedman, 1991), a linear
    combination of products of up to ``degree`` hinges
    $\max(0, \pm(x_j - t))$, fitted to minimise the ridge-penalised EY loss.
    As in R's ``earth``, a forward pass grows each basis by the hinge pair
    that best absorbs the EY gradient, refitting every view jointly in
    closed form, and a backward pass prunes it to ``nprune`` terms.
    ``earth``'s GCV has no EY counterpart, so choose ``nprune`` by
    cross-validation.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        degree: Maximum hinges per term; 1 is additive, 2 allows pairwise
            interactions. Per-view. Default is 1.
        nk: Maximum terms per view in the forward pass; None is ``earth``'s
            ``min(200, max(20, 2 * n_features))`` less its intercept.
            Per-view. Default is None.
        thresh: Stop the forward pass once a round lowers the loss by less
            than this fraction. Default is 0.001.
        minspan: Minimum support points between knots; 0 is Friedman's rule,
            None widens it to at most 20 knots per feature. Per-view. Default
            is None.
        endspan: Support points at either end that may not carry a knot,
            doubled for interactions; None is Friedman's rule. Per-view.
            Default is None.
        alpha: Ridge penalty on the coefficient of every basis function at
            unit variance, so that it ignores each feature's units. Per-view.
            Default is 0.01.
        nprune: Total terms, across views, kept by the backward pass; None
            keeps them all. Default is None.

    Attributes:
        encoders_: Fitted per-view encoders; ``encoders_[i].coef_`` has one
            row per entry of :meth:`basis_functions`.
        backward_path_: ``(view, term)`` in the order the backward pass
            deleted them.
        backward_loss_: Training EY loss after each number of deletions.
        n_removed_: Terms the backward pass pruned, the first ``n_removed_``
            of ``backward_path_``; zero when ``nprune`` is None.

    References:
        Chapman, J., Wang, H.-T., Wells, L., & Wiesner, J. (2021). CCA-Zoo: A
        collection of Regularized, Deep Learning based, Kernel, and
        Probabilistic CCA methods in a scikit-learn style framework. Journal
        of Open Source Software, 6(68), 3823.
        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.
        Friedman, J. H. (1991). Multivariate Adaptive Regression Splines.
        The Annals of Statistics, 19(1), 1-67.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.gam import MARSCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 5))
        >>> X2 = rng.standard_normal((200, 5))
        >>> model = MARSCCA(degree=2, nk=[10, 20]).fit([X1, X2])
        >>> len(model.basis_functions(0)) <= 10
        True
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "degree": [Interval(Integral, 1, None, closed="left"), "array-like"],
        "nk": [Interval(Integral, 1, None, closed="left"), "array-like", None],
        "thresh": [Interval(Real, 0, None, closed="left")],
        "minspan": [Interval(Integral, 0, None, closed="left"), "array-like", None],
        "endspan": [Interval(Integral, 0, None, closed="left"), "array-like", None],
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "nprune": [Interval(Integral, 1, None, closed="left"), None],
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        degree: int | list[int] = 1,
        nk: int | list[int | None] | None = None,
        thresh: float = 0.001,
        minspan: int | list[int | None] | None = None,
        endspan: int | list[int | None] | None = None,
        alpha: float | list[float] = 0.01,
        nprune: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.degree = degree
        self.nk = nk
        self.thresh = thresh
        self.minspan = minspan
        self.endspan = endspan
        self.alpha = alpha
        self.nprune = nprune

    def fit(self, views: list[ArrayLike], y: None = None) -> MARSCCA:
        """Fit the model: earth's forward pass, backward pass and final refit.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.

        Raises:
            ValueError: If ``nprune`` is below the number of views, or a view
                has no admissible knot.
        """
        views_ = self._setup_fit(views)
        m = self.n_views_
        if self.nprune is not None and self.nprune < m:
            raise ValueError(
                f"nprune={self.nprune} is fewer than the number of views "
                f"({m}); every view keeps at least one term."
            )
        k = self.n_components
        alpha_ = perview_parameter("alpha", self.alpha, 0.01, m)
        terms, raw_bases = self._forward_pass(views_, alpha_)

        removed, path_loss = _backward_path(_standardised(raw_bases), k, alpha_)
        view_of = np.repeat(np.arange(m), [len(t) for t in terms])
        column_of = np.concatenate([np.arange(len(t)) for t in terms])
        self.backward_path_: list[tuple[int, _Term]] = [
            (int(view_of[c]), terms[view_of[c]][column_of[c]]) for c in removed
        ]
        self.backward_loss_: np.ndarray = path_loss
        n_removed = 0 if self.nprune is None else max(len(view_of) - self.nprune, 0)
        gone = set(removed[:n_removed].tolist())
        keep = [
            np.array([c not in gone for c in np.flatnonzero(view_of == i)])
            for i in range(m)
        ]
        terms = [
            [t for t, kept in zip(ts, mask) if kept] for ts, mask in zip(terms, keep)
        ]
        raw_bases = [raw[:, mask] for raw, mask in zip(raw_bases, keep)]
        self.n_removed_: int = n_removed

        coefficients = penalised_basis_ey_closed_form(
            _standardised(raw_bases), k, alpha_
        )
        self.encoders_: list[_MarsEncoder] = [
            _MarsEncoder(t, raw.mean(axis=0), coef / raw.std(axis=0)[:, None])
            for t, raw, coef in zip(terms, raw_bases, coefficients)
        ]
        self._fit_maps_and_importances(views_)
        return self

    def _forward_pass(
        self, views: list[np.ndarray], alpha: list[float]
    ) -> tuple[list[list[_Term]], list[np.ndarray]]:
        """Earth's forward pass: grow each view's basis a hinge pair at a time.

        Each round adds, to every view still growing, the pair that most
        lowers the penalised EY loss given the other views' current fit, then
        refits every view in closed form. It stops at ``nk`` terms, or once a
        round lowers the loss by less than ``thresh``.

        Returns:
            Each view's terms and their raw (uncentred) basis columns.
        """
        k, m = self.n_components, len(views)
        max_degree = perview_parameter("degree", self.degree, 1, m)
        nk: list[int | None] = perview_parameter("nk", self.nk, None, m)
        max_terms = [
            _default_nk(X.shape[1]) if n is None else n for X, n in zip(views, nk)
        ]
        minspan: list[int | None] = perview_parameter("minspan", self.minspan, None, m)
        endspan: list[int | None] = perview_parameter("endspan", self.endspan, None, m)
        scorers = [_HingeScorer(X, a, b) for X, a, b in zip(views, minspan, endspan)]

        # earth starts from the intercept alone, where the EY gradient is
        # zero; start instead from linear PLS on the standardised views,
        # which, like earth, ignores each feature's units and the rows' order.
        representations = _pls_scores(
            [(X - X.mean(axis=0)) / _nonzero(X.std(axis=0)) for X in views], k
        )
        terms: list[list[_Term]] = [[] for _ in views]
        raw_bases = [np.zeros((X.shape[0], 0)) for X in views]
        parent_terms: list[list[_Term]] = [[()] for _ in views]
        growing = [True] * m
        previous_loss = None
        while any(growing):
            grads = ey_grad_z(representations)
            for i in [j for j in range(m) if growing[j]]:
                added = self._add_best_pair(
                    scorers[i],
                    terms[i],
                    parent_terms[i],
                    grads[i],
                    max_degree[i],
                    max_terms[i],
                    _exact_loss(raw_bases, i, k, alpha),
                )
                if added is None:
                    growing[i] = False
                else:
                    raw_bases[i] = np.column_stack([raw_bases[i], added])
                    growing[i] = len(terms[i]) < max_terms[i]

            empty = [i for i, raw in enumerate(raw_bases) if raw.shape[1] == 0]
            if empty:
                raise ValueError(
                    f"MARSCCA could not place a single hinge in view(s) {empty}: "
                    "every candidate knot is excluded or degenerate. Either the "
                    "view has too few samples for its endspan (Friedman's rule "
                    "keeps ~9-12 points free at each end) or its features are "
                    "constant; lower endspan or minspan for that view."
                )
            bases = _standardised(raw_bases)
            coefficients = penalised_basis_ey_closed_form(bases, k, alpha)
            representations = [b @ c for b, c in zip(bases, coefficients)]
            loss = ey_loss(representations)["objective"] + 0.5 * sum(
                a * float(np.sum(c**2)) for a, c in zip(alpha, coefficients)
            )
            if previous_loss is not None and previous_loss - loss < self.thresh * abs(
                loss
            ):
                break
            previous_loss = loss
        return terms, raw_bases

    @staticmethod
    def _add_best_pair(
        scorer: _HingeScorer,
        terms: list[_Term],
        parent_terms: list[_Term],
        grad: np.ndarray,
        max_degree: int,
        max_terms: int,
        exact_loss: Callable[[np.ndarray], float] | None,
    ) -> np.ndarray | None:
        """Append the best hinge pair to ``terms``.

        The :data:`_N_RESCORE` best candidates by gradient score are re-ranked by
        ``exact_loss`` when given.

        Returns:
            The new raw basis columns, shape (n_samples, 1 or 2), or None if no
            candidate is non-degenerate.
        """
        candidates = scorer.best_pairs(grad, _N_RESCORE if exact_loss else 1)
        if not candidates:
            return None
        options = []
        for _, parent, j, knot, keep in candidates:
            # With room for one more term only, a pair keeps its positive hinge.
            over_budget = len(terms) + sum(keep) > max_terms
            fitting = (True, False) if over_budget and keep[0] else keep
            new = [
                (*parent_terms[parent], (j, knot, sign))
                for sign, kept in zip((1, -1), fitting)
                if kept
            ]
            options.append((new, _evaluate_terms(scorer.X, new)))
        if exact_loss is not None and len(options) > 1:
            new_terms, columns = min(options, key=lambda o: exact_loss(o[1]))
        else:
            new_terms, columns = options[0]

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

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        return self.encoders_[view].predict_new(centred)

    def _feature_importances(self, views: list[np.ndarray]) -> list[np.ndarray]:
        """``earth``'s ``evimp``, with the EY loss in place of the RSS.

        Each backward-pass subset's loss decrease over the next smaller one is
        credited to every feature it uses.
        """
        members: list[set[tuple[int, _Term]]] = [
            {(i, t) for i, enc in enumerate(self.encoders_) for t in enc.terms_}
        ]
        for view, term in self.backward_path_[self.n_removed_ :]:
            members.append(members[-1] - {(view, term)})
        losses = [*self.backward_loss_[self.n_removed_ :], 0.0]
        importance = [np.zeros(p) for p in self.n_features_per_view_]
        for s, subset in enumerate(members):
            used = {(view, f) for view, term in subset for f, _, _ in term}
            for view, feature in used:
                importance[view][feature] += losses[s + 1] - losses[s]
        return importance

    def basis_functions(self, view: int) -> list[str]:
        """Readable form of one view's selected basis functions.

        Knots are in raw feature units, e.g. ``"h(x3 - 0.52) * h(1.1 - x0)"``
        with ``h(u) = max(0, u)``.

        Args:
            view: Index of the view.

        Returns:
            One string per basis function, matching the rows of
            ``encoders_[view].coef_``.
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
