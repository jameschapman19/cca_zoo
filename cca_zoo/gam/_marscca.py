"""MARSCCA — multivariate-adaptive-regression-spline Canonical Correlation Analysis."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    cheap_orthonormal_projection_weights,
    ey_grad_z,
    ridge_basis_ey_trust_krylov,
)
from cca_zoo._utils._validation import perview_parameter, validate_views

# A hinge factor (feature, knot, sign) is max(0, sign * (x[feature] - knot)).
_Factor = tuple[int, float, int]
_Term = tuple[_Factor, ...]

# Relative tolerance below which a candidate column counts as degenerate:
# identically zero on the training data, or already in the span of the
# current basis.
_DEGENERATE_TOL = 1e-8


def _evaluate_terms(X: np.ndarray, terms: list[_Term]) -> np.ndarray:
    """Raw (uncentred) MARS basis: one column per product-of-hinges term."""
    basis = np.ones((X.shape[0], len(terms)))
    for col, term in enumerate(terms):
        for feature, knot, sign in term:
            basis[:, col] *= np.maximum(0.0, sign * (X[:, feature] - knot))
    return basis


def _best_hinge_pair(
    parent: np.ndarray,
    x: np.ndarray,
    knots: np.ndarray,
    q: np.ndarray,
    grad: np.ndarray,
) -> tuple[float, float, tuple[bool, bool]]:
    r"""Best knot for the reflected hinge pair ``parent * h(±(x - t))``.

    Scores every candidate knot $t$ by how much of the current EY gradient
    $G$ the new columns can absorb once orthogonalised against the current
    basis (orthonormal columns ``q``): $\operatorname{tr}(G^\top P_H G)$,
    with $P_H$ the projection onto the orthogonalised pair. This is exactly
    classical MARS's forward-pass criterion — the reduction in residual sum
    of squares from adding the pair — with the least-squares residual
    replaced by the EY loss's negative gradient, the same functional-
    gradient reading :class:`~cca_zoo.tree.TreeCCA` uses to grow trees.

    When one hinge of the pair is degenerate (identically zero on the
    training data because the parent term vanishes on that side of the
    knot, or already in the basis span), the other is scored alone, the
    same way ``earth`` drops the unused half of a pair.

    Returns:
        Tuple ``(score, knot, keep)`` for the best knot, ``keep`` flagging
        which of the (positive, negative) hinges to add; ``score`` is
        ``-inf`` when every candidate is degenerate.
    """
    diff = x[:, None] - knots[None, :]
    hinges = [
        parent[:, None] * np.maximum(0.0, diff),
        parent[:, None] * np.maximum(0.0, -diff),
    ]
    hinges = [h - h.mean(axis=0) for h in hinges]
    raw_sq = [np.sum(h**2, axis=0) for h in hinges]
    ortho = [h - q @ (q.T @ h) for h in hinges]
    aa, bb = (np.sum(h**2, axis=0) for h in ortho)
    ab = np.sum(ortho[0] * ortho[1], axis=0)
    na, nb = (h.T @ grad for h in ortho)
    na_sq, nb_sq = np.sum(na**2, axis=1), np.sum(nb**2, axis=1)

    ok_a = aa > _DEGENERATE_TOL * raw_sq[0]
    ok_b = bb > _DEGENERATE_TOL * raw_sq[1]
    det = aa * bb - ab**2
    ok_pair = ok_a & ok_b & (det > _DEGENERATE_TOL * aa * bb)

    with np.errstate(divide="ignore", invalid="ignore"):
        pair = (bb * na_sq - 2 * ab * np.sum(na * nb, axis=1) + aa * nb_sq) / det
        single_a = na_sq / aa
        single_b = nb_sq / bb
    single_a = np.where(ok_a, single_a, -np.inf)
    single_b = np.where(ok_b, single_b, -np.inf)
    scores = np.where(ok_pair, pair, np.maximum(single_a, single_b))
    best = int(np.argmax(scores))
    if ok_pair[best]:
        keep = (True, True)
    else:
        keep = (
            bool(single_a[best] >= single_b[best]),
            bool(single_b[best] > single_a[best]),
        )
    return float(scores[best]), float(knots[best]), keep


class _MarsEncoder:
    """Per-view MARS encoder: a centred basis of products of hinge functions.

    Holds the terms selected by :class:`MARSCCA`'s forward pass and their
    fitted coefficients; the terms themselves are chosen, and the
    coefficients fit, by :class:`MARSCCA`, not by this class.
    """

    def __init__(self, X: np.ndarray, k: int) -> None:
        self.n, self.p = X.shape
        self.k = k
        self.terms_: list[_Term] = []
        self.basis_mean_: np.ndarray = np.zeros(0)
        self.coef_: np.ndarray = np.zeros((0, k))
        self._train_pred: np.ndarray = np.zeros((self.n, k))

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
    (:func:`_best_hinge_pair`; classical MARS's residual-sum-of-squares
    reduction, with the residual replaced by the EY loss's negative
    gradient), adds the best pair to each view in turn, then refits every
    view's coefficients jointly on the enlarged bases by the same exact
    trust-region Newton-CG solve GAMCCA uses
    (:func:`~cca_zoo._utils._ey.ridge_basis_ey_trust_krylov`), warm-started
    from a least-squares projection of the previous embeddings. Knots are
    therefore placed only where the cross-view signal needs them, and with
    ``max_degree >= 2`` a term can represent a genuine within-view
    interaction (e.g. $x_1 x_2$) that GAMCCA's additive structure cannot.

    The EY loss's all-zero embedding is a stationary point, so there is no
    gradient to select the very first terms against; every view is
    therefore warm-started with a random linear projection
    (:func:`~cca_zoo._utils._ey.cheap_orthonormal_projection_weights`),
    which the first refit replaces entirely.

    Note:
        Classical MARS follows the forward pass with a backward pruning
        pass scored by generalised cross-validation. GCV is a
        squared-error-residual criterion with no EY-loss counterpart, so
        MARSCCA has no pruning pass: model size is controlled by
        ``max_terms`` directly, and the ridge penalty ``alpha`` shrinks
        whichever terms turn out not to be needed. Tune both by
        cross-validation.

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
        n_candidate_knots: Number of candidate knots per feature, placed at
            interior quantiles of that feature's training values. Default
            is 20.
        alpha: Ridge penalty strength applied to every basis coefficient.
            Either a single float or a list of per-view floats. Default is
            0.1.
        max_iter: Maximum number of outer Newton iterations in each joint
            ``"trust-krylov"`` refit. Default is 100.
        tol: Gradient-norm convergence tolerance for each joint refit.
            Default is 1e-6.
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
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        max_terms: int | list[int] = 20,
        max_degree: int | list[int] = 1,
        n_candidate_knots: int = 20,
        alpha: float | list[float] = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int = 0,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.n_candidate_knots = n_candidate_knots
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
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
        max_terms_ = perview_parameter("max_terms", self.max_terms, 20, self.n_views_)
        max_degree_ = perview_parameter("max_degree", self.max_degree, 1, self.n_views_)
        alpha_ = perview_parameter("alpha", self.alpha, 0.1, self.n_views_)
        quantiles = np.linspace(0, 1, self.n_candidate_knots + 2)[1:-1]
        candidate_knots = [
            [np.unique(np.quantile(X[:, j], quantiles)) for j in range(X.shape[1])]
            for X in views_
        ]

        rng = np.random.default_rng(self.random_state)
        warm_start = cheap_orthonormal_projection_weights(views_, k, None, rng)
        representations = [X @ w for X, w in zip(views_, warm_start)]
        encoders = [_MarsEncoder(X, k) for X in views_]
        raw_bases = [np.zeros((X.shape[0], 0)) for X in views_]
        growing = [True] * self.n_views_

        while any(growing):
            grads = ey_grad_z(representations)
            for i, X in enumerate(views_):
                if not growing[i]:
                    continue
                added = self._add_best_pair(
                    X,
                    encoders[i],
                    raw_bases[i],
                    grads[i],
                    candidate_knots[i],
                    max_degree_[i],
                    max_terms_[i],
                )
                if added is None:
                    growing[i] = False
                    continue
                raw_bases[i] = np.column_stack([raw_bases[i], added])
                growing[i] = len(encoders[i].terms_) < max_terms_[i]

            bases = [raw - raw.mean(axis=0) for raw in raw_bases]
            coefficients = ridge_basis_ey_trust_krylov(
                bases,
                [
                    np.linalg.lstsq(basis, rep, rcond=None)[0]
                    for basis, rep in zip(bases, representations)
                ],
                alpha_,
                self.max_iter,
                self.tol,
            )
            representations = [b @ c for b, c in zip(bases, coefficients)]

        for enc, raw, coef, rep in zip(
            encoders, raw_bases, coefficients, representations
        ):
            enc.basis_mean_ = raw.mean(axis=0)
            enc.coef_ = coef
            enc._train_pred = rep

        self.encoders_: list[_MarsEncoder] = encoders
        return self

    @staticmethod
    def _add_best_pair(
        X: np.ndarray,
        encoder: _MarsEncoder,
        raw_basis: np.ndarray,
        grad: np.ndarray,
        knots: list[np.ndarray],
        max_degree: int,
        max_terms: int,
    ) -> np.ndarray | None:
        """Append the best-scoring hinge pair to ``encoder.terms_``.

        Returns:
            The new raw basis column(s), shape (n_samples, 1 or 2), or None
            if no candidate is non-degenerate.
        """
        centred = raw_basis - raw_basis.mean(axis=0)
        q = np.linalg.qr(centred)[0] if centred.shape[1] else centred
        parents: list[tuple[_Term, np.ndarray]] = [((), np.ones(X.shape[0]))]
        parents += [
            (term, raw_basis[:, col])
            for col, term in enumerate(encoder.terms_)
            if len(term) < max_degree
        ]

        best_score, best = -np.inf, None
        for term, parent in parents:
            used = {feature for feature, _, _ in term}
            for j in range(X.shape[1]):
                if j in used:
                    continue
                score, knot, keep = _best_hinge_pair(parent, X[:, j], knots[j], q, grad)
                if score > best_score:
                    best_score, best = score, (term, j, knot, keep)

        if best is None:
            return None
        term, j, knot, keep = best
        if len(encoder.terms_) + sum(keep) > max_terms:
            keep = (True, False)
        new_terms: list[_Term] = [
            (*term, (j, knot, sign)) for sign, kept in zip((1, -1), keep) if kept
        ]
        encoder.terms_ += new_terms
        return _evaluate_terms(X, new_terms)

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
            if sign > 0:
                return f"h(x{feature} - {raw_knot:.4g})"
            return f"h({raw_knot:.4g} - x{feature})"

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
