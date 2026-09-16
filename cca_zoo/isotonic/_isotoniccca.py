"""IsotonicCCA — monotonic-additive-model Canonical Correlation Analysis."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.isotonic import IsotonicRegression
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import ey_grad_z, random_orthogonal_embedding
from cca_zoo._utils._validation import validate_views


class _IsotonicEncoder:
    r"""Per-view additive-monotonic encoder: one boosted isotonic term per feature.

    Unlike :class:`~cca_zoo.gam._gamcca._GamEncoder`'s fixed B-spline
    basis (linear in a fixed set of coefficients, solvable by one joint
    Newton-CG solve), an isotonic step function has no fixed finite basis
    -- its breakpoints adapt to wherever the fitted target actually
    changes, the same way a decision tree's splits do. So, like
    :class:`~cca_zoo.tree.TreeCCA`, this is fit by functional gradient
    boosting: each round, for every feature independently, an
    :class:`~sklearn.isotonic.IsotonicRegression` is fit to that view's
    current EY gradient (:func:`~cca_zoo._utils._ey.ey_grad_z`) and its
    shrunk prediction is added to that feature's running contribution --
    "TreeCCA's boosting recipe" crossed with "GAMCCA's additive,
    per-feature structure".

    Each feature's monotonicity direction (``increasing_``) is fixed once,
    from the sign of its correlation with the *first* round's gradient,
    and reused for every subsequent round. This isn't a simplification of
    convenience: since a feature's total fitted contribution is the sum
    of every round's isotonic term for it, letting the direction flip
    between rounds would generally make that sum non-monotonic --
    defeating the entire reason to reach for isotonic regression instead
    of :class:`~cca_zoo.gam.GAMCCA`'s unconstrained splines. A sum of
    same-direction monotonic functions is itself monotonic; a fixed
    direction is what makes that guarantee hold for the whole fit, not
    just one round of it.
    """

    def __init__(self, X: np.ndarray, k: int, out_of_bounds: str) -> None:
        self.X = X
        self.n, self.p = X.shape
        self.k = k
        self.out_of_bounds = out_of_bounds
        self.models: list[list[list[IsotonicRegression]]] = [
            [[] for _ in range(self.p)] for _ in range(k)
        ]
        self.increasing_: np.ndarray | None = None
        self._train_pred = np.zeros((self.n, k))

    def predict(self) -> np.ndarray:
        """Encoder output on the training data, shape (n_samples, k)."""
        return self._train_pred

    def _resolve_directions(self, descent_target: np.ndarray) -> None:
        """Fix each (component, feature)'s monotonicity direction, once.

        Args:
            descent_target: The *negative* EY gradient (see :meth:`boost`) --
                directions reflect each feature's relationship to the
                embedding's own descent direction, not the raw gradient.
        """
        increasing = np.ones((self.k, self.p), dtype=bool)
        for c in range(self.k):
            for j in range(self.p):
                col = self.X[:, j]
                if col.std() > 1e-12:
                    corr = np.corrcoef(col, descent_target[:, c])[0, 1]
                    if np.isfinite(corr):
                        increasing[c, j] = corr >= 0
        self.increasing_ = increasing

    def boost(
        self,
        gradient: np.ndarray,
        learning_rate: float,
        subsample: float,
        rng: np.random.Generator,
    ) -> None:
        """Add one isotonic term per feature to every component, from the EY gradient.

        Each term is fit on a fresh random ``subsample`` of rows (stochastic
        gradient boosting, :class:`~cca_zoo.tree.TreeCCA`'s own
        ``subsample``) rather than every row every round. This isn't
        optional the way it is for a depth-limited tree or a fixed-knot
        spline: an isotonic fit has no built-in capacity limit (up to one
        step per distinct training value), and with ``p`` features each
        contributing their own term every round, an unregularised version
        of this reliably overfits worse the *more* rounds it runs for
        (verified directly -- held-out correlation on a purely monotonic
        synthetic relationship peaks early, then degrades, without this).

        Fits each term to (a subsample of) the *negative* gradient, not the
        raw gradient: minimising a loss means moving *against* its gradient,
        and unlike :class:`~cca_zoo.tree.TreeCCA`'s XGBoost/LightGBM
        backends -- whose Newton-step leaf values are already the correctly
        signed descent direction, an artifact of how those libraries'
        custom-objective protocol works -- there is no framework doing that
        negation on this class's behalf; it has to happen here (an earlier
        version of this method didn't, and its accumulated prediction grew
        without bound instead of converging -- verified directly by
        boosting against a *fixed* target, which should shrink the residual
        to zero and didn't, growing instead. That's now a regression test).

        Args:
            gradient: EY gradient for this view, shape (n_samples, k).
            learning_rate: Shrinkage applied to this round's fitted terms.
            subsample: Row fraction sampled (without replacement) for each
                round's fit, in ``(0, 1]``.
            rng: Random generator for the row subsample.
        """
        descent_target = -gradient
        if self.increasing_ is None:
            self._resolve_directions(descent_target)
        assert self.increasing_ is not None
        n_sub = max(2, int(round(subsample * self.n)))
        idx = rng.choice(self.n, n_sub, replace=False)
        for c in range(self.k):
            g = descent_target[:, c]
            for j in range(self.p):
                model = IsotonicRegression(
                    increasing=bool(self.increasing_[c, j]),
                    out_of_bounds=self.out_of_bounds,
                )
                model.fit(self.X[idx, j], g[idx])
                self.models[c][j].append(model)
                self._train_pred[:, c] += learning_rate * model.predict(self.X[:, j])

    def feature_term(
        self, feature_idx: int, x: np.ndarray, learning_rate: float
    ) -> np.ndarray:
        """Single feature's additive contribution, shape (n, k).

        Args:
            feature_idx: Index of the input feature.
            x: Raw (mean-centred) values for that feature, shape (n,).
            learning_rate: Same shrinkage used during boosting.

        Returns:
            Array of shape (n, k): the sum, over every boosting round, of
            that round's fitted isotonic term for this feature alone.
        """
        result = np.zeros((len(x), self.k))
        for c in range(self.k):
            for model in self.models[c][feature_idx]:
                result[:, c] += learning_rate * model.predict(x)
        return result

    def predict_new(self, X: np.ndarray, learning_rate: float) -> np.ndarray:
        """Encoder output for arbitrary (e.g. test) data, shape (n, k)."""
        result = np.zeros((X.shape[0], self.k))
        for j in range(self.p):
            result += self.feature_term(j, X[:, j], learning_rate)
        return result


class IsotonicCCA(BaseModel):
    r"""IsotonicCCA -- nonlinear multiview CCA with monotonic-additive encoders.

    Learns one nonlinear encoder $f_i$ per view -- like
    :class:`~cca_zoo.gam.GAMCCA`, additive across features,
    $f_i(x) = \sum_j s_{ij}(x_{ij})$ -- that jointly maximise the
    Eckart-Young (EY) unconstrained-CCA objective:

    $$
    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
    $$

    where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean pairwise
    cross-covariance (including $i = j$ terms) and $V$ the mean
    auto-covariance across all views (see :mod:`cca_zoo._utils._ey`, the
    same shared EY-loss machinery used by
    :class:`~cca_zoo.linear.gradient.CCAEY`, :class:`~cca_zoo.tree.TreeCCA`
    and :class:`~cca_zoo.gam.GAMCCA`). The difference from `GAMCCA` is the
    shape each term $s_{ij}$ is allowed to take: here it's constrained to
    be **monotonic**, fit via :class:`~sklearn.isotonic.IsotonicRegression`
    (PAVA) rather than an unconstrained B-spline. An isotonic step
    function has no fixed finite basis (its breakpoints adapt to the
    data), so unlike `GAMCCA`'s single joint Newton-CG solve, this is fit
    by the same **functional gradient boosting** recipe
    :class:`~cca_zoo.tree.TreeCCA` uses: each round, for every view in
    turn, one isotonic term is fit per feature against that view's current
    EY gradient and added (shrunk by ``learning_rate``) to that feature's
    running contribution -- Gauss-Seidel across views, exactly as in
    `TreeCCA`.

    Each feature's monotonicity direction is fixed once (from its
    correlation with the first round's gradient) and reused for every
    later round -- see :class:`_IsotonicEncoder` for why: a sum of
    same-direction monotonic terms is itself guaranteed monotonic; letting
    the direction vary round to round would not be. This is `GAMCCA`'s
    exact interpretability trade, sharpened: `GAMCCA`'s ``shape_function``
    shows an exact curve with no shape constraint at all; `IsotonicCCA`'s
    shows one that's provably monotonic throughout, at the cost of being
    unable to represent a genuinely non-monotonic effect (a feature with a
    U-shaped or periodic influence needs `GAMCCA`, `TreeCCA`, or
    :class:`~cca_zoo.gp.GaussianProcessCCA` instead).

    Note:
        Like `GAMCCA`, an additive encoder can't represent a genuine
        *interaction* between two features of the same view -- reach for
        `TreeCCA` or `GaussianProcessCCA` if cross-view structure only
        shows up through such interactions.

    References:
        Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D.
        (1972). Statistical Inference under Order Restrictions. Wiley.

        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Args:
        latent_dimensions: Number of latent components. Must not exceed the
            number of features in any view. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            Default is True.
        n_estimators: Number of boosting rounds. Default is 50.
        learning_rate: Shrinkage applied to each round's fitted isotonic
            terms. Default is 0.05 -- lower than
            :class:`~cca_zoo.tree.TreeCCA`'s 0.1, since each round here
            fits and sums *one term per feature* rather than one weak
            learner total, so the effective per-round step already scales
            with the view's own width.
        subsample: Row fraction (without replacement) used to fit each
            round's isotonic terms -- see :meth:`_IsotonicEncoder.boost`
            for why this one isn't optional the way it is for
            `TreeCCA`/`GAMCCA`, which have their own, different capacity
            controls (tree depth; ridge-penalised spline coefficients).
            Default is 0.5.
        out_of_bounds: How each fitted term extrapolates beyond its
            training range, passed straight through to
            :class:`~sklearn.isotonic.IsotonicRegression`. ``"clip"`` (the
            default) holds the boundary value constant, matching
            `GAMCCA`'s own ``extrapolation="constant"`` spline default.
        gauss_seidel: If True, re-predict view 1's embedding after updating
            its terms and use the fresh values when computing view 2's
            gradient (Gauss-Seidel); if False, both gradients are computed
            from the same stale embeddings (Jacobi). Default is True.
        random_state: Seed for the random-orthogonal initial embedding
            (isotonic regression itself is deterministic; this only seeds
            the starting point every boosting round builds on).

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 5))
        >>> X2 = rng.standard_normal((200, 5))
        >>> model = IsotonicCCA(latent_dimensions=2, n_estimators=20).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "subsample": [Interval(Real, 0, 1, closed="right")],
        "out_of_bounds": [StrOptions({"nan", "clip", "raise"})],
        "gauss_seidel": ["boolean"],
        "random_state": [Interval(Integral, 0, None, closed="left"), None],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        n_estimators: int = 50,
        learning_rate: float = 0.05,
        subsample: float = 0.5,
        out_of_bounds: str = "clip",
        gauss_seidel: bool = True,
        random_state: int | None = 0,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.out_of_bounds = out_of_bounds
        self.gauss_seidel = gauss_seidel
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> IsotonicCCA:
        """Fit IsotonicCCA by Gauss-Seidel functional gradient boosting.

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
        n_views = len(views_)

        rng = np.random.default_rng(self.random_state)
        base_margins = []
        projections = []
        for X in views_:
            bm, proj = random_orthogonal_embedding(X, k, rng)
            base_margins.append(bm)
            projections.append(proj)
        self._projections_: list[np.ndarray] = projections

        encoders = [_IsotonicEncoder(X, k, self.out_of_bounds) for X in views_]

        for _ in range(self.n_estimators):
            representations = [
                bm + enc.predict() for bm, enc in zip(base_margins, encoders)
            ]
            grads = ey_grad_z(representations)

            for view_idx in range(n_views):
                encoders[view_idx].boost(
                    grads[view_idx], self.learning_rate, self.subsample, rng
                )
                if self.gauss_seidel and view_idx < n_views - 1:
                    representations[view_idx] = (
                        base_margins[view_idx] + encoders[view_idx].predict()
                    )
                    grads = ey_grad_z(representations)

        self.encoders_: list[_IsotonicEncoder] = encoders
        return self

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Project views into the latent space using the fitted encoders.

        Args:
            views: List of arrays, each (n_samples, n_features_i), matching
                the number of views passed to ``fit``.

        Returns:
            List of arrays, each (n_samples, latent_dimensions).

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
        """
        check_is_fitted(self)
        validated = validate_views(views)
        centred = [v - m for v, m in zip(validated, self.means_)]
        result = []
        for v, enc, projection in zip(centred, self.encoders_, self._projections_):
            bm = v @ projection
            result.append(bm + enc.predict_new(v, self.learning_rate))
        return result

    def shape_function(self, view: int, feature: int, x: ArrayLike) -> np.ndarray:
        r"""Evaluate one feature's fitted additive term $s_j(x_j)$.

        Guaranteed monotonic in ``x`` (non-decreasing or non-increasing,
        per :class:`_IsotonicEncoder`'s fixed per-feature direction) --
        unlike :meth:`~cca_zoo.gam.GAMCCA.shape_function`, which has no
        such constraint.

        Args:
            view: Index of the view.
            feature: Index of the feature within that view (raw, i.e.
                un-centred column order).
            x: Raw (un-centred) values for that feature at which to evaluate
                the term, shape (n,).

        Returns:
            Array of shape (n, latent_dimensions): that feature's
            contribution alone, for every latent component.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
        """
        check_is_fitted(self)
        x_arr = np.asarray(x, dtype=float) - self.means_[view][feature]
        return self.encoders_[view].feature_term(feature, x_arr, self.learning_rate)

    @property
    def weights(self) -> list[np.ndarray]:
        """Not implemented for IsotonicCCA.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            NotImplementedError: IsotonicCCA encoders are boosted additive
                isotonic terms, not linear weight matrices. Use
                :meth:`shape_function` instead to inspect a fitted
                feature's (monotonic) contribution directly.
        """
        check_is_fitted(self)
        raise NotImplementedError(
            "IsotonicCCA has no linear weight matrices; its encoders are "
            "boosted additive isotonic terms (one monotonic curve per "
            "feature). Use the `shape_function` method instead to inspect "
            "a fitted feature's contribution directly."
        )
