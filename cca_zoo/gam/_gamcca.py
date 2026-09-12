"""GAMCCA — generalized-additive-model Canonical Correlation Analysis."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    ey_grad_z,
    random_orthogonal_embedding,
    rescale_grads_to_target_std,
)
from cca_zoo._utils._validation import validate_views


def _cubic_spline_basis(
    x: np.ndarray, knots: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """Truncated-power cubic regression spline basis for one feature.

    Args:
        x: Feature values (already scaled to unit standard deviation),
            shape (n_samples,).
        knots: Interior knot locations (quantiles of the training data, in
            the same scaled units), shape (n_knots,).
        lo: Training-data lower bound, used to clip inputs before
            evaluating the basis so that cubic extrapolation on unseen data
            cannot blow up.
        hi: Training-data upper bound (see ``lo``).

    Returns:
        Design matrix of shape (n_samples, 4 + n_knots): columns
        ``1, x, x^2, x^3`` followed by one ``(x - knot)_+^3`` column per
        interior knot.
    """
    xc = np.clip(x, lo, hi)
    cols = [np.ones_like(xc), xc, xc**2, xc**3]
    for knot in knots:
        cols.append(np.clip(xc - knot, 0.0, None) ** 3)
    return np.column_stack(cols)


def _ridge_pinv(basis: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge pseudo-inverse of a design matrix, column-norm-preconditioned.

    Cubic-spline basis columns (``x``, ``x^3``, truncated-cubic terms, ...)
    have wildly different natural scales, so a single ``ridge`` value only
    penalises them comparably after each column is rescaled to unit norm;
    the returned pseudo-inverse already undoes that rescaling, so
    ``basis @ (_ridge_pinv(basis, ridge) @ y)`` is the ridge fit of ``y`` in
    the *original* column scale.

    Args:
        basis: Design matrix, shape (n_samples, n_basis).
        ridge: Ridge penalty strength (applied after column-norm scaling,
            so it is comparable across features regardless of their raw
            scale).

    Returns:
        Pseudo-inverse matrix, shape (n_basis, n_samples).
    """
    col_scale = np.maximum(np.linalg.norm(basis, axis=0), 1e-8)
    normed = basis / col_scale
    gram = normed.T @ normed + ridge * np.eye(normed.shape[1])
    pinv_normed = np.linalg.solve(gram, normed.T)
    result: np.ndarray = pinv_normed / col_scale[:, None]
    return result


class _GamEncoder:
    r"""Per-view additive spline encoder, fit by componentwise L2 boosting.

    Each latent component is modelled as $f(x) = \\sum_j s_j(x_j)$, one
    cubic regression spline per input feature. Every boosting round performs
    one ridge-regularised least-squares fit of *all* per-feature spline
    bases at once against the (rescaled) EY gradient, shrunk by
    ``learning_rate`` and added to the running coefficients — i.e. L2Boosting
    (Bühlmann & Yu, 2003) with a penalised additive-spline base learner, used
    only during ``fit``.
    """

    def __init__(self, X: np.ndarray, k: int, n_knots: int, ridge: float) -> None:
        n, p = X.shape
        self.n = n
        self.p = p
        self.k = k
        self.scales_ = np.maximum(X.std(axis=0), 1e-8)
        scaled = X / self.scales_
        self.lo_ = scaled.min(axis=0)
        self.hi_ = scaled.max(axis=0)
        quantiles = np.linspace(0.0, 1.0, n_knots + 2)[1:-1]
        self.knots_ = [np.quantile(scaled[:, j], quantiles) for j in range(p)]
        bases = [
            _cubic_spline_basis(scaled[:, j], self.knots_[j], self.lo_[j], self.hi_[j])
            for j in range(p)
        ]
        self.df_: int = bases[0].shape[1]
        self._basis: np.ndarray = np.column_stack(bases)  # (n, p * df)
        self._pinv: np.ndarray = _ridge_pinv(self._basis, ridge)  # (p * df, n)
        self.coefs_: np.ndarray = np.zeros((p * self.df_, k))

    def predict(self) -> np.ndarray:
        """Encoder output on the training data, shape (n_samples, k)."""
        result: np.ndarray = self._basis @ self.coefs_
        return result

    def boost(self, gradient: np.ndarray, learning_rate: float) -> None:
        """Add one ridge-fitted, shrunk round to every feature's spline terms.

        Args:
            gradient: EY gradient for this view, shape (n_samples, k).
            learning_rate: Shrinkage applied to this round's fit.
        """
        beta = self._pinv @ (-gradient)  # (p * df, k)
        self.coefs_ += learning_rate * beta

    def _basis_new(self, X: np.ndarray) -> np.ndarray:
        scaled = X / self.scales_
        bases = [
            _cubic_spline_basis(scaled[:, j], self.knots_[j], self.lo_[j], self.hi_[j])
            for j in range(self.p)
        ]
        return np.column_stack(bases)

    def predict_new(self, X: np.ndarray) -> np.ndarray:
        """Encoder output for arbitrary (e.g. test) data, shape (n, k)."""
        result: np.ndarray = self._basis_new(X) @ self.coefs_
        return result

    def feature_term(self, feature_idx: int, x: np.ndarray) -> np.ndarray:
        """Single feature's additive contribution $s_j(x_j)$ at given values.

        Args:
            feature_idx: Index of the input feature (in the centred-view
                column order).
            x: Raw (mean-centred) values for that feature, shape (n,).

        Returns:
            Array of shape (n, k): this feature's spline term alone, for
            each latent component.
        """
        scaled = x / self.scales_[feature_idx]
        lo, hi = self.lo_[feature_idx], self.hi_[feature_idx]
        basis = _cubic_spline_basis(scaled, self.knots_[feature_idx], lo, hi)
        block = slice(feature_idx * self.df_, (feature_idx + 1) * self.df_)
        result: np.ndarray = basis @ self.coefs_[block]
        return result


class GAMCCA(BaseModel):
    r"""GAMCCA — nonlinear multiview CCA with generalized-additive-model encoders.

    Learns one nonlinear encoder $f_i$ per view — a generalized additive
    model (GAM), $f_i(x) = \sum_j s_{ij}(x_{ij})$, summing one univariate
    cubic-spline term per input feature — that jointly maximise the
    Eckart-Young (EY) unconstrained-CCA objective:

    $$
    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
    $$

    where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean pairwise
    cross-covariance (including $i = j$ terms) and $V$ the mean
    auto-covariance across all views (see :mod:`cca_zoo._utils._ey`, the
    same shared EY-loss machinery used by
    :class:`~cca_zoo.linear.gradient.CCA_EY`, :class:`~cca_zoo.deep.DCCA_EY`,
    and :class:`~cca_zoo.tree.TreeCCA`). The encoders are fit by alternating
    (Gauss-Seidel) L2Boosting (Bühlmann & Yu, 2003): each round, for every
    view in turn, the EY-loss gradient (rescaled to a fixed target standard
    deviation — see :func:`cca_zoo._utils._ey.rescale_grads_to_target_std`,
    needed since the analytic gradient's natural scale is far smaller than a
    well-conditioned regression target) is fit, *jointly across all of that
    view's features*, by a ridge-penalised cubic-regression-spline additive
    model, shrunk by ``learning_rate`` and added to the running per-feature
    coefficients — when ``gauss_seidel=True`` (default), the gradient is
    recomputed from the freshest embeddings before moving to the next view.
    Training starts from a random-orthogonal, unit-variance initial
    embedding per view, as for :class:`~cca_zoo.tree.TreeCCA`.

    Because each latent component decomposes exactly into one additive term
    per input feature, the fitted shape of any feature's contribution is
    available directly via :meth:`shape_function`, without a separate
    interpretability method such as SHAP — and, unlike a boosted-tree
    ensemble's step-function contributions, each term is a smooth curve
    by construction. This smoothness is also a genuine inductive bias, not
    just a cosmetic one: on data where the true per-feature relationship is
    smooth, GAMCCA reaches a given held-out canonical correlation in far
    fewer boosting rounds than :class:`~cca_zoo.tree.TreeCCA`, and can
    generalise better at a matched round budget (see the module's test
    suite for a worked example: a quadratic cross-view relationship where
    GAMCCA reaches a held-out correlation TreeCCA does not reach even at 4x
    the boosting rounds).

    Note:
        A GAM's additive structure assumes each feature contributes
        independently; it cannot represent a genuine *interaction* between
        two features of the same view (e.g. $x_1 x_2$) the way a
        multivariate tree split can. If cross-view structure only shows up
        through such interactions, expect :class:`~cca_zoo.tree.TreeCCA` to
        do better instead.

    References:
        Bühlmann, P., & Yu, B. (2003). Boosting with the L2 loss:
        regression and classification. Journal of the American Statistical
        Association, 98(462), 324-339.

        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Args:
        latent_dimensions: Number of latent components. Must not exceed the
            number of features in any view. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            Default is True.
        n_estimators: Number of boosting rounds. Default is 150.
        n_knots: Number of interior knots per feature's cubic regression
            spline (basis dimension is ``4 + n_knots``: intercept, linear,
            quadratic, cubic, plus one truncated-cubic term per knot).
            Default is 3.
        learning_rate: Boosting shrinkage applied to each round's ridge fit.
            Default is 0.3.
        ridge: Ridge penalty for each round's per-view spline fit, applied
            after normalising every basis column to unit norm (so it
            penalises all features/terms comparably regardless of their raw
            scale). Higher values give smoother, less wiggly per-feature
            curves at the cost of slower convergence, and are the main
            defence against overfitting a single view's noise. Default is
            0.1.
        gauss_seidel: If True, re-predict view 1's embedding after updating
            its encoder and use the fresh values when computing view 2's
            gradient (Gauss-Seidel); if False, both gradients are computed
            from the same stale embeddings (Jacobi). Default is True.
        random_state: Seed for drawing the random-orthogonal initial
            embedding. Default is 0.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 5))
        >>> X2 = rng.standard_normal((200, 5))
        >>> model = GAMCCA(latent_dimensions=2, n_estimators=20).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "n_knots": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "ridge": [Interval(Real, 0, None, closed="left")],
        "gauss_seidel": ["boolean"],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        n_estimators: int = 150,
        n_knots: int = 3,
        learning_rate: float = 0.3,
        ridge: float = 0.1,
        gauss_seidel: bool = True,
        random_state: int = 0,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.n_estimators = n_estimators
        self.n_knots = n_knots
        self.learning_rate = learning_rate
        self.ridge = ridge
        self.gauss_seidel = gauss_seidel
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> GAMCCA:
        """Fit the GAMCCA model.

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

        encoders = [_GamEncoder(X, k, self.n_knots, self.ridge) for X in views_]

        for _ in range(self.n_estimators):
            representations = [
                bm + enc.predict() for bm, enc in zip(base_margins, encoders)
            ]
            grads = rescale_grads_to_target_std(ey_grad_z(representations))

            for view_idx in range(n_views):
                encoders[view_idx].boost(grads[view_idx], self.learning_rate)
                if self.gauss_seidel and view_idx < n_views - 1:
                    representations[view_idx] = (
                        base_margins[view_idx] + encoders[view_idx].predict()
                    )
                    grads = rescale_grads_to_target_std(ey_grad_z(representations))

        self.encoders_: list[_GamEncoder] = encoders
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
            ValueError: If fewer than 2 views are provided.
        """
        check_is_fitted(self)
        validated = validate_views(views)
        centred = [v - m for v, m in zip(validated, self.means_)]
        result = []
        for v, enc, projection in zip(centred, self.encoders_, self._projections_):
            bm = v @ projection
            result.append(bm + enc.predict_new(v))
        return result

    def shape_function(self, view: int, feature: int, x: ArrayLike) -> np.ndarray:
        r"""Evaluate one feature's fitted additive term $s_j(x_j)$.

        Because GAMCCA's encoder is additive across features, each term can
        be inspected in isolation — the direct GAM analogue of
        :class:`~cca_zoo.tree.TreeCCA`'s split-gain feature importance, but
        exact and shape-preserving rather than a single importance score.

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
        return self.encoders_[view].feature_term(feature, x_arr)

    @property
    def weights(self) -> list[np.ndarray]:
        """Not implemented for GAMCCA.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            NotImplementedError: GAMCCA encoders are additive splines, not
                linear weight matrices. Use :meth:`shape_function` instead
                to inspect a feature's fitted contribution directly.
        """
        check_is_fitted(self)
        raise NotImplementedError(
            "GAMCCA has no linear weight matrices; its encoders are "
            "generalized additive models (one cubic spline per feature). "
            "Use the `shape_function` method instead to inspect a fitted "
            "feature's contribution directly."
        )
