"""GAMCCA — generalized-additive-model Canonical Correlation Analysis."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    ey_grad_z,
    random_orthogonal_embedding,
    rescale_grads_to_target_std,
)
from cca_zoo._utils._validation import validate_views


class _GamEncoder:
    """Per-view additive-spline encoder, fit by componentwise L2Boosting.

    Each latent component is modelled as a generalized additive model —
    one B-spline term per input feature — using only scikit-learn's own,
    already-required machinery: :class:`~sklearn.preprocessing.SplineTransformer`
    builds the per-feature B-spline design matrix (one contiguous block of
    columns per feature) and :class:`~sklearn.linear_model.Ridge` fits it,
    rather than reimplementing either. Every boosting round fits a fresh
    ridge regression of the (rescaled) EY gradient onto that fixed basis and
    accumulates it, shrunk by ``learning_rate`` — i.e. L2Boosting (Bühlmann
    & Yu, 2003) with a penalised-spline base learner, the same
    meta-algorithm :class:`~cca_zoo.tree.TreeCCA` uses with a decision-tree
    base learner instead. Used only during ``fit``.
    """

    def __init__(self, X: np.ndarray, k: int, n_knots: int, ridge: float) -> None:
        self.n, self.p = X.shape
        self.k = k
        self.ridge = ridge
        self._spline = SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            extrapolation="constant",
            include_bias=True,
        )
        self._basis: np.ndarray = self._spline.fit_transform(X)
        self.n_splines_: int = self._basis.shape[1] // self.p
        self.coefs_: np.ndarray = np.zeros((self._basis.shape[1], k))
        self._train_pred: np.ndarray = np.zeros((self.n, k))

    def predict(self) -> np.ndarray:
        """Encoder output on the training data, shape (n_samples, k)."""
        return self._train_pred

    def boost(self, gradient: np.ndarray, learning_rate: float) -> None:
        """Ridge-fit the negative gradient onto the spline basis and accumulate.

        Args:
            gradient: EY gradient for this view, shape (n_samples, k).
            learning_rate: Shrinkage applied to this round's fit.
        """
        target = -gradient
        model = Ridge(alpha=self.ridge, fit_intercept=False)
        model.fit(self._basis, target)
        coef = np.atleast_2d(model.coef_).T  # (n_basis, k)
        self.coefs_ += learning_rate * coef
        self._train_pred += learning_rate * (self._basis @ coef)

    def predict_new(self, X: np.ndarray) -> np.ndarray:
        """Encoder output for arbitrary (e.g. test) data, shape (n, k)."""
        basis = self._spline.transform(X)
        result: np.ndarray = basis @ self.coefs_
        return result

    def feature_term(self, feature_idx: int, x: np.ndarray) -> np.ndarray:
        """Single feature's additive contribution, shape (n, k).

        Args:
            feature_idx: Index of the input feature.
            x: Raw (mean-centred) values for that feature, shape (n,).

        Returns:
            Array of shape (n, k): this feature's term alone, for each
            latent component.
        """
        grid = np.zeros((len(x), self.p))
        grid[:, feature_idx] = x
        basis = self._spline.transform(grid)
        block = slice(
            feature_idx * self.n_splines_, (feature_idx + 1) * self.n_splines_
        )
        result: np.ndarray = basis[:, block] @ self.coefs_[block]
        return result


class GAMCCA(BaseModel):
    r"""GAMCCA — nonlinear multiview CCA with generalized-additive-model encoders.

    Learns one nonlinear encoder $f_i$ per view — a generalized additive
    model (GAM), $f_i(x) = \sum_j s_{ij}(x_{ij})$, summing one univariate
    B-spline term per input feature — that jointly maximise the
    Eckart-Young (EY) unconstrained-CCA objective:

    $$
    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
    $$

    where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean pairwise
    cross-covariance (including $i = j$ terms) and $V$ the mean
    auto-covariance across all views (see :mod:`cca_zoo._utils._ey`, the
    same shared EY-loss machinery used by
    :class:`~cca_zoo.linear.gradient.CCA_EY`, :class:`~cca_zoo.deep.DCCA_EY`,
    and :class:`~cca_zoo.tree.TreeCCA`). Rather than reimplementing spline
    fitting from scratch, each encoder is built entirely from scikit-learn's
    own, already-required machinery:
    :class:`~sklearn.preprocessing.SplineTransformer` builds the per-feature
    B-spline design matrix (one contiguous block of columns per feature,
    giving the additive structure) and :class:`~sklearn.linear_model.Ridge`
    fits it. The encoders are fit by alternating (Gauss-Seidel) L2Boosting
    (Bühlmann & Yu, 2003): each round, for every view in turn, the EY-loss
    gradient (rescaled to a fixed target standard deviation — see
    :func:`cca_zoo._utils._ey.rescale_grads_to_target_std`, needed since the
    analytic gradient's natural scale is far smaller than a well-conditioned
    regression target) is ridge-fit onto that view's fixed spline basis,
    shrunk by ``learning_rate`` and added to a running total — the same
    meta-algorithm :class:`~cca_zoo.tree.TreeCCA` uses with tree base
    learners instead of splines. With ``gauss_seidel=True`` (default), the
    gradient is recomputed from the freshest embeddings before moving to the
    next view. Training starts from a random-orthogonal, unit-variance
    initial embedding per view, as for :class:`~cca_zoo.tree.TreeCCA`. No
    optional dependency is required — everything here is already part of
    ``cca_zoo``'s required scikit-learn dependency.

    Because each latent component decomposes exactly into one additive term
    per input feature, the fitted shape of any feature's contribution is
    available directly via :meth:`shape_function`, without a separate
    interpretability method such as SHAP — and, unlike a boosted-tree
    ensemble's step-function contributions, each term is a smooth curve by
    construction. This smoothness is also a genuine inductive bias, not just
    a cosmetic one: on data where the true per-feature relationship is
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
        Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with
        B-splines and penalties. Statistical Science, 11(2), 89-121.

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
        n_knots: Number of knots per feature's B-spline term, passed
            straight through to ``sklearn.preprocessing.SplineTransformer(
            n_knots=...)``. Default is 5.
        learning_rate: Boosting shrinkage applied to each round's ridge fit.
            Default is 0.3.
        ridge: Ridge penalty for each round's per-view spline fit, passed
            straight through to ``sklearn.linear_model.Ridge(alpha=...)``.
            Higher values give smoother, less wiggly per-feature curves at
            the cost of slower convergence, and are the main defence
            against overfitting a single view's noise. Default is 0.1.
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
        "n_knots": [Interval(Integral, 2, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "ridge": [Interval(Real, 0, None, closed="left")],
        "gauss_seidel": ["boolean"],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        n_estimators: int = 150,
        n_knots: int = 5,
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
        an exact, shape-preserving curve rather than a single importance
        score.

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
            "generalized additive models (one B-spline term per feature). "
            "Use the `shape_function` method instead to inspect a fitted "
            "feature's contribution directly."
        )
