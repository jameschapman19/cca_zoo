"""GAMCCA — generalized-additive-model Canonical Correlation Analysis."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import SplineTransformer
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    cheap_orthonormal_projection_weights,
    ridge_basis_ey_trust_krylov,
)
from cca_zoo._utils._validation import perview_parameter, validate_views


class _GamEncoder:
    r"""Per-view additive-spline encoder: a fixed, centred B-spline basis.

    :class:`~sklearn.preprocessing.SplineTransformer` builds the per-feature
    B-spline design matrix once (one contiguous block of columns per
    feature) — the basis itself is fixed for the whole fit, never
    reimplemented. Centring it here (subtracting each basis column's own
    mean) is what makes $Z_i = \text{basis}_i B_i$ automatically zero-mean
    for *any* coefficients $B_i$, the same way centring the raw features
    already does for a plain linear encoder — no separate recentring step
    is needed anywhere downstream.

    The coefficients $B_i$ (``coef_``) are fit, jointly across every view, by
    a single trust-region Newton-CG solve
    (:func:`~cca_zoo._utils._ey.ridge_basis_ey_trust_krylov`), not by this
    class — it only builds and holds the fixed basis, and evaluates it
    (``predict``, ``predict_new``, ``feature_term``) once coefficients exist.
    """

    def __init__(self, X: np.ndarray, k: int, n_knots: int) -> None:
        self.n, self.p = X.shape
        self.k = k
        self._spline = SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            extrapolation="constant",
            include_bias=False,
        )
        raw_basis = self._spline.fit_transform(X)
        self.n_splines_: int = raw_basis.shape[1] // self.p
        self.basis_mean_: np.ndarray = raw_basis.mean(axis=0)
        self.basis_: np.ndarray = raw_basis - self.basis_mean_
        self.coef_: np.ndarray = np.zeros((self.basis_.shape[1], k))
        self._train_pred: np.ndarray = np.zeros((self.n, k))

    def predict(self) -> np.ndarray:
        """Encoder output on the training data, shape (n_samples, k)."""
        return self._train_pred

    def predict_new(self, X: np.ndarray) -> np.ndarray:
        """Encoder output for arbitrary (e.g. test) data, shape (n, k)."""
        basis = self._spline.transform(X) - self.basis_mean_
        result: np.ndarray = basis @ self.coef_
        return result

    def feature_term(self, feature_idx: int, x: np.ndarray) -> np.ndarray:
        """Single feature's additive contribution, shape (n, k).

        Args:
            feature_idx: Index of the input feature.
            x: Raw (mean-centred) values for that feature, shape (n,).

        Returns:
            Array of shape (n, k): this feature's term alone, for each
            latent component. Because the basis is centred per *column*
            (not just overall), each feature's own share of that centring
            is exactly ``basis_mean_[block]`` — so summing this over every
            feature reproduces :meth:`predict` exactly, with no leftover
            constant to split arbitrarily across features.
        """
        grid = np.zeros((len(x), self.p))
        grid[:, feature_idx] = x
        raw_basis = self._spline.transform(grid)
        block = slice(
            feature_idx * self.n_splines_, (feature_idx + 1) * self.n_splines_
        )
        centred_block = raw_basis[:, block] - self.basis_mean_[block]
        result: np.ndarray = centred_block @ self.coef_[block]
        return result


class GAMCCA(BaseModel):
    r"""GAMCCA — nonlinear multiview CCA with generalized-additive-model encoders.

    Learns one nonlinear encoder $f_i$ per view — a generalized additive
    model (GAM), $f_i(x) = \sum_j s_{ij}(x_{ij})$, summing one univariate
    B-spline term per input feature — that jointly minimise the
    elastic-net-penalised Eckart-Young (EY) objective:

    $$
    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
    $$

    where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean
    pairwise cross-covariance (including $i = j$ terms) and $V$
    the mean auto-covariance across all views (see
    :mod:`cca_zoo._utils._ey`, the same shared EY-loss machinery used by
    :class:`~cca_zoo.linear.gradient.CCAEY`, :class:`~cca_zoo.deep.DCCAEY`,
    :class:`~cca_zoo.tree.TreeCCA`, and :class:`~cca_zoo.sparse.ElasticNetCCA`).
    Writing $f_i(x) = \sum_j s_{ij}(x_{ij})$ as $\text{basis}_i(x) B_i$ for
    a fixed per-feature B-spline basis (:class:`_GamEncoder`, built by
    :class:`~sklearn.preprocessing.SplineTransformer`), fitting $B_1, \dots,
    B_M$ is conceptually a **P-IRLS** problem — the same repeated-penalised-
    quadratic-solve structure ``mgcv`` itself uses to fit a GAM, applied
    directly to the EY loss rather than a per-observation likelihood — but
    rather than hand-rolling that solve (or even cycling over views
    Gauss-Seidel-style), every view's coefficients are updated *jointly, in
    a single call*: $\mathcal{L}_{EY}$'s own exact gradient
    (:func:`~cca_zoo._utils._ey.ey_grad_z`) and exact Hessian-vector product
    across the *entire* stacked parameter vector $(B_1, \dots, B_M)$
    (:func:`~cca_zoo._utils._ey._ridge_basis_ey_hessp`) are handed straight to
    :func:`scipy.optimize.minimize`'s ``"trust-krylov"`` solver — a
    standard, off-the-shelf trust-region Newton-CG method — which performs
    all of its own outer Newton and inner Krylov iterations internally.
    $\mathcal{L}_{EY}$ is not convex, so this Hessian is only guaranteed
    positive semi-definite near the loss's own fixed point;
    ``"trust-krylov"`` handles the indefinite case directly, and works from
    the Hessian-vector product alone, never forming or inverting the
    $\left(\sum_i d_i k\right) \times \left(\sum_i d_i k\right)$ Hessian
    matrix explicitly. Each view's smoothing strength (``alpha``) is a fixed
    hyperparameter — there is no automatic smoothing-parameter search.

    Because each latent component still decomposes exactly into one
    additive term per input feature, the fitted shape of any feature's
    contribution remains available directly via ``shape_function`` — the
    GAM analogue of :class:`~cca_zoo.tree.TreeCCA`'s split-gain feature
    importance, but an exact curve rather than a single importance score.

    Note:
        A GAM's additive structure assumes each feature contributes
        independently; it cannot represent a genuine *interaction* between
        two features of the same view (e.g. $x_1 x_2$) the way a
        multivariate tree split or a joint kernel can. If cross-view
        structure only shows up through such interactions, expect
        :class:`~cca_zoo.tree.TreeCCA` or
        :class:`~cca_zoo.gp.GaussianProcessCCA` to do better instead.

    References:
        Wood, S. N. (2017). Generalized Additive Models: An Introduction
        with R (2nd ed.). Chapman and Hall/CRC.

        Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with
        B-splines and penalties. Statistical Science, 11(2), 89-121.

        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Args:
        latent_dimensions: Number of latent components. Must not exceed the
            number of features in any view. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            Default is True.
        n_knots: Number of knots per feature's B-spline term, passed
            straight through to ``sklearn.preprocessing.SplineTransformer(
            n_knots=...)``. Either a single value applied to every view or
            a list of per-view values. Default is 5.
        alpha: Ridge (smoothing) penalty strength(s) applied to every
            spline coefficient. Either a single float applied to every
            view or a list of per-view floats. Default is 0.1.
        max_iter: Maximum number of outer Newton iterations in the single
            joint ``"trust-krylov"`` solve (``scipy.optimize.minimize``'s own
            ``maxiter`` option). Default is 100.
        tol: Gradient-norm convergence tolerance for the joint solve
            (``scipy.optimize.minimize``'s own ``gtol`` option). Default is
            1e-6.
        random_state: Seed for the initial coefficients.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 5))
        >>> X2 = rng.standard_normal((200, 5))
        >>> model = GAMCCA(latent_dimensions=2).fit([X1, X2])
        >>> scores = model.transform([X1, X2])

        A different number of knots and penalty per view:

        >>> model = GAMCCA(latent_dimensions=2, n_knots=[5, 8], alpha=[0.1, 0.5]).fit(
        ...     [X1, X2]
        ... )
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "n_knots": [Interval(Integral, 2, None, closed="left"), "array-like"],
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        n_knots: int | list[int] = 5,
        alpha: float | list[float] = 0.1,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int = 0,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.n_knots = n_knots
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> GAMCCA:
        """Fit the GAMCCA model by one joint trust-region Newton-CG solve.

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
        n_knots_ = perview_parameter("n_knots", self.n_knots, 5, self.n_views_)
        alpha_ = perview_parameter("alpha", self.alpha, 0.1, self.n_views_)
        encoders = [_GamEncoder(X, k, nk) for X, nk in zip(views_, n_knots_)]
        bases = [enc.basis_ for enc in encoders]

        rng = np.random.default_rng(self.random_state)
        coefficients = ridge_basis_ey_trust_krylov(
            bases,
            cheap_orthonormal_projection_weights(bases, k, None, rng),
            alpha_,
            self.max_iter,
            self.tol,
        )
        representations = [basis @ coef for basis, coef in zip(bases, coefficients)]

        for enc, coef, rep in zip(encoders, coefficients, representations):
            enc.coef_ = coef
            enc._train_pred = rep

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
        return [enc.predict_new(v) for v, enc in zip(centred, self.encoders_)]

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
