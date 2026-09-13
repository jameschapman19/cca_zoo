"""GAMCCA — generalized-additive-model Canonical Correlation Analysis."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from sklearn.preprocessing import SplineTransformer
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    cheap_orthonormal_projection_weights,
    ey_cross_covariance,
    ey_grad_z,
    ey_loss,
)
from cca_zoo._utils._validation import validate_views


def _pirls_component_step(
    bases: list[np.ndarray],
    grams: list[np.ndarray],
    coefficients: list[np.ndarray],
    representations: list[np.ndarray],
    view_idx: int,
    component: int,
    ridge: float,
) -> None:
    r"""P-IRLS update of a single component's coefficients via trust-region Newton.

    Updates ``coefficients[view_idx][:, component]`` in place (and the
    matching column of ``representations[view_idx]``), solving
    $\mathcal{L}_{EY}$'s exact (not diagonal-approximated) restriction to
    that one component's coefficient vector $b$ to a local optimum — every
    other component and view held fixed. This is ``mgcv``'s own P-IRLS
    structure: a penalised weighted-least-squares-shaped solve per
    (view, component) per round. $\mathcal{L}_{EY}$ is not convex, so the
    Hessian below can be indefinite away from the loss's own fixed point;
    :func:`scipy.optimize.minimize`'s ``"trust-exact"`` solver handles that
    (and the accompanying step-acceptance logic) directly.

    Writing $Z_i = \text{bases}_i B_i$, the exact gradient and Hessian of
    the *penalised* EY loss with respect to $b = B_i[:, c]$ (bases$_i$
    already column-centred, other columns/views fixed) are:

    $$
    g = \text{bases}_i^\top \nabla_{Z_i}\mathcal{L}_{EY}[:, c] + \lambda b,
    \qquad
    H = \frac{4}{M(n-1)}(V_{cc}-1) G_i
      + \frac{4}{M^2(n-1)^2}\left((G_iB_i)(G_iB_i)^\top + (G_ib)(G_ib)^\top\right)
      + \lambda I
    $$

    where $G_i = \text{bases}_i^\top\text{bases}_i$ and $V$ is the current
    mean auto-covariance (see :func:`~cca_zoo._utils._ey.ey_cross_covariance`).
    The non-ridge part of $H$ has rank $\le k+1$.

    Args:
        bases: Fixed per-view (centred) design matrices.
        grams: Precomputed ``bases[i].T @ bases[i]`` per view.
        coefficients: Current per-view coefficient matrices, updated in
            place.
        representations: Current per-view embeddings
            (``bases[i] @ coefficients[i]``), updated in place.
        view_idx: Which view's component to update.
        component: Which latent component to update.
        ridge: Ridge penalty strength.
    """
    m = len(bases)
    n_minus_1 = bases[0].shape[0] - 1
    basis = bases[view_idx]
    gram = grams[view_idx]
    coef = coefficients[view_idx]
    b0 = coef[:, component].copy()

    def _set(b: np.ndarray) -> None:
        coef[:, component] = b
        representations[view_idx][:, component] = basis @ b

    def _fun(b: np.ndarray) -> float:
        _set(b)
        obj: float = ey_loss(representations)["objective"] + 0.5 * ridge * (b @ b)
        return obj

    def _grad(b: np.ndarray) -> np.ndarray:
        _set(b)
        grad_all = ey_grad_z(representations)
        grad: np.ndarray = basis.T @ grad_all[view_idx][:, component] + ridge * b
        return grad

    def _hess(b: np.ndarray) -> np.ndarray:
        _set(b)
        _, v = ey_cross_covariance(representations)
        alpha_v = (4.0 / (m * n_minus_1)) * (v[component, component] - 1.0)
        beta = 4.0 / (m**2 * n_minus_1**2)
        gb_all = gram @ coef
        u_c = gb_all[:, component]
        hess: np.ndarray = (
            alpha_v * gram
            + beta * (gb_all @ gb_all.T + np.outer(u_c, u_c))
            + ridge * np.eye(len(b))
        )
        return hess

    result = minimize(_fun, b0, jac=_grad, hess=_hess, method="trust-exact")
    _set(result.x)


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

    The coefficients $B_i$ (``coef_``) are fit by P-IRLS
    (:func:`_pirls_component_step`), not by this class — it only builds and
    holds the fixed basis, and evaluates it (``predict``, ``predict_new``,
    ``feature_term``) once coefficients exist.
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
    :class:`~sklearn.preprocessing.SplineTransformer`) turns fitting $B_i$
    into a **P-IRLS** recipe — the same iteration structure ``mgcv`` itself
    uses to fit a GAM, applied directly to the EY loss rather than a
    per-observation likelihood: for one latent component's coefficient
    vector $b = B_i[:, c]$ at a time (every other component and view held
    fixed), take a damped Newton step using $\mathcal{L}_{EY}$'s *exact*
    gradient and Hessian restricted to $b$ — a single penalised, ridge-shaped
    $d_i \times d_i$ linear solve, cycling through every component and view
    in turn (see :func:`_pirls_component_step` for the exact derivation).

    $\mathcal{L}_{EY}$ is not convex, so this Hessian is only guaranteed
    positive semi-definite near the loss's own fixed point; ``"trust-exact"``
    handles the indefinite case directly. Each view's smoothing strength
    (``alpha``) is a fixed hyperparameter — there is no automatic
    smoothing-parameter search.

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
            n_knots=...)``. Default is 5.
        alpha: Ridge (smoothing) penalty strength applied to every spline
            coefficient. Default is 0.1.
        max_iter: Maximum number of full P-IRLS sweeps (one Newton solve per
            view and component each). Default is 100.
        tol: Convergence tolerance on the penalised objective's change
            between consecutive sweeps. Default is 1e-6.
        random_state: Seed for the initial coefficients.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 5))
        >>> X2 = rng.standard_normal((200, 5))
        >>> model = GAMCCA(latent_dimensions=2).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "n_knots": [Interval(Integral, 2, None, closed="left")],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        n_knots: int = 5,
        alpha: float = 0.1,
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
        """Fit the GAMCCA model by P-IRLS on the EY loss.

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
        m = len(views_)
        encoders = [_GamEncoder(X, k, self.n_knots) for X in views_]
        bases = [enc.basis_ for enc in encoders]
        grams = [basis.T @ basis for basis in bases]

        rng = np.random.default_rng(self.random_state)
        coefficients = cheap_orthonormal_projection_weights(bases, k, None, rng)
        representations = [b @ c for b, c in zip(bases, coefficients)]

        prev_obj = np.inf
        for _ in range(self.max_iter):
            for view_idx in range(m):
                for component in range(k):
                    _pirls_component_step(
                        bases,
                        grams,
                        coefficients,
                        representations,
                        view_idx,
                        component,
                        self.alpha,
                    )
            penalty = 0.5 * self.alpha * sum(np.sum(c**2) for c in coefficients)
            obj = ey_loss(representations)["objective"] + penalty
            if abs(prev_obj - obj) < self.tol:
                break
            prev_obj = obj

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
