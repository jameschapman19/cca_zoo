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


def _flatten(mats: list[np.ndarray]) -> np.ndarray:
    """Concatenate per-view coefficient matrices into one parameter vector."""
    return np.concatenate([mat.ravel() for mat in mats])


def _unflatten(x: np.ndarray, dims: list[int], k: int) -> list[np.ndarray]:
    """Inverse of :func:`_flatten`: split a flat vector into per-view blocks."""
    mats = []
    offset = 0
    for d in dims:
        size = d * k
        mats.append(x[offset : offset + size].reshape(d, k))
        offset += size
    return mats


def _gamcca_joint_obj_grad(
    x: np.ndarray,
    bases: list[np.ndarray],
    grams: list[np.ndarray],
    cross: list[list[np.ndarray]],
    ridge: float,
    dims: list[int],
    k: int,
) -> tuple[float, np.ndarray]:
    r"""Penalised EY loss and gradient w.r.t. *every* view's coefficients at once.

    Writing $Z_i = \text{bases}_i B_i$ for every view $i$, this is
    $\mathcal{L}_{EY}(Z_1, \dots, Z_M) + \tfrac12\lambda\sum_i\lVert B_i\rVert_F^2$
    as a function of $B_1, \dots, B_M$ flattened and concatenated into one
    vector, with its exact analytic gradient
    $\text{bases}_i^\top\nabla_{Z_i}\mathcal{L}_{EY} + \lambda B_i$ per block
    — the same two ingredients (:func:`~cca_zoo._utils._ey.ey_loss` and
    :func:`~cca_zoo._utils._ey.ey_grad_z`) every other EY-loss model in this
    package already uses. ``grams`` and ``cross`` are unused here; they are
    accepted only so this function shares a call signature with
    :func:`_gamcca_joint_hessp`, which :func:`scipy.optimize.minimize` calls
    with the same ``args``.

    Args:
        x: Candidate coefficients for every view, flattened and concatenated.
        bases: Fixed per-view (centred) B-spline design matrices.
        grams: ``bases[i].T @ bases[i]`` per view; unused (see above).
        cross: ``cross[i][a] = bases[i].T @ bases[a]`` for every pair; unused.
        ridge: Ridge penalty strength.
        dims: Number of basis columns per view (``bases[i].shape[1]``).
        k: Number of latent components.

    Returns:
        Tuple ``(loss, grad)`` with ``grad`` flattened the same way as ``x``.
    """
    coefs = _unflatten(x, dims, k)
    reps = [basis @ b for basis, b in zip(bases, coefs)]
    loss = ey_loss(reps)["objective"] + 0.5 * ridge * sum(
        float(np.sum(b**2)) for b in coefs
    )
    grad_z = ey_grad_z(reps)
    grads = [basis.T @ gz + ridge * b for basis, gz, b in zip(bases, grad_z, coefs)]
    return loss, _flatten(grads)


def _gamcca_joint_hessp(
    x: np.ndarray,
    p: np.ndarray,
    bases: list[np.ndarray],
    grams: list[np.ndarray],
    cross: list[list[np.ndarray]],
    ridge: float,
    dims: list[int],
    k: int,
) -> np.ndarray:
    r"""Exact Hessian-vector product of the penalised EY loss over *every* view at once.

    Returns the exact action of the full joint Hessian — every view, every
    latent component, all updated together, no view held fixed — on a
    direction $P_1, \dots, P_M$, without ever forming the
    $\left(\sum_i d_i k\right) \times \left(\sum_i d_i k\right)$ Hessian
    matrix itself. This differentiates the already-exact embedding gradient
    (:func:`~cca_zoo._utils._ey.ey_grad_z`) once more, jointly in every
    view's direction $\text{bases}_i P_i$ simultaneously, and pulls each
    block of the result back through $\text{bases}_i^\top$. Writing
    $G_i = \text{bases}_i^\top\text{bases}_i$,
    $K_{ia} = \text{bases}_i^\top\text{bases}_a$ (``cross[i][a]``), and $V$
    for the current mean auto-covariance:

    $$
    dV = \frac{1}{M(n-1)}\sum_a\left(P_a^\top G_a B_a + B_a^\top G_a P_a\right),
    \qquad
    Hp_i = \frac{4}{M(n-1)}\left[G_i P_i V + (G_i B_i)\,dV
        - \sum_a K_{ia} P_a\right] + \lambda P_i.
    $$

    The $-\sum_a K_{ia}P_a$ term is what a single-view-at-a-time Hessian
    would miss: it captures how perturbing *any* view's coefficients changes
    every other view's gradient through their shared $S = \sum_a Z_a$, which
    is exactly what makes this a genuinely joint (not merely block-diagonal)
    Hessian-vector product. Verified against finite differences of
    :func:`~cca_zoo._utils._ey.ey_grad_z` for $M = 2, 3, 4$ views with
    unequal per-view dimensions and $k = 1, 2, 3$.

    Args:
        x: Point at which the Hessian is evaluated (every view's current
            coefficients, flattened and concatenated the same way as ``p``).
        p: Direction, flattened and concatenated the same way as ``x``.
        bases: Fixed per-view (centred) B-spline design matrices.
        grams: ``bases[i].T @ bases[i]`` per view, precomputed once.
        cross: ``cross[i][a] = bases[i].T @ bases[a]`` for every pair,
            precomputed once.
        ridge: Ridge penalty strength.
        dims: Number of basis columns per view (``bases[i].shape[1]``).
        k: Number of latent components.

    Returns:
        $\{Hp_i\}$, flattened and concatenated the same way as ``x``.
    """
    coefs = _unflatten(x, dims, k)
    directions = _unflatten(p, dims, k)
    m = len(bases)
    n_minus_1 = bases[0].shape[0] - 1
    reps = [basis @ b for basis, b in zip(bases, coefs)]
    _, v = ey_cross_covariance(reps)
    dv = sum(
        pi.T @ (grams[i] @ coefs[i]) + coefs[i].T @ (grams[i] @ pi)
        for i, pi in enumerate(directions)
    ) / (m * n_minus_1)
    scale = 4.0 / (m * n_minus_1)

    hessian_vector_products = []
    for i in range(m):
        term1 = grams[i] @ directions[i] @ v
        term2 = (grams[i] @ coefs[i]) @ dv
        term3 = sum(cross[i][a] @ directions[a] for a in range(m))
        hp_i = scale * (term1 + term2 - term3) + ridge * directions[i]
        hessian_vector_products.append(hp_i)
    return _flatten(hessian_vector_products)


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
    a single trust-region Newton-CG solve (:func:`_gamcca_joint_obj_grad`,
    :func:`_gamcca_joint_hessp`), not by this class — it only builds and
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
    :class:`~sklearn.preprocessing.SplineTransformer`), fitting $B_1, \dots,
    B_M$ is conceptually a **P-IRLS** problem — the same repeated-penalised-
    quadratic-solve structure ``mgcv`` itself uses to fit a GAM, applied
    directly to the EY loss rather than a per-observation likelihood — but
    rather than hand-rolling that solve (or even cycling over views
    Gauss-Seidel-style), every view's coefficients are updated *jointly, in
    a single call*: $\mathcal{L}_{EY}$'s own exact gradient
    (:func:`~cca_zoo._utils._ey.ey_grad_z`) and exact Hessian-vector product
    across the *entire* stacked parameter vector $(B_1, \dots, B_M)$
    (:func:`_gamcca_joint_hessp`) are handed straight to
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
            n_knots=...)``. Default is 5.
        alpha: Ridge (smoothing) penalty strength applied to every spline
            coefficient. Default is 0.1.
        max_iter: Maximum number of outer Newton iterations in the single
            joint ``"trust-krylov"`` solve (``scipy.optimize.minimize``'s own
            ``maxiter`` option). Default is 100.
        tol: Gradient-norm convergence tolerance for the joint solve
            (``scipy.optimize.minimize``'s own ``gtol`` option). Default is
            1e-6.
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
        m = len(views_)
        encoders = [_GamEncoder(X, k, self.n_knots) for X in views_]
        bases = [enc.basis_ for enc in encoders]
        dims = [basis.shape[1] for basis in bases]
        grams = [basis.T @ basis for basis in bases]
        cross = [[bases[i].T @ bases[a] for a in range(m)] for i in range(m)]

        rng = np.random.default_rng(self.random_state)
        coefficients = cheap_orthonormal_projection_weights(bases, k, None, rng)
        x0 = _flatten(coefficients)

        result = minimize(
            _gamcca_joint_obj_grad,
            x0,
            args=(bases, grams, cross, self.alpha, dims, k),
            jac=True,
            hessp=_gamcca_joint_hessp,
            method="trust-krylov",
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        coefficients = _unflatten(result.x, dims, k)
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
