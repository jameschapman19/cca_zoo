"""GAMCCA — generalized-additive-model Canonical Correlation Analysis."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import SplineTransformer
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    ey_cross_covariance,
    ey_grad_z,
    ey_loss,
    random_orthogonal_embedding,
)
from cca_zoo._utils._validation import validate_views


def _diag_hessian(
    Z_i: np.ndarray,
    V: np.ndarray,
    n_views: int,
    n_minus_1: int,
    floor_percentile: float,
) -> np.ndarray:
    r"""Diagonal (per-sample) approximation of the EY loss's Hessian.

    The exact Hessian of $\mathcal{L}_{EY}$ w.r.t. one view's embedding
    $Z_i$ (holding the other views fixed) is

    $$
    \frac{\partial^2 \mathcal{L}_{EY}}{\partial Z_i^2}
        = \frac{4}{M(n-1)}(V-1)\,P + \frac{8}{M^2(n-1)^2}\, Z_i Z_i^\top
    $$

    where $P = I - \tfrac1n\mathbf{1}\mathbf{1}^\top$ is the centring
    projection (needed because the EY loss re-centres every embedding
    internally, so it is invariant to shifting $Z_i$ by a constant — the
    true Hessian must annihilate that direction) and $V$ is the (diagonal
    of the) mean auto-covariance. This is an $(n, n)$ matrix — using its
    diagonal as a per-sample weight is the same simplification every
    P-IRLS-based GAM/GLM solver already makes (treating observations as
    independent); the ``P`` term drops out of the diagonal (its diagonal
    entries are all $1 - 1/n$, folded into the same additive constant as
    the $(V-1)$ term).

    That diagonal is usable directly only after floor-damping it: near the
    loss's own well-conditioned fixed point ($V \approx 1$), the $(V-1)$
    term vanishes and the diagonal collapses to just $Z_{i,m}^2$ for each
    sample $m$ — the diagonal slice of a rank-1 matrix, an extremely poor
    per-sample curvature estimate for most samples (verified empirically:
    the per-sample ratio ``gradient / raw diagonal`` swings across several
    orders of magnitude with sign changes). Flooring at a high percentile
    of its own values — rather than a fixed constant, which would need
    re-tuning for every ``(n_samples, n_views)`` combination — is a
    self-calibrating Levenberg-Marquardt-style damping: it behaves like an
    (approximately) uniform weight for typical samples and only lets the
    Hessian's real signal through for high-leverage outliers.

    Args:
        Z_i: Current embedding for this view, shape (n_samples, k).
        V: Current (k, k) mean auto-covariance matrix (see
            :func:`cca_zoo._utils._ey.ey_cross_covariance`).
        n_views: Number of views, $M$.
        n_minus_1: $n - 1$, the sample-covariance denominator.
        floor_percentile: Percentile (0-100) of the raw diagonal used as
            the damping floor.

    Returns:
        Array of shape (n_samples, k): positive per-sample weights, one
        per latent component.
    """
    v_diag = np.diag(V)
    a_coef = 4.0 / (n_views * n_minus_1) * (v_diag - 1.0)
    b_coef = 8.0 / (n_views**2 * n_minus_1**2)
    raw = a_coef[None, :] + b_coef * Z_i**2
    floor = np.maximum(np.percentile(raw, floor_percentile, axis=0), 1e-10)
    result: np.ndarray = np.maximum(raw, floor)
    return result


class _GamEncoder:
    r"""Per-view additive-spline encoder, fit by P-IRLS + GCV/REML-style search.

    Mirrors how GAM software such as ``mgcv`` actually fits a smooth term:
    :class:`~sklearn.preprocessing.SplineTransformer` builds the per-feature
    B-spline design matrix (one contiguous block of columns per feature) and
    :class:`~sklearn.linear_model.Ridge` / :class:`~sklearn.linear_model.RidgeCV`
    do the penalised fitting, rather than either being reimplemented. Two
    kinds of step are exposed, matching ``mgcv``'s own two nested loops:

    - :meth:`inner_step` — one penalised-iteratively-reweighted-least-squares
      (P-IRLS) Newton update of the spline coefficients at the *current,
      fixed* smoothing parameter (``alphas_``): ridge-regress the working
      response $Z_i - \nabla_i / h_i$ (a Newton step on the EY loss, where
      $\nabla_i$ is :func:`cca_zoo._utils._ey.ey_grad_z`'s gradient and $h_i$
      the diagonal-Hessian weight from :func:`_diag_hessian`) onto the fixed
      basis, weighted by $h_i$.
    - :meth:`outer_step` — re-selects the smoothing parameter itself via
      :class:`~sklearn.linear_model.RidgeCV`'s efficient leave-one-out
      cross-validation (the same statistical job GCV/REML do in ``mgcv``),
      at whatever working response/weights the inner loop has most recently
      converged to.

    Used only during ``fit``.
    """

    def __init__(self, X: np.ndarray, k: int, n_knots: int) -> None:
        self.n, self.p = X.shape
        self.k = k
        self._spline = SplineTransformer(
            n_knots=n_knots,
            degree=3,
            knots="quantile",
            extrapolation="constant",
            include_bias=True,
        )
        self._basis: np.ndarray = self._spline.fit_transform(X)
        self.n_splines_: int = self._basis.shape[1] // self.p
        self.models_: list[Ridge] = []
        self.alphas_: np.ndarray = np.ones(k)
        self.whiten_: np.ndarray = np.eye(k)
        self.raw_mean_: np.ndarray = np.zeros(k)
        self._train_pred: np.ndarray = np.zeros((self.n, k))

    def predict(self) -> np.ndarray:
        """Encoder output on the training data, shape (n_samples, k)."""
        return self._train_pred

    def _update_from_raw(self, raw: np.ndarray) -> None:
        """Whiten a raw (pre-decorrelation) prediction and cache it.

        The raw per-component fit is not itself guaranteed zero-mean (the
        basis's own intercept-like capacity is fit freely), so it is
        explicitly re-centred here; ``raw_mean_`` is cached so
        :meth:`predict_new` and :meth:`feature_term` apply the exact same
        correction rather than each output silently carrying its own,
        slightly different constant offset.

        Args:
            raw: Raw per-component predictions, shape (n_samples, k).
        """
        self.raw_mean_ = raw.mean(axis=0)
        centred = raw - self.raw_mean_
        cov = (centred.T @ centred) / (centred.shape[0] - 1)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-8)
        self.whiten_ = vecs @ np.diag(vals**-0.5) @ vecs.T
        self._train_pred = centred @ self.whiten_

    def inner_step(
        self, Z_self: np.ndarray, grad: np.ndarray, diag_hess: np.ndarray
    ) -> None:
        """One P-IRLS Newton update at the current, fixed ``alphas_``.

        Args:
            Z_self: This view's current embedding, shape (n_samples, k).
            grad: EY-loss gradient for this view (see
                :func:`cca_zoo._utils._ey.ey_grad_z`), shape (n_samples, k).
            diag_hess: Diagonal-Hessian weights (see :func:`_diag_hessian`),
                shape (n_samples, k).
        """
        raw_cols = []
        models = []
        for c in range(self.k):
            working_response = Z_self[:, c] - grad[:, c] / diag_hess[:, c]
            model = Ridge(alpha=self.alphas_[c], fit_intercept=False)
            model.fit(self._basis, working_response, sample_weight=diag_hess[:, c])
            models.append(model)
            raw_cols.append(model.predict(self._basis))
        self.models_ = models
        self._update_from_raw(np.column_stack(raw_cols))

    def outer_step(
        self,
        Z_self: np.ndarray,
        grad: np.ndarray,
        diag_hess: np.ndarray,
        alpha_grid: np.ndarray,
    ) -> np.ndarray:
        """Re-select the smoothing parameter via RidgeCV's efficient LOOCV.

        Args:
            Z_self: This view's current embedding, shape (n_samples, k).
            grad: EY-loss gradient for this view, shape (n_samples, k).
            diag_hess: Diagonal-Hessian weights, shape (n_samples, k).
            alpha_grid: Candidate smoothing parameters to search over.

        Returns:
            The newly selected smoothing parameter per component, shape (k,).
        """
        raw_cols = []
        models = []
        new_alphas = np.empty(self.k)
        for c in range(self.k):
            working_response = Z_self[:, c] - grad[:, c] / diag_hess[:, c]
            model = RidgeCV(alphas=alpha_grid)
            model.fit(self._basis, working_response, sample_weight=diag_hess[:, c])
            models.append(model)
            new_alphas[c] = model.alpha_
            raw_cols.append(model.predict(self._basis))
        self.models_ = models
        self.alphas_ = new_alphas
        self._update_from_raw(np.column_stack(raw_cols))
        return new_alphas

    def predict_new(self, X: np.ndarray) -> np.ndarray:
        """Encoder output for arbitrary (e.g. test) data, shape (n, k)."""
        basis = self._spline.transform(X)
        raw = np.column_stack([m.predict(basis) for m in self.models_])
        result: np.ndarray = (raw - self.raw_mean_) @ self.whiten_
        return result

    def feature_term(self, feature_idx: int, x: np.ndarray) -> np.ndarray:
        """Single feature's additive contribution, shape (n, k).

        Args:
            feature_idx: Index of the input feature.
            x: Raw (mean-centred) values for that feature, shape (n,).

        Returns:
            Array of shape (n, k): this feature's term alone, for each
            latent component, after the same whitening transform applied
            to the full encoder output. Whitening is linear, so the
            feature-wise decomposition of the raw fit carries through
            exactly — summing this over every feature reproduces
            :meth:`predict` exactly — except for the overall mean
            correction cached in ``raw_mean_``, which has no natural home
            in any single feature (it is a property of the whole additive
            sum) and is therefore split evenly across the ``p`` features.
        """
        grid = np.zeros((len(x), self.p))
        grid[:, feature_idx] = x
        basis = self._spline.transform(grid)
        block = slice(
            feature_idx * self.n_splines_, (feature_idx + 1) * self.n_splines_
        )
        raw = np.column_stack([basis[:, block] @ m.coef_[block] for m in self.models_])
        result: np.ndarray = (raw - self.raw_mean_ / self.p) @ self.whiten_
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
    and :class:`~cca_zoo.tree.TreeCCA`).

    Unlike those models, which reach the EY loss's optimum by many small
    (stochastic-)gradient steps, GAMCCA fits it the way GAM software such as
    ``mgcv`` fits an ordinary GAM: by directly Newton-optimising the loss
    with **P-IRLS** (penalised iteratively reweighted least squares) for the
    basis coefficients, wrapped in an **outer** loop that re-selects each
    view's smoothing parameter by (efficient, leave-one-out) cross-validation
    — the same statistical job ``mgcv``'s GCV/REML step does — and
    alternating the two until both the coefficients and the smoothing
    parameters stop changing:

    1. **Inner (P-IRLS)**: for the *current, fixed* smoothing parameters,
       repeatedly form a Newton step on $\mathcal{L}_{EY}$ for each view in
       turn — a working response $Z_i - \nabla_i / h_i$ (from the analytic
       gradient :func:`~cca_zoo._utils._ey.ey_grad_z` and a diagonal-Hessian
       weight, see :func:`_diag_hessian`) ridge-fit onto that view's fixed
       B-spline basis — cycling through every view until the EY loss itself
       stops moving.
    2. **Outer (GCV-style)**: only once the inner loop has converged, re-fit
       each view's smoothing parameter with :class:`~sklearn.linear_model.RidgeCV`
       at that converged state, then re-run the inner loop at the new
       parameters. Repeat until the smoothing parameters stabilise too.

    This is a genuine departure from the boosting-based recipe
    :class:`~cca_zoo.tree.TreeCCA` uses (necessary there because a
    tree ensemble has no closed-form single-shot fit to a moving target):
    GAMCCA never takes a small shrunk step, and has no `learning_rate` or
    `n_estimators` to tune — each view's smoothing strength is chosen
    automatically, the same way it would be for any other GAM. Everything
    here is built from scikit-learn's own, already-required machinery
    (``SplineTransformer``, ``Ridge``, ``RidgeCV``); no optional dependency
    or custom Newton/GCV solver is needed.

    Note:
        The Hessian used above is a *diagonal* (per-sample) approximation
        of the true, dense Hessian (which has a rank-1 correction beyond
        diagonal — every sample's curvature is coupled to every other's
        through $\operatorname{Var}(Z_i)$). That approximation needs its own
        damping: near the loss's own well-conditioned fixed point
        ($V \approx 1$), the diagonal collapses to a per-sample value with
        an enormous, sign-changing dynamic range, so it is floored at a high
        percentile of its own distribution (self-calibrating, rather than a
        fixed constant that would need re-tuning per dataset) — this acts
        like Levenberg-Marquardt damping, treating typical samples with an
        approximately uniform weight and only letting the Hessian's signal
        through for high-leverage outliers. The same diagonal Hessian does
        *not* carry over usefully to :class:`~cca_zoo.tree.TreeCCA`'s
        boosting: tree splits aggregate `grad`/`hess` sums *within each
        leaf* as if independent, which requires the Hessian to genuinely be
        (close to) diagonal — true for an ordinary GLM's per-observation
        likelihood, false here, where the curvature is fundamentally a
        global, rank-1 object. A ridge fit instead solves one global,
        basis-regularised system, so it isn't sensitive to the same mismatch.

        A GAM's additive structure also assumes each feature contributes
        independently; it cannot represent a genuine *interaction* between
        two features of the same view (e.g. $x_1 x_2$) the way a
        multivariate tree split can. If cross-view structure only shows up
        through such interactions, expect :class:`~cca_zoo.tree.TreeCCA` to
        do better instead.

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
        alphas: Candidate smoothing parameters searched by
            :class:`~sklearn.linear_model.RidgeCV` in the outer loop.
            Default (``None``) uses ``numpy.logspace(-6, 3, 10)``.
        max_inner_iter: Maximum P-IRLS rounds (cycling once through every
            view per round) per outer iteration. Default is 50.
        max_outer_iter: Maximum smoothing-parameter re-selection rounds.
            Default is 10.
        tol: Inner-loop convergence tolerance, on the change in the EY loss
            between successive full passes over all views. Default is 1e-4.
        hess_floor_percentile: Percentile (0-100) of each round's raw
            diagonal-Hessian values used to floor them (see
            :func:`_diag_hessian`). Default is 90.0.
        random_state: Seed for drawing the random-orthogonal initial
            embedding. Default is 0.

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
        "max_inner_iter": [Interval(Integral, 1, None, closed="left")],
        "max_outer_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "hess_floor_percentile": [Interval(Real, 0, 100, closed="both")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        n_knots: int = 5,
        alphas: ArrayLike | None = None,
        max_inner_iter: int = 50,
        max_outer_iter: int = 10,
        tol: float = 1e-4,
        hess_floor_percentile: float = 90.0,
        random_state: int = 0,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.n_knots = n_knots
        self.alphas = alphas
        self.max_inner_iter = max_inner_iter
        self.max_outer_iter = max_outer_iter
        self.tol = tol
        self.hess_floor_percentile = hess_floor_percentile
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
        n = views_[0].shape[0]
        n_minus_1 = n - 1
        alpha_grid = (
            np.asarray(self.alphas)
            if self.alphas is not None
            else np.logspace(-6, 3, 10)
        )

        rng = np.random.default_rng(self.random_state)
        encoders = [_GamEncoder(X, k, self.n_knots) for X in views_]
        representations = []
        for X in views_:
            bm, _ = random_orthogonal_embedding(X, k, rng)
            representations.append(bm)

        prev_alphas = [enc.alphas_.copy() for enc in encoders]
        for outer_it in range(self.max_outer_iter):
            # Outer loop: re-select each view's smoothing parameter via
            # RidgeCV's efficient LOOCV, at the current representations.
            for i in range(n_views):
                grad = ey_grad_z(representations)[i]
                _, V = ey_cross_covariance(representations)
                diag_hess = _diag_hessian(
                    representations[i],
                    V,
                    n_views,
                    n_minus_1,
                    self.hess_floor_percentile,
                )
                encoders[i].outer_step(representations[i], grad, diag_hess, alpha_grid)
                representations[i] = encoders[i].predict()

            # Inner loop: P-IRLS Newton steps at these now-fixed smoothing
            # parameters, cycling through every view, until the EY loss
            # itself stops moving (see _diag_hessian's docstring for why
            # this, rather than raw per-sample values, is the right
            # convergence signal to track).
            prev_obj = ey_loss(representations)["objective"]
            for _ in range(self.max_inner_iter):
                for i in range(n_views):
                    grad = ey_grad_z(representations)[i]
                    _, V = ey_cross_covariance(representations)
                    diag_hess = _diag_hessian(
                        representations[i],
                        V,
                        n_views,
                        n_minus_1,
                        self.hess_floor_percentile,
                    )
                    encoders[i].inner_step(representations[i], grad, diag_hess)
                    representations[i] = encoders[i].predict()
                obj = ey_loss(representations)["objective"]
                if abs(obj - prev_obj) < self.tol:
                    break
                prev_obj = obj

            alphas_now = [enc.alphas_.copy() for enc in encoders]
            alpha_change = max(
                np.max(np.abs(np.log(now) - np.log(prev)))
                for now, prev in zip(alphas_now, prev_alphas)
            )
            prev_alphas = alphas_now
            if outer_it > 0 and alpha_change < 0.05:
                break

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
