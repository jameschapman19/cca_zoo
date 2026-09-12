"""GPCCA — Gaussian-process Canonical Correlation Analysis."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    ey_cross_covariance,
    ey_diag_hessian,
    ey_grad_z,
    ey_loss,
    random_orthogonal_embedding,
)
from cca_zoo._utils._validation import validate_views


class _GpEncoder:
    r"""Per-view joint-kernel GP encoder, fit by inner/outer Newton + ML-II steps.

    Where :class:`~cca_zoo.gam._gamcca._GamEncoder` sums one univariate
    B-spline term per feature (additive, so it cannot see cross-feature
    interactions), this encoder fits one
    :class:`~sklearn.gaussian_process.GaussianProcessRegressor` per latent
    component directly on the raw (joint) feature vector, using an ARD
    (per-dimension-lengthscale) RBF kernel — a genuine, non-additive
    function of all of that view's features at once. The two step kinds
    mirror ``_GamEncoder``'s inner/outer split exactly, with the GP's own
    machinery standing in for ``Ridge``/``RidgeCV``:

    - :meth:`inner_step` — one Newton step on the EY loss (a working
      response $Z_i - \nabla_i / h_i$, from
      :func:`~cca_zoo._utils._ey.ey_grad_z` and
      :func:`~cca_zoo._utils._ey.ey_diag_hessian`) fit with the kernel's
      hyperparameters held fixed at their current value
      (``optimizer=None``), passing the diagonal-Hessian weights as the
      GP's per-sample ``alpha`` (heteroscedastic observation noise).
    - :meth:`outer_step` — re-fits the same working response with the
      kernel hyperparameters (lengthscales, signal variance) free to move,
      via the GP's own default marginal-likelihood optimisation — the
      direct GP analogue of GCV/REML smoothing-parameter search.

    Used only during ``fit``.
    """

    def __init__(self, X: np.ndarray, k: int) -> None:
        self.n, self.p = X.shape
        self.k = k
        self.X = X
        self.kernels_: list[Any] = [
            ConstantKernel(1.0) * RBF(length_scale=np.ones(self.p)) for _ in range(k)
        ]
        self.models_: list[GaussianProcessRegressor] = []
        self.whiten_: np.ndarray = np.eye(k)
        self.raw_mean_: np.ndarray = np.zeros(k)
        self._train_pred: np.ndarray = np.zeros((self.n, k))

    def predict(self) -> np.ndarray:
        """Encoder output on the training data, shape (n_samples, k)."""
        return self._train_pred

    def _update_from_raw(self, raw: np.ndarray) -> None:
        """Whiten a raw (pre-decorrelation) prediction and cache it.

        See :meth:`cca_zoo.gam._gamcca._GamEncoder._update_from_raw` for why
        this re-centring, and caching ``raw_mean_`` for reuse by
        :meth:`predict_new`, is necessary.
        """
        self.raw_mean_ = raw.mean(axis=0)
        centred = raw - self.raw_mean_
        cov = (centred.T @ centred) / (centred.shape[0] - 1)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-8)
        self.whiten_ = vecs @ np.diag(vals**-0.5) @ vecs.T
        self._train_pred = centred @ self.whiten_

    def _working_response_and_alpha(
        self, Z_self: np.ndarray, grad: np.ndarray, diag_hess: np.ndarray, c: int
    ) -> tuple[np.ndarray, np.ndarray]:
        working_response = Z_self[:, c] - grad[:, c] / diag_hess[:, c]
        alpha = np.clip(1.0 / diag_hess[:, c], 1e-6, 1e6)
        return working_response, alpha

    def inner_step(
        self, Z_self: np.ndarray, grad: np.ndarray, diag_hess: np.ndarray
    ) -> None:
        """One Newton update with the kernel hyperparameters held fixed.

        Args:
            Z_self: This view's current embedding, shape (n_samples, k).
            grad: EY-loss gradient for this view (see
                :func:`~cca_zoo._utils._ey.ey_grad_z`), shape (n_samples, k).
            diag_hess: Diagonal-Hessian weights (see
                :func:`~cca_zoo._utils._ey.ey_diag_hessian`), shape (n_samples, k).
        """
        raw_cols = []
        models = []
        for c in range(self.k):
            working_response, alpha = self._working_response_and_alpha(
                Z_self, grad, diag_hess, c
            )
            model = GaussianProcessRegressor(
                kernel=self.kernels_[c], alpha=alpha, optimizer=None
            )
            model.fit(self.X, working_response)
            models.append(model)
            raw_cols.append(model.predict(self.X))
        self.models_ = models
        self._update_from_raw(np.column_stack(raw_cols))

    def outer_step(
        self, Z_self: np.ndarray, grad: np.ndarray, diag_hess: np.ndarray
    ) -> None:
        """Re-fit the kernel hyperparameters via marginal-likelihood search.

        Args:
            Z_self: This view's current embedding, shape (n_samples, k).
            grad: EY-loss gradient for this view, shape (n_samples, k).
            diag_hess: Diagonal-Hessian weights, shape (n_samples, k).
        """
        raw_cols = []
        models = []
        for c in range(self.k):
            working_response, alpha = self._working_response_and_alpha(
                Z_self, grad, diag_hess, c
            )
            model = GaussianProcessRegressor(kernel=self.kernels_[c], alpha=alpha)
            model.fit(self.X, working_response)
            models.append(model)
            self.kernels_[c] = model.kernel_
            raw_cols.append(model.predict(self.X))
        self.models_ = models
        self._update_from_raw(np.column_stack(raw_cols))

    def predict_new(
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Encoder output for arbitrary (e.g. test) data, shape (n, k).

        Args:
            X: Input data, shape (n_samples, n_features).
            return_std: If True, also return the posterior standard
                deviation of each latent component, propagated through the
                (linear) whitening transform.

        Returns:
            Array of shape (n, k), or, if ``return_std`` is True, a tuple
            ``(mean, std)`` of two arrays each of shape (n, k).
        """
        if return_std:
            raw_mean = np.empty((X.shape[0], self.k))
            raw_std = np.empty((X.shape[0], self.k))
            for c, m in enumerate(self.models_):
                raw_mean[:, c], raw_std[:, c] = m.predict(X, return_std=True)
            mean = (raw_mean - self.raw_mean_) @ self.whiten_
            # Each component's raw GP posterior is fit independently, so
            # the raw covariance is diagonal; propagating variances through
            # the whitening transform is then a plain matrix product of
            # variances against squared whitening weights (no cross terms).
            var = raw_std**2 @ (self.whiten_**2)
            return mean, np.sqrt(var)
        raw = np.column_stack([m.predict(X) for m in self.models_])
        result: np.ndarray = (raw - self.raw_mean_) @ self.whiten_
        return result


class GPCCA(BaseModel):
    r"""GPCCA — nonlinear multiview CCA with Gaussian-process encoders.

    Learns one nonlinear encoder $f_i$ per view — a Gaussian process with
    an ARD (per-feature-lengthscale) RBF kernel over that view's *raw,
    joint* feature vector — that jointly maximise the Eckart-Young (EY)
    unconstrained-CCA objective:

    $$
    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
    $$

    where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean pairwise
    cross-covariance (including $i = j$ terms) and $V$ the mean
    auto-covariance across all views (see :mod:`cca_zoo._utils._ey`, the
    same shared EY-loss machinery used by
    :class:`~cca_zoo.linear.gradient.CCA_EY`, :class:`~cca_zoo.deep.DCCA_EY`,
    :class:`~cca_zoo.tree.TreeCCA`, and :class:`~cca_zoo.gam.GAMCCA`).

    Unlike :class:`~cca_zoo.gam.GAMCCA`'s per-feature additive splines, a
    GP with a joint kernel is not restricted to a sum of univariate terms:
    it can represent a genuine *interaction* between two features of the
    same view (e.g. cross-view structure that only shows up through
    $x_1 x_2$, not through $x_1$ or $x_2$ alone) directly and efficiently,
    the way a decision tree's multivariate splits can but an additive GAM
    structurally cannot. As a Bayesian model it also comes with calibrated
    predictive uncertainty for free: :meth:`transform` can return each
    latent component's posterior standard deviation alongside its mean.

    Fitting follows the same inner/outer split as
    :class:`~cca_zoo.gam.GAMCCA`'s P-IRLS/GCV recipe, with the GP's own
    machinery in place of ``Ridge``/``RidgeCV``:

    1. **Inner (fixed-kernel Newton steps)**: for the *current* kernel
       hyperparameters, repeatedly form a Newton step on
       $\mathcal{L}_{EY}$ for each view in turn — a working response
       $Z_i - \nabla_i / h_i$ (from the analytic gradient
       :func:`~cca_zoo._utils._ey.ey_grad_z` and a diagonal-Hessian weight,
       see :func:`~cca_zoo._utils._ey.ey_diag_hessian`) fit with
       :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
       (``optimizer=None``, so the kernel is held fixed), passing the
       diagonal-Hessian weights as the GP's per-sample ``alpha``
       (heteroscedastic observation noise) — cycling through every view
       until the EY loss itself stops moving.
    2. **Outer (marginal-likelihood kernel search)**: only once the inner
       loop has converged, re-fit each view's kernel hyperparameters at
       that converged working response via the GP's own default
       log-marginal-likelihood optimisation (the same statistical role
       GCV/REML plays for GAMCCA), then re-run the inner loop at the new
       kernel. Repeat until the kernel hyperparameters stabilise too.

    Everything here is built from scikit-learn's own, already-required
    ``GaussianProcessRegressor``/``RBF``/``ConstantKernel``; no optional
    dependency or custom solver is needed.

    Note:
        As with :class:`~cca_zoo.gam.GAMCCA`, the diagonal-Hessian weight
        is only a per-sample approximation of the loss's true (dense,
        rank-1-coupled) Hessian — see
        :func:`~cca_zoo._utils._ey.ey_diag_hessian`'s docstring for the
        full derivation and why it needs floor-damping. It is used here in
        exactly the role a GP's heteroscedastic ``alpha`` already exists
        to play (how much to trust each observation), so no further
        adaptation is needed beyond what :class:`~cca_zoo.gam.GAMCCA`
        already requires.

        A GP over the raw feature vector scales cubically in the number of
        training samples (exact GP inference); for very large datasets
        expect fitting to be markedly slower than
        :class:`~cca_zoo.gam.GAMCCA` or :class:`~cca_zoo.tree.TreeCCA`.

    References:
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes
        for Machine Learning. MIT Press.

        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Args:
        latent_dimensions: Number of latent components. Must not exceed the
            number of features in any view. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            Default is True.
        max_inner_iter: Maximum fixed-kernel Newton rounds (cycling once
            through every view per round) per outer iteration. Default is 15.
        max_outer_iter: Maximum kernel-hyperparameter re-selection rounds.
            Default is 4.
        tol: Inner-loop convergence tolerance, on the change in the EY loss
            between successive full passes over all views. Default is 1e-3.
        hess_floor_percentile: Percentile (0-100) of each round's raw
            diagonal-Hessian values used to floor them (see
            :func:`~cca_zoo._utils._ey.ey_diag_hessian`). Default is 90.0.
        random_state: Seed for drawing the random-orthogonal initial
            embedding. Default is 0.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 3))
        >>> X2 = rng.standard_normal((100, 3))
        >>> model = GPCCA(latent_dimensions=1).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
        >>> means, stds = model.transform([X1, X2], return_std=True)
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "max_inner_iter": [Interval(Integral, 1, None, closed="left")],
        "max_outer_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "hess_floor_percentile": [Interval(Real, 0, 100, closed="both")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        max_inner_iter: int = 15,
        max_outer_iter: int = 4,
        tol: float = 1e-3,
        hess_floor_percentile: float = 90.0,
        random_state: int = 0,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.max_inner_iter = max_inner_iter
        self.max_outer_iter = max_outer_iter
        self.tol = tol
        self.hess_floor_percentile = hess_floor_percentile
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> GPCCA:
        """Fit the GPCCA model.

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

        rng = np.random.default_rng(self.random_state)
        encoders = [_GpEncoder(X, k) for X in views_]
        representations = []
        for X in views_:
            bm, _ = random_orthogonal_embedding(X, k, rng)
            representations.append(bm)

        for outer_it in range(self.max_outer_iter):
            # Outer loop: re-select each view's kernel hyperparameters via
            # marginal-likelihood optimisation, at the current representations.
            for i in range(n_views):
                grad = ey_grad_z(representations)[i]
                _, V = ey_cross_covariance(representations)
                diag_hess = ey_diag_hessian(
                    representations[i],
                    V,
                    n_views,
                    n_minus_1,
                    self.hess_floor_percentile,
                )
                encoders[i].outer_step(representations[i], grad, diag_hess)
                representations[i] = encoders[i].predict()

            # Inner loop: fixed-kernel Newton steps, cycling through every
            # view, until the EY loss itself stops moving (see
            # ey_diag_hessian's docstring for why this, rather than raw
            # per-sample values, is the right convergence signal to track).
            prev_obj = ey_loss(representations)["objective"]
            for _ in range(self.max_inner_iter):
                for i in range(n_views):
                    grad = ey_grad_z(representations)[i]
                    _, V = ey_cross_covariance(representations)
                    diag_hess = ey_diag_hessian(
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

        self.encoders_: list[_GpEncoder] = encoders
        return self

    def transform(  # type: ignore[override]
        self, views: list[ArrayLike], return_std: bool = False
    ) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]:
        """Project views into the latent space using the fitted encoders.

        Args:
            views: List of arrays, each (n_samples, n_features_i), matching
                the number of views passed to ``fit``.
            return_std: If True, also return each view's posterior standard
                deviation, propagated through the whitening transform.

        Returns:
            List of arrays, each (n_samples, latent_dimensions); or, if
            ``return_std`` is True, a tuple ``(means, stds)`` of two such
            lists.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            ValueError: If fewer than 2 views are provided.
        """
        check_is_fitted(self)
        validated = validate_views(views)
        centred = [v - m for v, m in zip(validated, self.means_)]
        if return_std:
            means = []
            stds = []
            for v, enc in zip(centred, self.encoders_):
                mean, std = enc.predict_new(v, return_std=True)
                means.append(mean)
                stds.append(std)
            return means, stds
        return [enc.predict_new(v) for v, enc in zip(centred, self.encoders_)]

    @property
    def weights(self) -> list[np.ndarray]:
        """Not implemented for GPCCA.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            NotImplementedError: GPCCA encoders are Gaussian processes over
                the joint feature vector, not linear weight matrices, and
                have no per-feature decomposition analogous to
                :meth:`~cca_zoo.gam.GAMCCA.shape_function` (the kernel is
                not additive across features).
        """
        check_is_fitted(self)
        raise NotImplementedError(
            "GPCCA has no linear weight matrices; its encoders are "
            "Gaussian processes with a joint (non-additive) kernel over "
            "each view's raw features, so there is no per-feature "
            "decomposition to expose."
        )
