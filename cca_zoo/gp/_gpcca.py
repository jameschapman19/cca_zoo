"""GaussianProcessCCA — Gaussian-process Canonical Correlation Analysis."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.cluster import kmeans_plusplus
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.utils import deprecated
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import coordinate_descent_ey
from cca_zoo._utils._validation import validate_views


class _GpEncoder:
    r"""Per-view kernel encoder: a fixed, centred cross-kernel basis.

    Writes the encoder as $f_i(x) = k(x, Z_i)^\top B_i$ for a *fixed* set of
    basis points $Z_i$ (``inducing_``) and kernel $k$ — the standard
    Nyström/"subset of regressors" reduced-rank construction (Quiñonero-
    Candela & Rasmussen, 2005). ``inducing_`` is every training row when
    ``n_inducing`` is ``None`` or at least ``n_samples`` (exact inference);
    otherwise it is ``n_inducing`` rows selected via
    :func:`sklearn.cluster.kmeans_plusplus`'s seeding. Centring the
    cross-kernel with :class:`~sklearn.preprocessing.KernelCenterer` (fit on
    the inducing-point Gram matrix, the same textbook double-centring
    :class:`~sklearn.decomposition.KernelPCA` uses) is what makes
    $Z_i = \text{basis}_i B_i$ automatically zero-mean for *any*
    coefficients $B_i$ — no separate recentring step is needed anywhere
    downstream, unlike the whitening/recentring the old Newton-step fit
    required.

    The coefficients $B_i$ (``coef_``) are fit by
    :func:`~cca_zoo._utils._ey.coordinate_descent_ey` with
    ``ridge_matrices`` set to the centred inducing-point Gram matrix — the
    RKHS-norm penalty $B_i^\top K_{mm} B_i$ a Gaussian process's own
    posterior mean actually minimises, not a plain $\|B_i\|_2^2$ that would
    ignore the kernel's geometry — not by this class, which only builds and
    holds the fixed basis and evaluates it once coefficients exist.

    Predictive uncertainty (``predict_new(..., return_std=True)``) does not
    depend on $B_i$ at all — a standard GP/kernel-ridge fact, since the
    posterior variance formula only involves the kernel, the noise level,
    and the design points, never the fitted targets — so it is obtained
    directly from an actual :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
    fit on the inducing points with a placeholder (all-zero) target and
    ``optimizer=None``: only its ``predict(..., return_std=True)``'s second
    output is ever used. This computes the exact posterior standard
    deviation under the (uncentred) GP prior implied by the same kernel,
    noise level, and inducing points as the mean fit — a standard, easily
    verified sparse-GP variance approximation, computed independently of
    the centred-basis construction used for the mean.
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int,
        kernel: Kernel,
        ridge: float,
        n_inducing: int | None,
        random_state: int,
    ) -> None:
        self.n, self.p = X.shape
        self.k = k
        if n_inducing is None or n_inducing >= self.n:
            self.inducing_: np.ndarray = X
        else:
            _, idx = kmeans_plusplus(
                X, n_clusters=n_inducing, random_state=random_state
            )
            self.inducing_ = X[idx]

        self.kernel_: Kernel = kernel
        k_ind_raw = self.kernel_(self.inducing_, self.inducing_)
        self._centerer = KernelCenterer().fit(k_ind_raw)
        self.ridge_matrix_: np.ndarray = self._centerer.transform(k_ind_raw)
        self.basis_: np.ndarray = self._centerer.transform(
            self.kernel_(X, self.inducing_)
        )
        self.coef_: np.ndarray = np.zeros((self.inducing_.shape[0], k))
        self._train_pred: np.ndarray = np.zeros((self.n, k))
        self._variance_model = GaussianProcessRegressor(
            kernel=self.kernel_, alpha=ridge, optimizer=None
        ).fit(self.inducing_, np.zeros(self.inducing_.shape[0]))

    def predict(self) -> np.ndarray:
        """Encoder output on the training data, shape (n_samples, k)."""
        return self._train_pred

    def predict_new(
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Encoder output for arbitrary (e.g. test) data, shape (n, k).

        Args:
            X: Input data, shape (n_samples, n_features).
            return_std: If True, also return the posterior standard
                deviation of each latent component (identical across
                components, since all ``k`` share the same kernel/noise/
                inducing points — see the class docstring).

        Returns:
            Array of shape (n, k), or, if ``return_std`` is True, a tuple
            ``(mean, std)`` of two arrays each of shape (n, k).
        """
        basis = self._centerer.transform(self.kernel_(X, self.inducing_))
        mean: np.ndarray = basis @ self.coef_
        if not return_std:
            return mean
        _, std = self._variance_model.predict(X, return_std=True)
        return mean, np.tile(std[:, None], (1, self.k))


class GaussianProcessCCA(BaseModel):
    r"""GaussianProcessCCA — nonlinear multiview CCA with Gaussian-process encoders.

    Learns one nonlinear encoder $f_i$ per view — a Gaussian process with a
    joint (non-additive) kernel over that view's raw feature vector — that
    jointly minimise the ridge-penalised Eckart-Young (EY) objective:

    $$
    \mathcal{L}_{EY} = -2 \operatorname{tr}(C) + \operatorname{tr}(V V)
    $$

    where, for embeddings $Z_i = f_i(X_i)$, $C$ is the mean pairwise
    cross-covariance (including $i = j$ terms) and $V$ the mean
    auto-covariance across all views (see :mod:`cca_zoo._utils._ey`, the
    same shared EY-loss machinery used by
    :class:`~cca_zoo.linear.gradient.CCAEY`, :class:`~cca_zoo.deep.DCCAEY`,
    :class:`~cca_zoo.tree.TreeCCA`, :class:`~cca_zoo.gam.GAMCCA`, and
    :class:`~cca_zoo.sparse.ElasticNetCCA`). Writing $f_i(x) =
    k(x, Z_i)^\top B_i$ for a fixed kernel $k$ and fixed basis points $Z_i$
    (:class:`_GpEncoder`; every training row, or ``n_inducing`` of them
    selected via :func:`sklearn.cluster.kmeans_plusplus`) turns fitting
    $B_i$ into exactly the same problem
    :class:`~cca_zoo.gam.GAMCCA` solves for its spline coefficients, with an
    RKHS-norm penalty $B_i^\top K_{mm} B_i$ in place of a plain ridge:
    both are fit by :func:`~cca_zoo._utils._ey.coordinate_descent_ey` —
    cyclic coordinate descent directly on $\mathcal{L}_{EY}$, one scalar
    coefficient at a time, each solved to its exact global minimiser (see
    that function's docstring for the derivation).

    Unlike :class:`~cca_zoo.gam.GAMCCA`'s per-feature additive splines, a
    joint kernel is not restricted to a sum of univariate terms: it can
    represent a genuine *interaction* between two features of the same view
    directly, the way a decision tree's multivariate splits can but an
    additive GAM structurally cannot.

    This is a deliberate departure from the P-IRLS-plus-marginal-likelihood
    recipe GaussianProcessCCA used before: that scheme reached a Newton
    step by working in the $n$-dimensional embedding space $Z_i$ with a
    per-sample diagonal approximation of $\mathcal{L}_{EY}$'s Hessian
    (passed to :class:`~sklearn.gaussian_process.GaussianProcessRegressor`
    as heteroscedastic ``alpha``), which then needed a post-hoc whitening/
    decorrelation retraction to compensate for the curvature the diagonal
    approximation throws away (see
    :func:`~cca_zoo._utils._ey.ey_diag_hessian`'s docstring). Fitting
    directly in the encoder's own (much smaller, $m$-dimensional)
    coefficient space instead needs no such approximation or retraction:
    coordinate descent solves the *exact* (quartic, not linearised)
    restriction of $\mathcal{L}_{EY}$ to each coefficient. The trade-off is
    that kernel hyperparameters (lengthscales, signal variance) are now
    fixed — pass ``kernel`` explicitly, or tune it externally (e.g. with
    :class:`~sklearn.model_selection.GridSearchCV`, since this is an
    ordinary ``BaseEstimator``) — rather than automatically re-selected
    each round by the GP's own marginal-likelihood optimisation; that
    automatic search was specific to the working-response/Newton-step
    framing and has no direct analogue once fitting minimises the true
    quartic loss instead.

    As a Bayesian model it still comes with calibrated predictive
    uncertainty for free: :meth:`transform` can return each latent
    component's posterior standard deviation alongside its mean. This does
    not depend on the fitted coefficients at all — a standard GP fact, since
    posterior variance only involves the kernel, the noise level, and the
    design points — so it is computed directly by an actual
    :class:`~sklearn.gaussian_process.GaussianProcessRegressor` fit with a
    placeholder target purely to reuse its variance formula (see
    :class:`_GpEncoder`).

    Note:
        `n_inducing` no longer switches to a different, approximate
        *algorithm* the way it used to (the old Deterministic Training
        Conditional recipe): exact and sparse inference are now both the
        same reduced-rank ("subset of regressors") construction, differing
        only in how many basis points $Z_i$ are used ($n$, i.e. every
        training row, for exact; ``n_inducing`` < $n$, chosen by
        ``kmeans_plusplus``, for sparse) — so fitting still costs
        $O(n m^2 + m^3)$ for $m$ basis points, linear in $n$ once $m \ll n$.

    References:
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes
        for Machine Learning. MIT Press.

        Quiñonero-Candela, J., & Rasmussen, C. E. (2005). A Unifying View
        of Sparse Approximate Gaussian Process Regression. Journal of
        Machine Learning Research, 6, 1939-1959.

        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.

    Args:
        latent_dimensions: Number of latent components. Must not exceed the
            number of features in any view. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            Default is True.
        kernel: Fixed kernel used for every view. ``None`` (the default)
            uses ``ConstantKernel(1.0) * RBF(length_scale=np.ones(p))`` for
            each view's own feature count ``p``. If given explicitly, the
            same kernel (cloned per view) is used for every view, so its
            hyperparameters (e.g. an ARD ``length_scale`` array) must be
            compatible with every view's feature count.
        alpha: Ridge (RKHS-norm) penalty strength, also used as the noise
            level for the posterior-variance calculation. Default is 0.01.
        n_inducing: Number of basis ("inducing") points. ``None`` (the
            default) uses every training row (exact inference, appropriate
            up to a few thousand samples). For larger datasets, set this to
            a few hundred to make fitting scale as
            $O(n \, \text{n\_inducing}^2)$ instead of $O(n^3)$. Values at or
            above the number of training samples fall back to exact
            inference automatically.
        max_iter: Maximum number of full coordinate-descent sweeps (every
            view, feature, and component once each). Default is 100.
        tol: Convergence tolerance on the penalised objective's change
            between consecutive sweeps. Default is 1e-6.
        random_state: Seed for the initial coefficients and (if
            ``n_inducing`` is set) for selecting inducing points.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 3))
        >>> X2 = rng.standard_normal((100, 3))
        >>> model = GaussianProcessCCA(latent_dimensions=1).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
        >>> means, stds = model.transform([X1, X2], return_std=True)
        >>> # For larger datasets, cap inference cost with inducing points:
        >>> big_model = GaussianProcessCCA(latent_dimensions=1, n_inducing=200)
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "n_inducing": [None, Interval(Integral, 2, None, closed="left")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        kernel: Kernel | None = None,
        alpha: float = 0.01,
        n_inducing: int | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int = 0,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.kernel = kernel
        self.alpha = alpha
        self.n_inducing = n_inducing
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> GaussianProcessCCA:
        """Fit the GaussianProcessCCA model by coordinate descent on the EY loss.

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

        encoders = [
            _GpEncoder(
                X,
                k,
                (
                    clone(self.kernel)
                    if self.kernel is not None
                    else ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1]))
                ),
                self.alpha,
                self.n_inducing,
                self.random_state,
            )
            for X in views_
        ]

        rng = np.random.default_rng(self.random_state)
        coefficients, representations = coordinate_descent_ey(
            bases=[enc.basis_ for enc in encoders],
            k=k,
            alpha=self.alpha,
            l1_ratio=0.0,
            max_iter=self.max_iter,
            tol=self.tol,
            rng=rng,
            ridge_matrices=[enc.ridge_matrix_ for enc in encoders],
        )
        for enc, coef, rep in zip(encoders, coefficients, representations):
            enc.coef_ = coef
            enc._train_pred = rep

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
                deviation alongside its mean.

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
        """Not implemented for GaussianProcessCCA.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            NotImplementedError: GaussianProcessCCA encoders are Gaussian processes over
                the joint feature vector, not linear weight matrices, and
                have no per-feature decomposition analogous to
                :meth:`~cca_zoo.gam.GAMCCA.shape_function` (the kernel is
                not additive across features).
        """
        check_is_fitted(self)
        raise NotImplementedError(
            "GaussianProcessCCA has no linear weight matrices; its encoders are "
            "Gaussian processes with a joint (non-additive) kernel over "
            "each view's raw features, so there is no per-feature "
            "decomposition to expose."
        )


@deprecated(
    "Renamed to GaussianProcessCCA for sklearn-style naming "
    "(matching GaussianProcessRegressor/Classifier); use GaussianProcessCCA instead."
)
class GPCCA(GaussianProcessCCA):
    pass
