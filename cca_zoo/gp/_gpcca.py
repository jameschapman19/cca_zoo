"""Gaussian-process CCA."""

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
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import penalised_basis_ey_closed_form
from cca_zoo._utils._param_constraints import RANDOM_STATE
from cca_zoo._utils._validation import perview_parameter


class _GpEncoder:
    r"""Per-view kernel encoder $f(x) = k_c(x, U)^\top B$ on fixed inducing points $U$.

    The subset-of-regressors construction (Quiñonero-Candela & Rasmussen,
    2005): $U$ is every training row, or ``n_inducing`` rows seeded by
    :func:`~sklearn.cluster.kmeans_plusplus`, and $k_c$ is the kernel centred
    in feature space on $U$ with :class:`~sklearn.preprocessing.KernelCenterer`.
    With $k_c(U, U) = V \operatorname{diag}(\mu) V^\top$, writing
    $B = V \operatorname{diag}(\mu)^{-1/2} a$ turns the RKHS norm
    $\operatorname{tr}(B^\top k_c(U, U) B)$ into $\|a\|^2$ and the encoder
    into a linear map of the Nyström features $k_c(x, U) V
    \operatorname{diag}(\mu)^{-1/2}$. The posterior standard deviation does
    not depend on $B$, so it comes from a
    :class:`~sklearn.gaussian_process.GaussianProcessRegressor` fitted on $U$.
    """

    def __init__(
        self,
        X: np.ndarray,
        k: int,
        kernel: Kernel,
        ridge: float,
        n_inducing: int | None,
        random_state: int | None,
    ) -> None:
        self.k = k
        if n_inducing is None or n_inducing >= X.shape[0]:
            self.inducing_: np.ndarray = X
        else:
            _, idx = kmeans_plusplus(
                X, n_clusters=n_inducing, random_state=random_state
            )
            self.inducing_ = X[idx]

        self.kernel_: Kernel = kernel
        inducing_kernel = self.kernel_(self.inducing_, self.inducing_)
        self._centerer = KernelCenterer().fit(inducing_kernel)
        mu, vectors = np.linalg.eigh(self._centerer.transform(inducing_kernel))
        # Centring leaves the constant direction null; keep the numerically
        # positive spectrum, as numpy.linalg.matrix_rank does.
        keep = mu > mu.max() * len(mu) * np.finfo(mu.dtype).eps
        self._nystroem: np.ndarray = vectors[:, keep] / np.sqrt(mu[keep])
        self.coef_: np.ndarray = np.zeros((self.inducing_.shape[0], k))
        self._variance_model = GaussianProcessRegressor(
            kernel=self.kernel_, alpha=ridge, optimizer=None
        ).fit(self.inducing_, np.zeros(self.inducing_.shape[0]))

    def features(self, X: np.ndarray) -> np.ndarray:
        """Nyström features of ``X``, whose coefficients' norm is the RKHS norm."""
        features: np.ndarray = self.basis(X) @ self._nystroem
        return features

    def basis(self, X: np.ndarray) -> np.ndarray:
        """The centred cross-kernel of ``X`` with the inducing points."""
        basis: np.ndarray = self._centerer.transform(self.kernel_(X, self.inducing_))
        return basis

    def predict_new(self, X: np.ndarray) -> np.ndarray:
        """Encoder output for new data, shape (n, k)."""
        mean: np.ndarray = self.basis(X) @ self.coef_
        return mean

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Posterior standard deviation of the output, the same for every component."""
        _, std = self._variance_model.predict(X, return_std=True)
        return np.tile(std[:, None], (1, self.k))


class GaussianProcessCCA(BaseModel):
    r"""Nonlinear CCA with Gaussian-process encoders.

    Each view's encoder is $f_i(x) = k(x, U_i)^\top B_i$ for a kernel $k$ over
    the whole feature vector and inducing points $U_i$, so it can represent
    within-view interactions. Every view's $B_i$ minimises the EY loss
    (:mod:`cca_zoo._utils._ey`) plus the RKHS-norm penalty
    $\tfrac12 \alpha_i \operatorname{tr}(B_i^\top K_i B_i)$. On the Nyström
    features of the inducing points the penalty is a ridge, and the global
    minimiser is a generalized eigenproblem, solved in closed form. Fitting
    costs $O(n m^2 + m^3)$ for $m$ inducing points. Kernel hyperparameters are
    fixed; tune them by cross-validation.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        kernel: A kernel, cloned per view, or one per view; None uses
            ``ConstantKernel(1.0) * RBF(np.ones(n_features))``. Default is None.
        alpha: RKHS-norm penalty, also the noise level of the posterior
            variance. Per-view. Default is 0.01.
        n_inducing: Number of inducing points; None, or at least
            ``n_samples``, uses every training row. Per-view. Default is None.
        random_state: Seed for the inducing points. Default is None.

    Attributes:
        encoders_: Fitted per-view encoders, with ``inducing_``, ``kernel_``
            and ``coef_``.

    References:
        Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes
        for Machine Learning. MIT Press.

        Quiñonero-Candela, J., & Rasmussen, C. E. (2005). A Unifying View
        of Sparse Approximate Gaussian Process Regression. Journal of
        Machine Learning Research, 6, 1939-1959.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.gp import GaussianProcessCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 3))
        >>> X2 = rng.standard_normal((100, 3))
        >>> model = GaussianProcessCCA(alpha=[0.01, 0.1]).fit([X1, X2])
        >>> means, stds = model.transform([X1, X2]), model.posterior_std([X1, X2])
    """

    _components_bounded_by_features: ClassVar[bool] = False
    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "n_inducing": [None, Interval(Integral, 2, None, closed="left"), "array-like"],
        "kernel": [None, Kernel, list],
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        kernel: Kernel | list[Kernel | None] | None = None,
        alpha: float | list[float] = 0.01,
        n_inducing: int | list[int | None] | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.kernel = kernel
        self.alpha = alpha
        self.n_inducing = n_inducing
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> GaussianProcessCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        k = self.n_components

        kernel_ = perview_parameter("kernel", self.kernel, None, self.n_views_)
        alpha_ = perview_parameter("alpha", self.alpha, 0.01, self.n_views_)
        n_inducing_ = perview_parameter(
            "n_inducing", self.n_inducing, None, self.n_views_
        )

        encoders = [
            _GpEncoder(
                X,
                k,
                (
                    clone(kern)
                    if kern is not None
                    else ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1]))
                ),
                a,
                n_ind,
                self.random_state,
            )
            for X, kern, a, n_ind in zip(views_, kernel_, alpha_, n_inducing_)
        ]

        features = [enc.features(X) for enc, X in zip(encoders, views_)]
        # Inducing points other than the training rows leave the features
        # uncentred on the training data; the EY loss sees only covariances.
        features = [f - f.mean(axis=0) for f in features]
        coefficients = penalised_basis_ey_closed_form(features, k, alpha_)
        for enc, coef in zip(encoders, coefficients):
            enc.coef_ = enc._nystroem @ coef

        self.encoders_: list[_GpEncoder] = encoders
        return self._finish_fit(views_)

    def posterior_std(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Posterior standard deviation of each view's latent scores.

        The uncertainty of :meth:`transform`'s scores, larger away from the
        training data.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.

        Returns:
            One array of shape (n_samples, n_components) per view.
        """
        validated = self._check_views(views)
        return [
            enc.predict_std(v - m)
            for v, m, enc in zip(validated, self.means_, self.encoders_)
        ]

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        return self.encoders_[view].predict_new(centred)
