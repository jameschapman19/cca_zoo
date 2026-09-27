"""Gaussian-process CCA."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.cluster import kmeans_plusplus
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel
from sklearn.preprocessing import KernelCenterer
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import (
    cheap_orthonormal_projection_weights,
    ey_grad_z,
    ey_loss,
)
from cca_zoo._utils._param_constraints import RANDOM_STATE
from cca_zoo._utils._validation import perview_parameter


def _lbfgsb_joint_objective(
    x: np.ndarray,
    bases: list[np.ndarray],
    ridge_matrices: list[np.ndarray],
    shapes: list[tuple[int, int]],
    sizes: list[int],
    ridge: list[float],
) -> tuple[float, np.ndarray]:
    r"""RKHS-penalised EY loss and its gradient in every view's flattened coefficients.

    With $Z_i = \text{bases}_i B_i$, the loss is
    $\mathcal{L}_{EY} + \tfrac12 \sum_i \lambda_i \operatorname{tr}(B_i^\top M_i B_i)$
    and its gradient per view is
    $\text{bases}_i^\top \nabla_{Z_i} \mathcal{L}_{EY} + \lambda_i M_i B_i$.
    """
    coefficients = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        coefficients.append(x[offset : offset + size].reshape(shape))
        offset += size

    representations = [basis @ b for basis, b in zip(bases, coefficients)]
    loss = ey_loss(representations)["objective"]
    penalty = 0.5 * sum(
        r * np.sum(b * (m @ b)) for b, m, r in zip(coefficients, ridge_matrices, ridge)
    )
    grad_z = ey_grad_z(representations)
    grads = [
        basis.T @ gz + r * (m @ b)
        for basis, gz, m, b, r in zip(
            bases, grad_z, ridge_matrices, coefficients, ridge
        )
    ]
    grad = np.concatenate([g.ravel() for g in grads])
    return loss + penalty, grad


class _GpEncoder:
    r"""Per-view kernel encoder $f(x) = k(x, U)^\top B$ on fixed inducing points $U$.

    The subset-of-regressors construction (Quiñonero-Candela & Rasmussen,
    2005): $U$ is every training row, or ``n_inducing`` rows seeded by
    :func:`~sklearn.cluster.kmeans_plusplus`. The cross-kernel is centred with
    :class:`~sklearn.preprocessing.KernelCenterer`, so embeddings are
    zero-mean for any $B$. The posterior standard deviation does not depend
    on $B$, so it comes from a
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
        """Encoder output for new data, shape (n, k), optionally with its std.

        Args:
            X: Input data, shape (n, n_features).
            return_std: Whether to also return the posterior standard deviation,
                the same for every component.

        Returns:
            The output, or ``(mean, std)``, each of shape (n, k).
        """
        basis = self._centerer.transform(self.kernel_(X, self.inducing_))
        mean: np.ndarray = basis @ self.coef_
        if not return_std:
            return mean
        _, std = self._variance_model.predict(X, return_std=True)
        return mean, np.tile(std[:, None], (1, self.k))


class GaussianProcessCCA(BaseModel):
    r"""Nonlinear CCA with Gaussian-process encoders.

    Each view's encoder is $f_i(x) = k(x, U_i)^\top B_i$ for a kernel $k$ over
    the whole feature vector and inducing points $U_i$, so it can represent
    within-view interactions. Every view's $B_i$ is fitted jointly by
    L-BFGS-B to minimise the EY loss (:mod:`cca_zoo._utils._ey`) plus the
    RKHS-norm penalty $\tfrac12 \alpha_i \operatorname{tr}(B_i^\top K_i B_i)$.
    Fitting costs $O(n m^2 + m^3)$ for $m$ inducing points. Kernel
    hyperparameters are fixed; tune them by cross-validation.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        kernel: A kernel, cloned per view, or one per view; None uses
            ``ConstantKernel(1.0) * RBF(np.ones(n_features))``. Default is None.
        alpha: RKHS-norm penalty, also the noise level of the posterior
            variance. Per-view. Default is 0.01.
        n_inducing: Number of inducing points; None, or at least
            ``n_samples``, uses every training row. Per-view. Default is None.
        max_iter: Maximum L-BFGS-B iterations. Default is 1000.
        tol: L-BFGS-B ``ftol``. Default is 1e-6.
        random_state: Seed for the initial coefficients and inducing points.
            Default is None.

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
        >>> means, stds = model.transform([X1, X2], return_std=True)
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "n_inducing": [None, Interval(Integral, 2, None, closed="left"), "array-like"],
        "kernel": [None, Kernel, list],
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        kernel: Kernel | list[Kernel | None] | None = None,
        alpha: float | list[float] = 0.01,
        n_inducing: int | list[int | None] | None = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.kernel = kernel
        self.alpha = alpha
        self.n_inducing = n_inducing
        self.max_iter = max_iter
        self.tol = tol
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

        bases = [enc.basis_ for enc in encoders]
        ridge_matrices = [enc.ridge_matrix_ for enc in encoders]

        rng = np.random.default_rng(self.random_state)
        coefficients0 = cheap_orthonormal_projection_weights(bases, k, None, rng)
        shapes = [c.shape for c in coefficients0]
        sizes = [c.size for c in coefficients0]
        x0 = np.concatenate([c.ravel() for c in coefficients0])

        result = minimize(
            _lbfgsb_joint_objective,
            x0,
            args=(bases, ridge_matrices, shapes, sizes, alpha_),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        coefficients = []
        offset = 0
        for shape, size in zip(shapes, sizes):
            coefficients.append(result.x[offset : offset + size].reshape(shape))
            offset += size

        representations = [basis @ coef for basis, coef in zip(bases, coefficients)]

        for enc, coef, rep in zip(encoders, coefficients, representations):
            enc.coef_ = coef
            enc._train_pred = rep

        self.encoders_: list[_GpEncoder] = encoders
        return self

    def transform(  # type: ignore[override]
        self, views: list[ArrayLike], return_std: bool = False
    ) -> list[np.ndarray] | tuple[list[np.ndarray], list[np.ndarray]]:
        """Project views into the latent space.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            return_std: Whether to also return the posterior standard deviations.
                Default is False.

        Returns:
            One array of shape (n_samples, n_components) per view, or
            ``(means, stds)`` of two such lists.
        """
        if not return_std:
            return super().transform(views)
        validated = self._check_views(views)
        centred = [v - m for v, m in zip(validated, self.means_)]
        means = []
        stds = []
        for v, enc in zip(centred, self.encoders_):
            mean, std = enc.predict_new(v, return_std=True)
            means.append(mean)
            stds.append(std)
        return means, stds

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, self.encoders_[view].predict_new(centred))
