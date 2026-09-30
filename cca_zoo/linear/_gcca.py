"""Generalized CCA."""

from __future__ import annotations

import math
from typing import Any, ClassVar

from numpy.typing import ArrayLike
from sklearn.utils._array_api import device, get_namespace

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import covariance, psd_inverse_sqrt
from cca_zoo._utils._param_constraints import RIDGE_PARAMETER
from cca_zoo._utils._validation import perview_parameter


class GCCA(BaseModel):
    r"""Generalized CCA: views correlated with a shared latent variable.

    $$
    \min_{T,\, w_i} \sum_i \mu_i \Bigl(
        w_i^\top C_i w_i - \tfrac{2}{n - 1} w_i^\top X_i^\top T \Bigr)
    \quad \text{subject to} \quad \tfrac{1}{n - 1} T^\top T = I,
    $$

    with $C_i = (1 - c_i) \Sigma_{ii} + c_i I$: at $c_i = 0$, the squared
    error of regressing $T$ on each view. Each view's weights are its
    regularised regression onto $T$, $w_i = C_i^{-1} X_i^\top T / (n - 1)$,
    and $T$ holds the top eigenvectors of $\sum_i \mu_i X_i C_i^{-1}
    X_i^\top$, found from the SVD of the stacked whitened views.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's covariance towards the identity,
            in ``[0, 1]``: 0 is CCA and 1 is PLS. Per-view. Default is 0.
        view_weights: Weight $\mu_i$ of each view; None weights them
            equally. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Carroll, J. D. (1968). Generalization of canonical correlation analysis
        to three or more sets of variables. Proceedings of the 76th Annual
        Convention of the American Psychological Association, 3, 227-228.
        Kettenring, J. R. (1971). Canonical analysis of several sets of
        variables. Biometrika, 58(3), 433-451.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import GCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> X3 = rng.standard_normal((50, 6))
        >>> model = GCCA(n_components=2).fit([X1, X2, X3])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "shrinkage": RIDGE_PARAMETER,
        "view_weights": [None, "array-like"],
    }

    _EPS: ClassVar[float] = 1e-6
    _supports_array_api: ClassVar[bool] = True

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float | list[float] = 0.0,
        view_weights: list[float] | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.shrinkage = shrinkage
        self.view_weights = view_weights

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        sample_weight: ArrayLike | None = None,
    ) -> GCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.
            sample_weight: Weight of each sample; an integer weight is the
                same as repeating the sample. None weights samples equally.

        Returns:
            self.
        """
        views_ = self._setup_fit(views, sample_weight)
        xp, _ = get_namespace(*views_)
        c_ = perview_parameter("shrinkage", self.shrinkage, 0.0, self.n_views_)
        mu = perview_parameter("view_weights", self.view_weights, 1.0, self.n_views_)

        # Q = sum_i mu_i X_i cov_i^{-1} X_i^T is H H^T for the stacked
        # whitened views H = [sqrt(mu_i) X_i cov_i^{-1/2}], so its top
        # eigenvectors are H's top left singular vectors: an n x sum(p_i)
        # SVD in place of an n x n eigenproblem.
        whiteners = []
        for v, ci in zip(views_, c_):
            identity = xp.eye(v.shape[1], dtype=v.dtype, device=device(v))
            cov = (1.0 - ci) * covariance(v) + ci * identity
            whiteners.append(psd_inverse_sqrt(cov, self._EPS))
        whitened = [math.sqrt(mi) * v @ r for v, r, mi in zip(views_, whiteners, mu)]
        U = xp.linalg.svd(xp.concat(whitened, axis=1), full_matrices=False)[0]
        # Unit-variance shared latent, so the scores' scale does not depend on
        # the number of samples.
        T = U[:, : self.n_components] * math.sqrt(self.n_samples_ - 1)
        # Each view's weights are its regularised regression onto T,
        # cov_i^{-1} X_i' T / (n - 1): least squares at shrinkage 0.
        self.weights_: list[Any] = [
            r @ (r @ (v.T @ T)) / (self.n_samples_ - 1)
            for v, r in zip(views_, whiteners)
        ]
        self._fit_maps_and_importances(views_)
        return self
