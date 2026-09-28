"""Multiset CCA."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag
from sklearn.decomposition import PCA

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._param_constraints import RIDGE_PARAMETER
from cca_zoo._utils._validation import perview_parameter


class MCCA(BaseModel):
    r"""Multiset CCA: the sum of pairwise covariances under ridge constraints.

    $$
    \max_{w} \sum_{i \neq j} w_i^\top X_i^\top X_j w_j
    \quad \text{subject to} \quad
    w_i^\top \bigl((1 - c_i) X_i^\top X_i + c_i I\bigr) w_i = 1,
    $$

    solved as the generalized eigenproblem $A v = \lambda B v$, with $A$ the
    between-view and $B$ the regularised within-view covariance blocks.
    ``shrinkage=0`` is CCA and ``shrinkage=1`` is PLS.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of each view's covariance towards the identity,
            in ``[0, 1]``: 0 is CCA and 1 is PLS. Per-view. Default is 0.
        pca: Whether to solve in each view's principal components, which is
            faster and more stable for wide data. Default is True.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Kettenring, J. R. (1971). Canonical analysis of several sets of
        variables. Biometrika, 58(3), 433-451.

        Vinod, H. D. (1976). Canonical ridge and econometrics of joint
        production. Journal of Econometrics, 4(2), 147-166.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import MCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> X3 = rng.standard_normal((50, 6))
        >>> model = MCCA(n_components=2, shrinkage=0.1).fit([X1, X2, X3])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "shrinkage": RIDGE_PARAMETER,
        "pca": ["boolean"],
    }

    _EPS: ClassVar[float] = 1e-6

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float | list[float] = 0.0,
        pca: bool = True,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.shrinkage = shrinkage
        self.pca = pca

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        sample_weight: ArrayLike | None = None,
    ) -> MCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.
            sample_weight: Weight of each sample; an integer weight is the
                same as repeating the sample. None weights samples equally.

        Returns:
            self.
        """
        views_: list[np.ndarray] = self._setup_fit(views, sample_weight)
        c_ = perview_parameter("shrinkage", self.shrinkage, 0.0, self.n_views_)

        if self.pca:
            pca_models = [PCA().fit(v) for v in views_]
            views_pca = [m.transform(v) for m, v in zip(pca_models, views_)]
            A = self._build_A(views_pca)
            B = self._build_B_pca(pca_models, c_)
        else:
            A = self._build_A(views_)
            B = self._build_B(views_, c_)

        splits = np.cumsum([v.shape[1] for v in (views_pca if self.pca else views_)])
        _, eigvecs = gevp(A, B, self.n_components)

        raw_weights = np.split(eigvecs, splits[:-1], axis=0)
        if self.pca:
            self.weights_: list[np.ndarray] = [
                m.components_.T @ w for m, w in zip(pca_models, raw_weights)
            ]
        else:
            self.weights_ = raw_weights
        return self._finish_fit(views_)

    # ------------------------------------------------------------------
    # Matrix construction helpers (overridable by subclasses)
    # ------------------------------------------------------------------

    def _build_A(self, views: list[np.ndarray]) -> np.ndarray:
        """Between-view covariance block matrix, with zero diagonal blocks."""
        all_views = np.hstack(views)
        A = np.cov(all_views, rowvar=False)
        A -= block_diag(*[np.cov(v, rowvar=False) for v in views])
        return A / len(views)

    def _build_B(self, views: list[np.ndarray], c: list[float]) -> np.ndarray:
        """Block-diagonal ridge-blended within-view covariance."""
        blocks = [
            (1.0 - c[i]) * np.cov(v, rowvar=False) + c[i] * np.eye(v.shape[1])
            for i, v in enumerate(views)
        ]
        B: np.ndarray = np.asarray(block_diag(*blocks))
        min_eig = np.linalg.eigvalsh(B).min()
        if min_eig < self._EPS:
            B += (self._EPS - min_eig) * np.eye(B.shape[0])
        return np.asarray(B / len(views))

    def _build_B_pca(
        self,
        pca_models: list[PCA],
        c: list[float],
    ) -> np.ndarray:
        """Diagonal ``B`` from the PCA explained variances."""
        blocks = [
            np.diag((1.0 - c[i]) * m.explained_variance_ + c[i])
            for i, m in enumerate(pca_models)
        ]
        B: np.ndarray = np.asarray(block_diag(*blocks))
        min_eig = np.linalg.eigvalsh(B).min()
        if min_eig < self._EPS:
            B += (self._EPS - min_eig) * np.eye(B.shape[0])
        return np.asarray(B / len(pca_models))
