"""Multiset CCA."""

from __future__ import annotations

from itertools import accumulate, pairwise
from typing import Any, ClassVar

from numpy.typing import ArrayLike
from sklearn.decomposition import PCA
from sklearn.utils._array_api import device, get_namespace

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import block_diag, covariance, gevp
from cca_zoo._utils._param_constraints import RIDGE_PARAMETER
from cca_zoo._utils._validation import perview_parameter


def _eye(n: int, like: Any) -> Any:
    """The n-by-n identity in the namespace, dtype and device of ``like``."""
    xp, _ = get_namespace(like)
    return xp.eye(n, dtype=like.dtype, device=device(like))


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
    _supports_array_api: ClassVar[bool] = True

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
        views_ = self._setup_fit(views, sample_weight)
        c_ = perview_parameter("shrinkage", self.shrinkage, 0.0, self.n_views_)

        if self.pca:
            pca_models = [PCA().fit(v) for v in views_]
            solved_views = [m.transform(v) for m, v in zip(pca_models, views_)]
            A = self._build_A(solved_views)
            B = self._build_B_pca(pca_models, c_)
        else:
            solved_views = views_
            A = self._build_A(views_)
            B = self._build_B(views_, c_)
        _, eigvecs = gevp(A, B, self.n_components)

        edges = list(accumulate((v.shape[1] for v in solved_views), initial=0))
        weights = [eigvecs[a:b, :] for a, b in pairwise(edges)]
        if self.pca:
            weights = [m.components_.T @ w for m, w in zip(pca_models, weights)]
        self.weights_: list[Any] = weights
        return self._finish_fit(views_)

    # ------------------------------------------------------------------
    # Matrix construction helpers (overridable by subclasses)
    # ------------------------------------------------------------------

    def _build_A(self, views: list[Any]) -> Any:
        """Between-view covariance block matrix, with zero diagonal blocks."""
        xp, _ = get_namespace(*views)
        A = covariance(xp.concat(views, axis=1))
        A = A - block_diag([covariance(v) for v in views])
        return A / len(views)

    def _build_B(self, views: list[Any], c: list[float]) -> Any:
        """Block-diagonal ridge-blended within-view covariance."""
        blocks = [
            (1.0 - ci) * covariance(v) + ci * _eye(v.shape[1], v)
            for v, ci in zip(views, c)
        ]
        return self._floored(block_diag(blocks) / len(views))

    def _build_B_pca(self, pca_models: list[PCA], c: list[float]) -> Any:
        """Diagonal ``B`` from the PCA explained variances."""
        blocks = [
            _eye(v.shape[0], v) * ((1.0 - ci) * v + ci)
            for v, ci in zip((m.explained_variance_ for m in pca_models), c)
        ]
        return self._floored(block_diag(blocks) / len(pca_models))

    def _floored(self, B: Any) -> Any:
        """``B`` with its spectrum raised to at least ``_EPS / n_views``."""
        xp, _ = get_namespace(B)
        floor = self._EPS / self.n_views_
        min_eig = float(xp.min(xp.linalg.eigvalsh(B)))
        return B + max(0.0, floor - min_eig) * _eye(B.shape[0], B)
