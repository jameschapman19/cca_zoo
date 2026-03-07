"""Multiset CCA (MCCA) — generalised eigendecomposition approach."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag
from sklearn.decomposition import PCA

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._validation import perview_parameter


class MCCA(BaseModel):
    r"""Multiset Canonical Correlation Analysis.

    Finds linear projections of multiple (>=2) views that maximise the sum of
    pairwise cross-view covariances subject to within-view variance constraints.
    A ridge regularisation parameter ``c`` controls the trade-off between
    correlation and variance explained.

    The primal objective is:

    .. math::

        \max_{\mathbf{w}} \sum_{i \neq j} \mathbf{w}_i^\top X_i^\top X_j
        \mathbf{w}_j

        \text{subject to } \mathbf{w}_i^\top
        \bigl((1-c_i) X_i^\top X_i + c_i I\bigr) \mathbf{w}_i = 1

    This is solved as a generalised eigenvalue problem:

    .. math::

        A \mathbf{v} = \lambda B \mathbf{v}

    where :math:`A` is the between-view block covariance matrix and :math:`B`
    is the block-diagonal regularised within-view covariance matrix.

    When ``pca=True`` (default), each view is first reduced to its principal
    components, which makes the problem numerically stable for
    high-dimensional data and allows an efficient closed-form :math:`B`.

    References:
        Kettenring, J. R. (1971). Canonical analysis of several sets of
        variables. *Biometrika*, 58(3), 433–451.

        Vinod, H. D. (1976). Canonical ridge and econometrics of joint
        production. *Journal of Econometrics*, 4(2), 147–166.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default True.
        c: Ridge regularisation parameter(s).  Either a single float applied
            to all views or a list of per-view floats in ``[0, 1]``.
            ``c=0`` gives standard CCA constraints; ``c=1`` gives sphering
            (PLS-like).  Default is 0.
        pca: Whether to apply full PCA whitening as a pre-processing step
            before solving the eigenvalue problem.  Highly recommended for
            high-dimensional data.  Default is True.
        eps: Small constant added to the eigenvalues of B to ensure positive
            definiteness.  Default is 1e-6.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 10))
        >>> X2 = rng.standard_normal((50, 8))
        >>> X3 = rng.standard_normal((50, 6))
        >>> model = MCCA(latent_dimensions=2).fit([X1, X2, X3])
        >>> scores = model.transform([X1, X2, X3])
    """

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        c: float | list[float] = 0.0,
        pca: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.c = c
        self.pca = pca
        self.eps = eps

    def fit(self, views: list[ArrayLike], y: None = None) -> MCCA:
        """Fit the MCCA model.

        Args:
            views: List of arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        c_ = perview_parameter("c", self.c, 0.0, self.n_views_)

        if self.pca:
            pca_models = [PCA().fit(v) for v in views_]
            views_pca = [m.transform(v) for m, v in zip(pca_models, views_)]
            A = self._build_A(views_pca)
            B = self._build_B_pca(pca_models, c_)
        else:
            A = self._build_A(views_)
            B = self._build_B(views_, c_)

        splits = np.cumsum([v.shape[1] for v in (views_pca if self.pca else views_)])
        _, eigvecs = gevp(A, B, self.latent_dimensions)

        raw_weights = np.split(eigvecs, splits[:-1], axis=0)
        if self.pca:
            self.weights_: list[np.ndarray] = [
                m.components_.T @ w for m, w in zip(pca_models, raw_weights)
            ]
        else:
            self.weights_ = raw_weights
        return self

    # ------------------------------------------------------------------
    # Matrix construction helpers (overridable by subclasses)
    # ------------------------------------------------------------------

    def _build_A(self, views: list[np.ndarray]) -> np.ndarray:
        """Build the between-view covariance block matrix A.

        Args:
            views: Centred (and optionally PCA-projected) view arrays.

        Returns:
            Symmetric matrix of shape (sum_features, sum_features).
        """
        all_views = np.hstack(views)
        A = np.cov(all_views, rowvar=False)
        A -= block_diag(*[np.cov(v, rowvar=False) for v in views])
        return A / len(views)

    def _build_B(self, views: list[np.ndarray], c: list[float]) -> np.ndarray:
        """Build the regularised within-view covariance block matrix B.

        Args:
            views: Centred view arrays.
            c: Per-view regularisation parameters.

        Returns:
            Symmetric positive-definite matrix of shape (sum_features, sum_features).
        """
        blocks = [
            (1.0 - c[i]) * np.cov(v, rowvar=False) + c[i] * np.eye(v.shape[1])
            for i, v in enumerate(views)
        ]
        B: np.ndarray = np.asarray(block_diag(*blocks))
        min_eig = np.linalg.eigvalsh(B).min()
        if min_eig < self.eps:
            B += (self.eps - min_eig) * np.eye(B.shape[0])
        return np.asarray(B / len(views))

    def _build_B_pca(
        self,
        pca_models: list[PCA],
        c: list[float],
    ) -> np.ndarray:
        """Build B using the PCA explained variances (diagonal form).

        Args:
            pca_models: Fitted PCA models for each view.
            c: Per-view regularisation parameters.

        Returns:
            Block-diagonal matrix of shape (sum_components, sum_components).
        """
        blocks = [
            np.diag((1.0 - c[i]) * m.explained_variance_ + c[i])
            for i, m in enumerate(pca_models)
        ]
        B: np.ndarray = np.asarray(block_diag(*blocks))
        min_eig = np.linalg.eigvalsh(B).min()
        if min_eig < self.eps:
            B += (self.eps - min_eig) * np.eye(B.shape[0])
        return np.asarray(B / len(pca_models))
