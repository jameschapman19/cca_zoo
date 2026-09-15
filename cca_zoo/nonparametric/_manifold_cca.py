r"""ManifoldCCA — transductive multiview CCA over a shared manifold operator."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag
from scipy.sparse import issparse
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from sklearn.kernel_ridge import KernelRidge
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._param_constraints import POSITIVE_EPS
from cca_zoo._utils._validation import validate_views


def _centering_matrix(n: int) -> np.ndarray:
    r"""The (scaled) sample-covariance operator of the identity 'feature map'.

    $\frac{1}{n-1}\bigl(I_n - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top\bigr)$ --
    if each training point's "feature vector" were literally its own
    one-hot indicator (so the projection *is* the per-sample embedding,
    with nothing left to learn but which n-vector it should be), this is
    that identity feature map's own covariance, i.e. exactly what
    :class:`~cca_zoo.linear.MCCA`'s ``_build_A`` would compute from it.
    """
    return (np.eye(n) - np.ones((n, n)) / n) / (n - 1)


def _floor_min_eig(M: np.ndarray, eps: float) -> np.ndarray:
    """Add a multiple of the identity so ``M``'s smallest eigenvalue is >= eps."""
    min_eig = np.linalg.eigvalsh(M).min()
    if min_eig < eps:
        M = M + (eps - min_eig) * np.eye(M.shape[0])
    return M


def _laplacian_operator(
    v: np.ndarray, n_neighbors: int, affinity: str, gamma: float | None
) -> np.ndarray:
    r"""Graph Laplacian $L = D - W$ of a k-NN affinity graph over ``v``.

    Reuses :class:`sklearn.manifold.SpectralEmbedding` purely for its own
    (well-tested) affinity-graph construction -- the k-NN search, symmetrisation
    and (for ``affinity="rbf"``) heat-kernel weighting -- then builds the
    graph Laplacian directly via :func:`scipy.sparse.csgraph.laplacian`,
    rather than using ``SpectralEmbedding``'s own single-view spectral
    solution, since what's needed here is the raw operator to plug into a
    *joint multiview* eigenproblem (see :class:`ManifoldCCA`), not a
    finished single-view embedding.
    """
    se = SpectralEmbedding(
        n_components=1, n_neighbors=n_neighbors, affinity=affinity, gamma=gamma
    ).fit(v)
    W = se.affinity_matrix_
    L = sparse_laplacian(W, normed=True)
    return np.asarray(L.toarray() if issparse(L) else L)


def _lle_operator(v: np.ndarray, n_neighbors: int, reg: float) -> np.ndarray:
    r"""LLE operator $M = (I - W)^\top (I - W)$ from local reconstruction weights.

    ``W[a, :]`` reconstructs point ``a`` as the best (sum-to-one) linear
    combination of its ``n_neighbors`` nearest neighbours -- Roweis & Saul
    (2000)'s original barycentric-weight construction, hand-implemented
    here (rather than reused from :class:`sklearn.manifold.LocallyLinearEmbedding`,
    which doesn't expose ``W``/``M`` as public API) so the raw operator is
    available for the joint eigenproblem below.
    """
    n = v.shape[0]
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(v)
    indices = nn.kneighbors(v, return_distance=False)[:, 1:]

    W = np.zeros((n, n))
    for i in range(n):
        neighbours = v[indices[i]] - v[i]
        gram = neighbours @ neighbours.T
        gram += reg * max(np.trace(gram), 1e-12) * np.eye(n_neighbors)
        w = np.linalg.solve(gram, np.ones(n_neighbors))
        W[i, indices[i]] = w / w.sum()

    I_minus_W = np.eye(n) - W
    return I_minus_W.T @ I_minus_W


class ManifoldCCA(BaseModel):
    r"""ManifoldCCA -- transductive multiview CCA over a shared manifold operator.

    Every other multiview method in this library maximises cross-view
    covariance subject to a within-view *covariance* constraint
    (:class:`~cca_zoo.linear.MCCA`'s ridge-blended sample covariance,
    :class:`~cca_zoo.linear.GraphicalLassoCCA`'s sparse-precision
    covariance, :class:`~cca_zoo.nonparametric.KCCA`'s regularised kernel
    Gram matrix). Spectral manifold-learning methods
    (:class:`sklearn.manifold.SpectralEmbedding`,
    :class:`sklearn.manifold.LocallyLinearEmbedding`) instead constrain a
    single view's embedding against a *graph operator* $M$ built from that
    view's own local neighbourhood structure -- the graph Laplacian
    ($M = D - W$, small $\operatorname{tr}(Y^\top M Y)$ means neighbouring
    points map to nearby embeddings) or the LLE reconstruction operator
    ($M = (I-W)^\top(I-W)$, small $\operatorname{tr}(Y^\top M Y)$ means each
    point's embedding is well reconstructed from its neighbours'). Both are
    themselves generalised eigenproblems of exactly the same
    "maximise-subject-to-a-quadratic-constraint" shape as
    :class:`~cca_zoo.linear.MCCA` -- just with $M$ replacing a covariance
    matrix, and with no covariance available at all in the usual sense,
    since there's no feature map: the "weight" *is* the per-training-point
    embedding.

    ``ManifoldCCA`` solves the resulting **joint, multiview** version:

    $$
    \max_{Z_1, \dots, Z_M} \sum_{i \neq j} \operatorname{Cov}(Z_i, Z_j)
    \quad \text{subject to} \quad Z_i^\top M_i Z_i = I \; \forall i
    $$

    where $Z_i \in \mathbb{R}^{n \times k}$ is view $i$'s embedding of the
    *training* points and $M_i$ is that view's own graph operator --
    the same joint-eigenproblem construction :class:`~cca_zoo.linear.MCCA`
    and :class:`~cca_zoo.nonparametric.KCCA` use, but with each view's
    "feature map" being the identity (so the projection weight found by the
    solver *is* $Z_i$ directly -- see :func:`_centering_matrix`) and its
    within-view block $M_i$ instead of a covariance.

    Since $Z_i$ is only ever defined at the training points (there is no
    feature map to apply to new data), out-of-sample projection is *not*
    the graph operator's own Nystrom extension (which would need
    re-deriving for the joint, multiview eigenproblem above, not just the
    single-view one the literature covers) -- it is instead a per-view
    :class:`~sklearn.kernel_ridge.KernelRidge` (RBF) regression fit from
    each view's raw training features onto its own training embedding
    $Z_i$, evaluated on new data. This is an approximation of the "true"
    spectral extension, not a re-derivation of it -- a well-established
    alternative in its own right for out-of-sample spectral embeddings, and
    one that doesn't risk a subtly wrong bespoke formula.

    Note:
        Unlike :class:`~sklearn.manifold.LocallyLinearEmbedding`,
        Hessian-LLE (``method="hessian"``) and LTSA
        (``method="ltsa"``) are not implemented here -- both need a local
        Hessian/tangent-space estimate per point (local PCA plus a
        polynomial basis, then a null-space projection) that's
        substantially more involved to get right than the graph Laplacian
        or LLE's own reconstruction weights, and are left for a future
        extension rather than shipped undertested. Isomap-flavoured
        (geodesic-distance) regularisation is achievable today via
        :class:`~cca_zoo.nonparametric.KCCA` with a precomputed geodesic
        Gram matrix in place of a standard kernel, so isn't duplicated here.

        ``inverse_transform``/``predict`` are not supported (same as
        :class:`~cca_zoo.nonparametric.KCCA`): both rely on
        ``BaseModel``'s default view-loading fit, which assumes
        ``weights_[i]`` has shape ``(n_features_i, k)``; here it is
        ``(n_train_samples, k)`` (the training embedding itself), since
        there is no feature-space weight vector to speak of.

        Solves an $(nM) \times (nM)$ dense generalised eigenproblem
        ($n$ = training samples, $M$ = number of views), the same cost
        profile as :class:`~cca_zoo.nonparametric.KCCA` -- intended for
        moderate training-set sizes, not the very large-$n$ regime.

    References:
        Roweis, S. T., & Saul, L. K. (2000). Nonlinear dimensionality
        reduction by locally linear embedding. Science, 290(5500), 2323-2326.

        Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for
        dimensionality reduction and data representation. Neural
        Computation, 15(6), 1373-1396.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default
            True.
        method: ``"laplacian"`` (graph Laplacian, matching
            :class:`~sklearn.manifold.SpectralEmbedding`) or ``"lle"``
            (locally linear embedding operator). Default ``"laplacian"``.
        n_neighbors: Number of neighbours used to build each view's graph.
            Default 10.
        affinity: ``"nearest_neighbors"`` or ``"rbf"``, passed to
            :class:`~sklearn.manifold.SpectralEmbedding` when
            ``method="laplacian"``. Ignored for ``method="lle"``. Default
            ``"nearest_neighbors"``.
        gamma: RBF kernel coefficient, used only when ``method="laplacian"``
            and ``affinity="rbf"``. Default ``None`` (sklearn's own
            ``1 / n_features`` default).
        lle_reg: Regularisation added to each point's local reconstruction
            Gram matrix, used only when ``method="lle"``. Default 1e-3.
        extrapolator_alpha: Ridge strength for the per-view
            :class:`~sklearn.kernel_ridge.KernelRidge` out-of-sample
            extension. Default 1.0.
        eps: Small constant added to each $M_i$'s eigenvalues to ensure
            positive definiteness. Default 1e-6.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((60, 8))
        >>> X2 = rng.standard_normal((60, 6))
        >>> model = ManifoldCCA(method="laplacian", n_neighbors=8).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "method": [StrOptions({"laplacian", "lle"})],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "affinity": [StrOptions({"nearest_neighbors", "rbf"})],
        "gamma": [Interval(Real, 0, None, closed="neither"), None],
        "lle_reg": [Interval(Real, 0, None, closed="left")],
        "extrapolator_alpha": [Interval(Real, 0, None, closed="neither")],
        "eps": POSITIVE_EPS,
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        method: str = "laplacian",
        n_neighbors: int = 10,
        affinity: str = "nearest_neighbors",
        gamma: float | None = None,
        lle_reg: float = 1e-3,
        extrapolator_alpha: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.method = method
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.gamma = gamma
        self.lle_reg = lle_reg
        self.extrapolator_alpha = extrapolator_alpha
        self.eps = eps

    def _operator(self, v: np.ndarray) -> np.ndarray:
        if self.method == "laplacian":
            return _laplacian_operator(v, self.n_neighbors, self.affinity, self.gamma)
        return _lle_operator(v, self.n_neighbors, self.lle_reg)

    def fit(self, views: list[ArrayLike], y: None = None) -> ManifoldCCA:
        """Fit ManifoldCCA by a joint generalised eigenproblem over per-view graphs.

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
        n = self.n_samples_
        m = self.n_views_

        operators = [_floor_min_eig(self._operator(v), self.eps) for v in views_]
        B = np.asarray(block_diag(*operators)) / m
        A = np.kron(np.ones((m, m)) - np.eye(m), _centering_matrix(n)) / m

        _, eigvecs = gevp(A, B, self.latent_dimensions)
        embedding = list(np.split(eigvecs, m, axis=0))
        self.weights_: list[np.ndarray] = embedding

        self._extrapolators_: list[KernelRidge] = [
            KernelRidge(kernel="rbf", alpha=self.extrapolator_alpha).fit(v, z)
            for v, z in zip(views_, embedding)
        ]
        return self

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Project new views via each view's fitted out-of-sample extension.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            List of arrays, each of shape (n_samples, latent_dimensions).

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
        """
        check_is_fitted(self)
        validated = validate_views(views)
        centred = [v - m for v, m in zip(validated, self.means_)]
        return [
            reg.predict(v) for reg, v in zip(self._extrapolators_, centred)
        ]
