"""Multiview CCA constrained by per-view graph operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag, null_space
from scipy.sparse import issparse
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.spatial.distance import cdist
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._param_validation import Interval, StrOptions

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._param_constraints import RIDGE_PARAMETER
from cca_zoo._utils._validation import perview_parameter

#: Least ``1 - eigenvalue`` of a Laplacian eigenvector kept in the basis. The
#: Nystrom extension divides by it, so an eigenvector near or past 1, barely
#: smoother than noise, would amplify a new point's embedding without bound.
_NYSTROM_MIN_MU = 0.05


def _centering_matrix(n: int) -> np.ndarray:
    """Covariance operator ``(I - 11'/n) / (n - 1)`` of per-sample embeddings."""
    return (np.eye(n) - np.ones((n, n)) / n) / (n - 1)


def _smooth_basis(
    operator: np.ndarray, n_components: int, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """The ``n_components`` eigenvectors of ``operator`` with the smallest eigenvalues.

    Truncating to the smoothest eigenvectors is what regularises the fit, as
    in :class:`~sklearn.manifold.SpectralEmbedding`; the full operator has
    enough freedom to fit correlation to noise.

    Args:
        operator: Symmetric operator on the complement of the constant
            vector, shape (n-1, n-1).
        n_components: Number of eigenvectors to keep.
        eps: Floor on each kept eigenvalue.

    Returns:
        ``(basis, eigenvalues)`` of shapes (n-1, n_components) and
        (n_components,).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    k = min(n_components, operator.shape[0])
    basis = eigenvectors[:, :k]
    floored = np.maximum(eigenvalues[:k], eps)
    return basis, floored


def _extendable(
    basis: np.ndarray, eigenvalues: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """The Laplacian eigenvectors the Nystrom extension can carry to new points.

    Raises:
        ValueError: If fewer than ``n_components`` remain.
    """
    keep = 1.0 - eigenvalues >= _NYSTROM_MIN_MU
    if keep.sum() < n_components:
        raise ValueError(
            f"Only {keep.sum()} of a view's graph eigenvectors are smooth enough "
            f"to extend to new points, fewer than n_components={n_components}; "
            "raise n_neighbors."
        )
    return basis[:, keep], eigenvalues[keep]


def _orthonormal_complement(null_vector: np.ndarray) -> np.ndarray:
    """Orthonormal basis, shape (n, n-1), of the complement of a null vector.

    Every graph operator has a trivial null direction, the constant vector
    for LLE and ``D^{1/2} 1`` for the normalised Laplacian; it is projected
    out before the eigensolve, as ``SpectralEmbedding(drop_first=True)``
    drops it, so that the kept basis is exactly the operator's eigenvectors.
    """
    return np.asarray(null_space(null_vector[np.newaxis, :]))


def _laplacian_affinity(
    v: np.ndarray, n_neighbors: int, affinity: str, gamma: float | None
) -> tuple[np.ndarray, float | None]:
    """Training affinity matrix ``W`` and the resolved RBF ``gamma``.

    Built by :class:`~sklearn.manifold.SpectralEmbedding`; its row sums are
    needed again for the Nystrom extension.
    """
    se = SpectralEmbedding(
        n_components=1, n_neighbors=n_neighbors, affinity=affinity, gamma=gamma
    ).fit(v)
    W = se.affinity_matrix_
    W = np.asarray(W.toarray() if issparse(W) else W)
    resolved_gamma = getattr(se, "gamma_", gamma)
    return W, resolved_gamma


def _normalised_laplacian(W: np.ndarray) -> np.ndarray:
    r"""The normalised graph Laplacian $L = I - D^{-1/2}WD^{-1/2}$ of affinity $W$."""
    L = sparse_laplacian(W, normed=True)
    return np.asarray(L.toarray() if issparse(L) else L)


def _laplacian_new_point_affinity(
    v_new: np.ndarray, v_train: np.ndarray, state: _LaplacianViewState
) -> np.ndarray:
    """New-to-training affinities, by the rule that built the training graph.

    SpectralEmbedding's nearest-neighbour graph is ``(A + A') / 2`` for the
    k-NN connectivity ``A`` including each point itself, so a new point is
    joined to a training point with weight 1/2 for each of: the training
    point being among its k nearest, and it being within the training
    point's k-th neighbour distance. The Laplacian ignores self-loops, so a
    point's affinity to itself is dropped: a training point gets its own row
    of the graph back.
    """
    distances = cdist(v_new, v_train)
    # Distances from recentred training points carry rounding error.
    tolerance = 1e-9 * np.abs(v_train).max()
    if state.nn is None:
        affinity = np.asarray(rbf_kernel(v_new, v_train, gamma=state.gamma))
    else:
        graph = state.nn.kneighbors_graph(v_new, mode="connectivity")
        outgoing = np.asarray(graph.toarray() if issparse(graph) else graph)
        affinity = 0.5 * (outgoing + (distances <= state.radii + tolerance))
    affinity[distances <= tolerance] = 0.0
    return affinity


def _barycenter_weights(
    query: np.ndarray, reference: np.ndarray, indices: np.ndarray, reg: float
) -> np.ndarray:
    """Sum-to-one weights reconstructing each query point from its neighbours.

    Args:
        query: Points to reconstruct, shape (n_query, n_features).
        reference: Points to reconstruct from, shape (n_reference, n_features).
        indices: Neighbour indices into ``reference``, shape (n_query,
            n_neighbors).
        reg: Regularisation of each local Gram matrix, relative to its trace.

    Returns:
        Weights of shape (n_query, n_neighbors).
    """
    n_query, n_neighbors = indices.shape
    weights = np.empty((n_query, n_neighbors))
    for i in range(n_query):
        neighbours = reference[indices[i]] - query[i]
        gram = neighbours @ neighbours.T
        gram += reg * max(np.trace(gram), 1e-12) * np.eye(n_neighbors)
        w = np.linalg.solve(gram, np.ones(n_neighbors))
        weights[i] = w / w.sum()
    return weights


def _lle_operator(v: np.ndarray, n_neighbors: int, reg: float) -> np.ndarray:
    """LLE operator ``(I - W)'(I - W)`` from barycentric reconstruction weights."""
    n = v.shape[0]
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(v)
    indices = nn.kneighbors(v, return_distance=False)[:, 1:]
    weights = _barycenter_weights(v, v, indices, reg)

    W = np.zeros((n, n))
    for i in range(n):
        W[i, indices[i]] = weights[i]

    I_minus_W = np.eye(n) - W
    return I_minus_W.T @ I_minus_W


@dataclass
class _LaplacianViewState:
    """Per-view artifacts the Laplacian Nystrom extension needs at transform time."""

    degrees: np.ndarray
    gamma: float | None
    nn: NearestNeighbors | None
    radii: np.ndarray | None
    full_basis: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    mu: np.ndarray = field(default_factory=lambda: np.empty(0))
    block: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


def _laplacian_parts(
    v: np.ndarray, n_neighbors: int, affinity: str, gamma: float | None
) -> tuple[np.ndarray, _LaplacianViewState]:
    """A view's normalised Laplacian, and what its Nystrom extension needs."""
    W, resolved_gamma = _laplacian_affinity(v, n_neighbors, affinity, gamma)
    nn = (
        NearestNeighbors(n_neighbors=n_neighbors).fit(v)
        if affinity == "nearest_neighbors"
        else None
    )
    state = _LaplacianViewState(
        # Degrees without self-loops, which the Laplacian ignores.
        degrees=W.sum(axis=1) - np.diag(W),
        gamma=resolved_gamma,
        nn=nn,
        radii=None if nn is None else nn.kneighbors(v)[0][:, -1],
    )
    return _normalised_laplacian(W), state


class ManifoldCCA(BaseModel):
    r"""Multiview CCA with a graph-operator constraint per view.

    Maximises cross-view covariance of the training embeddings, restricted to
    each view's smoothest eigenvectors of a graph Laplacian or LLE operator
    ``M_i`` built from that view's neighbourhoods, subject to a blend of each
    view's variance and its roughness on its graph:

    $$
    \max_{Z_1, \dots, Z_M} \sum_{i \neq j} \operatorname{Cov}(Z_i, Z_j)
    \quad \text{subject to} \quad
    \sum_i (1 - c_i) \operatorname{Cov}(Z_i, Z_i) + c_i Z_i^\top M_i Z_i = I.
    $$

    The roughness is rescaled so its trace over the kept eigenvectors equals
    their variance's, so that ``shrinkage`` means the same for either operator.
    This is :class:`~cca_zoo.linear.MCCA` with ``shrinkage=c`` on each view's
    operator eigenvectors scaled by the inverse square root of their rescaled
    eigenvalues: ``shrinkage=0`` is CCA between the views' smooth graph
    coordinates, and ``shrinkage=1`` PLS, which trades correlation for
    smoothness. With identical views and ``shrinkage > 0`` it reduces to the
    view's :class:`~sklearn.manifold.LocallyLinearEmbedding` or, up to centring,
    :class:`~sklearn.manifold.SpectralEmbedding`.
    New points are embedded with each method's standard out-of-sample extension:
    LLE barycentric weights for ``method="lle"``, the Nystrom extension for
    ``method="laplacian"``; either returns a training point's own embedding.
    Fitting solves a dense ``(n M) x (n M)``
    eigenproblem, so it suits moderate sample sizes.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Blend of each view's constraint from its embedding's variance
            (0, CCA) to its roughness on the view's graph (1, PLS), as
            :class:`~cca_zoo.linear.MCCA`'s ``shrinkage``. Per-view. Default is
            0.1.
        method: ``"laplacian"`` or ``"lle"``, for every view. Default is
            ``"laplacian"``.
        n_neighbors: Neighbours in each view's graph. Per-view. Default is 10.
        affinity: ``"nearest_neighbors"`` or ``"rbf"``, for ``method="laplacian"``.
            Per-view. Default is ``"nearest_neighbors"``.
        gamma: RBF kernel coefficient for ``affinity="rbf"``; ``None`` uses
            ``1 / n_features``. Per-view. Default is None.
        reg: Regularisation of each local reconstruction for
            ``method="lle"``. Per-view. Default is 1e-3.
        n_operator_components: Smoothest operator eigenvectors kept per view
            before the joint solve; fewer is stronger regularisation. ``None``
            uses ``max(4 * n_components, 40)``, clipped to ``n_samples - 1``.
            For ``method="laplacian"``, eigenvectors with eigenvalue above
            0.95, which the Nystrom extension cannot carry to new points, are
            dropped. Per-view. Default is None.

    Attributes:
        embedding_: Training embedding of each view, shape (n_samples, n_components).
        views_fit_: The centred training views, from which a new view's
            embedding is interpolated.

    References:
        Roweis, S. T., & Saul, L. K. (2000). Nonlinear dimensionality reduction
        by locally linear embedding. Science, 290(5500), 2323-2326.

        Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality
        reduction and data representation. Neural Computation, 15(6), 1373-1396.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.nonparametric import ManifoldCCA
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((60, 8)), rng.standard_normal((60, 6))
        >>> model = ManifoldCCA(n_neighbors=8).fit([X1, X2])
        >>> Z1, Z2 = model.transform([X1, X2])
    """

    _components_bounded_by_features: ClassVar[bool] = False
    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "shrinkage": RIDGE_PARAMETER,
        "method": [StrOptions({"laplacian", "lle"})],
        "n_neighbors": [Interval(Integral, 1, None, closed="left"), "array-like"],
        "affinity": [StrOptions({"nearest_neighbors", "rbf"}), "array-like"],
        "gamma": [Interval(Real, 0, None, closed="neither"), None, "array-like"],
        "reg": [Interval(Real, 0, None, closed="left"), "array-like"],
        "n_operator_components": [
            Interval(Integral, 1, None, closed="left"),
            None,
            "array-like",
        ],
    }

    _EPS: ClassVar[float] = 1e-6

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float | list[float] = 0.1,
        method: str = "laplacian",
        n_neighbors: int | list[int] = 10,
        affinity: str | list[str] = "nearest_neighbors",
        gamma: float | list[float | None] | None = None,
        reg: float | list[float] = 1e-3,
        n_operator_components: int | list[int | None] | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.shrinkage = shrinkage
        self.method = method
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.gamma = gamma
        self.reg = reg
        self.n_operator_components = n_operator_components

    def _resolve_n_operator_components(
        self, n_operator_components: int | None, n: int
    ) -> int:
        if n_operator_components is not None:
            return min(n_operator_components, n - 1)
        return min(max(4 * self.n_components, 40), n - 1)

    def fit(self, views: list[ArrayLike], y: None = None) -> ManifoldCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        n = self.n_samples_
        m = self.n_views_

        c_ = perview_parameter("shrinkage", self.shrinkage, 0.1, m)
        n_neighbors_ = perview_parameter("n_neighbors", self.n_neighbors, 10, m)
        affinity_ = perview_parameter("affinity", self.affinity, "nearest_neighbors", m)
        gamma_ = perview_parameter("gamma", self.gamma, None, m)
        reg_ = perview_parameter("reg", self.reg, 1e-3, m)
        n_operator_components_ = perview_parameter(
            "n_operator_components", self.n_operator_components, None, m
        )

        C = _centering_matrix(n)

        k_ops = [
            self._resolve_n_operator_components(c, n) for c in n_operator_components_
        ]
        full_bases = []
        eigenvalue_blocks = []
        laplacian_parts: list[_LaplacianViewState] = []
        lle_nn: list[NearestNeighbors] = []
        for v, nn_i, aff_i, gamma_i, reg_i, k_op in zip(
            views_, n_neighbors_, affinity_, gamma_, reg_, k_ops
        ):
            if self.method == "laplacian":
                operator, state = _laplacian_parts(v, nn_i, aff_i, gamma_i)
                null_vector = np.sqrt(state.degrees)
                laplacian_parts.append(state)
            else:
                operator = _lle_operator(v, nn_i, reg_i)
                null_vector = np.ones(n)
                lle_nn.append(NearestNeighbors(n_neighbors=nn_i + 1).fit(v))

            P = _orthonormal_complement(null_vector)
            basis, eigenvalues = _smooth_basis(P.T @ operator @ P, k_op, self._EPS)
            if self.method == "laplacian":
                basis, eigenvalues = _extendable(basis, eigenvalues, self.n_components)
            full_bases.append(P @ basis)
            eigenvalue_blocks.append(eigenvalues)

        # Between-view covariances of the smooth bases, as MCCA's A.
        stacked = np.hstack(full_bases)
        A = stacked.T @ C @ stacked
        A -= block_diag(*[b.T @ C @ b for b in full_bases])
        # Each view's variance blended with its roughness, U' M U = diag(ev),
        # the roughness rescaled to the variance's trace so that shrinkage
        # means the same for operators of any scale.
        variances = [b.T @ C @ b for b in full_bases]
        B = block_diag(
            *[
                (1 - c) * v + c * np.trace(v) / ev.sum() * np.diag(ev)
                for v, ev, c in zip(variances, eigenvalue_blocks, c_)
            ]
        )
        _, eigvecs = gevp(A / m, B / m, self.n_components)
        sizes = [b.shape[1] for b in full_bases]
        blocks = np.split(eigvecs, np.cumsum(sizes)[:-1], axis=0)
        embedding = [fb @ blk for fb, blk in zip(full_bases, blocks)]
        # The Laplacian's basis is orthogonal to D^{1/2} 1, not to the
        # constant: centre the embedding, and new points by the same means.
        self._embedding_means_: list[np.ndarray] = [e.mean(axis=0) for e in embedding]
        self.embedding_: list[np.ndarray] = [
            e - m for e, m in zip(embedding, self._embedding_means_)
        ]
        self.views_fit_: list[np.ndarray] = views_
        self._n_neighbors_: list[int] = n_neighbors_
        self._reg_: list[float] = reg_

        for state, basis, eigenvalues, block in zip(
            laplacian_parts, full_bases, eigenvalue_blocks, blocks
        ):
            state.full_basis, state.mu, state.block = basis, 1.0 - eigenvalues, block
        self._laplacian_state_: list[_LaplacianViewState] | None = (
            laplacian_parts if self.method == "laplacian" else None
        )
        self._lle_state_: list[NearestNeighbors] | None = (
            lle_nn if self.method == "lle" else None
        )
        self._fit_maps_and_importances(views_)
        return self

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        v_train = self.views_fit_[view]
        n_neighbors = self._n_neighbors_[view]
        if self.method == "lle":
            assert self._lle_state_ is not None
            distances, indices = self._lle_state_[view].kneighbors(
                centred, n_neighbors=n_neighbors
            )
            weights = _barycenter_weights(centred, v_train, indices, self._reg_[view])
            embedded: np.ndarray = np.einsum(
                "qn,qnk->qk", weights, self.embedding_[view][indices]
            )
            # A training point is its own reconstruction: it gets its embedding.
            same = distances[:, 0] <= 1e-9 * np.abs(v_train).max()
            embedded[same] = self.embedding_[view][indices[same, 0]]
            return embedded
        assert self._laplacian_state_ is not None
        state = self._laplacian_state_[view]
        W_new = _laplacian_new_point_affinity(centred, v_train, state)
        degrees_new = np.maximum(W_new.sum(axis=1), 1e-12)
        K_tilde = W_new / np.sqrt(np.outer(degrees_new, state.degrees))
        extended: np.ndarray = (
            K_tilde @ state.full_basis
        ) / state.mu @ state.block - self._embedding_means_[view]
        return extended
