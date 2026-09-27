"""Multiview CCA constrained by per-view graph operators."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag, null_space
from scipy.sparse import issparse
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._param_validation import Interval, StrOptions

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import gevp
from cca_zoo._utils._param_constraints import POSITIVE_EPS
from cca_zoo._utils._validation import perview_parameter

#: Floor for the Laplacian Nystrom extension's 1/mu rescaling (mu = 1 -
#: eigenvalue), independent of the class's own (much smaller) ``eps``: that
#: one only needs to keep _smooth_basis's B block positive-definite for the
#: joint solve, and using it here too would let a component with eigenvalue
#: near 1 (barely "smooth" at all -- see _LaplacianViewState) blow up its
#: Nystrom contribution by a factor of 1e6, contaminating every other,
#: well-behaved component once _block_ combines them. 0.05 caps that
#: amplification at 20x instead.
_NYSTROM_MU_FLOOR = 0.05


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


def _orthonormal_complement_of_ones(n: int) -> np.ndarray:
    """Orthonormal basis, shape (n, n-1), of the complement of the constant vector.

    The constant vector is a null direction of both the operators and the
    centring reward, so it is projected out before the eigensolve, as
    ``SpectralEmbedding(drop_first=True)`` does.
    """
    return np.asarray(null_space(np.ones((1, n))))


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


def _laplacian_operator(
    v: np.ndarray, n_neighbors: int, affinity: str, gamma: float | None
) -> np.ndarray:
    r"""Graph Laplacian $L = I - D^{-1/2}WD^{-1/2}$ of a k-NN graph over ``v``."""
    W, _ = _laplacian_affinity(v, n_neighbors, affinity, gamma)
    return _normalised_laplacian(W)


def _laplacian_new_point_affinity(
    v_new: np.ndarray,
    v_train: np.ndarray,
    affinity: str,
    gamma: float | None,
    n_neighbors: int,
    nn: NearestNeighbors | None,
) -> np.ndarray:
    """New-to-training affinities, by the rule that built the training graph.

    For nearest-neighbour graphs each new point connects to its
    ``n_neighbors`` nearest training points, without resymmetrising.
    """
    if affinity == "rbf":
        return np.asarray(rbf_kernel(v_new, v_train, gamma=gamma))
    assert nn is not None
    graph = nn.kneighbors_graph(v_new, n_neighbors=n_neighbors, mode="connectivity")
    return np.asarray(graph.toarray() if issparse(graph) else graph)


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

    full_basis: np.ndarray
    mu: np.ndarray
    block: np.ndarray
    degrees: np.ndarray
    gamma: float | None
    nn: NearestNeighbors | None


class ManifoldCCA(BaseModel):
    r"""Multiview CCA with a graph-operator constraint per view.

    Maximises cross-view covariance of the training embeddings subject to each
    view's spectral-embedding constraint, a graph Laplacian or LLE operator
    ``M_i`` built from that view's neighbourhoods:

    $$
    \max_{Z_1, \dots, Z_M} \sum_{i \neq j} \operatorname{Cov}(Z_i, Z_j)
    \quad \text{subject to} \quad Z_i^\top M_i Z_i = I.
    $$

    With identical views it reduces to :class:`~sklearn.manifold.SpectralEmbedding`.
    New points are embedded with each method's standard out-of-sample extension:
    LLE barycentric weights for ``method="lle"``, the Nystrom extension for
    ``method="laplacian"``. Fitting solves a dense ``(n M) x (n M)``
    eigenproblem, so it suits moderate sample sizes.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means before fitting. Default is True.
        method: ``"laplacian"`` or ``"lle"``, for every view. Default is
            ``"laplacian"``.
        n_neighbors: Neighbours in each view's graph. Per-view. Default is 10.
        affinity: ``"nearest_neighbors"`` or ``"rbf"``, for ``method="laplacian"``.
            Per-view. Default is ``"nearest_neighbors"``.
        gamma: RBF kernel coefficient for ``affinity="rbf"``; ``None`` uses
            ``1 / n_features``. Per-view. Default is None.
        lle_reg: Regularisation of each local reconstruction for
            ``method="lle"``. Per-view. Default is 1e-3.
        n_operator_components: Smoothest operator eigenvectors kept per view
            before the joint solve; fewer is stronger regularisation. ``None``
            uses ``max(4 * n_components, 10)``, clipped to ``n_samples - 1``.
            Per-view. Default is None.
        eps: Floor on the kept operator eigenvalues. Default is 1e-6.

    Attributes:
        embedding_: Training embedding of each view, shape (n_samples, n_components).
        train_views_: The centred training views, from which a new view's
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

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "method": [StrOptions({"laplacian", "lle"})],
        "n_neighbors": [Interval(Integral, 1, None, closed="left"), "array-like"],
        "affinity": [StrOptions({"nearest_neighbors", "rbf"}), "array-like"],
        "gamma": [Interval(Real, 0, None, closed="neither"), None, "array-like"],
        "lle_reg": [Interval(Real, 0, None, closed="left"), "array-like"],
        "n_operator_components": [
            Interval(Integral, 1, None, closed="left"),
            None,
            "array-like",
        ],
        "eps": POSITIVE_EPS,
    }

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        method: str = "laplacian",
        n_neighbors: int | list[int] = 10,
        affinity: str | list[str] = "nearest_neighbors",
        gamma: float | list[float | None] | None = None,
        lle_reg: float | list[float] = 1e-3,
        n_operator_components: int | list[int | None] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.method = method
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.gamma = gamma
        self.lle_reg = lle_reg
        self.n_operator_components = n_operator_components
        self.eps = eps

    def _resolve_n_operator_components(
        self, n_operator_components: int | None, n: int
    ) -> int:
        if n_operator_components is not None:
            return min(n_operator_components, n - 1)
        return min(max(4 * self.n_components, 10), n - 1)

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

        n_neighbors_ = perview_parameter("n_neighbors", self.n_neighbors, 10, m)
        affinity_ = perview_parameter("affinity", self.affinity, "nearest_neighbors", m)
        gamma_ = perview_parameter("gamma", self.gamma, None, m)
        lle_reg_ = perview_parameter("lle_reg", self.lle_reg, 1e-3, m)
        n_operator_components_ = perview_parameter(
            "n_operator_components", self.n_operator_components, None, m
        )

        # Project onto the constant vector's orthogonal complement first --
        # see _orthonormal_complement_of_ones -- so the shared (near-)null
        # direction every operator and the reward both have never enters
        # the solve at all, rather than being merely floored.
        P = _orthonormal_complement_of_ones(n)
        C_reduced = P.T @ _centering_matrix(n) @ P

        k_ops = [
            self._resolve_n_operator_components(c, n) for c in n_operator_components_
        ]
        bases = []
        full_bases = []
        eigenvalue_blocks = []
        laplacian_degrees: list[np.ndarray] = []
        laplacian_gamma: list[float | None] = []
        laplacian_nn: list[NearestNeighbors | None] = []
        lle_nn: list[NearestNeighbors] = []
        for v, nn_i, aff_i, gamma_i, lle_reg_i, k_op in zip(
            views_, n_neighbors_, affinity_, gamma_, lle_reg_, k_ops
        ):
            if self.method == "laplacian":
                W, resolved_gamma = _laplacian_affinity(v, nn_i, aff_i, gamma_i)
                operator = _normalised_laplacian(W)
                laplacian_degrees.append(W.sum(axis=1))
                laplacian_gamma.append(resolved_gamma)
                laplacian_nn.append(
                    NearestNeighbors(n_neighbors=nn_i).fit(v)
                    if aff_i == "nearest_neighbors"
                    else None
                )
            else:
                operator = _lle_operator(v, nn_i, lle_reg_i)
                lle_nn.append(NearestNeighbors(n_neighbors=nn_i + 1).fit(v))

            reduced_operator = P.T @ operator @ P
            basis, eigenvalues = _smooth_basis(reduced_operator, k_op, self.eps)
            bases.append(basis)
            full_bases.append(P @ basis)
            eigenvalue_blocks.append(eigenvalues)

        offsets = np.concatenate([[0], np.cumsum(k_ops)])
        B = np.asarray(block_diag(*[np.diag(ev) for ev in eigenvalue_blocks])) / m
        A = np.zeros((offsets[-1], offsets[-1]))
        for i in range(m):
            for j in range(m):
                if i != j:
                    block = bases[i].T @ C_reduced @ bases[j]
                    A[offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]] = block
        A /= m

        _, eigvecs = gevp(A, B, self.n_components)
        blocks = list(np.split(eigvecs, offsets[1:-1], axis=0))
        embedding = [fb @ blk for fb, blk in zip(full_bases, blocks)]
        self.embedding_: list[np.ndarray] = embedding
        self.train_views_: list[np.ndarray] = views_
        self._n_neighbors_: list[int] = n_neighbors_
        self._affinity_: list[str] = affinity_
        self._lle_reg_: list[float] = lle_reg_

        if self.method == "laplacian":
            self._laplacian_state_: list[_LaplacianViewState] | None = [
                _LaplacianViewState(
                    full_basis=full_bases[i],
                    mu=np.maximum(1.0 - eigenvalue_blocks[i], _NYSTROM_MU_FLOOR),
                    block=blocks[i],
                    degrees=laplacian_degrees[i],
                    gamma=laplacian_gamma[i],
                    nn=laplacian_nn[i],
                )
                for i in range(m)
            ]
            self._lle_state_: list[NearestNeighbors] | None = None
        else:
            self._laplacian_state_ = None
            self._lle_state_ = lle_nn
        return self._finish_fit(views_)

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        v_train = self.train_views_[view]
        n_neighbors = self._n_neighbors_[view]
        if self.method == "lle":
            assert self._lle_state_ is not None
            indices = self._lle_state_[view].kneighbors(
                centred, n_neighbors=n_neighbors, return_distance=False
            )
            weights = _barycenter_weights(
                centred, v_train, indices, self._lle_reg_[view]
            )
            embedded: np.ndarray = np.einsum(
                "qn,qnk->qk", weights, self.embedding_[view][indices]
            )
            return embedded
        assert self._laplacian_state_ is not None
        state = self._laplacian_state_[view]
        W_new = _laplacian_new_point_affinity(
            centred, v_train, self._affinity_[view], state.gamma, n_neighbors, state.nn
        )
        degrees_new = np.maximum(W_new.sum(axis=1), 1e-12)
        K_tilde = W_new / np.sqrt(np.outer(degrees_new, state.degrees))
        extended: np.ndarray = (K_tilde @ state.full_basis) / state.mu @ state.block
        return extended
