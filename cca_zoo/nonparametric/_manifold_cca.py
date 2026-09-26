r"""ManifoldCCA — transductive multiview CCA over a shared manifold operator."""

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
    r"""The (scaled) sample-covariance operator of the identity 'feature map'.

    $\frac{1}{n-1}\bigl(I_n - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top\bigr)$ --
    if each training point's "feature vector" were literally its own
    one-hot indicator (so the projection *is* the per-sample embedding,
    with nothing left to learn but which n-vector it should be), this is
    that identity feature map's own covariance, i.e. exactly what
    :class:`~cca_zoo.linear.MCCA`'s ``_build_A`` would compute from it.
    """
    return (np.eye(n) - np.ones((n, n)) / n) / (n - 1)


def _smooth_basis(
    operator: np.ndarray, n_components: int, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""The ``n_components`` eigenvectors of ``operator`` with the smallest eigenvalues.

    Every method :class:`ManifoldCCA` is built on --
    :class:`~sklearn.manifold.SpectralEmbedding`,
    :class:`~sklearn.manifold.LocallyLinearEmbedding` -- keeps only a
    handful of components, never all $n-1$: on genuine low-dimensional
    manifold data there's a spectral *gap* between a few small
    ("smooth"/"structure") eigenvalues and a bulk of larger ("noise")
    ones, and truncating to the small side is where the denoising
    actually happens. Solving the *full* untruncated operator (as an
    earlier version of this module did) hands each view $n-1$ genuine
    degrees of freedom to search for a matching direction against the
    other view -- with a reward that (see :func:`_centering_matrix`)
    doesn't know anything about either view's actual content, that much
    freedom is enough to fabricate spurious cross-view correlation out of
    pure noise, the classic small-$n$-large-effective-dimension CCA
    failure mode every ridge-regularised method elsewhere in this library
    exists to avoid -- except a ridge blend toward the identity doesn't
    fix it here, since the identity is already full rank. A hard
    truncation is what's actually needed, matching every other spectral
    method's own convention.

    Args:
        operator: Symmetric operator, shape (n-1, n-1) (already projected
            onto the constant vector's orthogonal complement).
        n_components: Number of smallest-eigenvalue eigenvectors to keep.
        eps: Floor applied to each kept eigenvalue individually.

    Returns:
        Tuple ``(basis, eigenvalues)``: ``basis`` has shape
        ``(n-1, n_components)`` with orthonormal columns; ``eigenvalues``
        has shape ``(n_components,)``.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    k = min(n_components, operator.shape[0])
    basis = eigenvectors[:, :k]
    floored = np.maximum(eigenvalues[:k], eps)
    return basis, floored


def _orthonormal_complement_of_ones(n: int) -> np.ndarray:
    r"""An (n, n-1) orthonormal basis of the constant vector's orthogonal complement.

    Every operator :class:`ManifoldCCA` plugs in as a within-view
    constraint -- the graph Laplacian, the LLE reconstruction operator --
    has the constant vector $\mathbf{1}$ in its exact (Laplacian) or
    near-exact (LLE) null space: a uniform shift carries zero graph
    energy under either. :func:`_centering_matrix` (the shared
    between-view reward) also annihilates $\mathbf{1}$ exactly. Left in,
    $\mathbf{1}$ is therefore a near-simultaneous null direction of
    *both* sides of the generalised eigenproblem -- the classical source
    of spurious blow-up in a generalised ``eigh(A, B)`` solve (that
    direction's "eigenvalue" is a 0/0 ratio, and floating-point noise in
    the numerator and denominator can amplify it into something that
    dwarfs every genuine eigenvalue, silently stealing the requested
    top-``k`` slots with numerical junk rather than real structure).
    Projecting the whole problem onto this complement before solving
    (see :meth:`ManifoldCCA.fit`) removes the shared degeneracy outright,
    matching :class:`~sklearn.manifold.SpectralEmbedding`'s own
    ``drop_first=True`` default -- :class:`~sklearn.manifold.LocallyLinearEmbedding`
    never returns the trivial constant solution either.
    """
    return np.asarray(null_space(np.ones((1, n))))


def _laplacian_affinity(
    v: np.ndarray, n_neighbors: int, affinity: str, gamma: float | None
) -> tuple[np.ndarray, float | None]:
    r"""Training affinity matrix $W$ and resolved RBF ``gamma``.

    Reuses :class:`sklearn.manifold.SpectralEmbedding` purely for its own
    (well-tested) affinity-graph construction -- the k-NN search,
    symmetrisation and (for ``affinity="rbf"``) heat-kernel weighting --
    rather than reimplementing it. Returned separately from the Laplacian
    itself (see :func:`_normalised_laplacian`) since $W$'s row sums (the
    degree matrix) are also exactly what :meth:`ManifoldCCA.transform`
    needs to extend a fitted eigenvector to a new point (the classical
    Nystrom formula, e.g. Bengio et al. 2003) -- information the
    normalised Laplacian $I - D^{-1/2}WD^{-1/2}$ alone no longer carries
    separately.
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
    r"""New-to-training affinity, built by the same rule as the training graph.

    For ``affinity="rbf"`` this is exact -- the identical dense RBF kernel
    evaluation :func:`_laplacian_affinity` used for training, just off the
    diagonal block. For ``affinity="nearest_neighbors"`` it's the standard
    Nystrom simplification (also how
    :meth:`~sklearn.manifold.LocallyLinearEmbedding.transform` handles its
    own out-of-sample case): a new point's *own* one-directional
    connectivity to its ``n_neighbors`` nearest training points, rather than
    attempting to retroactively symmetrise against training points that
    might now also consider the new point a neighbour -- doing that
    properly would mean rebuilding the whole training graph per query point.
    """
    if affinity == "rbf":
        return np.asarray(rbf_kernel(v_new, v_train, gamma=gamma))
    assert nn is not None
    graph = nn.kneighbors_graph(v_new, n_neighbors=n_neighbors, mode="connectivity")
    return np.asarray(graph.toarray() if issparse(graph) else graph)


def _barycenter_weights(
    query: np.ndarray, reference: np.ndarray, indices: np.ndarray, reg: float
) -> np.ndarray:
    r"""Barycentric (sum-to-one) reconstruction weights of each query point.

    ``weights[a, :]`` reconstructs ``query[a]`` as the best linear
    combination of ``reference[indices[a]]`` -- Roweis & Saul (2000)'s
    original construction. Used both to build the training LLE operator
    (``query = reference = v``, each point reconstructed from its own
    neighbours) and, unchanged, to extend a new point at
    :meth:`ManifoldCCA.transform` time (``query`` = new data,
    ``reference`` = training data) -- exactly how
    :meth:`~sklearn.manifold.LocallyLinearEmbedding.transform` reuses its
    own training-time weight computation for new points too.

    Args:
        query: Points to reconstruct, shape (n_query, n_features).
        reference: Points to reconstruct from, shape (n_reference, n_features).
        indices: Neighbour indices into ``reference`` for each query point,
            shape (n_query, n_neighbors).
        reg: Regularisation added to each point's local reconstruction Gram
            matrix, relative to its trace.

    Returns:
        Array of shape (n_query, n_neighbors), each row summing to 1.
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
    r"""LLE operator $M = (I - W)^\top (I - W)$ from local reconstruction weights.

    ``W[a, :]`` reconstructs point ``a`` as the best (sum-to-one) linear
    combination of its ``n_neighbors`` nearest neighbours (see
    :func:`_barycenter_weights`), hand-implemented here (rather than reused
    from :class:`sklearn.manifold.LocallyLinearEmbedding`, which doesn't
    expose ``W``/``M`` as public API) so the raw operator is available for
    the joint eigenproblem below.
    """
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

    Solved after projecting both sides onto $\mathbf{1}^\perp$ (see
    :func:`_orthonormal_complement_of_ones`): every $M_i$ has the constant
    vector in its (near-)null space, as does the reward, so leaving it in
    would put a spurious, numerically unstable 0/0-type direction at the
    top of the spectrum -- the same reason
    :class:`~sklearn.manifold.SpectralEmbedding` always discards its own
    trivial constant solution. One consequence worth checking directly: if
    two views are given *identical* data, $A$ (via the shared reward)
    restricted to $\mathbf{1}^\perp$ is proportional to the identity, so
    the joint problem's solution for that view is exactly the ordinary
    Rayleigh-quotient minimiser of $M_i$ alone -- i.e. it reduces exactly
    to plain single-view spectral embedding of that view, which is the
    right sanity check for "is this actually the natural multiview
    generalisation" (verified directly against
    :class:`~sklearn.manifold.SpectralEmbedding`'s own affinity
    construction in the tests).

    Since $Z_i$ is only ever defined at the training points (there is no
    feature map to apply to new data), out-of-sample projection reuses each
    method's own established extension rather than a generic auxiliary
    model bolted on afterward:

    - ``method="lle"``: a new point's barycentric reconstruction weights
      (see :func:`_barycenter_weights`) against its ``n_neighbors`` nearest
      *training* points, applied directly to those training points' rows of
      the fitted $Z_i$ -- exactly
      :meth:`~sklearn.manifold.LocallyLinearEmbedding.transform`'s own
      mechanism, reusing the identical weight computation training used.
      Valid because that formula is linear in the training embedding: with
      $Z_i = Y_i B_i$ ($Y_i$ the per-view smooth basis, $B_i$ the joint
      solve's combination -- see :meth:`fit`), applying new-point weights to
      $Y_i$ first and then $B_i$, or to $Z_i = Y_i B_i$ directly, are the
      same computation.
    - ``method="laplacian"``: the classical Nystrom extension (Bengio et
      al. 2003) of each kept eigenvector individually -- $y_k(x) =
      \tfrac{1}{\mu_k}\sum_j \tilde{W}(x, x_j)\, y_k(x_j)$, $\tilde W$ the
      degree-normalised new-to-training affinity (built by the same rule as
      the training graph, see :func:`_laplacian_new_point_affinity`) and
      $\mu_k = 1 - \lambda_k$ -- then the *same* $B_i$ combination used at
      training time. Unlike the LLE case, this can't collapse to a single
      "apply weights to $Z_i$" step, since each eigenvector has its own
      $\mu_k$ rescaling *before* $B_i$ mixes them.

    Both are exact consequences of what :meth:`fit` actually solved, not a
    separately-fit approximation of it.

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

        There is no feature-space weight matrix: the fitted training
        embedding is ``embedding_[i]``, shape ``(n_train_samples, k)``, as
        in :class:`~sklearn.manifold.LocallyLinearEmbedding`.
        ``inverse_transform`` and ``predict`` work through the
        out-of-sample extension, like every model's.

        ``method`` itself is not a per-view parameter, unlike every other
        constructor argument: mixing ``"laplacian"`` and ``"lle"`` across
        views is not mathematically ruled out (the joint eigenproblem in
        :meth:`fit` only ever consumes each view's own basis/eigenvalues,
        regardless of which operator produced them), but ``transform``'s
        out-of-sample extension dispatches on ``method`` once for every
        view at once, and would need its own per-view branch and
        per-view-typed fitted state to support a genuine mix -- a
        larger, separate change from exposing this class's already
        per-view-independent operator hyperparameters.

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
            (locally linear embedding operator), the same for every view.
            Default ``"laplacian"``.
        n_neighbors: Number of neighbours used to build each view's graph.
            Either a single value applied to every view or a list of
            per-view values. Default 10.
        affinity: ``"nearest_neighbors"`` or ``"rbf"``, passed to
            :class:`~sklearn.manifold.SpectralEmbedding` when
            ``method="laplacian"``. Ignored for ``method="lle"``. Either a
            single value applied to every view or a list of per-view
            values. Default ``"nearest_neighbors"``.
        gamma: RBF kernel coefficient(s), used only when
            ``method="laplacian"`` and ``affinity="rbf"``. Either a single
            float (or ``None``) applied to every view or a list of
            per-view values. Default ``None`` (sklearn's own
            ``1 / n_features`` default).
        lle_reg: Regularisation added to each point's local reconstruction
            Gram matrix, used only when ``method="lle"``. Either a single
            float applied to every view or a list of per-view floats.
            Default 1e-3.
        n_operator_components: Number of each view's own smallest-eigenvalue
            operator components kept before the joint eigenproblem is
            solved (see :func:`_smooth_basis`) -- effectively this class's
            regularisation strength, the same role ``c`` plays elsewhere
            in this library, just the opposite direction: smaller keeps
            fewer, "smoother" candidate directions per view (more
            regularised, more denoised, but liable to discard real
            structure if set too small); the untruncated limit
            (``n_operator_components = n_samples - 1``) hands each view as
            many free directions as there are training points, which --
            like *any* unregularised multivariate CCA at that
            dimensionality-to-sample-size ratio -- fabricates spurious
            cross-view correlation out of pure noise (see
            ``tests/nonparametric/test_manifold_cca.py``'s comparison
            against plain unregularised ``MCCA`` at the same nominal
            dimensionality). Either a single value (or ``None``) applied to
            every view or a list of per-view values. Default ``None``:
            ``max(4 * latent_dimensions, 10)``, clipped to
            ``n_samples - 1``, independently per view.
        eps: Floor applied to each kept operator eigenvalue (see
            :func:`_smooth_basis`) to ensure positive definiteness. Default
            1e-6.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((60, 8))
        >>> X2 = rng.standard_normal((60, 6))
        >>> model = ManifoldCCA(method="laplacian", n_neighbors=8).fit([X1, X2])
        >>> scores = model.transform([X1, X2])

        A different neighbourhood size and regularisation strength per view:

        >>> model = ManifoldCCA(
        ...     method="laplacian", n_neighbors=[8, 12], n_operator_components=[10, 15]
        ... ).fit([X1, X2])
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
        latent_dimensions: int = 1,
        center: bool = True,
        method: str = "laplacian",
        n_neighbors: int | list[int] = 10,
        affinity: str | list[str] = "nearest_neighbors",
        gamma: float | list[float | None] | None = None,
        lle_reg: float | list[float] = 1e-3,
        n_operator_components: int | list[int | None] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
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
        return min(max(4 * self.latent_dimensions, 10), n - 1)

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

        _, eigvecs = gevp(A, B, self.latent_dimensions)
        blocks = list(np.split(eigvecs, offsets[1:-1], axis=0))
        embedding = [fb @ blk for fb, blk in zip(full_bases, blocks)]
        self.embedding_: list[np.ndarray] = embedding
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
        return self

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Project new views via each method's own out-of-sample extension.

        ``method="lle"`` reuses :func:`_barycenter_weights` against each new
        point's nearest training points (exactly
        :meth:`~sklearn.manifold.LocallyLinearEmbedding.transform`'s own
        mechanism); ``method="laplacian"`` uses the classical Nystrom
        extension of each kept eigenvector (Bengio et al. 2003) followed by
        the same combination :meth:`fit` used. See the class docstring.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            List of arrays, each of shape (n_samples, latent_dimensions).

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
        """
        return super().transform(views)

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        v_train = self._views_fit_[view]
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
