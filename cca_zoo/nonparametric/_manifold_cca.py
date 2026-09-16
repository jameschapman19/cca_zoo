r"""ManifoldCCA — transductive multiview CCA over a shared manifold operator."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import block_diag, null_space
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
            dimensionality). Default ``None``: ``max(4 * latent_dimensions,
            10)``, clipped to ``n_samples - 1``.
        extrapolator_alpha: Ridge strength for the per-view
            :class:`~sklearn.kernel_ridge.KernelRidge` out-of-sample
            extension. Default 1.0.
        eps: Floor applied to each kept operator eigenvalue (see
            :func:`_smooth_basis`) to ensure positive definiteness. Default
            1e-6.

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
        "n_operator_components": [Interval(Integral, 1, None, closed="left"), None],
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
        n_operator_components: int | None = None,
        extrapolator_alpha: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.method = method
        self.n_neighbors = n_neighbors
        self.affinity = affinity
        self.gamma = gamma
        self.lle_reg = lle_reg
        self.n_operator_components = n_operator_components
        self.extrapolator_alpha = extrapolator_alpha
        self.eps = eps

    def _operator(self, v: np.ndarray) -> np.ndarray:
        if self.method == "laplacian":
            return _laplacian_operator(v, self.n_neighbors, self.affinity, self.gamma)
        return _lle_operator(v, self.n_neighbors, self.lle_reg)

    def _resolve_n_operator_components(self, n: int) -> int:
        if self.n_operator_components is not None:
            return min(self.n_operator_components, n - 1)
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

        # Project onto the constant vector's orthogonal complement first --
        # see _orthonormal_complement_of_ones -- so the shared (near-)null
        # direction every operator and the reward both have never enters
        # the solve at all, rather than being merely floored.
        P = _orthonormal_complement_of_ones(n)
        C_reduced = P.T @ _centering_matrix(n) @ P

        k_op = self._resolve_n_operator_components(n)
        bases = []
        eigenvalue_blocks = []
        for v in views_:
            reduced_operator = P.T @ self._operator(v) @ P
            basis, eigenvalues = _smooth_basis(reduced_operator, k_op, self.eps)
            bases.append(basis)
            eigenvalue_blocks.append(eigenvalues)

        B = np.asarray(block_diag(*[np.diag(ev) for ev in eigenvalue_blocks])) / m
        A = np.zeros((k_op * m, k_op * m))
        for i in range(m):
            for j in range(m):
                if i != j:
                    block = bases[i].T @ C_reduced @ bases[j]
                    A[i * k_op : (i + 1) * k_op, j * k_op : (j + 1) * k_op] = block
        A /= m

        _, eigvecs = gevp(A, B, self.latent_dimensions)
        embedding = [
            P @ (basis @ block)
            for basis, block in zip(bases, np.split(eigvecs, m, axis=0))
        ]
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
