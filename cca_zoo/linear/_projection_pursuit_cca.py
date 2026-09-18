r"""ProjectionPursuitCCA -- robust multiview CCA via projection pursuit."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.covariance import MinCovDet
from sklearn.utils._param_validation import Interval, StrOptions

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import deflate
from cca_zoo._utils._param_constraints import POSITIVE_INT

IndexFn = Callable[[np.ndarray, np.ndarray], float]


def _angles_to_unit_vector(theta: np.ndarray, p: int) -> np.ndarray:
    r"""Hyperspherical-angle parametrisation of a unit vector in $\R^p$.

    Recovers a $p$-dimensional unit-norm vector from $p-1$ unconstrained
    angles, so a projection direction can be searched over with an ordinary
    unconstrained optimiser rather than one that has to respect a norm
    constraint. Follows the recursive construction of
    :cite:`branco2005robust` (their Section 2, worked in reverse from polar
    to Cartesian coordinates):

    $$
    a_{(2)} = (\cos\theta_1, \sin\theta_1), \qquad
    a_{(j)} = (a_{(j-1)} \sin\theta_{j-1},\ \cos\theta_{j-1})
    \quad (2 < j \le p).
    $$

    Unlike the cited paper, which restricts each angle's range to make the
    vector unique (up to a global sign), the ranges here are left
    unconstrained: :func:`sin`/:func:`cos` are already periodic, so an
    unconstrained optimiser can reach every direction on the sphere without
    needing the restriction, at the cost of the parametrisation no longer
    being one-to-one (harmless for a search that only cares about the
    optimum it finds, not the coordinates that got it there).

    Args:
        theta: Angles, shape ``(p - 1,)``.
        p: Target dimensionality. ``p == 1`` returns ``[1.0]`` directly
            (no angle needed -- a 1-dimensional unit vector has no freedom
            beyond an overall sign, which the *other* view's direction can
            already absorb for a single pair).

    Returns:
        Unit-norm vector, shape ``(p,)``.
    """
    if p == 1:
        return np.ones(1)
    a = np.array([np.cos(theta[0]), np.sin(theta[0])])
    for j in range(1, p - 1):
        a = np.concatenate([a * np.sin(theta[j]), [np.cos(theta[j])]])
    return a


def spearman_projection_index(u: np.ndarray, v: np.ndarray) -> float:
    r"""Spearman rank correlation between two projected univariate scores.

    The projection index behind ``projection_index="spearman"``: the
    correlation between the *ranks* of ``u`` and ``v`` rather than their raw
    values, so it does not rely on any symmetry or moment condition the way
    Pearson correlation does, and is insensitive to any outlier's exact
    magnitude -- only its rank matters. This is ``PP-SPM`` in
    :cite:`branco2005robust`, who find it the strongest of the projection
    indices they compare, with good efficiency in both the presence and
    absence of contamination.

    Args:
        u: Projected scores for one view, shape ``(n,)``.
        v: Projected scores for another view, shape ``(n,)``.

    Returns:
        Spearman's rho, in ``[-1, 1]`` (``0.0`` if either score is constant,
        where rank correlation is undefined).
    """
    rho, _ = spearmanr(u, v)
    return 0.0 if np.isnan(rho) else float(rho)


def mcd_projection_index(
    u: np.ndarray, v: np.ndarray, support_fraction: float, random_state: int | None
) -> float:
    r"""Correlation derived from a bivariate minimum covariance determinant fit.

    The projection index behind ``projection_index="mcd"``: fit the MCD
    estimator (:class:`~sklearn.covariance.MinCovDet`) to the 2-dimensional
    ``(u, v)`` scatter and read the correlation off its robust covariance
    estimate, rather than the ordinary (non-robust) sample covariance. This
    is ``PP-MCD`` in :cite:`branco2005robust`; the same paper's own
    conclusion is that it is a reasonable, faster-to-compute alternative to
    ``PP-SPM`` when computation time matters more than squeezing out the
    last bit of efficiency.

    Args:
        u: Projected scores for one view, shape ``(n,)``.
        v: Projected scores for another view, shape ``(n,)``.
        support_fraction: Passed straight through to
            :class:`~sklearn.covariance.MinCovDet`.
        random_state: Passed straight through to
            :class:`~sklearn.covariance.MinCovDet`.

    Returns:
        The MCD-based correlation coefficient, in ``[-1, 1]`` (``0.0`` if
        the fit is degenerate, e.g. a near-zero robust variance).
    """
    z = np.column_stack([u, v])
    try:
        cov = (
            MinCovDet(support_fraction=support_fraction, random_state=random_state)
            .fit(z)
            .covariance_
        )
    except ValueError:
        return 0.0
    denom = np.sqrt(cov[0, 0] * cov[1, 1])
    return 0.0 if denom < 1e-12 else float(cov[0, 1] / denom)


class ProjectionPursuitCCA(BaseModel):
    r"""ProjectionPursuitCCA -- robust multiview CCA via projection pursuit.

    Every other estimator in :mod:`cca_zoo.linear` -- robust or not -- is
    built from a cross- or auto-*covariance* statistic of the projected
    views, computed once the projection directions are (implicitly or
    explicitly) fixed. Projection pursuit inverts that: it never forms a
    covariance matrix at all, instead searching *directly* over candidate
    projection directions for the pair that maximises a robust bivariate
    correlation measure (the *projection index*) between the resulting
    univariate scores, following the classical projection-pursuit paradigm
    of :cite:`huber1985projection` as carried over to CCA specifically by
    :cite:`branco2005robust`.

    For two views this reduces to their exact problem: find unit vectors
    $\mathbf{a}, \mathbf{b}$ maximising $\operatorname{PI}(X\mathbf{a},
    Y\mathbf{b})$ for a robust correlation measure $\operatorname{PI}$.
    ``ProjectionPursuitCCA`` generalises this to $M \geq 2$ views by
    maximising the *average* projection index over every pair of views
    (the same generalisation :class:`~cca_zoo.linear.RANSACCCA`'s
    consensus score and :class:`~cca_zoo.linear.MCCA`'s sum-of-pairwise
    objective both make):

    $$
    \max_{\|\mathbf{a}_1\| = \dots = \|\mathbf{a}_M\| = 1}
    \frac{1}{\binom{M}{2}} \sum_{i < j}
    \operatorname{PI}(X_i \mathbf{a}_i, X_j \mathbf{a}_j).
    $$

    Two projection indices are available: ``"spearman"`` (default, Spearman
    rank correlation -- see :func:`spearman_projection_index`) and ``"mcd"``
    (a minimum-covariance-determinant-based correlation -- see
    :func:`mcd_projection_index`).

    Fit by direct numerical search: each unit vector is parametrised by
    $p_i - 1$ unconstrained angles (:func:`_angles_to_unit_vector`), and
    the (generally non-smooth -- a rank correlation changes discontinuously
    wherever two projected scores swap rank order) objective above is
    maximised over the stacked angle vector by Powell's method
    (derivative-free, unlike the L-BFGS-B used elsewhere in this package,
    since the objective's gradient is undefined almost everywhere) from
    ``n_restarts`` random starting points, keeping the best. Later latent
    dimensions are fit the same way on views deflated by the previously
    found directions (:func:`~cca_zoo._utils._linalg.deflate`, the same
    Gram-Schmidt convention :mod:`cca_zoo.sparse`'s ALS-based methods use).

    Note:
        Unlike :class:`~cca_zoo.linear.HuberCCA` (leverage-based) and
        :class:`~cca_zoo.linear.RANSACCCA`/:class:`~cca_zoo.linear.TrimmedCCA`
        (both relational, via a per-sample loss or agreement score),
        projection pursuit does not single out individual bad rows at all
        -- its robustness comes entirely from the projection index itself
        being insensitive to a handful of extreme values, whatever kind of
        contamination produced them. There is accordingly no
        ``inlier_mask_`` to inspect after fitting.

        The random-restart search is not guaranteed to find the global
        optimum (the objective is non-convex and, for ``"spearman"``,
        genuinely discontinuous), and its cost scales with ``n_restarts``
        times the cost of one projection-index evaluation times
        ``latent_dimensions``; ``"mcd"`` is markedly more expensive per
        evaluation than ``"spearman"`` since it re-fits a robust covariance
        estimator at every candidate direction.

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        projection_index: ``"spearman"`` (default) or ``"mcd"`` -- which
            robust bivariate correlation measure to maximise; see
            :func:`spearman_projection_index` and
            :func:`mcd_projection_index`.
        mcd_support_fraction: Passed to :class:`~sklearn.covariance.MinCovDet`
            when ``projection_index="mcd"``; ignored otherwise. Default 0.75.
        n_restarts: Random restarts of the direction search per latent
            dimension; the best-scoring result is kept. Default 10.
        max_iter: Maximum Powell iterations per restart. Default 200.
        tol: Convergence tolerance for Powell's method (``xtol``/``ftol``).
            Default 1e-6.
        random_state: Seed for the random restarts (and for ``"mcd"``'s own
            random subsampling).

    References:
        Huber, P. J. (1985). Projection pursuit. The Annals of Statistics,
        13(2), 435-475.

        Branco, J. A., Croux, C., Filzmoser, P., & Oliveira, M. R. (2005).
        Robust canonical correlations: A comparative study. Computational
        Statistics, 20(2), 203-229.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 6))
        >>> X2 = rng.standard_normal((200, 5))
        >>> model = ProjectionPursuitCCA(latent_dimensions=1, random_state=0)
        >>> model = model.fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "projection_index": [StrOptions({"spearman", "mcd"})],
        "mcd_support_fraction": [Interval(Real, 0, 1, closed="right")],
        "n_restarts": POSITIVE_INT,
        "max_iter": POSITIVE_INT,
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        projection_index: str = "spearman",
        mcd_support_fraction: float = 0.75,
        n_restarts: int = 10,
        max_iter: int = 200,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.projection_index = projection_index
        self.mcd_support_fraction = mcd_support_fraction
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _make_index_fn(self, rng: np.random.Generator) -> IndexFn:
        """Resolve ``projection_index`` into a callable ``(u, v) -> float``."""
        if self.projection_index == "spearman":
            return spearman_projection_index
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        support_fraction = self.mcd_support_fraction
        return lambda u, v: mcd_projection_index(u, v, support_fraction, seed)

    def _fit_directions(
        self,
        views: list[np.ndarray],
        pairs: list[tuple[int, int]],
        index_fn: IndexFn,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Search for one unit-norm direction per view maximising the index.

        Args:
            views: Current (deflated) view arrays.
            pairs: Every ``(i, j)`` pair of view indices, ``i < j``.
            index_fn: Projection index, ``(u, v) -> float``.
            rng: Random generator for the restarts.

        Returns:
            One unit-norm direction per view, matching ``views``' order.
        """
        ps = [v.shape[1] for v in views]
        sizes = [max(p - 1, 0) for p in ps]
        offsets = np.cumsum([0] + sizes)

        def unpack(theta: np.ndarray) -> list[np.ndarray]:
            return [
                _angles_to_unit_vector(theta[offsets[i] : offsets[i + 1]], ps[i])
                for i in range(len(ps))
            ]

        def objective(theta: np.ndarray) -> float:
            directions = unpack(theta)
            scores = [views[i] @ directions[i] for i in range(len(ps))]
            total = sum(index_fn(scores[i], scores[j]) for i, j in pairs)
            return -total / len(pairs)

        n_theta = int(offsets[-1])
        if n_theta == 0:
            # Every view is already 1-dimensional: nothing left to search.
            return unpack(np.zeros(0))

        best_value = np.inf
        best_theta = np.zeros(n_theta)
        for _ in range(self.n_restarts):
            theta0 = rng.uniform(0.0, 2 * np.pi, size=n_theta)
            result = minimize(
                objective,
                theta0,
                method="Powell",
                options={
                    "maxiter": self.max_iter,
                    "xtol": self.tol,
                    "ftol": self.tol,
                },
            )
            if result.fun < best_value:
                best_value, best_theta = float(result.fun), result.x
        return unpack(best_theta)

    def fit(self, views: list[ArrayLike], y: None = None) -> ProjectionPursuitCCA:
        """Fit ProjectionPursuitCCA by direct search over projection directions.

        Args:
            views: List of 2 or more arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        index_fn = self._make_index_fn(rng)
        pairs = list(combinations(range(self.n_views_), 2))

        weights: list[np.ndarray] = [
            np.zeros((p, self.latent_dimensions)) for p in self.n_features_in_
        ]
        deflated = [v.copy() for v in views_]
        for d in range(self.latent_dimensions):
            directions = self._fit_directions(deflated, pairs, index_fn, rng)
            for i, a in enumerate(directions):
                weights[i][:, d] = a
            deflated = deflate(deflated, directions)
        self.weights_ = weights
        return self
