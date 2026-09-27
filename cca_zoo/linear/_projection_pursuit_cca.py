"""Robust multiview CCA by projection pursuit."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.covariance import MinCovDet
from sklearn.utils._param_validation import Interval, StrOptions

from cca_zoo._base import BaseModel
from cca_zoo._utils._linalg import deflate
from cca_zoo._utils._param_constraints import POSITIVE_INT, RANDOM_STATE

IndexFn = Callable[[np.ndarray, np.ndarray], float]


def _angles_to_unit_vector(theta: np.ndarray, p: int) -> np.ndarray:
    r"""Unit vector in $\mathbb{R}^p$ from $p - 1$ unconstrained angles.

    The hyperspherical parametrisation of Branco et al. (2005), with the
    angles left unrestricted so an unconstrained optimiser reaches every
    direction.

    Args:
        theta: Angles, shape (p - 1,).
        p: Dimension.

    Returns:
        Unit vector, shape (p,).
    """
    if p == 1:
        return np.ones(1)
    # Unrolled, entry k >= 2 is cos(theta_{k-1}) times the product of every
    # later angle's sine, and the first two entries share the product of
    # them all: suffix products of the sines.
    suffix = np.append(np.cumprod(np.sin(theta[:0:-1]))[::-1], 1.0)
    head = np.concatenate([[np.cos(theta[0]), np.sin(theta[0])], np.cos(theta[1:])])
    result: np.ndarray = head * np.concatenate([suffix[:1], suffix])
    return result


def spearman_projection_index(u: np.ndarray, v: np.ndarray) -> float:
    """Spearman rank correlation of two projected scores (``PP-SPM``).

    Args:
        u: Scores of one view, shape (n,).
        v: Scores of another view, shape (n,).

    Returns:
        Spearman's rho, or 0 if either score is constant.
    """
    # Pearson correlation of the average ranks, as spearmanr computes it
    # without its per-call wrapper overhead (most of a fit's time).
    ranks_u = rankdata(u) - (len(u) + 1) / 2
    ranks_v = rankdata(v) - (len(v) + 1) / 2
    scale = np.sqrt((ranks_u @ ranks_u) * (ranks_v @ ranks_v))
    return 0.0 if scale == 0 else float(ranks_u @ ranks_v / scale)


def mcd_projection_index(
    u: np.ndarray, v: np.ndarray, support_fraction: float, random_state: int | None
) -> float:
    """Correlation from a bivariate minimum covariance determinant fit (``PP-MCD``).

    Faster than :func:`spearman_projection_index` on large data, and somewhat
    less efficient.

    Args:
        u: Scores of one view, shape (n,).
        v: Scores of another view, shape (n,).
        support_fraction: Passed to :class:`~sklearn.covariance.MinCovDet`.
        random_state: Passed to :class:`~sklearn.covariance.MinCovDet`.

    Returns:
        The robust correlation, or 0 if either robust variance is zero.
    """
    cov = (
        MinCovDet(support_fraction=support_fraction, random_state=random_state)
        .fit(np.column_stack([u, v]))
        .covariance_
    )
    denom = np.sqrt(cov[0, 0] * cov[1, 1])
    return 0.0 if denom < 1e-12 else float(cov[0, 1] / denom)


class ProjectionPursuitCCA(BaseModel):
    r"""Robust multiview CCA by projection pursuit.

    Searches directly for unit directions maximising a robust correlation,
    the projection index PI, averaged over pairs of views:

    $$
    \max_{\|a_1\| = \dots = \|a_M\| = 1}
    \frac{1}{\binom{M}{2}} \sum_{i < j} \operatorname{PI}(X_i a_i, X_j a_j).
    $$

    Each direction is parametrised by angles and the objective, which is not
    smooth, is maximised by Powell's method from ``n_init`` random starts.
    Later dimensions are fitted on deflated views. Robustness comes from the
    index, so no rows are flagged as outliers.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        projection_index: ``"spearman"`` or ``"mcd"``. Default is
            ``"spearman"``.
        mcd_support_fraction: ``MinCovDet`` support fraction for
            ``projection_index="mcd"``. Default is 0.75.
        n_init: Random restarts per dimension. Default is 10.
        max_iter: Maximum Powell iterations per restart. Default is 200.
        tol: Powell ``xtol`` and ``ftol``. Default is 1e-6.
        random_state: Seed for the restarts. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    References:
        Branco, J. A., Croux, C., Filzmoser, P., & Oliveira, M. R. (2005).
        Robust canonical correlations: A comparative study. Computational
        Statistics, 20(2), 203-229.

        Alfons, A., Croux, C., & Filzmoser, P. (2017). Robust maximum
        association estimators. Journal of the American Statistical
        Association, 112(517), 436-445.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import ProjectionPursuitCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 6))
        >>> X2 = rng.standard_normal((200, 5))
        >>> model = ProjectionPursuitCCA(random_state=0).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "projection_index": [StrOptions({"spearman", "mcd"})],
        "mcd_support_fraction": [Interval(Real, 0, 1, closed="right")],
        "n_init": POSITIVE_INT,
        "max_iter": POSITIVE_INT,
        "tol": [Interval(Real, 0, None, closed="neither")],
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        projection_index: str = "spearman",
        mcd_support_fraction: float = 0.75,
        n_init: int = 10,
        max_iter: int = 200,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.projection_index = projection_index
        self.mcd_support_fraction = mcd_support_fraction
        self.n_init = n_init
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
        """One unit direction per view maximising the mean pairwise index."""
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
        for _ in range(self.n_init):
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
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_: list[np.ndarray] = self._setup_fit(views)
        rng = np.random.default_rng(self.random_state)
        index_fn = self._make_index_fn(rng)
        pairs = list(combinations(range(self.n_views_), 2))

        weights: list[np.ndarray] = [
            np.zeros((p, self.n_components)) for p in self.n_features_in_
        ]
        deflated = [v.copy() for v in views_]
        for d in range(self.n_components):
            directions = self._fit_directions(deflated, pairs, index_fn, rng)
            for i, a in enumerate(directions):
                weights[i][:, d] = a
            deflated = deflate(deflated, directions)
        self.weights_ = weights
        return self
