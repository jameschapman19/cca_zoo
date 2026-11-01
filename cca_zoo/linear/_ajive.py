"""Angle-based Joint and Individual Variation Explained."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted

from cca_zoo._base import BaseModel
from cca_zoo._utils._param_constraints import POSITIVE_INT, RANDOM_STATE
from cca_zoo._utils._validation import perview_parameter

_RANK_LIST: list[Any] = [None, Interval(Integral, 0, None, closed="left"), "array-like"]


def _optimal_rank(singular_values: np.ndarray, n: int, p: int) -> int:
    """Singular values above Gavish and Donoho's hard threshold, noise unknown.

    The threshold is ``omega(beta)`` times the median singular value, with
    ``beta`` the aspect ratio of the ``n`` by ``p`` matrix and ``omega`` their
    polynomial approximation to the optimal coefficient.
    """
    beta = min(n, p) / max(n, p)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    return int(np.sum(singular_values > omega * np.median(singular_values)))


class AJIVE(BaseModel):
    r"""Angle-based Joint and Individual Variation Explained.

    Splits each centred view into variation shared with the other views, variation
    of its own, and noise:

    $$
    X_i = \underbrace{Z_J B_i}_{\text{joint}}
        + \underbrace{Z_{I,i} L_i^\top}_{\text{individual}} + E_i.
    $$

    $Z_J$ holds one set of joint scores for all views. The steps are those of
    Feng et al. (2018):

    1. Each view's signal subspace is its top ``signal_ranks`` left singular
       vectors.
    2. The joint score space is spanned by the top left singular vectors of
       the signal subspaces stacked side by side. Their squared singular values
       lie between 0 and the number of views, and measure how well a
       direction lies in all the signal subspaces.
    3. Each view's joint part is its projection onto the joint scores. The
       individual part is the rank-``individual_ranks`` approximation of the
       rest, which is orthogonal to the joint scores in sample space.

    ``n_components`` is the joint rank, a number or ``"auto"``. With
    ``"auto"``, a squared singular value counts as joint when it exceeds the
    ``1 - alpha`` quantile of the largest one for random signal subspaces of
    the same ranks (the random direction bound of Feng et al.). Their Wedin
    bound, for the estimation error of the signal subspaces, is not used. When
    ranks are not given, each is the number of singular values above Gavish and
    Donoho's (2014) optimal hard threshold.

    With each view cut to its signal subspace, the joint scores are those of CCA for
    two views, the normalised sum of the two canonical variates, and of multiset CCA
    (:class:`~cca_zoo.linear.GCCA`) for more: a hard rank cut stands in for CCA's ridge
    penalty, and the individual structure and variance accounting come on top.

    The estimator follows the package's multiview API with the joint scores as
    the latent space: ``transform`` estimates each view's joint scores by
    regressing its centred rows on the view's joint and individual loadings
    together, so that individual variation does not leak into them.
    :meth:`transform_individual` gives the individual scores from the same
    regression. For the training views these are close to, but not exactly,
    ``joint_scores_`` and ``individual_scores_``.

    Args:
        n_components: Number of joint components, or ``"auto"`` to select them
            by the random direction bound. Default is 1.
        center: Whether to centre each view. Default is True.
        signal_ranks: Rank of each view's signal subspace: a number for all
            views, one per view, or None for Gavish and Donoho's threshold.
            Raised to ``n_components`` where it is lower. Default is None.
        individual_ranks: Rank of each view's individual structure, as
            ``signal_ranks``; 0 for none. Default is None.
        alpha: Level of the random direction bound, for ``n_components="auto"``.
            Default is 0.05.
        n_null_draws: Random signal subspaces drawn for that bound. Default is
            100.
        random_state: Seed for those draws. Default is None.

    Attributes:
        weights_: Weights of each view's joint scores, shape
            (n_features_i, n_components_).
        individual_weights_: Weights of each view's individual scores, shape
            (n_features_i, n_individual_i).
        joint_scores_: Joint scores of the training samples, unit variance and
            uncorrelated, shape (n_samples, n_components_).
        joint_loadings_: Joint loadings $B_i^\top$ of each view, shape
            (n_features_i, n_components_).
        individual_scores_: Individual scores of each view for the training
            samples, unit variance and uncorrelated, shape
            (n_samples, n_individual_i).
        individual_loadings_: Individual loadings $L_i$ of each view, shape
            (n_features_i, n_individual_i).
        signal_ranks_: Signal rank used for each view.
        joint_variance_explained_: Fraction of each view's variance in each
            joint component, shape (n_views, n_components_). Components are
            orthogonal, so the fractions add up.
        individual_variance_explained_: Fraction of each view's variance in
            each of its individual components, one array per view.
        noise_variance_explained_: Fraction of each view's variance in
            neither, shape (n_views,). For each view, the joint, individual
            and noise fractions add up to 1.

    References:
        Feng, Q., Jiang, M., Hannig, J., & Marron, J. S. (2018). Angle-based
        joint and individual variation explained. Journal of Multivariate
        Analysis, 166, 241-265.
        Lock, E. F., Hoadley, K. A., Marron, J. S., & Nobel, A. B. (2013).
        Joint and individual variation explained (JIVE) for integrated
        analysis of multiple data types. The Annals of Applied Statistics,
        7(1), 523-542.
        Gavish, M., & Donoho, D. L. (2014). The optimal hard threshold for
        singular values is 4/sqrt(3). IEEE Transactions on Information
        Theory, 60(8), 5040-5053.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import AJIVE
        >>> rng = np.random.default_rng(0)
        >>> joint = rng.standard_normal((100, 1))
        >>> views = [
        ...     joint @ rng.standard_normal((1, p))
        ...     + rng.standard_normal((100, 2)) @ rng.standard_normal((2, p))
        ...     + 0.3 * rng.standard_normal((100, p))
        ...     for p in (20, 30)
        ... ]
        >>> model = AJIVE(n_components="auto", random_state=0).fit(views)
        >>> model.n_components_
        1
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "n_components": [
            Interval(Integral, 1, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "signal_ranks": _RANK_LIST,
        "individual_ranks": _RANK_LIST,
        "alpha": [Interval(Real, 0, 1, closed="neither")],
        "n_null_draws": POSITIVE_INT,
        "random_state": RANDOM_STATE,
    }

    # "auto" is not a number to compare with the narrowest view; fit checks.
    _components_bounded_by_features: ClassVar[bool] = False

    def __init__(
        self,
        n_components: int | Literal["auto"] = 1,
        *,
        center: bool = True,
        signal_ranks: int | list[int] | None = None,
        individual_ranks: int | list[int] | None = None,
        alpha: float = 0.05,
        n_null_draws: int = 100,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)  # type: ignore[arg-type]
        self.signal_ranks = signal_ranks
        self.individual_ranks = individual_ranks
        self.alpha = alpha
        self.n_null_draws = n_null_draws
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> AJIVE:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        n = self.n_samples_
        fixed = None if isinstance(self.n_components, str) else self.n_components
        narrowest = min(self.n_features_per_view_)
        if fixed is not None and not fixed <= min(narrowest, n - 1):
            raise ValueError(
                f"n_components={fixed} must be at most {min(narrowest, n - 1)}, "
                "the number of features in the narrowest view or one less than "
                "the number of samples."
            )

        # 1. Each view's signal subspace.
        svds = [np.linalg.svd(v, full_matrices=False) for v in views_]
        requested = perview_parameter(
            "signal_ranks", _as_list(self.signal_ranks), None, self.n_views_
        )
        signal_ranks = []
        for (_, s, _), r, p in zip(svds, requested, self.n_features_per_view_):
            rank = _optimal_rank(s, n, p) if r is None else r
            signal_ranks.append(min(max(rank, fixed or 1), n - 1, p))
        bases = [u[:, :r] for (u, _, _), r in zip(svds, signal_ranks)]

        # 2. The joint score space.
        stacked_u, stacked_s, _ = np.linalg.svd(
            np.concatenate(bases, axis=1), full_matrices=False
        )
        joint_rank = fixed
        if joint_rank is None:
            bound = self._random_direction_bound(signal_ranks, n)
            joint_rank = int(np.sum(stacked_s**2 > bound))
            if joint_rank == 0:
                raise ValueError(
                    "No squared singular value of the stacked signal subspaces "
                    f"exceeds the random direction bound of {bound:.3f}, so no "
                    "joint structure was found. Set n_components to choose a "
                    "joint rank, or signal_ranks to widen the signal subspaces."
                )
        joint_scores = stacked_u[:, :joint_rank] * np.sqrt(n - 1)

        # 3. Each view's joint part, then its individual part.
        individual_requested = perview_parameter(
            "individual_ranks", _as_list(self.individual_ranks), None, self.n_views_
        )
        self.weights_: list[np.ndarray] = []
        self.joint_loadings_: list[np.ndarray] = []
        self.individual_scores_: list[np.ndarray] = []
        self.individual_loadings_: list[np.ndarray] = []
        self.individual_weights_: list[np.ndarray] = []
        joint_var, individual_var, noise_var = [], [], []
        for v, r, p in zip(views_, individual_requested, self.n_features_per_view_):
            joint_loadings = joint_scores.T @ v / (n - 1)  # (k, p)
            residual = v - joint_scores @ joint_loadings
            u, s, vt = np.linalg.svd(residual, full_matrices=False)
            rank = _optimal_rank(s, n, p) if r is None else min(r, len(s))
            total = np.sum(v**2)
            individual_loadings = vt[:rank].T * s[:rank] / np.sqrt(n - 1)  # (p, r)
            # Row x = z B + s L' + noise, solved for (z, s) together.
            solved = np.linalg.pinv(
                np.concatenate([joint_loadings.T, individual_loadings], axis=1)
            ).T
            self.weights_.append(solved[:, :joint_rank])
            self.individual_weights_.append(solved[:, joint_rank:])
            self.joint_loadings_.append(joint_loadings.T)
            self.individual_scores_.append(u[:, :rank] * np.sqrt(n - 1))
            self.individual_loadings_.append(individual_loadings)
            joint_var.append((n - 1) * np.sum(joint_loadings**2, axis=1) / total)
            individual_var.append(s[:rank] ** 2 / total)
            noise_var.append((np.sum(residual**2) - np.sum(s[:rank] ** 2)) / total)
        self.signal_ranks_: list[int] = signal_ranks
        self.joint_scores_: np.ndarray = joint_scores
        self.joint_variance_explained_: np.ndarray = np.array(joint_var)
        self.individual_variance_explained_: list[np.ndarray] = individual_var
        self.noise_variance_explained_: np.ndarray = np.array(noise_var)
        self._fit_maps_and_importances(views_)
        return self

    def _random_direction_bound(self, signal_ranks: list[int], n: int) -> float:
        """The ``1 - alpha`` quantile of the top squared singular value under chance.

        Random signal subspaces of the views' ranks, in the space of centred
        samples, stand in for views with no shared structure.
        """
        rng = check_random_state(self.random_state)
        top = np.empty(self.n_null_draws)
        for d in range(self.n_null_draws):
            bases = []
            for r in signal_ranks:
                g = rng.standard_normal((n, r))
                bases.append(np.linalg.qr(g - g.mean(axis=0))[0])
            top[d] = np.linalg.norm(np.concatenate(bases, axis=1), ord=2) ** 2
        return float(np.quantile(top, 1 - self.alpha))

    def transform_individual(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Each view's individual scores.

        Each centred row is regressed on the view's joint and individual
        loadings together, as for ``transform``.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            List of arrays, each of shape (n_samples, n_individual_i).
        """
        check_is_fitted(self)
        return [
            (v - self.means_[i]) @ self.individual_weights_[i]
            for i, v in enumerate(self._check_views(views))
        ]


def _as_list(value: Any) -> Any:
    """An array-like of ranks as a list, which ``perview_parameter`` takes."""
    return list(value) if isinstance(value, np.ndarray | tuple) else value
