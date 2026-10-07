"""Ridge CCA with the shrinkage chosen by cross-validation."""

from __future__ import annotations

import warnings
from numbers import Integral
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import check_cv
from sklearn.utils._param_validation import Interval, StrOptions

from cca_zoo._utils._linalg import covariance_eigenbasis, truncated_svd
from cca_zoo._utils._validation import validate_views
from cca_zoo.linear._ridge_cca import RidgeCCA

#: Shrinkage grid used when ``shrinkages`` is None: 0, then 1e-4 to 1 by thirds
#: of a decade.
DEFAULT_SHRINKAGES = np.concatenate([[0.0], np.logspace(-4, 0, 13)])

#: With ``n_components="auto"``, the most components considered.
AUTO_MAX_COMPONENTS = 10

#: With ``n_components="auto"``, a component is kept when its mean held-out
#: correlation exceeds this many multiples of ``1 / sqrt(n_samples)``. A null
#: component's mean over the folds has standard deviation about 1.3 times
#: ``1 / sqrt(n_samples)`` once the shrinkage is also chosen on the same
#: folds, so 3 keeps the false-positive rate at 5% or below.
AUTO_THRESHOLD = 3.0


def _held_out_correlations(
    train: list[np.ndarray],
    test: list[np.ndarray],
    shrinkages: np.ndarray,
    n_components: int,
    center: bool,
) -> np.ndarray:
    """Held-out canonical correlation of each component at each shrinkage.

    Shape (len(shrinkages), n_components); a component the fold's data are too
    small to have is NaN.

    Each view's eigenbasis, the cross-covariance in those bases and the test
    scores in them are computed once; a shrinkage then only rescales them
    and takes the SVD of the (rank_1, rank_2) cross-covariance, instead of a
    refit of the whole model.
    """
    if center:
        means = [v.mean(axis=0) for v in train]
        train = [v - m for v, m in zip(train, means)]
        test = [v - m for v, m in zip(test, means)]
    bases = [covariance_eigenbasis(v) for v in train]
    z_train = [v @ V for v, (_, V) in zip(train, bases)]
    z_test = [v @ V for v, (_, V) in zip(test, bases)]
    cross = z_train[0].T @ z_train[1] / (train[0].shape[0] - 1)
    k = min(n_components, *cross.shape)
    scores = np.full((len(shrinkages), n_components), np.nan)
    for i, c in enumerate(shrinkages):
        d = [((1.0 - c) * lam + c) ** -0.5 for lam, _ in bases]
        U, _, Vt = truncated_svd(d[0][:, None] * cross * d[1], k)
        a = (z_test[0] * d[0]) @ U
        b = (z_test[1] * d[1]) @ Vt.T
        a, b = a - a.mean(axis=0), b - b.mean(axis=0)
        scores[i, :k] = (a * b).sum(axis=0) / np.sqrt(
            (a**2).sum(axis=0) * (b**2).sum(axis=0)
        )
    return scores


class RidgeCCACV(RidgeCCA):
    r"""Ridge CCA whose shrinkage is chosen by cross-validated canonical correlation.

    Equivalent to ``GridSearchCV(RidgeCCA(...), {"shrinkage": shrinkages})``
    with the default scorer, but much faster: each view's eigendecomposition
    is computed once per fold rather than once per fold per shrinkage, and
    each shrinkage then costs one small SVD. The same shrinkage is used for
    both views; tune per-view shrinkage with ``GridSearchCV``.

    With ``n_components="auto"``, the number of components is also chosen
    from the held-out correlations: the shrinkage maximises the summed
    positive held-out correlation of the first ten components, and the model
    keeps the leading components whose mean held-out correlation exceeds
    ``3 / sqrt(n_samples)``. On simulated data with known canonical
    correlations this recovers the true number whenever the samples can
    resolve it, and under-counts weak components when they cannot, rather
    than over-counting. If not even the first component clears the
    threshold, one component is kept and a warning is raised.

    Args:
        n_components: Number of latent dimensions, or ``"auto"`` to choose it
            from the held-out correlations. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkages: Candidate shrinkages in ``[0, 1]``. None uses 0 and 13
            values log-spaced from 1e-4 to 1.
        cv: Cross-validation splitter or number of folds, as in
            :func:`sklearn.model_selection.check_cv`. Default is 5.

    Attributes:
        shrinkage_: The shrinkage with the best mean held-out correlation.
        cv_scores_: Per candidate shrinkage, the mean held-out canonical
            correlation over the folds and components; with
            ``n_components="auto"``, the sum over components of its positive
            part.
        cv_component_scores_: Mean held-out correlation over the folds of each
            component, shape (n_shrinkages, n_components), with
            ``AUTO_MAX_COMPONENTS`` components for ``"auto"``.
        weights_: Weight matrix of each view, shape (n_features_i, n_components).

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import RidgeCCACV
        >>> rng = np.random.default_rng(0)
        >>> z = rng.standard_normal((100, 1))
        >>> X1 = z @ rng.standard_normal((1, 20)) + rng.standard_normal((100, 20))
        >>> X2 = z @ rng.standard_normal((1, 15)) + rng.standard_normal((100, 15))
        >>> model = RidgeCCACV(n_components=1).fit([X1, X2])
        >>> 0 <= model.shrinkage_ <= 1
        True
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **{
            k: v for k, v in RidgeCCA._parameter_constraints.items() if k != "shrinkage"
        },
        "n_components": [
            Interval(Integral, 1, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "shrinkages": [None, "array-like"],
        "cv": ["cv_object", Integral],
    }

    def __init__(
        self,
        n_components: int | str = 1,
        *,
        center: bool = True,
        shrinkages: ArrayLike | None = None,
        cv: Any = 5,
    ) -> None:
        super().__init__(n_components=n_components, center=center)  # type: ignore[arg-type]
        self.shrinkages = shrinkages
        self.cv = cv

    def _shrinkage_per_view(self) -> list[float]:
        return [self.shrinkage_] * 2

    def _fit_components(self) -> int:
        return self._selected_components

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        sample_weight: ArrayLike | None = None,
    ) -> RidgeCCACV:
        """Choose the shrinkage by cross-validation and fit at it.

        Args:
            views: Two arrays of shape (n_samples, n_features_i).
            y: Ignored.
            sample_weight: Must be None: weighted cross-validation is not
                supported.

        Returns:
            self.

        Raises:
            ValueError: If there are not exactly two views, or a
                ``sample_weight`` is given.
        """
        if sample_weight is not None:
            raise ValueError(f"{type(self).__name__} does not support sample_weight.")
        self._validate_params()
        validated = validate_views(views, ensure_min_samples=2)
        if len(validated) != 2:
            raise ValueError(
                f"{type(self).__name__} requires exactly 2 views, got {len(validated)}."
            )
        shrinkages = np.asarray(
            DEFAULT_SHRINKAGES if self.shrinkages is None else self.shrinkages,
            dtype=float,
        )
        auto = isinstance(self.n_components, str)
        scored = AUTO_MAX_COMPONENTS if auto else self.n_components
        folds = list(check_cv(self.cv).split(validated[0]))
        per_fold = [
            _held_out_correlations(
                [v[train] for v in validated],
                [v[test] for v in validated],
                shrinkages,
                scored,
                self.center,
            )
            for train, test in folds
        ]
        self.cv_component_scores_: np.ndarray = np.nanmean(per_fold, axis=0)
        scores = np.nan_to_num(self.cv_component_scores_)
        self.cv_scores_: np.ndarray = (
            np.maximum(scores, 0).sum(axis=1) if auto else scores.mean(axis=1)
        )
        best = int(np.argmax(self.cv_scores_))
        self.shrinkage_: float = float(shrinkages[best])
        self._selected_components: int = (
            self._count_components(self.cv_component_scores_[best], len(validated[0]))
            if auto
            else self.n_components
        )
        return super().fit(views)

    @staticmethod
    def _count_components(held_out: np.ndarray, n_samples: int) -> int:
        """The leading components whose held-out correlation clears the threshold."""
        clears = held_out > AUTO_THRESHOLD / np.sqrt(n_samples)
        leading = len(clears) if clears.all() else int(np.argmin(clears))
        if leading == 0:
            warnings.warn(
                "No component's held-out correlation exceeds chance; keeping one.",
                UserWarning,
                stacklevel=3,
            )
        return max(leading, 1)
