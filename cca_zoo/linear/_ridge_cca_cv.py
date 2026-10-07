"""Ridge CCA with the shrinkage chosen by cross-validation."""

from __future__ import annotations

from numbers import Integral
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.model_selection import check_cv

from cca_zoo._utils._linalg import covariance_eigenbasis, truncated_svd
from cca_zoo._utils._validation import validate_views
from cca_zoo.linear._ridge_cca import RidgeCCA

#: Shrinkage grid used when ``shrinkages`` is None: 0, then 1e-4 to 1 by thirds
#: of a decade.
DEFAULT_SHRINKAGES = np.concatenate([[0.0], np.logspace(-4, 0, 13)])


def _held_out_correlations(
    train: list[np.ndarray],
    test: list[np.ndarray],
    shrinkages: np.ndarray,
    n_components: int,
    center: bool,
) -> np.ndarray:
    """Mean held-out canonical correlation at each shrinkage, shape (len(shrinkages),).

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
    scores = np.empty(len(shrinkages))
    for i, c in enumerate(shrinkages):
        d = [((1.0 - c) * lam + c) ** -0.5 for lam, _ in bases]
        U, _, Vt = truncated_svd(d[0][:, None] * cross * d[1], k)
        a = (z_test[0] * d[0]) @ U
        b = (z_test[1] * d[1]) @ Vt.T
        a, b = a - a.mean(axis=0), b - b.mean(axis=0)
        scores[i] = np.mean(
            (a * b).sum(axis=0) / np.sqrt((a**2).sum(axis=0) * (b**2).sum(axis=0))
        )
    return scores


class RidgeCCACV(RidgeCCA):
    r"""Ridge CCA whose shrinkage is chosen by cross-validated canonical correlation.

    Equivalent to ``GridSearchCV(RidgeCCA(...), {"shrinkage": shrinkages})``
    with the default scorer, but much faster: each view's eigendecomposition
    is computed once per fold rather than once per fold per shrinkage, and
    each shrinkage then costs one small SVD. The same shrinkage is used for
    both views; tune per-view shrinkage with ``GridSearchCV``.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkages: Candidate shrinkages in ``[0, 1]``. None uses 0 and 13
            values log-spaced from 1e-4 to 1.
        cv: Cross-validation splitter or number of folds, as in
            :func:`sklearn.model_selection.check_cv`. Default is 5.

    Attributes:
        shrinkage_: The shrinkage with the best mean held-out correlation.
        cv_scores_: Mean held-out canonical correlation over the folds, one
            per candidate shrinkage.
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
        "shrinkages": [None, "array-like"],
        "cv": ["cv_object", Integral],
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkages: ArrayLike | None = None,
        cv: Any = 5,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.shrinkages = shrinkages
        self.cv = cv

    def _shrinkage_per_view(self) -> list[float]:
        return [self.shrinkage_] * 2

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
        folds = list(check_cv(self.cv).split(validated[0]))
        per_fold = [
            _held_out_correlations(
                [v[train] for v in validated],
                [v[test] for v in validated],
                shrinkages,
                self.n_components,
                self.center,
            )
            for train, test in folds
        ]
        self.cv_scores_: np.ndarray = np.mean(per_fold, axis=0)
        self.shrinkage_: float = float(shrinkages[np.argmax(self.cv_scores_)])
        return super().fit(views)
