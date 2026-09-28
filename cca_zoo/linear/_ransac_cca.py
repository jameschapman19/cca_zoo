"""Robust multiview CCA by random sample consensus."""

from __future__ import annotations

from itertools import combinations
from numbers import Integral, Real
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._param_constraints import (
    POSITIVE_INT,
    RANDOM_STATE,
    RIDGE_PARAMETER,
)
from cca_zoo.linear._mcca import MCCA


def _cross_view_agreement(representations: list[np.ndarray]) -> np.ndarray:
    r"""Each sample's contribution to the cross-covariance, averaged over view pairs.

    Projections are standardised, then the products $\tilde z_i[s] \tilde z_j[s]$
    are summed over components. Samples the model explains score positive.

    Args:
        representations: One array of shape (n_samples, k) per view.

    Returns:
        Agreement of each sample, shape (n_samples,).
    """
    standardised = [
        (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-12) for z in representations
    ]
    pairs = list(combinations(range(len(standardised)), 2))
    products = [standardised[i] * standardised[j] for i, j in pairs]
    total: np.ndarray = np.sum(products, axis=0)
    result: np.ndarray = total.sum(axis=1) / len(pairs)
    return result


class RANSACCCA(BaseModel):
    """Robust multiview CCA by random sample consensus.

    As :class:`~sklearn.linear_model.RANSACRegressor`: fits
    :class:`~cca_zoo.linear.MCCA` on random subsets of ``min_samples`` rows,
    scores each candidate by the total positive cross-view agreement of
    every sample, and refits on the inliers of the best one. Targets samples
    whose cross-view relationship is wrong but whose magnitudes are ordinary,
    which leverage-based :class:`~cca_zoo.linear.HuberCCA` cannot see. The
    number of trials adapts to the inlier fraction found, up to
    ``max_trials``.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of every internal MCCA fit. Default is 0.1.
        min_samples: Subset size, as a fraction in ``(0, 1]`` or a count.
            Default is 0.25.
        residual_threshold: Minimum agreement of an inlier; None is 0.
            Default is None.
        max_trials: Maximum subsets tried. Default is 200.
        stop_probability: Stop once a subset at least as clean as the best
            is found with this probability. Default is 0.99.
        random_state: Seed for the subset draws. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, n_components).
        inlier_mask_: Boolean mask of the consensus set.
        n_trials_: Number of subsets tried.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import RANSACCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 8))
        >>> X2 = rng.standard_normal((200, 6))
        >>> model = RANSACCCA(random_state=0).fit([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "shrinkage": RIDGE_PARAMETER,
        "min_samples": [
            Interval(Real, 0, 1, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
        "residual_threshold": [Interval(Real, None, None, closed="neither"), None],
        "max_trials": POSITIVE_INT,
        "stop_probability": [Interval(Real, 0, 1, closed="both")],
        "random_state": RANDOM_STATE,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float | list[float] = 0.1,
        min_samples: int | float = 0.25,
        residual_threshold: float | None = None,
        max_trials: int = 200,
        stop_probability: float = 0.99,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.shrinkage = shrinkage
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.stop_probability = stop_probability
        self.random_state = random_state

    def _resolve_min_samples(self, n: int) -> int:
        """``min_samples`` as a count in ``[1, n]``."""
        if isinstance(self.min_samples, Integral):
            resolved = int(self.min_samples)
        else:
            resolved = max(1, int(round(self.min_samples * n)))
        return min(resolved, n)

    def fit(self, views: list[ArrayLike], y: None = None) -> RANSACCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        n = self.n_samples_
        k = self.n_components
        min_samples = self._resolve_min_samples(n)
        threshold = 0.0 if self.residual_threshold is None else self.residual_threshold
        rng = np.random.default_rng(self.random_state)

        best_mask: np.ndarray | None = None
        best_score = -np.inf
        dynamic_max_trials = self.max_trials
        trial = 0
        while trial < dynamic_max_trials:
            idx = rng.choice(n, min_samples, replace=False)
            candidate = MCCA(n_components=k, shrinkage=self.shrinkage).fit(
                [v[idx] for v in views_]
            )
            agreement = _cross_view_agreement(
                candidate._transform_arrays(cast("list[ArrayLike]", views_))
            )
            score = float(np.clip(agreement, 0.0, None).sum())
            if score > best_score:
                best_score = score
                best_mask = agreement >= threshold
                w = max(float(best_mask.sum()) / n, 1e-10)
                denom = np.clip(1.0 - w**min_samples, 1e-12, 1 - 1e-12)
                dynamic_max_trials = min(
                    self.max_trials,
                    int(np.ceil(np.log(1 - self.stop_probability) / np.log(denom))),
                )
            trial += 1

        assert best_mask is not None
        final = MCCA(n_components=k, shrinkage=self.shrinkage).fit(
            [v[best_mask] for v in views_]
        )
        self.weights_: list[np.ndarray] = final.weights_
        self.inlier_mask_: np.ndarray = best_mask
        self.n_trials_: int = trial
        return self._finish_fit(views_)
