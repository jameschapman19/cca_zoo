r"""RANSACCCA — robust multiview CCA via random sample consensus."""

from __future__ import annotations

from itertools import combinations
from numbers import Integral, Real
from typing import Any, ClassVar, cast

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._param_constraints import POSITIVE_INT, RIDGE_PARAMETER
from cca_zoo.linear._mcca import MCCA


def _cross_view_agreement(representations: list[np.ndarray]) -> np.ndarray:
    r"""Per-sample cross-view agreement, averaged over every pair of views.

    Each view's projection is standardised to zero mean and unit variance
    (over all ``n`` samples, so this is comparable across candidate models
    with otherwise arbitrary weight scales), then for every pair of views
    $(i, j)$ and every sample $s$, the elementwise product
    $\tilde{z}_i[s] \cdot \tilde{z}_j[s]$ (summed over latent dimensions)
    measures whether that sample supports a positive relationship between
    the two views under this candidate's weights -- literally that
    sample's own contribution to the cross-covariance trace the EY reward
    term rewards in aggregate (see :mod:`cca_zoo._utils._ey`), just
    evaluated one sample at a time instead of summed over all of them.
    Averaged over every view pair when there are more than two views.

    A genuinely well-fit sample scores positive; a sample whose cross-view
    relationship the candidate's weights don't capture (independent noise,
    or a relationship pointing the wrong way) scores at or below zero --
    which is why :class:`RANSACCCA` defaults ``residual_threshold`` to 0,
    not a value estimated from the data.

    Args:
        representations: List of M arrays, each of shape (n_samples, k).

    Returns:
        Array of shape (n_samples,).
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
    r"""RANSACCCA -- robust multiview CCA via random sample consensus.

    The EY-loss family already has :class:`~cca_zoo.linear.HuberCCA` for
    robustness against high-*leverage* samples: points whose combined
    magnitude across views dominates a covariance-based statistic simply
    by being large. That leaves a different failure mode untouched: a
    subset of samples whose cross-view relationship is wrong -- mismatched,
    corrupted, or drawn from an altogether different relationship -- while
    remaining completely ordinary in magnitude within each view on its own,
    so nothing about their individual norm flags them as unusual.
    :class:`~sklearn.linear_model.RANSACRegressor` was built for exactly
    this in ordinary regression (structured minorities that leverage-based
    downweighting can't see), and the same idea carries over here.

    Fit follows classical RANSAC: repeatedly draw a random subset of
    ``min_samples`` rows, fit a candidate model on just that subset with
    :class:`~cca_zoo.linear.MCCA` (a fast closed-form generalised
    eigenvalue solve, needed since this repeats many times), and score the
    candidate by projecting *every* sample through it and summing each
    sample's positive :func:`_cross_view_agreement` -- literally, how much
    of the data this candidate's direction actually explains. The
    best-scoring candidate's inlier set (samples with agreement at or
    above ``residual_threshold``) is kept, and the final model is refit on
    that consensus set alone. The number of trials adapts to the best
    inlier fraction found so far via the standard RANSAC formula, capped
    at ``max_trials``.

    Note:
        Like :class:`~cca_zoo.linear.MCCA`, this is not convex, and here
        there is the added instability of the random subset draws
        themselves: different ``random_state`` seeds can find different
        consensus sets, especially when the "wrong" relationship is
        supported by close to half the data (see the class's tests for a
        worked example of where this method helps and where the problem
        itself becomes too ambiguous for any method to resolve reliably).

    Args:
        latent_dimensions: Number of latent dimensions. Default is 1.
        center: Whether to subtract column means. Default is True.
        c: Ridge regularisation passed to every internal
            :class:`~cca_zoo.linear.MCCA` fit (candidate, and final refit).
            A small positive value keeps small-subset candidate fits
            well-posed; see :class:`~cca_zoo.linear.MCCA`. Default is 0.1.
        min_samples: Size of each random candidate subset, as a fraction
            of the training set (float in ``(0, 1]``) or an absolute count
            (int). Default is 0.25.
        residual_threshold: Minimum :func:`_cross_view_agreement` for a
            sample to count as an inlier. Default (``None``) is 0 -- the
            statistic's own zero point under no real relationship, so no
            data-dependent calibration is needed.
        max_trials: Maximum number of random subsets to try. Default is 200.
        stop_probability: Trials stop early, before ``max_trials``, once
            the standard RANSAC formula estimates this probability of
            already having found a subset at least as clean as the best
            one seen so far. Default is 0.99.
        random_state: Seed for the random subset draws.

    Examples:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 8))
        >>> X2 = rng.standard_normal((200, 6))
        >>> model = RANSACCCA(latent_dimensions=1, random_state=0).fit([X1, X2])
        >>> scores = model.transform([X1, X2])
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "c": RIDGE_PARAMETER,
        "min_samples": [
            Interval(Real, 0, 1, closed="right"),
            Interval(Integral, 1, None, closed="left"),
        ],
        "residual_threshold": [Interval(Real, None, None, closed="neither"), None],
        "max_trials": POSITIVE_INT,
        "stop_probability": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        c: float | list[float] = 0.1,
        min_samples: int | float = 0.25,
        residual_threshold: float | None = None,
        max_trials: int = 200,
        stop_probability: float = 0.99,
        random_state: int | None = None,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.c = c
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.stop_probability = stop_probability
        self.random_state = random_state

    def _resolve_min_samples(self, n: int) -> int:
        """Resolve ``min_samples`` (fraction or count) to an absolute count.

        Args:
            n: Number of training samples.

        Returns:
            Absolute subset size, at least 1 and at most ``n``.
        """
        if isinstance(self.min_samples, Integral):
            resolved = int(self.min_samples)
        else:
            resolved = max(1, int(round(self.min_samples * n)))
        return min(resolved, n)

    def fit(self, views: list[ArrayLike], y: None = None) -> RANSACCCA:
        """Fit RANSACCCA by random sample consensus over MCCA candidate fits.

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
        k = self.latent_dimensions
        min_samples = self._resolve_min_samples(n)
        threshold = 0.0 if self.residual_threshold is None else self.residual_threshold
        rng = np.random.default_rng(self.random_state)

        best_mask: np.ndarray | None = None
        best_score = -np.inf
        dynamic_max_trials = self.max_trials
        trial = 0
        while trial < dynamic_max_trials:
            idx = rng.choice(n, min_samples, replace=False)
            candidate = MCCA(latent_dimensions=k, c=self.c).fit(
                [v[idx] for v in views_]
            )
            agreement = _cross_view_agreement(
                candidate.transform(cast("list[ArrayLike]", views_))
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
        final = MCCA(latent_dimensions=k, c=self.c).fit([v[best_mask] for v in views_])
        self.weights_: list[np.ndarray] = final.weights_
        self.inlier_mask_: np.ndarray = best_mask
        self.n_trials_: int = trial
        return self
