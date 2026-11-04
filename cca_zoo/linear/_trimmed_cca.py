"""Robust multiview CCA by concentration steps."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval, Options

from cca_zoo._utils._convergence import warn_if_not_converged
from cca_zoo._utils._ey import weight_gram_mean
from cca_zoo._utils._param_constraints import (
    POSITIVE_INT,
)
from cca_zoo.linear.gradient import CCAEY


def _per_sample_terms(
    zs: list[np.ndarray], b: float, c: float, h: int
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Per-sample terms of the one-component CCAEY loss on a size-``h`` subset.

    For fixed weights the loss on a subset $S$ is
    $\sum_{s \in S} \sigma(s) + K (\sum_{s \in S} e(s))^2$ plus a constant,
    for any number of views. With more than one component the penalty is a
    matrix quadratic form, so :class:`TrimmedCCA` is limited to one.

    Args:
        zs: Per-view projections, each of shape (n,).
        b: The weight-Gram scalar.
        c: Ridge blend in ``[0, 1]``.
        h: Subset size.

    Returns:
        ``(sigma, e, K)``: two arrays of shape (n,) and a scalar.
    """
    m = len(zs)
    total: np.ndarray = np.sum(zs, axis=0)
    e: np.ndarray = np.sum([z**2 for z in zs], axis=0) / m
    r = total**2 / m
    rho = -2 * r + 2 * c * e
    sigma = rho / (h - 1) + 2 * c * (1 - c) * b * e / (h - 1)
    k_coef = (1 - c) ** 2 / (h - 1) ** 2
    return sigma, e, k_coef


def _select(
    zs: list[np.ndarray], b: float, c: float, h: int, n_bisect: int = 60
) -> np.ndarray:
    r"""The ``h`` samples minimising the CCAEY loss for the current weights.

    Ranking by $\sigma(s) + \mu e(s)$ is exact once
    $\mu = 2K \sum_{s \in S(\mu)} e(s)$; the sum is monotone in $\mu$, so
    the multiplier is found by bisection.

    Args:
        zs: Per-view projections, each of shape (n,).
        b: The weight-Gram scalar.
        c: Ridge blend in ``[0, 1]``.
        h: Number of samples to keep.
        n_bisect: Bisection iterations.

    Returns:
        Sorted indices of the kept samples.
    """
    sigma, e, k_coef = _per_sample_terms(zs, b, c, h)

    def h_of(mu: float) -> tuple[float, np.ndarray]:
        subset = np.argsort(sigma + mu * e)[:h]
        return 2 * k_coef * e[subset].sum() - mu, subset

    mu_lo = 0.0
    mu_hi = max(1.0, 2 * k_coef * h * e.max())
    h_hi, _ = h_of(mu_hi)
    tries = 0
    while h_hi > 0 and tries < 60:
        mu_hi *= 2
        h_hi, _ = h_of(mu_hi)
        tries += 1

    kept = None
    for _ in range(n_bisect):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        h_mid, subset_mid = h_of(mu_mid)
        if h_mid >= 0:
            mu_lo, kept = mu_mid, subset_mid
        else:
            mu_hi = mu_mid
    if kept is None:
        _, kept = h_of(mu_lo)
    return np.sort(kept)


# L-BFGS-B iterations allowed each refit: scipy's default, the concentration
# steps being counted by max_iter.
_REFIT_MAX_ITER = 15_000


class TrimmedCCA(CCAEY):
    """Robust multiview CCA by concentration steps.

    :class:`~cca_zoo.linear.gradient.CCAEY` fitted by concentration steps: as
    in least trimmed squares, alternates between keeping the ``h`` rows with
    the lowest CCAEY loss and refitting CCAEY on them. Neither step increases
    the loss. The best of ``n_init`` random starts is kept. Suited to heavy
    contamination when the clean fraction is roughly known. Supports one
    latent dimension. The concentration step is Rousseeuw and Van Driessen's
    (1999), from their fast minimum covariance determinant algorithm.

    Args:
        n_components: Number of latent dimensions; must be 1. Default is 1.
        center: Whether to centre each view. Default is True.
        shrinkage: Shrinkage of the covariances towards the identity, one value
            for every view as in ``CCAEY``, in ``[0, 1]``: 0 is CCA and 1 is PLS.
            Default is 0.1, since each fit sees only ``support_fraction`` of
            the rows.
        support_fraction: Fraction of rows kept, in ``(0, 1]``; a prior on the clean
            fraction. Default is 0.75.
        n_init: Random restarts. Default is 10.
        max_iter: Maximum concentration steps per restart. Default is 30.
        tol: L-BFGS-B ``ftol`` for each refit. Default is 1e-8.
        random_state: Seed for the restarts. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, 1).
        inlier_mask_: Boolean mask of the kept training rows.
        n_iter_: Concentration steps of the best restart.

    References:
        Chapman, J., Wang, H.-T., Wells, L., & Wiesner, J. (2021). CCA-Zoo: A
        collection of Regularized, Deep Learning based, Kernel, and
        Probabilistic CCA methods in a scikit-learn style framework. Journal
        of Open Source Software, 6(68), 3823.
        Chapman, J., Wells, L., & Lawry Aguila, A. (2024). Unconstrained
        Stochastic CCA: Unifying Multiview and Self-Supervised Learning.
        arXiv:2310.01012.
        Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for
        the minimum covariance determinant estimator. Technometrics,
        41(3), 212-223.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import TrimmedCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 8))
        >>> X2 = rng.standard_normal((200, 6))
        >>> model = TrimmedCCA(support_fraction=0.7, random_state=0).fit([X1, X2])
        >>> int(model.inlier_mask_.sum())
        140
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **CCAEY._parameter_constraints,
        "n_components": [Options(Integral, {1})],
        "support_fraction": [Interval(Real, 0, 1, closed="right")],
        "n_init": POSITIVE_INT,
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        center: bool = True,
        shrinkage: float = 0.1,
        support_fraction: float = 0.75,
        n_init: int = 10,
        max_iter: int = 30,
        tol: float = 1e-8,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            n_components=n_components,
            center=center,
            shrinkage=shrinkage,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.support_fraction = support_fraction
        self.n_init = n_init

    def fit(self, views: list[ArrayLike], y: None = None) -> TrimmedCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        views_ = self._setup_fit(views)
        n = self.n_samples_
        h = max(2, round(self.support_fraction * n))
        rng = np.random.default_rng(self.random_state)
        best_weights: list[np.ndarray] | None = None
        best_mask: np.ndarray | None = None
        best_obj = np.inf
        best_converged = False
        for _ in range(self.n_init):
            weights = []
            for xv in views_:
                w = rng.standard_normal((xv.shape[1], 1))
                w /= np.linalg.norm(w)
                weights.append(w)

            kept = np.sort(rng.choice(n, h, replace=False))
            xs_kept = [xv[kept] for xv in views_]
            weights, _ = self._minimise(xs_kept, weights, _REFIT_MAX_ITER)
            cur_obj = self._loss(xs_kept, weights)

            converged = False
            for n_iter in range(1, self.max_iter + 1):
                zs = [(xv @ w).ravel() for xv, w in zip(views_, weights)]
                b = weight_gram_mean(weights)[0, 0]
                new_kept = _select(zs, b, self.shrinkage, h)
                if np.array_equal(new_kept, kept):
                    converged = True
                    break
                xs_new = [xv[new_kept] for xv in views_]
                new_weights, _ = self._minimise(xs_new, weights, _REFIT_MAX_ITER)
                new_obj = self._loss(xs_new, new_weights)
                if new_obj > cur_obj + 1e-10:
                    converged = True
                    break
                kept, weights, cur_obj = new_kept, new_weights, new_obj

            if cur_obj < best_obj:
                best_obj, best_weights, best_mask = cur_obj, weights, kept
                self.n_iter_: int = n_iter
                best_converged = converged

        assert best_weights is not None
        assert best_mask is not None
        warn_if_not_converged(self, best_converged)
        self.weights_: list[np.ndarray] = best_weights
        self.inlier_mask_: np.ndarray = np.zeros(n, dtype=bool)
        self.inlier_mask_[best_mask] = True
        self._fit_maps_and_importances(views_)
        return self

    def _loss(self, views: list[np.ndarray], weights: list[np.ndarray]) -> float:
        """The CCAEY loss of ``weights`` on ``views``."""
        return self._objective(views, [v @ w for v, w in zip(views, weights)], weights)
