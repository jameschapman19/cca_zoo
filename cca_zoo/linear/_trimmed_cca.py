r"""TrimmedCCA — robust multiview CCA via concentration steps."""

from __future__ import annotations

from numbers import Real
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from sklearn.utils._param_validation import Interval

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import weight_gram_mean
from cca_zoo._utils._param_constraints import POSITIVE_INT, RIDGE_PARAMETER
from cca_zoo.linear.gradient import CCAEY


def _per_sample_terms(
    z1: np.ndarray, z2: np.ndarray, b: float, c: float, h: int
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Per-sample decomposition of CCAEY(c)'s loss, restricted to a size-h subset.

    For fixed weights (so fixed projections ``z1``, ``z2`` and weight-Gram
    scalar ``b``), the loss restricted to a kept subset $S$ of size $h$
    decomposes as

    $$
    \mathcal{L}(S) = \sum_{s \in S} \sigma(s)
        + K \Big(\sum_{s \in S} e(s)\Big)^2 + \text{const}
    $$

    -- every term additive over $S$ except the squared sum, the same
    algebraic shape as a knapsack relaxation (see :func:`_select`).
    Derived by expanding CCAEY's mean pairwise cross-covariance and mean
    auto-covariance as per-sample sums over the $h - 1$ denominator.

    Args:
        z1: First view's projection, shape (n,).
        z2: Second view's projection, shape (n,).
        b: Weight-Gram scalar (mean of $w_1^\top w_1$, $w_2^\top w_2$).
        c: Ridge blend in ``[0, 1]``, matching ``CCAEY``'s own ``c``.
        h: Subset size the loss will be restricted to.

    Returns:
        Tuple ``(sigma, e, k_coef)``: ``sigma`` and ``e`` are arrays of
        shape (n,), ``k_coef`` is the scalar $K$.
    """
    e = 0.5 * (z1**2 + z2**2)
    r = 0.5 * (z1 + z2) ** 2
    rho = -2 * r + 2 * c * e
    sigma = rho / (h - 1) + 2 * c * (1 - c) * b * e / (h - 1)
    k_coef = (1 - c) ** 2 / (h - 1) ** 2
    return sigma, e, k_coef


def _select(
    z1: np.ndarray, z2: np.ndarray, b: float, c: float, h: int, n_bisect: int = 60
) -> np.ndarray:
    r"""The h samples minimising CCAEY(c)'s loss, for the current weights.

    Solved via a Lagrangian relaxation of the "additive term + K * (additive
    sum)^2" structure :func:`_per_sample_terms` exposes: for a multiplier
    $\mu$, ranking samples by $\sigma(s) + \mu e(s)$ and keeping the best
    $h$ gives the exact minimiser once $\mu$ is self-consistent
    ($\mu = 2K \sum_{s \in S(\mu)} e(s)$). Since $\sum_{s \in S(\mu)} e(s)$
    is non-increasing in $\mu$, bisection on this self-consistency
    condition finds it reliably -- unlike naive fixed-point iteration,
    which can cycle. Verified against brute-force combinatorial search:
    exact in the large majority of trials, with a small bounded gap from
    a ranking tie in the rest, closed operationally by the caller's own
    safeguard (never accept a selection that doesn't actually improve the
    objective).

    Args:
        z1: First view's projection, shape (n,).
        z2: Second view's projection, shape (n,).
        b: Weight-Gram scalar.
        c: Ridge blend in ``[0, 1]``.
        h: Number of samples to keep.
        n_bisect: Bisection iterations for the multiplier search.

    Returns:
        Sorted integer array of the ``h`` kept sample indices.
    """
    sigma, e, k_coef = _per_sample_terms(z1, z2, b, c, h)

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


def _refit(
    model: CCAEY,
    x1: np.ndarray,
    x2: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-minimise CCAEY(c)'s exact loss on (x1, x2), warm-started at (w1, w2).

    Monotone by construction: L-BFGS-B's line search never accepts a step
    that increases the objective, and it starts exactly at the incoming
    weights' own value.
    """
    shapes = [w1.shape, w2.shape]
    sizes = [w1.size, w2.size]

    def unflatten(x: np.ndarray) -> list[np.ndarray]:
        out = []
        offset = 0
        for shape, size in zip(shapes, sizes):
            out.append(x[offset : offset + size].reshape(shape))
            offset += size
        return out

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        weights = unflatten(x)
        representations = [x1 @ weights[0], x2 @ weights[1]]
        obj = model._objective([x1, x2], representations, weights)
        grads = model._derivative([x1, x2], representations, weights)
        grad = np.concatenate([g.ravel() for g in grads])
        return obj, grad

    x0 = np.concatenate([w1.ravel(), w2.ravel()])
    result = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"ftol": tol})
    w1_new, w2_new = unflatten(result.x)
    return w1_new, w2_new


def _objective_value(
    model: CCAEY, x1: np.ndarray, x2: np.ndarray, w1: np.ndarray, w2: np.ndarray
) -> float:
    """CCAEY(c)'s exact loss at (w1, w2) on (x1, x2)."""
    representations = [x1 @ w1, x2 @ w2]
    return model._objective([x1, x2], representations, [w1, w2])


class TrimmedCCA(BaseModel):
    r"""TrimmedCCA -- robust multiview CCA via concentration steps.

    :class:`~cca_zoo.linear.RANSACCCA` searches for a clean subset by
    drawing many small random candidates and keeping the best-scoring
    one -- a good strategy while contamination stays well below its
    search's own odds of ever drawing a clean-enough sample. As
    contamination approaches the ~50% breakdown point, that search
    degrades: a random small subset becomes close to a coin flip on
    being usably clean, however many trials are tried. ``TrimmedCCA``
    instead uses concentration steps in the style of Rousseeuw's Least
    Trimmed Squares / Minimum Covariance Determinant: starting from a
    large random subset of ``h`` rows (``h_frac`` of the data), it
    alternates

    1. **select**: rank every sample by its own contribution to
       :class:`~cca_zoo.linear.gradient.CCAEY`'s exact loss (for the
       *current* weights) and keep the best ``h`` -- solved via a
       Lagrangian relaxation of the loss's own algebraic structure (see
       :func:`_select`), not an absolute per-sample threshold like
       ``RANSACCCA``'s;
    2. **refit**: re-minimise ``CCAEY``'s exact loss restricted to the
       kept ``h`` rows, warm-started at the current weights via
       L-BFGS-B.

    Each step only ever accepts a subset/weight pair that doesn't
    increase the loss (refit is monotone by construction; a selection
    step that fails to improve is rejected and the loop stops there) --
    the classical C-step argument, applied to CCAEY's real objective
    rather than a proxy score. Repeated over ``n_starts`` random restarts
    (this objective is non-convex, so a single start can land on a poor
    local optimum), keeping the lowest-loss result.

    Note:
        ``h_frac`` is not learned from the data -- like
        :class:`sklearn.covariance.MinCovDet`'s ``support_fraction``, it
        is a prior on how much of the training data you expect is
        contaminated, set before fitting. Too high wastes some of a
        fixed-size budget on good rows discarded unnecessarily when
        contamination is actually low; too low forces contaminated rows
        into every fit once true contamination exceeds ``1 - h_frac``.
        It cannot be chosen by cross-validating a downstream metric,
        since that would need labels for which rows are contaminated --
        exactly what's unknown.

        ``TrimmedCCA`` currently supports exactly 2 views and
        ``latent_dimensions=1``: the selection rule's closed-form
        derivation is specific to that case. ``RANSACCCA`` (via
        :class:`~cca_zoo.linear.MCCA`) supports any number of views and
        latent dimensions directly, and matches or beats ``TrimmedCCA``
        away from the ~50% breakdown regime -- reach for ``TrimmedCCA``
        specifically when contamination is expected to be heavy and
        ``h_frac`` can be set close to the true clean fraction.

    Args:
        latent_dimensions: Must be 1 (the only value currently
            supported).
        center: Whether to subtract column means. Default True.
        c: Ridge blend in ``[0, 1]``, same semantics as
            :class:`~cca_zoo.linear.gradient.CCAEY`'s own ``c``. Default
            0.1 (the unregularised ``c=0`` can be poorly conditioned once
            a concentration step's ``h``-sized subset doesn't outnumber
            the combined feature count by a healthy margin; see
            ``CCAEY``'s own note on this).
        h_frac: Fraction of rows kept every concentration step, in
            ``(0, 1]``. Default 0.75.
        n_starts: Random restarts; the lowest-loss result is kept.
            Default 10.
        max_iter: Maximum concentration steps per restart. Default 30.
        tol: Convergence tolerance for each refit's L-BFGS-B call, passed
            as ``ftol`` (see ``CCAEY``'s own docstring for why this
            matters). Default 1e-8.
        random_state: Seed for the random restarts.

    References:
        Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for
        the minimum covariance determinant estimator. Technometrics,
        41(3), 212-223.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 8))
        >>> X2 = rng.standard_normal((200, 6))
        >>> model = TrimmedCCA(h_frac=0.7, random_state=0).fit([X1, X2])
        >>> inliers = model.inlier_mask_  # boolean array over the training rows
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "c": RIDGE_PARAMETER,
        "h_frac": [Interval(Real, 0, 1, closed="right")],
        "n_starts": POSITIVE_INT,
        "max_iter": POSITIVE_INT,
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        latent_dimensions: int = 1,
        center: bool = True,
        c: float = 0.1,
        h_frac: float = 0.75,
        n_starts: int = 10,
        max_iter: int = 30,
        tol: float = 1e-8,
        random_state: int | None = None,
    ) -> None:
        super().__init__(latent_dimensions=latent_dimensions, center=center)
        self.c = c
        self.h_frac = h_frac
        self.n_starts = n_starts
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> TrimmedCCA:
        """Fit TrimmedCCA by concentration steps on CCAEY's exact loss.

        Args:
            views: List of exactly 2 arrays, each (n_samples, n_features_i).
            y: Ignored.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If the number of views isn't exactly 2, or
                ``latent_dimensions`` isn't 1 (see the class's ``Note``).
        """
        views_ = self._setup_fit(views)
        if self.n_views_ != 2:
            raise ValueError(
                f"TrimmedCCA currently supports exactly 2 views, got {self.n_views_}."
            )
        if self.latent_dimensions != 1:
            raise ValueError(
                "TrimmedCCA currently supports only latent_dimensions=1, "
                f"got {self.latent_dimensions}."
            )
        x1, x2 = views_
        n = self.n_samples_
        h = max(2, int(round(self.h_frac * n)))
        rng = np.random.default_rng(self.random_state)
        # CCAEY's own _objective/_derivative back both the selection score
        # and the refit, rather than a second implementation of the loss
        # living here -- .fit() is never called on it, only these two.
        model = CCAEY(latent_dimensions=1, c=self.c)

        best_w1: np.ndarray | None = None
        best_w2: np.ndarray | None = None
        best_mask: np.ndarray | None = None
        best_obj = np.inf
        for _ in range(self.n_starts):
            w1 = rng.standard_normal((x1.shape[1], 1))
            w1 /= np.linalg.norm(w1)
            w2 = rng.standard_normal((x2.shape[1], 1))
            w2 /= np.linalg.norm(w2)

            kept = np.sort(rng.choice(n, h, replace=False))
            w1, w2 = _refit(model, x1[kept], x2[kept], w1, w2, self.tol)
            cur_obj = _objective_value(model, x1[kept], x2[kept], w1, w2)

            for _ in range(self.max_iter):
                z1 = (x1 @ w1).ravel()
                z2 = (x2 @ w2).ravel()
                b = weight_gram_mean([w1, w2])[0, 0]
                new_kept = _select(z1, z2, b, self.c, h)
                if np.array_equal(new_kept, kept):
                    break
                w1_new, w2_new = _refit(
                    model, x1[new_kept], x2[new_kept], w1, w2, self.tol
                )
                new_obj = _objective_value(
                    model, x1[new_kept], x2[new_kept], w1_new, w2_new
                )
                if new_obj > cur_obj + 1e-10:
                    break
                kept, w1, w2, cur_obj = new_kept, w1_new, w2_new, new_obj

            if cur_obj < best_obj:
                best_obj, best_w1, best_w2, best_mask = cur_obj, w1, w2, kept

        assert best_w1 is not None
        assert best_w2 is not None
        assert best_mask is not None
        self.weights_: list[np.ndarray] = [best_w1, best_w2]
        self.inlier_mask_: np.ndarray = np.zeros(n, dtype=bool)
        self.inlier_mask_[best_mask] = True
        return self
