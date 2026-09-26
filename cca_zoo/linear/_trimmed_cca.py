"""Robust multiview CCA by concentration steps."""

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


def _refit(
    model: CCAEY,
    xs: list[np.ndarray],
    weights: list[np.ndarray],
    tol: float,
) -> list[np.ndarray]:
    """Re-minimise the CCAEY loss on ``xs`` by L-BFGS-B, starting from ``weights``."""
    shapes = [w.shape for w in weights]
    sizes = [w.size for w in weights]

    def unflatten(x: np.ndarray) -> list[np.ndarray]:
        out = []
        offset = 0
        for shape, size in zip(shapes, sizes):
            out.append(x[offset : offset + size].reshape(shape))
            offset += size
        return out

    def fun(x: np.ndarray) -> tuple[float, np.ndarray]:
        ws = unflatten(x)
        representations = [xv @ w for xv, w in zip(xs, ws)]
        obj = model._objective(xs, representations, ws)
        grads = model._derivative(xs, representations, ws)
        grad = np.concatenate([g.ravel() for g in grads])
        return obj, grad

    x0 = np.concatenate([w.ravel() for w in weights])
    result = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"ftol": tol})
    return unflatten(result.x)


def _objective_value(
    model: CCAEY, xs: list[np.ndarray], weights: list[np.ndarray]
) -> float:
    """The CCAEY loss at ``weights`` on ``xs``."""
    representations = [xv @ w for xv, w in zip(xs, weights)]
    return model._objective(xs, representations, weights)


class TrimmedCCA(BaseModel):
    """Robust multiview CCA by concentration steps.

    As in least trimmed squares, alternates between keeping the ``h`` rows
    with the lowest :class:`~cca_zoo.linear.gradient.CCAEY` loss and
    refitting on them. Neither step increases the loss. The best of
    ``n_init`` random starts is kept. Suited to heavy contamination when
    the clean fraction is roughly known. Supports one latent dimension.

    Args:
        n_components: Number of latent dimensions; must be 1. Default is 1.
        center: Whether to centre each view. Default is True.
        c: Ridge blend in ``[0, 1]``, as in ``CCAEY``. Default is 0.1.
        h_frac: Fraction of rows kept, in ``(0, 1]``; a prior on the clean
            fraction. Default is 0.75.
        n_init: Random restarts. Default is 10.
        max_iter: Maximum concentration steps per restart. Default is 30.
        tol: L-BFGS-B ``ftol`` for each refit. Default is 1e-8.
        random_state: Seed for the restarts. Default is None.

    Attributes:
        weights_: Weight matrix of each view, shape (n_features_i, 1).
        inlier_mask_: Boolean mask of the kept training rows.

    References:
        Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for
        the minimum covariance determinant estimator. Technometrics,
        41(3), 212-223.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.linear import TrimmedCCA
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((200, 8))
        >>> X2 = rng.standard_normal((200, 6))
        >>> model = TrimmedCCA(h_frac=0.7, random_state=0).fit([X1, X2])
        >>> int(model.inlier_mask_.sum())
        140
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        **BaseModel._parameter_constraints,
        "c": RIDGE_PARAMETER,
        "h_frac": [Interval(Real, 0, 1, closed="right")],
        "n_init": POSITIVE_INT,
        "max_iter": POSITIVE_INT,
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        n_components: int = 1,
        center: bool = True,
        c: float = 0.1,
        h_frac: float = 0.75,
        n_init: int = 10,
        max_iter: int = 30,
        tol: float = 1e-8,
        random_state: int | None = None,
    ) -> None:
        super().__init__(n_components=n_components, center=center)
        self.c = c
        self.h_frac = h_frac
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, views: list[ArrayLike], y: None = None) -> TrimmedCCA:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.

        Raises:
            ValueError: If ``n_components`` is not 1.
        """
        views_ = self._setup_fit(views)
        if self.n_components != 1:
            raise ValueError(
                "TrimmedCCA currently supports only n_components=1, "
                f"got {self.n_components}."
            )
        n = self.n_samples_
        h = max(2, int(round(self.h_frac * n)))
        rng = np.random.default_rng(self.random_state)
        # CCAEY's own _objective/_derivative back both the selection score
        # and the refit, rather than a second implementation of the loss
        # living here -- .fit() is never called on it, only these two.
        model = CCAEY(n_components=1, c=self.c)

        best_weights: list[np.ndarray] | None = None
        best_mask: np.ndarray | None = None
        best_obj = np.inf
        for _ in range(self.n_init):
            weights = []
            for xv in views_:
                w = rng.standard_normal((xv.shape[1], 1))
                w /= np.linalg.norm(w)
                weights.append(w)

            kept = np.sort(rng.choice(n, h, replace=False))
            xs_kept = [xv[kept] for xv in views_]
            weights = _refit(model, xs_kept, weights, self.tol)
            cur_obj = _objective_value(model, xs_kept, weights)

            for _ in range(self.max_iter):
                zs = [(xv @ w).ravel() for xv, w in zip(views_, weights)]
                b = weight_gram_mean(weights)[0, 0]
                new_kept = _select(zs, b, self.c, h)
                if np.array_equal(new_kept, kept):
                    break
                xs_new = [xv[new_kept] for xv in views_]
                new_weights = _refit(model, xs_new, weights, self.tol)
                new_obj = _objective_value(model, xs_new, new_weights)
                if new_obj > cur_obj + 1e-10:
                    break
                kept, weights, cur_obj = new_kept, new_weights, new_obj

            if cur_obj < best_obj:
                best_obj, best_weights, best_mask = cur_obj, weights, kept

        assert best_weights is not None
        assert best_mask is not None
        self.weights_: list[np.ndarray] = best_weights
        self.inlier_mask_: np.ndarray = np.zeros(n, dtype=bool)
        self.inlier_mask_[best_mask] = True
        return self
