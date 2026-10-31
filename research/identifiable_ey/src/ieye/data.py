"""Two-view synthetic data with independent non-Gaussian shared sources.

Source ``i`` has unit variance and excess kurtosis ``kurtosis[i]``. View ``v``
observes it through ``y_vi = sqrt(rho_i) s_i + sqrt(1 - rho_i) e_vi`` with
Gaussian ``e``, so the canonical correlations are exactly ``rho``. The
observation is ``x_v = P_v [y_v; g_v]`` with ``g_v`` independent Gaussian
private dimensions and ``P_v`` a random invertible mixing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln

MIN_EXCESS_KURTOSIS = -1.2  # the uniform distribution, the limit of shape -> inf


def ggd_excess_kurtosis(shape: float) -> float:
    """Excess kurtosis of the generalised Gaussian with the given shape."""
    return float(np.exp(gammaln(5 / shape) + gammaln(1 / shape) - 2 * gammaln(3 / shape)) - 3)


def ggd_shape(excess_kurtosis: float) -> float:
    """Shape of the generalised Gaussian with the given excess kurtosis."""
    if excess_kurtosis <= MIN_EXCESS_KURTOSIS:
        raise ValueError(f"excess kurtosis must exceed {MIN_EXCESS_KURTOSIS}")
    return brentq(lambda b: ggd_excess_kurtosis(b) - excess_kurtosis, 0.05, 200.0)


def sample_sources(
    n: int, kurtosis: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Independent unit-variance sources, shape (n, k), with the given excess kurtoses."""
    columns = []
    for kappa in kurtosis:
        if kappa == 0:
            columns.append(rng.standard_normal(n))
            continue
        beta = ggd_shape(kappa)
        # |x|^beta / alpha^beta ~ Gamma(1/beta); alpha sets unit variance.
        magnitude = rng.gamma(1 / beta, size=n) ** (1 / beta)
        alpha = np.exp(0.5 * (gammaln(1 / beta) - gammaln(3 / beta)))
        columns.append(alpha * magnitude * rng.choice([-1.0, 1.0], size=n))
    return np.stack(columns, axis=1)


def _random_mixing(p: int, rng: np.random.Generator) -> np.ndarray:
    """A random invertible matrix with condition number at most e."""
    q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    return q @ np.diag(np.exp(rng.uniform(-0.5, 0.5, p)))


@dataclass(frozen=True)
class TwoViewModel:
    """A fixed population: sample from it, and read off the truth."""

    rho: np.ndarray
    kurtosis: np.ndarray
    mixings: tuple[np.ndarray, np.ndarray]

    @classmethod
    def make(
        cls,
        rho: np.ndarray,
        kurtosis: np.ndarray,
        n_features: int,
        rng: np.random.Generator,
    ) -> TwoViewModel:
        """Draw the mixings; ``rho`` and ``kurtosis`` have one entry per source."""
        rho, kurtosis = np.asarray(rho, float), np.asarray(kurtosis, float)
        if rho.shape != kurtosis.shape:
            raise ValueError("rho and kurtosis must have the same shape")
        if not (0 < rho).all() or not (rho < 1).all():
            raise ValueError("rho must lie in (0, 1)")
        if n_features < len(rho):
            raise ValueError("n_features must be at least the number of sources")
        return cls(rho, kurtosis, tuple(_random_mixing(n_features, rng) for _ in range(2)))

    @property
    def k(self) -> int:
        """Number of shared sources."""
        return len(self.rho)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, list[np.ndarray]]:
        """``(sources, [x_1, x_2])``, shapes (n, k) and (n, p)."""
        s = sample_sources(n, self.kurtosis, rng)
        p = self.mixings[0].shape[0]
        views = []
        for mixing in self.mixings:
            e = rng.standard_normal((n, self.k))
            y = np.sqrt(self.rho) * s + np.sqrt(1 - self.rho) * e
            g = rng.standard_normal((n, p - self.k))
            views.append(np.hstack([y, g]) @ mixing.T)
        return s, views

    def view_signal_mixing(self, v: int) -> np.ndarray:
        """Columns of ``P_v`` that carry the shared sources, shape (p, k)."""
        return self.mixings[v][:, : self.k]

    def canonical_weights(self, v: int) -> np.ndarray:
        """Population CCA weights of view ``v``, shape (p, k)."""
        return np.linalg.inv(self.mixings[v])[: self.k].T
