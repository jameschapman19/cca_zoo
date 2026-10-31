"""Multi-omics-like views with shared and private factors and missing views.

Each feature has unit variance: a share ``shared`` from factors common to all
views, a share ``private`` from factors of its own view, and the rest noise.
Whole views are then removed from samples, completely at random (MCAR) or with
a probability that rises with the first shared factor (MNAR), as when the
samples that lack an assay differ biologically from those that have it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


@dataclass(frozen=True)
class Dataset:
    """Complete noisy views, what was observed, and the truth."""

    views: list[np.ndarray]
    observed: np.ndarray  # (n_samples, n_views) bool
    shared_factors: np.ndarray  # (n_samples, n_shared)
    shared_signal: list[np.ndarray]  # each view's noise-free shared part

    @property
    def masked(self) -> list[np.ndarray]:
        """The views with the rows of unobserved samples set to NaN."""
        return [
            np.where(o[:, None], v, np.nan) for v, o in zip(self.views, self.observed.T)
        ]


def simulate(
    seed: int,
    *,
    n_samples: int = 300,
    widths: tuple[int, ...] = (200, 300, 150),
    n_shared: int = 2,
    n_private: int = 2,
    shared: float = 0.15,
    private: float = 0.4,
    missing_rate: float = 0.0,
    mechanism: str = "mcar",
    density: float = 1.0,
) -> Dataset:
    """Draw one dataset.

    Args:
        seed: Random seed.
        n_samples: Number of samples.
        widths: Features per view.
        n_shared: Factors common to every view.
        n_private: Factors specific to each view.
        shared: Share of each feature's variance from the shared factors.
        private: Share from the view's private factors.
        missing_rate: Mean probability that a sample lacks a given view.
        mechanism: ``"mcar"`` or ``"mnar"``.
        density: Fraction of features each factor loads on; 1 is dense loadings,
            less is sparse, with the loadings rescaled to keep the variance
            shares on average.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_samples, n_shared))
    views, signals = [], []
    for p in widths:
        own = rng.standard_normal((n_samples, n_private))
        signal = z @ _loadings(rng, n_shared, p, density) * np.sqrt(shared / n_shared)
        signals.append(signal)
        views.append(
            signal
            + own @ _loadings(rng, n_private, p, density) * np.sqrt(private / n_private)
            + rng.standard_normal((n_samples, p)) * np.sqrt(1 - shared - private)
        )
    observed = _observed(rng, z[:, 0], len(widths), missing_rate, mechanism)
    return Dataset(views, observed, z, signals)


def _loadings(rng: np.random.Generator, k: int, p: int, density: float) -> np.ndarray:
    """Loadings of ``k`` factors on ``p`` features, each non-zero with ``density``."""
    return (
        rng.standard_normal((k, p)) * (rng.random((k, p)) < density) / np.sqrt(density)
    )


def _observed(
    rng: np.random.Generator,
    driver: np.ndarray,
    n_views: int,
    missing_rate: float,
    mechanism: str,
) -> np.ndarray:
    """Which views each sample has; every sample keeps at least one."""
    if mechanism == "mcar":
        p_missing = np.full((len(driver), n_views), missing_rate)
    elif mechanism == "mnar":
        # The first view is missing more for high values of the driver, the
        # second less, the third more: shift each so its mean rate is right.
        signs = np.resize([1.0, -1.0], n_views)
        columns = []
        for sign in signs:
            logit = 1.5 * sign * driver
            shift = (
                brentq(
                    lambda s, logit=logit: expit(s + logit).mean() - missing_rate,
                    -30,
                    30,
                )
                if 0 < missing_rate < 1
                else (30 if missing_rate else -30)
            )
            columns.append(expit(shift + logit))
        p_missing = np.stack(columns, axis=1)
    else:
        raise ValueError(f"mechanism must be 'mcar' or 'mnar', got {mechanism!r}.")
    observed = rng.random(p_missing.shape) >= p_missing
    for i in np.flatnonzero(~observed.any(axis=1)):
        observed[i, rng.integers(n_views)] = True
    return observed
