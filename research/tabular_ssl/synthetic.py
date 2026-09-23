"""Synthetic tables with a controllable split between shared and idiosyncratic signal.

A table has ``d_shared`` columns that are noisy, monotonically distorted
measurements of a ``k``-dimensional latent state, plus ``d_idio`` independent
columns. The target mixes a nonlinear function of the latent state (weight
``alpha``) with the same function of four idiosyncratic columns (``1-alpha``).
``alpha=1`` is the multi-view-redundant regime in which unlabeled rows carry
information about the label; ``alpha=0`` is the regime where no SSL can help.
``nonmonotone=True`` passes half the shared measurements through ``|u|``,
which a Gaussian copula cannot linearise.
"""

from __future__ import annotations

import numpy as np

_DISTORTIONS = [
    lambda u: u,
    np.exp,
    lambda u: u**3,
    lambda u: np.tanh(2 * u),
    lambda u: np.floor(2 * u),  # discretised
]


def _signal(A: np.ndarray) -> np.ndarray:
    return A[:, 0] + np.sin(2 * A[:, 1]) + A[:, 2] * A[:, 3]


def make_table(
    n: int,
    alpha: float,
    k: int = 4,
    d_shared: int = 24,
    d_idio: int = 8,
    noise: float = 0.5,
    nonmonotone: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``(X, y)`` for a regression task; see the module docstring."""
    structure = np.random.default_rng(1000 + seed)  # fixed per table family
    loadings = structure.standard_normal((k, d_shared))
    loadings /= np.linalg.norm(loadings, axis=0)
    distort = structure.integers(len(_DISTORTIONS), size=d_shared + d_idio)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, k))
    U = z @ loadings + noise * rng.standard_normal((n, d_shared))
    if nonmonotone:
        U[:, ::2] = np.abs(U[:, ::2])
    E = rng.standard_normal((n, d_idio))
    raw = np.hstack([U, E])
    X = np.column_stack([_DISTORTIONS[g](raw[:, j]) for j, g in enumerate(distort)])
    y = alpha * _signal(z) + (1 - alpha) * _signal(E[:, :4])
    return X, y + 0.1 * rng.standard_normal(n)
