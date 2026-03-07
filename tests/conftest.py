"""Shared pytest fixtures for the cca-zoo test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def two_views() -> list[np.ndarray]:
    """Two random views with 50 samples, 10 and 8 features respectively."""
    rng = np.random.default_rng(0)
    return [rng.standard_normal((50, 10)), rng.standard_normal((50, 8))]


@pytest.fixture
def three_views() -> list[np.ndarray]:
    """Three random views with 50 samples and 10, 8, 6 features."""
    rng = np.random.default_rng(0)
    return [
        rng.standard_normal((50, 10)),
        rng.standard_normal((50, 8)),
        rng.standard_normal((50, 6)),
    ]


@pytest.fixture
def two_views_small() -> list[np.ndarray]:
    """Two small random views (30 samples, 5 features) for kernel methods."""
    rng = np.random.default_rng(0)
    return [rng.standard_normal((30, 5)), rng.standard_normal((30, 5))]


@pytest.fixture
def three_views_small() -> list[np.ndarray]:
    """Three small random views (20 samples, 5 features) for kernel methods."""
    rng = np.random.default_rng(0)
    return [
        rng.standard_normal((20, 5)),
        rng.standard_normal((20, 5)),
        rng.standard_normal((20, 5)),
    ]


@pytest.fixture
def correlated_views() -> list[np.ndarray]:
    """Two views sharing a latent structure, giving higher canonical correlations."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((50, 2))
    x1 = z @ rng.standard_normal((2, 10)) + 0.1 * rng.standard_normal((50, 10))
    x2 = z @ rng.standard_normal((2, 8)) + 0.1 * rng.standard_normal((50, 8))
    return [x1, x2]


@pytest.fixture
def two_views_test() -> list[np.ndarray]:
    """An independent test set matching the two_views fixture dimensionality."""
    rng = np.random.default_rng(42)
    return [rng.standard_normal((20, 10)), rng.standard_normal((20, 8))]
