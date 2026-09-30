"""Shared pytest fixtures for the cca-zoo test suite."""

from __future__ import annotations

import os

# Each pytest-xdist worker gets one thread: BLAS, torch and XLA would otherwise
# each spread over every core and contend across workers. Set before they load.
if "PYTEST_XDIST_WORKER" in os.environ:
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_var] = "1"
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )

import numpy as np
import pytest

from tests._helpers import linear_views


@pytest.fixture
def two_views() -> list[np.ndarray]:
    """Two random views with 50 samples, 10 and 8 features respectively."""
    rng = np.random.default_rng(0)
    return [rng.standard_normal((50, 10)), rng.standard_normal((50, 8))]


@pytest.fixture
def two_views_small() -> list[np.ndarray]:
    """Two small random views (30 samples, 5 features) for kernel methods."""
    rng = np.random.default_rng(0)
    return [rng.standard_normal((30, 5)), rng.standard_normal((30, 5))]


@pytest.fixture
def correlated_views() -> list[np.ndarray]:
    """Two views sharing two factors, giving high canonical correlations."""
    return linear_views(0, 50, (10, 8), noise=0.1)
