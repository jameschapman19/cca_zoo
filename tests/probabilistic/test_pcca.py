"""Tests for probabilistic CCA (ProbabilisticCCA).

All tests are marked slow and require numpyro + jax.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip the entire module if numpyro is not installed
numpyro = pytest.importorskip("numpyro", reason="numpyro is not installed")
jax = pytest.importorskip("jax", reason="jax is not installed")

# Also skip if the probabilistic module fails to import (e.g. dependency
# on v2 internals not yet ported)
pcca_module = pytest.importorskip(
    "cca_zoo.probabilistic",
    reason="cca_zoo.probabilistic could not be imported",
)


# ---------------------------------------------------------------------------
# Import ProbabilisticCCA — skip if not present in the module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pcca_class() -> type:
    """Return the ProbabilisticCCA class, or skip the test if unavailable."""
    if not hasattr(pcca_module, "ProbabilisticCCA"):
        pytest.skip("ProbabilisticCCA not found in cca_zoo.probabilistic")
    return pcca_module.ProbabilisticCCA  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Rotational-symmetry alignment (regression: posterior mean used to be
# biased toward zero by un-aligned draws — see align_posterior_rotation)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_pcca_posterior_draws_are_rotation_aligned(pcca_class: type) -> None:
    """The posterior mean W shouldn't be shrunk by cross-draw rotational drift.

    ``||mean(W)||_F^2`` and ``mean(||W||_F^2)`` are both rotation-invariant
    quantities that should be nearly equal if every draw agrees on a common
    rotation (any gap is pure rotational cancellation in the mean, not
    signal). Before aligning draws via ``align_posterior_rotation``, this
    ratio measured ~0.81 on a similar problem; this guards against
    regressing back to that.
    """
    rng = np.random.default_rng(0)
    n = 100
    z = rng.standard_normal((n, 2))
    x1 = z @ rng.standard_normal((2, 6)) + 0.1 * rng.standard_normal((n, 6))
    x2 = z @ rng.standard_normal((2, 5)) + 0.1 * rng.standard_normal((n, 5))

    model = pcca_class(
        n_components=2, n_warmup=500, n_posterior_samples=1000, random_state=0
    ).fit([x1, x2])

    w_samples = np.concatenate(
        [model.posterior_samples_[f"W_{i}"] for i in range(2)], axis=1
    )
    frob_of_mean = np.linalg.norm(w_samples.mean(axis=0), ord="fro") ** 2
    mean_of_frob = np.mean(np.linalg.norm(w_samples, ord="fro", axis=(1, 2)) ** 2)
    ratio = frob_of_mean / mean_of_frob
    assert ratio > 0.95, f"Expected near-coherent draws (ratio ~1.0), got {ratio}"
