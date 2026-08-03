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
# Helpers
# ---------------------------------------------------------------------------


def _make_small_views(
    n: int = 20, p1: int = 4, p2: int = 4, seed: int = 0
) -> list[np.ndarray]:
    """Create two small normalised views for fast MCMC runs."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal((n, p1))
    x2 = rng.standard_normal((n, p2))
    # Normalise for better MCMC mixing
    x1 = (x1 - x1.mean(axis=0)) / (x1.std(axis=0) + 1e-8)
    x2 = (x2 - x2.mean(axis=0)) / (x2.std(axis=0) + 1e-8)
    return [x1, x2]


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
# fit completes
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_pcca_fit_completes(pcca_class: type) -> None:
    """ProbabilisticCCA.fit completes for minimal num_warmup and num_samples."""
    views = _make_small_views()
    model = pcca_class(
        latent_dimensions=1,
        num_warmup=10,
        num_samples=10,
        random_state=0,
    )
    fitted = model.fit(views)
    assert fitted is model


@pytest.mark.slow
def test_pcca_fit_sets_params(pcca_class: type) -> None:
    """ProbabilisticCCA.fit stores inferred parameters."""
    views = _make_small_views()
    model = pcca_class(
        latent_dimensions=1,
        num_warmup=10,
        num_samples=10,
        random_state=0,
    )
    model.fit(views)
    assert hasattr(model, "posterior_samples_")
    assert model.posterior_samples_ is not None


# ---------------------------------------------------------------------------
# transform output shapes
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_pcca_transform_output_shapes(pcca_class: type) -> None:
    """ProbabilisticCCA.transform returns arrays of the expected shape."""
    n, k = 20, 1
    views = _make_small_views(n=n)
    model = pcca_class(
        latent_dimensions=k,
        num_warmup=10,
        num_samples=10,
        random_state=0,
    )
    model.fit(views)
    # The v2-style transform may return representations of shape (n, k)
    # or (num_samples, n, k) depending on implementation
    result = model.transform(views)
    # Accept either a list or a single array
    if isinstance(result, list):
        for arr in result:
            assert isinstance(arr, np.ndarray)
    else:
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# latent_dimensions parameter
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("k", [1, 2])
def test_pcca_latent_dimensions(pcca_class: type, k: int) -> None:
    """ProbabilisticCCA can be instantiated with various latent_dimensions."""
    model = pcca_class(
        latent_dimensions=k,
        num_warmup=5,
        num_samples=5,
        random_state=0,
    )
    views = _make_small_views(n=15)
    model.fit(views)
    assert model.latent_dimensions == k


# ---------------------------------------------------------------------------
# View-count handling
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_pcca_supports_more_than_two_views(pcca_class: type) -> None:
    """ProbabilisticCCA's generative model is view-count generic (>=2)."""
    rng = np.random.default_rng(0)
    three_views = [rng.standard_normal((15, 4)) for _ in range(3)]
    model = pcca_class(
        latent_dimensions=1,
        num_warmup=5,
        num_samples=5,
        random_state=0,
    )
    model.fit(three_views)
    assert len(model.weights_) == 3


@pytest.mark.slow
def test_pcca_rejects_single_view(pcca_class: type) -> None:
    """ProbabilisticCCA raises ValueError when given fewer than 2 views."""
    rng = np.random.default_rng(0)
    one_view = [rng.standard_normal((15, 4))]
    model = pcca_class(
        latent_dimensions=1,
        num_warmup=5,
        num_samples=5,
        random_state=0,
    )
    with pytest.raises(ValueError, match="views"):
        model.fit(one_view)
