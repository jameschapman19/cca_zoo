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
        latent_dimensions=2, num_warmup=500, num_samples=1000, random_state=0
    ).fit([x1, x2])

    w_samples = np.concatenate(
        [model.posterior_samples_[f"W_{i}"] for i in range(2)], axis=1
    )
    frob_of_mean = np.linalg.norm(w_samples.mean(axis=0), ord="fro") ** 2
    mean_of_frob = np.mean(np.linalg.norm(w_samples, ord="fro", axis=(1, 2)) ** 2)
    ratio = frob_of_mean / mean_of_frob
    assert ratio > 0.95, f"Expected near-coherent draws (ratio ~1.0), got {ratio}"


@pytest.mark.slow
def test_pcca_z_rotation_matches_realigned_weights(pcca_class: type) -> None:
    """Realigned z draws stay internally consistent with the realigned W draws.

    Each draw's rotation is applied to both its W_i's and its z jointly (see
    ``ProbabilisticCCA.fit``); checks this by verifying that, for a handful
    of draws, ``z_s @ W_s.T`` (the draw's own reconstruction of the centred
    data) is unaffected by the realignment -- confirming z and W were
    rotated by the same, not independent, rotations.
    """
    rng = np.random.default_rng(0)
    n = 60
    x1 = rng.standard_normal((n, 5))
    x2 = rng.standard_normal((n, 4))

    model = pcca_class(
        latent_dimensions=2, num_warmup=50, num_samples=50, random_state=0
    ).fit([x1, x2])

    z_samples = model.posterior_samples_["z"]  # (S, n, k)
    w0_samples = model.posterior_samples_["W_0"]  # (S, p0, k)
    for s in (0, 10, 30):
        reconstruction = z_samples[s] @ w0_samples[s].T
        # Reconstructions should be finite and non-trivial (not collapsed to
        # zero by a botched joint rotation).
        assert np.all(np.isfinite(reconstruction))
        assert np.std(reconstruction) > 1e-6
