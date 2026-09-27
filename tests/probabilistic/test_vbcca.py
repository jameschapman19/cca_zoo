"""Tests for VariationalBayesCCA.

All tests are marked slow and require numpyro + jax.
"""

from __future__ import annotations

import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro", reason="numpyro is not installed")
jax = pytest.importorskip("jax", reason="jax is not installed")

pcca_module = pytest.importorskip(
    "cca_zoo.probabilistic",
    reason="cca_zoo.probabilistic could not be imported",
)


def _make_small_views(
    n: int = 20, p1: int = 4, p2: int = 4, seed: int = 0
) -> list[np.ndarray]:
    """Create two small normalised views for fast SVI runs."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal((n, p1))
    x2 = rng.standard_normal((n, p2))
    x1 = (x1 - x1.mean(axis=0)) / (x1.std(axis=0) + 1e-8)
    x2 = (x2 - x2.mean(axis=0)) / (x2.std(axis=0) + 1e-8)
    return [x1, x2]


def _make_low_rank_views(
    n: int = 100, true_k: int = 2, p1: int = 6, p2: int = 5, seed: int = 0
) -> tuple[list[np.ndarray], np.ndarray]:
    """Two views sharing exactly ``true_k`` latent dimensions, plus noise.

    Returns the views and the true generating latent factor ``z``, so tests
    can check recovery of it directly.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, true_k))
    x1 = z @ rng.standard_normal((true_k, p1)) + 0.1 * rng.standard_normal((n, p1))
    x2 = z @ rng.standard_normal((true_k, p2)) + 0.1 * rng.standard_normal((n, p2))
    return [x1, x2], z


@pytest.fixture(scope="module")
def vbcca_class() -> type:
    """Return the VariationalBayesCCA class, or skip if unavailable."""
    if not hasattr(pcca_module, "VariationalBayesCCA"):
        pytest.skip("VariationalBayesCCA not found in cca_zoo.probabilistic")
    return pcca_module.VariationalBayesCCA  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# fit completes
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_vbcca_losses_decrease(vbcca_class: type) -> None:
    """The ELBO loss should be lower at the end of SVI than at the start."""
    views = _make_small_views(n=40)
    model = vbcca_class(n_components=1, max_iter=300, random_state=0).fit(views)
    assert model.losses_[-1] < model.losses_[0]


# ---------------------------------------------------------------------------
# ARD behaviour
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_vbcca_ard_shrinks_unsupported_dimensions(vbcca_class: type) -> None:
    """With more n_components than true shared factors, ARD should shrink the rest.

    The relevance score (posterior mean ARD precision) for the two true
    shared dimensions should end up substantially smaller than for the
    single spurious dimension, since a smaller precision means a wider
    (less-shrunk) prior on that column's loadings.
    """
    true_k = 2
    views, _ = _make_low_rank_views(n=150, true_k=true_k, seed=0)
    model = vbcca_class(n_components=3, max_iter=2000, random_state=0).fit(views)
    relevance = model.ard_relevance_
    assert relevance.shape == (3,)
    spurious_relevance = np.max(relevance)
    true_relevance = np.sort(relevance)[:true_k]
    assert spurious_relevance > 2 * np.max(true_relevance)


# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_vbcca_recovers_true_latent_factor(vbcca_class: type) -> None:
    """VariationalBayesCCA's inferred z recovers the true generating latent factor.

    Unlike deterministic CCA, this model's shared latent space has no
    per-axis identifiability: z ~ N(0, I) is rotationally symmetric, so the
    *individual* recovered axes need not align with the *individual* true
    generating axes (or with each other's per-view projections — see
    ``test_shared_scoring.py``'s docstring). What should hold regardless of
    that rotation is that the recovered 2-D z jointly spans the same
    subspace as the true 2-D z, which a CCA between the two (invariant to
    any invertible linear transform of either side) can check directly.
    """
    from cca_zoo.linear import CCA

    views, z_true = _make_low_rank_views(n=150, true_k=2, seed=0)
    model = vbcca_class(n_components=2, max_iter=2000, random_state=0).fit(views)
    z_hat = model.posterior_mean(views)

    recovery = CCA(n_components=2).fit([z_true, z_hat]).score([z_true, z_hat])
    assert np.all(recovery > 0.8), f"Expected near-total recovery, got {recovery}"
