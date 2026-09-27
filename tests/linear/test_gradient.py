"""Tests for EY-loss CCA variants: PLSEY, CCAEY, StochasticCCAEY."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA, CCAEY, MCCA, PLS, PLSEY
from tests._helpers import canonical_correlations


@pytest.fixture
def three_correlated_views() -> list[np.ndarray]:
    """Three views sharing a latent structure, for multiview correctness checks."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((300, 2))
    x1 = z @ rng.standard_normal((2, 10)) + 0.1 * rng.standard_normal((300, 10))
    x2 = z @ rng.standard_normal((2, 8)) + 0.1 * rng.standard_normal((300, 8))
    x3 = z @ rng.standard_normal((2, 6)) + 0.1 * rng.standard_normal((300, 6))
    return [x1, x2, x3]


# ---------------------------------------------------------------------------
# CCAEY's `c` ridge parameter continuously blends its loss towards PLSEY's
# (see cca_zoo._utils._ey.weight_gram_mean); PLSEY is implemented as a thin
# CCAEY subclass with `c` fixed at 1, mirroring how CCA/PLS are thin RidgeCCA
# subclasses with `c` fixed at 0/1. StochasticCCAEY inherits the same `c`
# from CCAEY, fit by mini-batch momentum SGD instead of full-batch L-BFGS-B.
# ---------------------------------------------------------------------------


def test_pls_ey_loss_matches_cca_ey_at_c_equal_one(
    two_views: list[np.ndarray],
) -> None:
    """PLSEY's derivative/objective are exactly CCAEY's own formula at c=1.

    PLSEY does not fit identical weights to CCAEY(c=1) given the same seed,
    since the two deliberately use different initial-weights strategies
    (see cca_zoo._utils._ey.random_orthonormal_weights vs.
    cheap_orthonormal_projection_weights) -- this checks the invariant that
    actually matters instead: the shared loss/gradient formula itself.
    """
    k = 2
    rng = np.random.default_rng(0)
    weights = [rng.standard_normal((v.shape[1], k)) for v in two_views]
    representations = [v @ w for v, w in zip(two_views, weights)]
    pls = PLSEY(n_components=k)
    cca_c1 = CCAEY(n_components=k, c=1.0)
    grads_pls = pls._derivative(two_views, representations, weights)
    grads_cca = cca_c1._derivative(two_views, representations, weights)
    for a, b in zip(grads_pls, grads_cca):
        np.testing.assert_array_equal(a, b)
    assert pls._objective(two_views, representations, weights) == cca_c1._objective(
        two_views, representations, weights
    )


def test_cca_ey_c_zero_matches_shared_ey_gradient(
    two_views: list[np.ndarray],
) -> None:
    """c=0 (the default) reduces exactly to the shared, unregularised ey_grad_z."""
    from cca_zoo._utils._ey import ey_grad_z

    k = 2
    model = CCAEY(n_components=k, c=0.0, random_state=0)
    views_ = model._setup_fit(two_views)
    rng = np.random.default_rng(0)
    weights = [rng.standard_normal((v.shape[1], k)) for v in views_]
    representations = [v @ w for v, w in zip(views_, weights)]

    grads = model._derivative(views_, representations, weights)
    z_grads = ey_grad_z(representations)
    expected = [(v - v.mean(axis=0)).T @ zg for v, zg in zip(views_, z_grads)]
    for a, b in zip(grads, expected):
        np.testing.assert_allclose(a, b, atol=1e-10)


# ---------------------------------------------------------------------------
# Correctness / optimality
#
# The EY-loss gradient computed here (see cca_zoo._utils._ey) is verified
# against finite-difference gradients; these tests additionally check that
# a converged fit recovers the same canonical correlations as the exact
# eigendecomposition solutions.
# ---------------------------------------------------------------------------


def test_cca_ey_matches_cca(correlated_views: list[np.ndarray]) -> None:
    """Converged CCAEY recovers the same correlations as exact CCA."""
    k = 2
    s_cca = canonical_correlations(
        CCA(n_components=k).fit(correlated_views), correlated_views
    )
    s_ey = canonical_correlations(
        CCAEY(n_components=k, random_state=0).fit(correlated_views),
        correlated_views,
    )
    # L-BFGS-B does not guarantee components are returned in
    # descending-correlation order, unlike the exact eigendecomposition.
    np.testing.assert_allclose(
        sorted(s_ey, reverse=True), sorted(s_cca, reverse=True), atol=0.05
    )


def test_pls_ey_matches_pls(correlated_views: list[np.ndarray]) -> None:
    """Converged PLSEY recovers the same correlations as exact PLS."""
    k = 2
    s_pls = canonical_correlations(
        PLS(n_components=k).fit(correlated_views), correlated_views
    )
    s_ey = canonical_correlations(
        PLSEY(n_components=k, random_state=0).fit(correlated_views),
        correlated_views,
    )
    np.testing.assert_allclose(
        sorted(s_ey, reverse=True), sorted(s_pls, reverse=True), atol=0.05
    )


def test_cca_ey_matches_mcca_for_three_views(
    three_correlated_views: list[np.ndarray],
) -> None:
    """CCAEY, given 3 views directly, recovers the same correlations as exact MCCA."""
    k = 2
    s_mcca = canonical_correlations(
        MCCA(n_components=k).fit(three_correlated_views), three_correlated_views
    )
    s_ey = canonical_correlations(
        CCAEY(n_components=k, random_state=0).fit(three_correlated_views),
        three_correlated_views,
    )
    np.testing.assert_allclose(
        sorted(s_ey, reverse=True), sorted(s_mcca, reverse=True), atol=0.05
    )


# ---------------------------------------------------------------------------
# Initial weights: PLSEY gets plain orthonormal weights (matching its own
# weight-space penalty); CCAEY gets a cheap, data-informed init giving
# exactly unit-variance, uncorrelated projections on the full dataset
# instead (a cheap stand-in for classical CCA's full whitening step, and a
# direct counter to the near-null-direction divergence risk noted in
# CCAEY's own docstring). StochasticCCAEY inherits CCAEY's initialiser but
# projects on one mini-batch instead of the full dataset, since a
# full-batch pass is exactly what it exists to avoid.
# ---------------------------------------------------------------------------


def test_pls_ey_initial_weights_are_orthonormal(two_views: list[np.ndarray]) -> None:
    """PLSEY's initial weights have exactly orthonormal columns per view."""
    k = 2
    rng = np.random.default_rng(0)
    model = PLSEY(n_components=k, random_state=0)
    weights = model._initial_weights(two_views, rng)
    for w in weights:
        np.testing.assert_allclose(w.T @ w, np.eye(k), atol=1e-10)


def test_cca_ey_initial_weights_give_orthonormal_projections(
    two_views: list[np.ndarray],
) -> None:
    """CCAEY's initial weights give unit-variance, uncorrelated projections.

    On the full dataset. Unlike PLSEY's plain weight-orthonormal init,
    CCAEY's own initial *weights* are not themselves orthonormal in
    general -- what's orthonormal is the projection.
    """
    k = 2
    rng = np.random.default_rng(0)
    model = CCAEY(n_components=k, random_state=0)
    weights = model._initial_weights(two_views, rng)
    for view, w in zip(two_views, weights):
        z = view @ w
        np.testing.assert_allclose(z.T @ z, np.eye(k), atol=1e-8)
