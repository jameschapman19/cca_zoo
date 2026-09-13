"""Tests for EY-loss CCA variants: PLSEY, CCAEY, StochasticCCAEY."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA, CCAEY, MCCA, PLS, PLSEY, HuberCCA, StochasticCCAEY

FULL_BATCH_MODELS = [PLSEY, CCAEY, HuberCCA]
ALL_GRADIENT_MODELS = [PLSEY, CCAEY, StochasticCCAEY, HuberCCA]

# Use fewer iterations for speed in tests
_FIT_KWARGS: dict = dict(latent_dimensions=1, max_iter=50, random_state=0)


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
# fit completes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_two_view_fit_completes(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Fit completes on two-view data without error."""
    model = ModelClass(**_FIT_KWARGS)
    fitted = model.fit(two_views)
    assert fitted is model
    assert hasattr(model, "weights_")


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_three_view_fit_completes(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """CCAEY (and its subclasses) fit directly on three-view data."""
    model = ModelClass(**_FIT_KWARGS)
    fitted = model.fit(three_views)
    assert fitted is model


# ---------------------------------------------------------------------------
# transform output shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_transform_shapes_two_view(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """Transform returns list of (n_samples, latent_dimensions) arrays."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    result = model.transform(two_views)
    assert len(result) == len(two_views)
    for arr, view in zip(result, two_views):
        assert arr.shape == (view.shape[0], k)


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_transform_shapes_multi_view(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """Transform returns correct shapes for three views."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(
        three_views
    )
    result = model.transform(three_views)
    assert len(result) == len(three_views)
    for arr, view in zip(result, three_views):
        assert arr.shape == (view.shape[0], k)


# ---------------------------------------------------------------------------
# fit_transform consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", FULL_BATCH_MODELS)
def test_fit_transform_consistency(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """fit_transform equals fit().transform()."""
    kwargs = dict(latent_dimensions=1, max_iter=50, random_state=0)
    result_ft = ModelClass(**kwargs).fit_transform(two_views)
    result_sep = ModelClass(**kwargs).fit(two_views).transform(two_views)
    for a, b in zip(result_ft, result_sep):
        np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-10)


# ---------------------------------------------------------------------------
# score shape and range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_score_shape(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Score returns array of shape (latent_dimensions,)."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    s = model.score(two_views)
    assert s.shape == (k,)


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_score_shape_multi_view(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """Score returns array of shape (latent_dimensions,) for multi-view."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(
        three_views
    )
    s = model.score(three_views)
    assert s.shape == (k,)


# get_params/set_params roundtrip behaviour is exercised generically for
# every model in the package by tests/test_sklearn_compat.py.


# ---------------------------------------------------------------------------
# weights shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_weights_shapes_two_view(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Weights shapes are (n_features_i, latent_dimensions) per view."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    w = model.weights
    assert len(w) == len(two_views)
    for weight, view in zip(w, two_views):
        assert weight.shape == (view.shape[1], k)


# ---------------------------------------------------------------------------
# get_factor_loadings shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", FULL_BATCH_MODELS)
def test_get_factor_loadings_shapes(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """get_factor_loadings returns (n_features_i, k) per view."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    loadings = model.get_factor_loadings(two_views)
    assert len(loadings) == len(two_views)
    for loading, view in zip(loadings, two_views):
        assert loading.shape == (view.shape[1], k)


# ---------------------------------------------------------------------------
# StochasticCCAEY-specific: mini-batch training
# ---------------------------------------------------------------------------


def test_mini_batch_training(two_views: list[np.ndarray]) -> None:
    """StochasticCCAEY trains without error with batch_size < n_samples."""
    model = StochasticCCAEY(
        latent_dimensions=1, max_iter=20, batch_size=16, random_state=0
    )
    model.fit(two_views)
    result = model.transform(two_views)
    assert len(result) == 2
    for arr, view in zip(result, two_views):
        assert arr.shape == (view.shape[0], 1)
    assert all(np.all(np.isfinite(w)) for w in model.weights_)


def test_full_batch_training(two_views: list[np.ndarray]) -> None:
    """StochasticCCAEY trains without error with batch_size=None (full batch)."""
    model = StochasticCCAEY(
        latent_dimensions=1, max_iter=20, batch_size=None, random_state=0
    )
    model.fit(two_views)
    result = model.transform(two_views)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_reproducibility_same_random_state(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """Same random_state gives identical weights."""
    kwargs = dict(latent_dimensions=1, max_iter=50, random_state=123)
    w1 = ModelClass(**kwargs).fit(two_views).weights
    w2 = ModelClass(**kwargs).fit(two_views).weights
    for a, b in zip(w1, w2):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_different_seeds_give_different_results(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """Different random_state values generally give different initial weights."""
    kwargs_a = dict(latent_dimensions=1, max_iter=2, random_state=0)
    kwargs_b = dict(latent_dimensions=1, max_iter=2, random_state=999)
    w1 = ModelClass(**kwargs_a).fit(two_views).weights
    w2 = ModelClass(**kwargs_b).fit(two_views).weights
    # At least one weight matrix should differ
    any_different = any(not np.allclose(a, b) for a, b in zip(w1, w2))
    assert any_different


# ---------------------------------------------------------------------------
# CCAEY's `c` ridge parameter continuously blends its loss towards PLSEY's
# (see cca_zoo._utils._ey.weight_gram_mean); PLSEY is implemented as a thin
# CCAEY subclass with `c` fixed at 1, mirroring how CCA/PLS are thin rCCA
# subclasses with `c` fixed at 0/1. StochasticCCAEY inherits the same `c`
# from CCAEY, fit by mini-batch momentum SGD instead of full-batch L-BFGS-B.
# ---------------------------------------------------------------------------


def test_cca_ey_accepts_c_pls_ey_does_not() -> None:
    """C blends CCAEY towards PLSEY; PLSEY fixes it at 1, unexposed."""
    assert CCAEY().get_params()["c"] == 0.0
    assert "c" not in PLSEY().get_params()
    assert StochasticCCAEY().get_params()["c"] == 0.0


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
    pls = PLSEY(latent_dimensions=k)
    cca_c1 = CCAEY(latent_dimensions=k, c=1.0)
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
    model = CCAEY(latent_dimensions=k, c=0.0, random_state=0)
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
# center=False
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_center_false(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """All gradient models work with center=False."""
    model = ModelClass(latent_dimensions=1, max_iter=20, center=False, random_state=0)
    model.fit(two_views)
    result = model.transform(two_views)
    assert len(result) == 2


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
    s_cca = CCA(latent_dimensions=k).fit(correlated_views).score(correlated_views)
    s_ey = (
        CCAEY(latent_dimensions=k, random_state=0)
        .fit(correlated_views)
        .score(correlated_views)
    )
    # L-BFGS-B does not guarantee components are returned in
    # descending-correlation order, unlike the exact eigendecomposition.
    np.testing.assert_allclose(
        sorted(s_ey, reverse=True), sorted(s_cca, reverse=True), atol=0.05
    )


def test_pls_ey_matches_pls(correlated_views: list[np.ndarray]) -> None:
    """Converged PLSEY recovers the same correlations as exact PLS."""
    k = 2
    s_pls = PLS(latent_dimensions=k).fit(correlated_views).score(correlated_views)
    s_ey = (
        PLSEY(latent_dimensions=k, random_state=0)
        .fit(correlated_views)
        .score(correlated_views)
    )
    np.testing.assert_allclose(
        sorted(s_ey, reverse=True), sorted(s_pls, reverse=True), atol=0.05
    )


def test_cca_ey_matches_mcca_for_three_views(
    three_correlated_views: list[np.ndarray],
) -> None:
    """CCAEY, given 3 views directly, recovers the same correlations as exact MCCA."""
    k = 2
    s_mcca = (
        MCCA(latent_dimensions=k)
        .fit(three_correlated_views)
        .score(three_correlated_views)
    )
    s_ey = (
        CCAEY(latent_dimensions=k, random_state=0)
        .fit(three_correlated_views)
        .score(three_correlated_views)
    )
    np.testing.assert_allclose(
        sorted(s_ey, reverse=True), sorted(s_mcca, reverse=True), atol=0.05
    )


@pytest.mark.parametrize("ModelClass", [PLSEY, CCAEY])
def test_gradient_models_find_high_correlation(
    ModelClass: type, correlated_views: list[np.ndarray]
) -> None:
    """Full-batch EY models find substantial correlation on correlated views."""
    s = (
        ModelClass(latent_dimensions=1, max_iter=500, random_state=0)
        .fit(correlated_views)
        .score(correlated_views)
    )
    assert np.all(s > 0.8), f"{ModelClass.__name__} got low correlation: {s}"


def test_stochastic_cca_ey_finds_high_correlation(
    correlated_views: list[np.ndarray],
) -> None:
    """StochasticCCAEY finds substantial correlation on correlated views."""
    s = (
        StochasticCCAEY(latent_dimensions=1, max_iter=500, random_state=0)
        .fit(correlated_views)
        .score(correlated_views)
    )
    assert np.all(s > 0.8), f"StochasticCCAEY got low correlation: {s}"


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
    model = PLSEY(latent_dimensions=k, random_state=0)
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
    model = CCAEY(latent_dimensions=k, random_state=0)
    weights = model._initial_weights(two_views, rng)
    for view, w in zip(two_views, weights):
        z = view @ w
        np.testing.assert_allclose(z.T @ z, np.eye(k), atol=1e-8)


def test_stochastic_cca_ey_initial_weights_give_orthonormal_projections(
    two_views: list[np.ndarray],
) -> None:
    """StochasticCCAEY's initial weights give orthonormal projections.

    On one mini-batch, rather than the full dataset.
    """
    k = 2
    bs = 16
    rng = np.random.default_rng(0)
    model = StochasticCCAEY(latent_dimensions=k, batch_size=bs, random_state=0)
    weights = model._initial_weights(two_views, rng)
    # Re-derive, with a fresh rng in the same state, exactly which rows the
    # initialiser sampled, so the projection can be checked on that batch.
    rng2 = np.random.default_rng(0)
    n = two_views[0].shape[0]
    idx = rng2.choice(n, bs, replace=False)
    for view, w in zip(two_views, weights):
        z = view[idx] @ w
        np.testing.assert_allclose(z.T @ z, np.eye(k), atol=1e-8)


def test_cca_ey_initial_weights_differ_from_pls_ey(
    two_views: list[np.ndarray],
) -> None:
    """The two initialisers give different weights from the same seed."""
    k = 2
    rng_pls = np.random.default_rng(0)
    rng_cca = np.random.default_rng(0)
    w_pls = PLSEY(latent_dimensions=k)._initial_weights(two_views, rng_pls)
    w_cca = CCAEY(latent_dimensions=k)._initial_weights(two_views, rng_cca)
    assert any(not np.allclose(a, b) for a, b in zip(w_pls, w_cca))


# ---------------------------------------------------------------------------
# MCCAEY is deprecated: CCAEY now supports 2 or more views directly.
# ---------------------------------------------------------------------------


def test_mccaey_deprecated_alias_still_works(
    three_correlated_views: list[np.ndarray],
) -> None:
    """MCCAEY still works (as a thin CCAEY subclass) but warns FutureWarning."""
    from cca_zoo.linear import MCCAEY

    with pytest.warns(FutureWarning):
        model = MCCAEY(latent_dimensions=1, random_state=0)
    model.fit(three_correlated_views)
    assert isinstance(model, CCAEY)
    result = model.transform(three_correlated_views)
    assert len(result) == 3
