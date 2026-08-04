"""Tests for gradient-descent CCA variants: PLS_EY, CCA_EY, MCCA_EY."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA, CCA_EY, MCCA, MCCA_EY, PLS, PLS_EY

GRADIENT_MODELS_TWO_VIEW = [PLS_EY, CCA_EY]
GRADIENT_MODELS_MULTI_VIEW = [MCCA_EY]
ALL_GRADIENT_MODELS = GRADIENT_MODELS_TWO_VIEW + GRADIENT_MODELS_MULTI_VIEW

# Use fewer iterations for speed in tests
_FIT_KWARGS: dict = dict(latent_dimensions=1, max_iter=50, random_state=0)


@pytest.fixture
def three_correlated_views() -> list[np.ndarray]:
    """Three views sharing a latent structure, for MCCA_EY correctness checks."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((300, 2))
    x1 = z @ rng.standard_normal((2, 10)) + 0.1 * rng.standard_normal((300, 10))
    x2 = z @ rng.standard_normal((2, 8)) + 0.1 * rng.standard_normal((300, 8))
    x3 = z @ rng.standard_normal((2, 6)) + 0.1 * rng.standard_normal((300, 6))
    return [x1, x2, x3]


# ---------------------------------------------------------------------------
# fit completes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_TWO_VIEW)
def test_two_view_fit_completes(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Fit completes on two-view data without error."""
    model = ModelClass(**_FIT_KWARGS)
    fitted = model.fit(two_views)
    assert fitted is model


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_two_view_fit_completes_all(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """All gradient models fit on two-view data."""
    model = ModelClass(**_FIT_KWARGS)
    model.fit(two_views)
    assert hasattr(model, "weights_")


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_MULTI_VIEW)
def test_three_view_fit_completes(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """MCCA_EY fits on three-view data."""
    model = ModelClass(**_FIT_KWARGS)
    fitted = model.fit(three_views)
    assert fitted is model


# ---------------------------------------------------------------------------
# transform output shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_TWO_VIEW)
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


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_MULTI_VIEW)
def test_transform_shapes_multi_view(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """MCCA_EY transform returns correct shapes for three views."""
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


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_TWO_VIEW)
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


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_TWO_VIEW)
def test_score_shape(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Score returns array of shape (latent_dimensions,)."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    s = model.score(two_views)
    assert s.shape == (k,)


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_MULTI_VIEW)
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


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_TWO_VIEW)
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


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_TWO_VIEW)
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
# Mini-batch training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_TWO_VIEW)
def test_mini_batch_training(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Models train without error with batch_size=16."""
    model = ModelClass(latent_dimensions=1, max_iter=20, batch_size=16, random_state=0)
    model.fit(two_views)
    result = model.transform(two_views)
    assert len(result) == 2
    for arr, view in zip(result, two_views):
        assert arr.shape == (view.shape[0], 1)


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_TWO_VIEW)
def test_full_batch_training(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Models train without error with batch_size=None (full batch)."""
    model = ModelClass(
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
# No upfront whitening (regression: CCA_EY used to whiten the full dataset
# with a full-batch SVD before any gradient step, defeating the entire
# point of a mini-batch/streaming stochastic method; MCCA_EY inherited the
# same bug via CCA_EY.fit(). PLS_EY, TreeCCA, and DCCA_EY all apply the same
# shared EY loss directly to raw embeddings with no such step, and the
# family's own docs describe it as needing no full-batch preprocessing.)
#
# CCA_EY keeps a ``c`` ridge parameter that continuously blends its loss
# towards PLS_EY's (see cca_zoo._utils._ey.weight_gram_mean); PLS_EY is
# implemented as a thin CCA_EY subclass with ``c`` fixed at 1, mirroring how
# CCA/PLS are thin rCCA subclasses with ``c`` fixed at 0/1.
# ---------------------------------------------------------------------------


def test_cca_ey_module_does_not_reference_svd_whiten() -> None:
    """CCA_EY's source no longer calls the full-batch whitening routine."""
    import inspect

    from cca_zoo.linear.gradient import _cca_ey

    source = inspect.getsource(_cca_ey)
    assert "svd_whiten" not in source


def test_cca_ey_accepts_c_pls_ey_does_not() -> None:
    """C blends CCA_EY towards PLS_EY; PLS_EY fixes it at 1, unexposed."""
    assert CCA_EY().get_params()["c"] == 0.0
    assert "c" not in PLS_EY().get_params()


def test_pls_ey_loss_matches_cca_ey_at_c_equal_one(
    two_views: list[np.ndarray],
) -> None:
    """PLS_EY's derivative/objective are exactly CCA_EY's own formula at c=1.

    PLS_EY no longer fits identical weights to CCA_EY(c=1) given the same
    seed, since the two now deliberately use different initial-weights
    strategies (see cca_zoo._utils._ey.random_orthonormal_weights vs.
    cheap_orthonormal_projection_weights) -- this checks the invariant that
    actually matters instead: the shared loss/gradient formula itself.
    """
    k = 2
    rng = np.random.default_rng(0)
    weights = [rng.standard_normal((v.shape[1], k)) for v in two_views]
    representations = [v @ w for v, w in zip(two_views, weights)]
    pls = PLS_EY(latent_dimensions=k)
    cca_c1 = CCA_EY(latent_dimensions=k, c=1.0)
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
    model = CCA_EY(latent_dimensions=k, c=0.0, random_state=0)
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
# converged gradient descent recovers the same canonical correlations as the
# exact eigendecomposition solutions, which the pre-fix implementation did
# not (it optimised a different, unrelated objective).
# ---------------------------------------------------------------------------


def test_cca_ey_matches_cca(correlated_views: list[np.ndarray]) -> None:
    """Converged CCA_EY recovers the same correlations as exact CCA."""
    k = 2
    s_cca = CCA(latent_dimensions=k).fit(correlated_views).score(correlated_views)
    s_ey = (
        CCA_EY(latent_dimensions=k, max_iter=1000, random_state=0)
        .fit(correlated_views)
        .score(correlated_views)
    )
    # Gradient descent does not guarantee components are returned in
    # descending-correlation order, unlike the exact eigendecomposition.
    np.testing.assert_allclose(
        sorted(s_ey, reverse=True), sorted(s_cca, reverse=True), atol=0.05
    )


def test_pls_ey_matches_pls(correlated_views: list[np.ndarray]) -> None:
    """Converged PLS_EY recovers the same correlations as exact PLS."""
    k = 2
    s_pls = PLS(latent_dimensions=k).fit(correlated_views).score(correlated_views)
    s_ey = (
        PLS_EY(latent_dimensions=k, max_iter=1000, random_state=0)
        .fit(correlated_views)
        .score(correlated_views)
    )
    np.testing.assert_allclose(
        sorted(s_ey, reverse=True), sorted(s_pls, reverse=True), atol=0.05
    )


def test_mcca_ey_matches_mcca(three_correlated_views: list[np.ndarray]) -> None:
    """Converged MCCA_EY recovers the same correlations as exact MCCA."""
    k = 2
    s_mcca = (
        MCCA(latent_dimensions=k)
        .fit(three_correlated_views)
        .score(three_correlated_views)
    )
    s_ey = (
        MCCA_EY(latent_dimensions=k, max_iter=1000, random_state=0)
        .fit(three_correlated_views)
        .score(three_correlated_views)
    )
    np.testing.assert_allclose(
        sorted(s_ey, reverse=True), sorted(s_mcca, reverse=True), atol=0.05
    )


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_gradient_models_find_high_correlation(
    ModelClass: type, correlated_views: list[np.ndarray]
) -> None:
    """All gradient models find substantial correlation on correlated views."""
    s = (
        ModelClass(latent_dimensions=1, max_iter=500, random_state=0)
        .fit(correlated_views)
        .score(correlated_views)
    )
    assert np.all(s > 0.8), f"{ModelClass.__name__} got low correlation: {s}"


# ---------------------------------------------------------------------------
# Initial weights: PLS_EY gets plain orthonormal weights (matching its own
# weight-space penalty); CCA_EY (and MCCA_EY, which inherits it) gets a
# cheap, data-informed init giving exactly unit-variance, uncorrelated
# projections on the first mini-batch instead (a cheap stand-in for
# classical CCA's full whitening step, and a direct counter to the
# near-null-direction divergence risk noted in CCA_EY's own docstring).
# ---------------------------------------------------------------------------


def test_pls_ey_initial_weights_are_orthonormal(two_views: list[np.ndarray]) -> None:
    """PLS_EY's initial weights have exactly orthonormal columns per view."""
    k = 2
    rng = np.random.default_rng(0)
    model = PLS_EY(latent_dimensions=k, random_state=0)
    weights = model._initial_weights(two_views, rng)
    for w in weights:
        np.testing.assert_allclose(w.T @ w, np.eye(k), atol=1e-10)


def test_cca_ey_initial_weights_give_orthonormal_projections(
    two_views: list[np.ndarray],
) -> None:
    """CCA_EY's initial weights give unit-variance, uncorrelated projections.

    Unlike PLS_EY's plain weight-orthonormal init, CCA_EY's own initial
    *weights* are not themselves orthonormal in general -- what's
    orthonormal is the projection onto the mini-batch used to build them.
    """
    k = 2
    bs = 16
    rng = np.random.default_rng(0)
    model = CCA_EY(latent_dimensions=k, batch_size=bs, random_state=0)
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
    w_pls = PLS_EY(latent_dimensions=k)._initial_weights(two_views, rng_pls)
    w_cca = CCA_EY(latent_dimensions=k)._initial_weights(two_views, rng_cca)
    assert any(not np.allclose(a, b) for a, b in zip(w_pls, w_cca))


def test_mcca_ey_initial_weights_give_orthonormal_projections(
    three_correlated_views: list[np.ndarray],
) -> None:
    """MCCA_EY inherits CCA_EY's data-informed initialiser unchanged."""
    k = 2
    bs = 32
    rng = np.random.default_rng(0)
    model = MCCA_EY(latent_dimensions=k, batch_size=bs, random_state=0)
    weights = model._initial_weights(three_correlated_views, rng)
    rng2 = np.random.default_rng(0)
    n = three_correlated_views[0].shape[0]
    idx = rng2.choice(n, bs, replace=False)
    for view, w in zip(three_correlated_views, weights):
        z = view[idx] @ w
        np.testing.assert_allclose(z.T @ z, np.eye(k), atol=1e-8)


# ---------------------------------------------------------------------------
# _post_step hook (lets a proximal-gradient variant subclass without
# duplicating _gradient_descent)
# ---------------------------------------------------------------------------


def test_post_step_default_is_identity(two_views: list[np.ndarray]) -> None:
    """The unmodified base hook does not change fitted weights."""
    baseline = CCA_EY(**_FIT_KWARGS).fit(two_views).weights_

    class _NoOpVariant(CCA_EY):
        def _post_step(self, weights):
            return weights

    variant = _NoOpVariant(**_FIT_KWARGS).fit(two_views).weights_
    for a, b in zip(baseline, variant):
        np.testing.assert_allclose(a, b)


def test_post_step_override_is_applied_during_training(
    two_views: list[np.ndarray],
) -> None:
    """A _post_step override (hard-thresholding) actually runs every
    iteration, not just once at the end -- checked by confirming the
    thresholded entries are exactly zero in the final fit, which a
    single post-hoc clip would also produce, combined with a call
    counter that must fire max_iter times."""
    calls = []

    class _HardThresholdVariant(CCA_EY):
        def _post_step(self, weights):
            calls.append(1)
            return [np.where(np.abs(w) < 0.05, 0.0, w) for w in weights]

    model = _HardThresholdVariant(**_FIT_KWARGS).fit(two_views)
    assert len(calls) == _FIT_KWARGS["max_iter"]
    for w in model.weights_:
        small = np.abs(w) < 0.05
        assert np.all(w[small] == 0.0)
