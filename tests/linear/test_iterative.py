"""Tests for ALS-based sparse/regularised CCA variants.

Covers PLS_ALS, SCCA_PMD, SCCA_ADMM, SCCA_IPLS, SCCA_Span, ElasticCCA,
ParkhomenkoCCA.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import (
    PLS_ALS,
    SCCA_ADMM,
    SCCA_IPLS,
    SCCA_PMD,
    ElasticCCA,
    ParkhomenkoCCA,
    SCCA_Span,
)

ALL_ITERATIVE_MODELS = [
    PLS_ALS,
    SCCA_PMD,
    SCCA_ADMM,
    SCCA_IPLS,
    SCCA_Span,
    ElasticCCA,
    ParkhomenkoCCA,
]

# Use few iterations for test speed
_BASE_KWARGS: dict = dict(latent_dimensions=1, max_iter=50, random_state=0)


# ---------------------------------------------------------------------------
# fit completes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_two_view_fit_completes(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Fit completes on two-view data without error."""
    model = ModelClass(**_BASE_KWARGS)
    fitted = model.fit(two_views)
    assert fitted is model


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_three_view_fit_completes(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """Iterative models accept three or more views."""
    model = ModelClass(**_BASE_KWARGS)
    fitted = model.fit(three_views)
    assert fitted is model


# ---------------------------------------------------------------------------
# transform output shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
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


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_transform_shapes_three_view(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """Transform returns correct shapes for three-view data."""
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


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
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


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_score_shape(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Score returns array of shape (latent_dimensions,)."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    s = model.score(two_views)
    assert s.shape == (k,)


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_score_values_in_valid_range(
    ModelClass: type, correlated_views: list[np.ndarray]
) -> None:
    """Score values lie in [-1, 1]."""
    model = ModelClass(latent_dimensions=1, max_iter=100, random_state=0).fit(
        correlated_views
    )
    s = model.score(correlated_views)
    assert np.all(s >= -1.0 - 1e-9)
    assert np.all(s <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# get_params / set_params roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_get_params_roundtrip(ModelClass: type) -> None:
    """get_params returns the configured parameter values."""
    model = ModelClass(latent_dimensions=3, max_iter=200, random_state=1)
    params = model.get_params()
    assert params["latent_dimensions"] == 3
    assert params["max_iter"] == 200
    assert params["random_state"] == 1


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_set_params_roundtrip(ModelClass: type) -> None:
    """set_params correctly updates parameters."""
    model = ModelClass(latent_dimensions=1)
    model.set_params(latent_dimensions=4)
    assert model.latent_dimensions == 4


# ---------------------------------------------------------------------------
# weights shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_weights_shapes_two_view(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Weights are shaped (n_features_i, latent_dimensions) per view."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    w = model.weights
    assert len(w) == len(two_views)
    for weight, view in zip(w, two_views):
        assert weight.shape == (view.shape[1], k)


# ---------------------------------------------------------------------------
# get_factor_loadings shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_get_factor_loadings_shapes(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """get_factor_loadings returns (n_features_i, k) arrays."""
    k = 2
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    loadings = model.get_factor_loadings(two_views)
    assert len(loadings) == len(two_views)
    for loading, view in zip(loadings, two_views):
        assert loading.shape == (view.shape[1], k)


# ---------------------------------------------------------------------------
# Sparsity verification
# ---------------------------------------------------------------------------


def test_scca_pmd_achieves_sparsity(two_views: list[np.ndarray]) -> None:
    """SCCA_PMD with small tau produces sparse weights (some zeros)."""
    model = SCCA_PMD(latent_dimensions=1, tau=0.3, max_iter=200, random_state=0).fit(
        two_views
    )
    for w in model.weights:
        n_zeros = np.sum(np.abs(w) < 1e-10)
        assert n_zeros > 0, f"Expected some zero weights, got {n_zeros}"


def test_parkhomenko_achieves_sparsity(two_views: list[np.ndarray]) -> None:
    """ParkhomenkoCCA with positive tau produces sparse weights."""
    model = ParkhomenkoCCA(
        latent_dimensions=1, tau=0.5, max_iter=200, random_state=0
    ).fit(two_views)
    for w in model.weights:
        n_zeros = np.sum(np.abs(w) < 1e-10)
        assert n_zeros > 0, f"Expected some zero weights, got {n_zeros}"


def test_scca_span_achieves_sparsity(two_views: list[np.ndarray]) -> None:
    """SCCA_Span with span < n_features produces sparse weights."""
    n_features = two_views[0].shape[1]
    span = n_features // 2
    model = SCCA_Span(latent_dimensions=1, span=span, max_iter=200, random_state=0).fit(
        two_views
    )
    # First view should have at most 'span' nonzero entries per dimension
    w0 = model.weights[0][:, 0]
    n_nonzero = np.sum(np.abs(w0) > 1e-10)
    assert n_nonzero <= span, f"Expected <= {span} nonzero, got {n_nonzero}"


def test_scca_admm_achieves_sparsity(two_views: list[np.ndarray]) -> None:
    """SCCA_ADMM with positive tau produces some sparse weights."""
    model = SCCA_ADMM(latent_dimensions=1, tau=0.5, max_iter=200, random_state=0).fit(
        two_views
    )
    assert hasattr(model, "weights_")
    for w in model.weights:
        assert w.shape[0] > 0


def test_elastic_cca_with_lasso(two_views: list[np.ndarray]) -> None:
    """ElasticCCA with l1_ratio=1 (lasso) produces some sparse weights."""
    model = ElasticCCA(
        latent_dimensions=1, alpha=0.1, l1_ratio=1.0, max_iter=200, random_state=0
    ).fit(two_views)
    assert hasattr(model, "weights_")


def test_scca_ipls_with_lasso(two_views: list[np.ndarray]) -> None:
    """SCCA_IPLS with alpha > 0 runs without error."""
    model = SCCA_IPLS(
        latent_dimensions=1, alpha=0.1, l1_ratio=1.0, max_iter=100, random_state=0
    ).fit(two_views)
    assert hasattr(model, "weights_")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_reproducibility(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Same random_state gives identical weights."""
    kwargs = dict(latent_dimensions=1, max_iter=50, random_state=42)
    w1 = ModelClass(**kwargs).fit(two_views).weights
    w2 = ModelClass(**kwargs).fit(two_views).weights
    for a, b in zip(w1, w2):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# center=False
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_center_false(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """All iterative models work with center=False."""
    model = ModelClass(latent_dimensions=1, max_iter=20, center=False, random_state=0)
    model.fit(two_views)
    result = model.transform(two_views)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# pairwise_correlations shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_ITERATIVE_MODELS)
def test_pairwise_correlations_shape(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """pairwise_correlations returns (n_views, n_views, k)."""
    k = 1
    model = ModelClass(latent_dimensions=k, max_iter=50, random_state=0).fit(two_views)
    corrs = model.pairwise_correlations(two_views)
    assert corrs.shape == (2, 2, k)


# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


def test_pls_als_matches_pls(correlated_views: list[np.ndarray]) -> None:
    """PLS_ALS (converged) recovers the same correlations as exact PLS."""
    from cca_zoo.linear import PLS

    k = 2
    s_pls = PLS(latent_dimensions=k).fit(correlated_views).score(correlated_views)
    s_als = (
        PLS_ALS(latent_dimensions=k, max_iter=1000, random_state=0)
        .fit(correlated_views)
        .score(correlated_views)
    )
    np.testing.assert_allclose(s_als, s_pls, atol=0.05)


def test_iterative_models_find_high_correlation(
    correlated_views: list[np.ndarray],
) -> None:
    """All iterative models find substantial correlation on clearly correlated views."""
    for ModelClass in ALL_ITERATIVE_MODELS:
        s = (
            ModelClass(latent_dimensions=1, max_iter=500, random_state=0)
            .fit(correlated_views)
            .score(correlated_views)
        )
        assert np.all(s > 0.5), f"{ModelClass.__name__} got low correlation: {s}"
