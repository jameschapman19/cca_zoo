"""Tests for eigendecomposition-based linear CCA methods.

Covers CCA, rCCA, PLS, MCCA, GCCA, TCCA.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA, GCCA, MCCA, PLS, TCCA, rCCA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TWO_VIEW_MODELS = [CCA, rCCA, PLS]
MULTI_VIEW_MODELS = [MCCA, GCCA, TCCA]
ALL_EIGEN_MODELS = TWO_VIEW_MODELS + MULTI_VIEW_MODELS

# Models that accept random_state
_MODELS_WITH_RANDOM_STATE = {TCCA}


def _make_multi_view_model(ModelClass: type, latent_dimensions: int = 1) -> object:
    """Construct a multi-view model, passing random_state only if supported."""
    if ModelClass in _MODELS_WITH_RANDOM_STATE:
        return ModelClass(latent_dimensions=latent_dimensions, random_state=0)
    return ModelClass(latent_dimensions=latent_dimensions)


# ---------------------------------------------------------------------------
# Two-view fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_EIGEN_MODELS)
def test_two_view_fit_completes(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Fit completes without error on two-view data."""
    model = ModelClass(latent_dimensions=1)
    fitted = model.fit(two_views)
    assert fitted is model


# ---------------------------------------------------------------------------
# Three-view fit (multi-view models only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", [MCCA, GCCA])
def test_three_view_fit_completes_no_random_state(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """Fit completes without error on three-view data (MCCA/GCCA)."""
    model = ModelClass(latent_dimensions=1)
    fitted = model.fit(three_views)
    assert fitted is model


def test_tcca_three_view_fit_completes(three_views: list[np.ndarray]) -> None:
    """TCCA fit completes without error on three-view data."""
    model = TCCA(latent_dimensions=1, random_state=0)
    fitted = model.fit(three_views)
    assert fitted is model


@pytest.mark.parametrize("ModelClass", [CCA, rCCA, PLS])
def test_two_view_models_reject_three_views(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """CCA, rCCA, PLS raise ValueError when given 3 views."""
    with pytest.raises(ValueError):
        ModelClass(latent_dimensions=1).fit(three_views)


# ---------------------------------------------------------------------------
# transform output shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", TWO_VIEW_MODELS)
def test_two_view_transform_shapes(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """Transform returns list of (n_samples, latent_dimensions) arrays."""
    k = 2
    model = ModelClass(latent_dimensions=k).fit(two_views)
    result = model.transform(two_views)
    assert len(result) == len(two_views)
    for arr, view in zip(result, two_views):
        assert arr.shape == (view.shape[0], k)


@pytest.mark.parametrize("ModelClass", MULTI_VIEW_MODELS)
def test_multi_view_transform_shapes(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """Transform on 3 views returns list of (n_samples, latent_dimensions) arrays."""
    k = 2
    model = _make_multi_view_model(ModelClass, latent_dimensions=k).fit(three_views)
    result = model.transform(three_views)
    assert len(result) == len(three_views)
    for arr, view in zip(result, three_views):
        assert arr.shape == (view.shape[0], k)


# ---------------------------------------------------------------------------
# fit_transform == fit().transform()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", TWO_VIEW_MODELS)
def test_fit_transform_consistency_two_view(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """fit_transform output equals fit().transform() numerically."""
    result_ft = ModelClass(latent_dimensions=1).fit_transform(two_views)
    result_sep = ModelClass(latent_dimensions=1).fit(two_views).transform(two_views)
    for a, b in zip(result_ft, result_sep):
        np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-10)


@pytest.mark.parametrize("ModelClass", MULTI_VIEW_MODELS)
def test_fit_transform_consistency_multi_view(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """fit_transform equals fit().transform() for multi-view models."""
    result_ft = _make_multi_view_model(ModelClass, latent_dimensions=1).fit_transform(
        three_views
    )
    result_sep = (
        _make_multi_view_model(ModelClass, latent_dimensions=1)
        .fit(three_views)
        .transform(three_views)
    )
    for a, b in zip(result_ft, result_sep):
        np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-10)


# ---------------------------------------------------------------------------
# score shape and range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", TWO_VIEW_MODELS)
def test_score_shape_two_view(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Score returns array of shape (latent_dimensions,)."""
    k = 2
    model = ModelClass(latent_dimensions=k).fit(two_views)
    s = model.score(two_views)
    assert s.shape == (k,)


@pytest.mark.parametrize("ModelClass", TWO_VIEW_MODELS)
def test_score_values_in_valid_range_two_view(
    ModelClass: type, correlated_views: list[np.ndarray]
) -> None:
    """Score values are in [-1, 1]."""
    model = ModelClass(latent_dimensions=2).fit(correlated_views)
    s = model.score(correlated_views)
    assert np.all(s >= -1.0 - 1e-9)
    assert np.all(s <= 1.0 + 1e-9)


@pytest.mark.parametrize("ModelClass", MULTI_VIEW_MODELS)
def test_score_shape_multi_view(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """Score returns array of shape (latent_dimensions,) for multi-view models."""
    k = 2
    model = _make_multi_view_model(ModelClass, latent_dimensions=k).fit(three_views)
    s = model.score(three_views)
    assert s.shape == (k,)


# ---------------------------------------------------------------------------
# get_params / set_params roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_EIGEN_MODELS)
def test_get_params_roundtrip(ModelClass: type) -> None:
    """get_params returns the correct parameter values."""
    model = ModelClass(latent_dimensions=3)
    params = model.get_params()
    assert params["latent_dimensions"] == 3


@pytest.mark.parametrize("ModelClass", ALL_EIGEN_MODELS)
def test_set_params_roundtrip(ModelClass: type) -> None:
    """set_params correctly updates model parameters."""
    model = ModelClass(latent_dimensions=1)
    model.set_params(latent_dimensions=3)
    assert model.latent_dimensions == 3


# ---------------------------------------------------------------------------
# weights shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", TWO_VIEW_MODELS)
def test_weights_shape_two_view(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """Weights has correct shapes (n_features_i, latent_dimensions) per view."""
    k = 2
    model = ModelClass(latent_dimensions=k).fit(two_views)
    w = model.weights
    assert len(w) == len(two_views)
    for weight, view in zip(w, two_views):
        assert weight.shape == (view.shape[1], k)


@pytest.mark.parametrize("ModelClass", MULTI_VIEW_MODELS)
def test_weights_shape_multi_view(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """Weights shapes are correct for multi-view models."""
    k = 2
    model = _make_multi_view_model(ModelClass, latent_dimensions=k).fit(three_views)
    w = model.weights
    assert len(w) == len(three_views)
    for weight, view in zip(w, three_views):
        assert weight.shape == (view.shape[1], k)


# ---------------------------------------------------------------------------
# get_factor_loadings shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", TWO_VIEW_MODELS)
def test_get_factor_loadings_shapes_two_view(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """get_factor_loadings returns one (n_features_i, latent_dims) array per view."""
    k = 2
    model = ModelClass(latent_dimensions=k).fit(two_views)
    loadings = model.get_factor_loadings(two_views)
    assert len(loadings) == len(two_views)
    for loading, view in zip(loadings, two_views):
        assert loading.shape == (view.shape[1], k)


@pytest.mark.parametrize("ModelClass", MULTI_VIEW_MODELS)
def test_get_factor_loadings_shapes_multi_view(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """get_factor_loadings shapes are correct for multi-view models."""
    k = 2
    model = _make_multi_view_model(ModelClass, latent_dimensions=k).fit(three_views)
    loadings = model.get_factor_loadings(three_views)
    assert len(loadings) == len(three_views)
    for loading, view in zip(loadings, three_views):
        assert loading.shape == (view.shape[1], k)


# ---------------------------------------------------------------------------
# Multiple latent dimensions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", TWO_VIEW_MODELS)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_multiple_latent_dimensions(
    ModelClass: type, k: int, two_views: list[np.ndarray]
) -> None:
    """Models work correctly for various latent_dimensions values."""
    model = ModelClass(latent_dimensions=k).fit(two_views)
    result = model.transform(two_views)
    for arr, view in zip(result, two_views):
        assert arr.shape == (view.shape[0], k)


# ---------------------------------------------------------------------------
# rCCA-specific: c parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [0.0, 0.1, 0.5, 1.0])
def test_rcca_c_parameter(c: float, two_views: list[np.ndarray]) -> None:
    """RCCA works for various values of the ridge parameter c."""
    model = rCCA(latent_dimensions=1, c=c).fit(two_views)
    result = model.transform(two_views)
    assert len(result) == 2


def test_rcca_per_view_c_parameter(two_views: list[np.ndarray]) -> None:
    """RCCA accepts per-view c=[c1, c2]."""
    model = rCCA(latent_dimensions=1, c=[0.1, 0.3]).fit(two_views)
    result = model.transform(two_views)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# MCCA-specific: c parameter and pca flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pca", [True, False])
def test_mcca_pca_flag(pca: bool, two_views: list[np.ndarray]) -> None:
    """MCCA works with both pca=True and pca=False."""
    model = MCCA(latent_dimensions=1, pca=pca).fit(two_views)
    result = model.transform(two_views)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# GCCA-specific: view_weights
# ---------------------------------------------------------------------------


def test_gcca_view_weights(three_views: list[np.ndarray]) -> None:
    """GCCA accepts per-view weights."""
    model = GCCA(latent_dimensions=1, view_weights=[1.0, 1.0, 2.0]).fit(three_views)
    result = model.transform(three_views)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# TCCA-specific: random_state reproducibility
# ---------------------------------------------------------------------------


def test_tcca_reproducibility(three_views: list[np.ndarray]) -> None:
    """TCCA with same random_state gives identical weights."""
    w1 = TCCA(latent_dimensions=1, random_state=42).fit(three_views).weights
    w2 = TCCA(latent_dimensions=1, random_state=42).fit(three_views).weights
    for a, b in zip(w1, w2):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# CCA correlated views: score should be high
# ---------------------------------------------------------------------------


def test_cca_high_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """CCA should find high correlation on views sharing a latent structure."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    s = model.score(correlated_views)
    assert np.all(s > 0.5), f"Expected high correlation, got {s}"


# ---------------------------------------------------------------------------
# center=False runs without error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", TWO_VIEW_MODELS)
def test_center_false_two_view(ModelClass: type, two_views: list[np.ndarray]) -> None:
    """All two-view models run correctly with center=False."""
    model = ModelClass(latent_dimensions=1, center=False).fit(two_views)
    result = model.transform(two_views)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# pairwise_correlations shape
# ---------------------------------------------------------------------------


def test_pairwise_correlations_shape_two_view(
    two_views: list[np.ndarray],
) -> None:
    """pairwise_correlations returns shape (n_views, n_views, k) for two views."""
    k = 2
    model = CCA(latent_dimensions=k).fit(two_views)
    corrs = model.pairwise_correlations(two_views)
    assert corrs.shape == (2, 2, k)


def test_pairwise_correlations_shape_three_view(
    three_views: list[np.ndarray],
) -> None:
    """pairwise_correlations returns shape (n_views, n_views, k) for three views."""
    k = 2
    model = MCCA(latent_dimensions=k).fit(three_views)
    corrs = model.pairwise_correlations(three_views)
    assert corrs.shape == (3, 3, k)
