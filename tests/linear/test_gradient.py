"""Tests for gradient-descent CCA variants: PLS_EY, CCA_EY, MCCA_EY."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA_EY, MCCA_EY, PLS_EY

GRADIENT_MODELS_TWO_VIEW = [PLS_EY, CCA_EY]
GRADIENT_MODELS_MULTI_VIEW = [MCCA_EY]
ALL_GRADIENT_MODELS = GRADIENT_MODELS_TWO_VIEW + GRADIENT_MODELS_MULTI_VIEW

# Use fewer iterations for speed in tests
_FIT_KWARGS: dict = dict(latent_dimensions=1, max_iter=50, random_state=0)


# ---------------------------------------------------------------------------
# fit completes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_TWO_VIEW)
def test_two_view_fit_completes(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """fit completes on two-view data without error."""
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
    """transform returns list of (n_samples, latent_dimensions) arrays."""
    k = 2
    model = ModelClass(
        latent_dimensions=k, max_iter=50, random_state=0
    ).fit(two_views)
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
    model = ModelClass(
        latent_dimensions=k, max_iter=50, random_state=0
    ).fit(three_views)
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
    """score returns array of shape (latent_dimensions,)."""
    k = 2
    model = ModelClass(
        latent_dimensions=k, max_iter=50, random_state=0
    ).fit(two_views)
    s = model.score(two_views)
    assert s.shape == (k,)


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_MULTI_VIEW)
def test_score_shape_multi_view(
    ModelClass: type, three_views: list[np.ndarray]
) -> None:
    """score returns array of shape (latent_dimensions,) for multi-view."""
    k = 2
    model = ModelClass(
        latent_dimensions=k, max_iter=50, random_state=0
    ).fit(three_views)
    s = model.score(three_views)
    assert s.shape == (k,)


# ---------------------------------------------------------------------------
# get_params / set_params roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_get_params_roundtrip(ModelClass: type) -> None:
    """get_params returns correct parameter values."""
    model = ModelClass(latent_dimensions=3, max_iter=100, random_state=7)
    params = model.get_params()
    assert params["latent_dimensions"] == 3
    assert params["max_iter"] == 100
    assert params["random_state"] == 7


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_set_params_roundtrip(ModelClass: type) -> None:
    """set_params correctly updates parameters."""
    model = ModelClass(latent_dimensions=1)
    model.set_params(latent_dimensions=4)
    assert model.latent_dimensions == 4


# ---------------------------------------------------------------------------
# weights shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_TWO_VIEW)
def test_weights_shapes_two_view(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """weights shapes are (n_features_i, latent_dimensions) per view."""
    k = 2
    model = ModelClass(
        latent_dimensions=k, max_iter=50, random_state=0
    ).fit(two_views)
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
    model = ModelClass(
        latent_dimensions=k, max_iter=50, random_state=0
    ).fit(two_views)
    loadings = model.get_factor_loadings(two_views)
    assert len(loadings) == len(two_views)
    for loading, view in zip(loadings, two_views):
        assert loading.shape == (view.shape[1], k)


# ---------------------------------------------------------------------------
# Mini-batch training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_TWO_VIEW)
def test_mini_batch_training(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """Models train without error with batch_size=16."""
    model = ModelClass(
        latent_dimensions=1, max_iter=20, batch_size=16, random_state=0
    )
    model.fit(two_views)
    result = model.transform(two_views)
    assert len(result) == 2
    for arr, view in zip(result, two_views):
        assert arr.shape == (view.shape[0], 1)


@pytest.mark.parametrize("ModelClass", GRADIENT_MODELS_TWO_VIEW)
def test_full_batch_training(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
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
# CCA_EY: c parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [0.0, 0.1, 0.5])
def test_cca_ey_c_parameter(c: float, two_views: list[np.ndarray]) -> None:
    """CCA_EY runs without error for different values of c."""
    model = CCA_EY(latent_dimensions=1, max_iter=20, c=c, random_state=0)
    model.fit(two_views)
    assert hasattr(model, "weights_")


# ---------------------------------------------------------------------------
# center=False
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_GRADIENT_MODELS)
def test_center_false(
    ModelClass: type, two_views: list[np.ndarray]
) -> None:
    """All gradient models work with center=False."""
    model = ModelClass(
        latent_dimensions=1, max_iter=20, center=False, random_state=0
    )
    model.fit(two_views)
    result = model.transform(two_views)
    assert len(result) == 2
