"""Tests for nonparametric (kernel-based) CCA methods: KCCA, KGCCA, KTCCA."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA
from cca_zoo.nonparametric import KCCA, KGCCA, KTCCA

ALL_KERNEL_MODELS = [KCCA, KGCCA, KTCCA]

# Only KTCCA has a random_state parameter
_KERNEL_MODELS_WITH_RANDOM_STATE = {KTCCA}


def _make_kernel_model(
    ModelClass: type,
    latent_dimensions: int = 1,
    c: float = 0.1,
    **kwargs: object,
) -> object:
    """Construct a kernel model, passing random_state=0 only if supported."""
    if ModelClass in _KERNEL_MODELS_WITH_RANDOM_STATE:
        return ModelClass(
            latent_dimensions=latent_dimensions, c=c, random_state=0, **kwargs
        )
    return ModelClass(latent_dimensions=latent_dimensions, c=c, **kwargs)


# ---------------------------------------------------------------------------
# fit completes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_KERNEL_MODELS)
def test_two_view_fit_completes(
    ModelClass: type, two_views_small: list[np.ndarray]
) -> None:
    """Fit completes on two-view data without error."""
    model = ModelClass(latent_dimensions=1, c=0.1)
    fitted = model.fit(two_views_small)
    assert fitted is model


@pytest.mark.parametrize("ModelClass", ALL_KERNEL_MODELS)
def test_three_view_fit_completes(
    ModelClass: type, three_views_small: list[np.ndarray]
) -> None:
    """Fit completes on three-view data without error."""
    model = _make_kernel_model(ModelClass, latent_dimensions=1, c=0.1)
    fitted = model.fit(three_views_small)
    assert fitted is model


# ---------------------------------------------------------------------------
# transform output shapes (training data)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_KERNEL_MODELS)
def test_transform_shapes_training_data(
    ModelClass: type, two_views_small: list[np.ndarray]
) -> None:
    """Transform on training data returns (n_samples, latent_dimensions) arrays."""
    k = 2
    model = _make_kernel_model(ModelClass, latent_dimensions=k).fit(two_views_small)
    result = model.transform(two_views_small)
    assert len(result) == len(two_views_small)
    n = two_views_small[0].shape[0]
    for arr in result:
        assert arr.shape == (n, k)


# ---------------------------------------------------------------------------
# transform on held-out test data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", [KCCA, KGCCA, KTCCA])
def test_transform_on_test_data(
    ModelClass: type, two_views_small: list[np.ndarray]
) -> None:
    """Transform returns correct shapes for new (unseen) test samples."""
    rng = np.random.default_rng(99)
    test_views = [rng.standard_normal((10, 5)), rng.standard_normal((10, 5))]
    k = 1
    model = _make_kernel_model(ModelClass, latent_dimensions=k).fit(two_views_small)
    result = model.transform(test_views)
    assert len(result) == 2
    for arr in result:
        assert arr.shape == (10, k)


# ---------------------------------------------------------------------------
# fit_transform consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_KERNEL_MODELS)
def test_fit_transform_consistency(
    ModelClass: type, two_views_small: list[np.ndarray]
) -> None:
    """fit_transform equals fit().transform() numerically."""
    m1 = _make_kernel_model(ModelClass, latent_dimensions=1)
    m2 = _make_kernel_model(ModelClass, latent_dimensions=1)
    result_ft = m1.fit_transform(two_views_small)
    result_sep = m2.fit(two_views_small).transform(two_views_small)
    for a, b in zip(result_ft, result_sep):
        np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-10)


# ---------------------------------------------------------------------------
# score shape and range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_KERNEL_MODELS)
def test_score_shape(ModelClass: type, two_views_small: list[np.ndarray]) -> None:
    """Score returns array of shape (latent_dimensions,)."""
    k = 2
    model = _make_kernel_model(ModelClass, latent_dimensions=k).fit(two_views_small)
    s = model.score(two_views_small)
    assert s.shape == (k,)


@pytest.mark.parametrize("ModelClass", ALL_KERNEL_MODELS)
def test_score_values_in_range(
    ModelClass: type, two_views_small: list[np.ndarray]
) -> None:
    """Score values lie in [-1, 1]."""
    model = _make_kernel_model(ModelClass, latent_dimensions=1).fit(two_views_small)
    s = model.score(two_views_small)
    assert np.all(s >= -1.0 - 1e-9)
    assert np.all(s <= 1.0 + 1e-9)


# get_params/set_params roundtrip behaviour is exercised generically for
# every model in the package by tests/test_sklearn_compat.py.


# ---------------------------------------------------------------------------
# weights shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_KERNEL_MODELS)
def test_weights_shapes(ModelClass: type, two_views_small: list[np.ndarray]) -> None:
    """Kernel model weights are dual variables of shape (n_samples, k)."""
    k = 2
    model = _make_kernel_model(ModelClass, latent_dimensions=k).fit(two_views_small)
    w = model.weights
    n = two_views_small[0].shape[0]
    assert len(w) == len(two_views_small)
    for weight in w:
        assert weight.shape == (n, k)


# ---------------------------------------------------------------------------
# get_factor_loadings shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_KERNEL_MODELS)
def test_get_factor_loadings_shapes(
    ModelClass: type, two_views_small: list[np.ndarray]
) -> None:
    """get_factor_loadings returns (n_features_i, k) arrays."""
    k = 2
    model = _make_kernel_model(ModelClass, latent_dimensions=k).fit(two_views_small)
    loadings = model.get_factor_loadings(two_views_small)
    assert len(loadings) == len(two_views_small)
    for loading, view in zip(loadings, two_views_small):
        assert loading.shape == (view.shape[1], k)


# ---------------------------------------------------------------------------
# Different kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_kcca_different_kernels(kernel: str, two_views_small: list[np.ndarray]) -> None:
    """KCCA works with linear, rbf, and polynomial kernels."""
    model = KCCA(latent_dimensions=1, c=0.1, kernel=kernel).fit(two_views_small)
    result = model.transform(two_views_small)
    n = two_views_small[0].shape[0]
    for arr in result:
        assert arr.shape == (n, 1)


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_kgcca_different_kernels(
    kernel: str, two_views_small: list[np.ndarray]
) -> None:
    """KGCCA works with linear and rbf kernels."""
    model = KGCCA(latent_dimensions=1, c=0.1, kernel=kernel).fit(two_views_small)
    result = model.transform(two_views_small)
    n = two_views_small[0].shape[0]
    for arr in result:
        assert arr.shape == (n, 1)


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
def test_ktcca_different_kernels(
    kernel: str, two_views_small: list[np.ndarray]
) -> None:
    """KTCCA works with linear and rbf kernels."""
    model = KTCCA(latent_dimensions=1, c=0.1, kernel=kernel, random_state=0).fit(
        two_views_small
    )
    result = model.transform(two_views_small)
    n = two_views_small[0].shape[0]
    for arr in result:
        assert arr.shape == (n, 1)


# ---------------------------------------------------------------------------
# Per-view kernel specification
# ---------------------------------------------------------------------------


def test_kcca_per_view_kernel(two_views_small: list[np.ndarray]) -> None:
    """KCCA accepts per-view kernel specification as a list."""
    model = KCCA(latent_dimensions=1, c=0.1, kernel=["linear", "rbf"]).fit(
        two_views_small
    )
    result = model.transform(two_views_small)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# center=False
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_KERNEL_MODELS)
def test_center_false(ModelClass: type, two_views_small: list[np.ndarray]) -> None:
    """All kernel models work with center=False."""
    model = _make_kernel_model(ModelClass, latent_dimensions=1, center=False)
    model.fit(two_views_small)
    result = model.transform(two_views_small)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# pairwise_correlations shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ModelClass", ALL_KERNEL_MODELS)
def test_pairwise_correlations_shape(
    ModelClass: type, two_views_small: list[np.ndarray]
) -> None:
    """pairwise_correlations returns (n_views, n_views, k)."""
    k = 1
    model = _make_kernel_model(ModelClass, latent_dimensions=k).fit(two_views_small)
    corrs = model.pairwise_correlations(two_views_small)
    assert corrs.shape == (2, 2, k)


# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


def test_kcca_linear_kernel_matches_cca(correlated_views: list[np.ndarray]) -> None:
    """KCCA with a linear kernel recovers the same correlations as CCA (small c)."""
    k = 2
    s_cca = CCA(latent_dimensions=k).fit(correlated_views).score(correlated_views)
    s_kcca = (
        KCCA(latent_dimensions=k, kernel="linear", c=1e-4)
        .fit(correlated_views)
        .score(correlated_views)
    )
    np.testing.assert_allclose(s_kcca, s_cca, atol=1e-3)


def test_kcca_finds_high_correlation(correlated_views: list[np.ndarray]) -> None:
    """KCCA with rbf kernel finds high correlation on clearly correlated views."""
    s = (
        KCCA(latent_dimensions=1, c=0.01, kernel="rbf")
        .fit(correlated_views)
        .score(correlated_views)
    )
    assert np.all(s > 0.8), f"Expected high correlation, got {s}"


def test_kcca_regularisation_reduces_correlation(
    correlated_views: list[np.ndarray],
) -> None:
    """Higher regularisation c gives lower (or equal) training correlation."""
    s_low = (
        KCCA(latent_dimensions=1, kernel="linear", c=1e-4)
        .fit(correlated_views)
        .score(correlated_views)
    )
    s_high = (
        KCCA(latent_dimensions=1, kernel="linear", c=10.0)
        .fit(correlated_views)
        .score(correlated_views)
    )
    assert s_low[0] >= s_high[0] - 1e-6
