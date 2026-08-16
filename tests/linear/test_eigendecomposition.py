"""Tests for eigendecomposition-based linear CCA methods.

Covers CCA, rCCA, PLS, MCCA, GCCA, TCCA.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA, CCAR3, GCCA, GRCCA, MCCA, PLS, TCCA, PartialCCA, rCCA

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


# get_params/set_params roundtrip behaviour (sklearn.BaseEstimator machinery,
# unmodified by cca_zoo) is exercised generically for every model in the
# package by tests/test_sklearn_compat.py, rather than per-class here.


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
    """CCA finds near-perfect correlation on low-noise correlated views (SNR ~10)."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    s = model.score(correlated_views)
    assert np.all(s > 0.95), f"Expected near-perfect correlation, got {s}"


def test_cca_perfect_correlation_identical_views() -> None:
    """CCA on identical views (X1 == X2) should give correlation == 1.0."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 5))
    model = CCA(latent_dimensions=3).fit([x, x])
    s = model.score([x, x])
    np.testing.assert_allclose(s, 1.0, atol=1e-6)


def test_cca_correlations_are_decreasing(correlated_views: list[np.ndarray]) -> None:
    """Canonical correlations are returned in non-increasing order."""
    s = CCA(latent_dimensions=2).fit(correlated_views).score(correlated_views)
    assert s[0] >= s[1] - 1e-10


def test_rcca_zero_regularisation_matches_cca(
    correlated_views: list[np.ndarray],
) -> None:
    """RCCA with c=0 should give the same correlations as CCA."""
    s_cca = CCA(latent_dimensions=2).fit(correlated_views).score(correlated_views)
    s_rcca = (
        rCCA(latent_dimensions=2, c=0.0).fit(correlated_views).score(correlated_views)
    )
    np.testing.assert_allclose(s_rcca, s_cca, atol=1e-6)


@pytest.mark.parametrize("ModelClass", [CCA, PLS, MCCA, GCCA])
def test_model_finds_high_correlation_on_correlated_views(
    ModelClass: type, correlated_views: list[np.ndarray]
) -> None:
    """All unregularised models find high correlation on clearly correlated views."""
    model = _make_multi_view_model(ModelClass, latent_dimensions=2).fit(
        correlated_views
    )
    s = model.score(correlated_views)
    assert np.all(s > 0.8), f"{ModelClass.__name__} got low correlation: {s}"


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


# ---------------------------------------------------------------------------
# Mathematical properties / correctness
# ---------------------------------------------------------------------------


def test_mcca_two_views_matches_cca(correlated_views: list[np.ndarray]) -> None:
    """MCCA with two views is equivalent to CCA (same canonical correlations)."""
    k = 2
    s_cca = CCA(latent_dimensions=k).fit(correlated_views).score(correlated_views)
    s_mcca = MCCA(latent_dimensions=k).fit(correlated_views).score(correlated_views)
    np.testing.assert_allclose(s_mcca, s_cca, atol=1e-6)


def test_cca_canonical_variates_are_uncorrelated() -> None:
    """CCA canonical variates are orthogonal across dimensions."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 10))
    k = 3
    model = CCA(latent_dimensions=k).fit([x, x])
    Z = model.transform([x, x])
    for z in Z:
        corr = np.corrcoef(z.T)
        off_diag = corr - np.eye(k)
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-6)


def test_tcca_finds_high_correlation(correlated_views: list[np.ndarray]) -> None:
    """TCCA finds high correlation on clearly correlated views."""
    s = (
        TCCA(latent_dimensions=1, random_state=0)
        .fit(correlated_views)
        .score(correlated_views)
    )
    assert np.all(s > 0.8), f"Expected high correlation, got {s}"


# ---------------------------------------------------------------------------
# PartialCCA
# ---------------------------------------------------------------------------


def test_partial_cca_fit_transform(two_views: list[np.ndarray]) -> None:
    """PartialCCA fits and transforms with confound variables removed."""
    rng = np.random.default_rng(1)
    partials = rng.standard_normal((50, 3))
    k = 2
    model = PartialCCA(latent_dimensions=k).fit(two_views, partials=partials)
    result = model.transform(two_views, partials=partials)
    assert len(result) == 2
    for arr, view in zip(result, two_views):
        assert arr.shape == (view.shape[0], k)


def test_partial_cca_requires_partials(two_views: list[np.ndarray]) -> None:
    """PartialCCA.fit raises ValueError when partials is not provided."""
    with pytest.raises(ValueError, match="partials"):
        PartialCCA(latent_dimensions=1).fit(two_views)


def test_partial_cca_transform_without_partials_falls_back(
    two_views: list[np.ndarray],
) -> None:
    """PartialCCA.transform without partials falls back to a plain projection."""
    rng = np.random.default_rng(1)
    partials = rng.standard_normal((50, 3))
    model = PartialCCA(latent_dimensions=1).fit(two_views, partials=partials)
    result = model.transform(two_views)
    assert len(result) == 2


def test_partial_cca_score_and_fit_transform(two_views: list[np.ndarray]) -> None:
    """PartialCCA supports score() and fit_transform() with partials."""
    rng = np.random.default_rng(1)
    partials = rng.standard_normal((50, 3))
    model = PartialCCA(latent_dimensions=2)
    ft = model.fit_transform(two_views, partials=partials)
    fit_then_transform = (
        PartialCCA(latent_dimensions=2)
        .fit(two_views, partials=partials)
        .transform(two_views, partials=partials)
    )
    for a, b in zip(ft, fit_then_transform):
        np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-10)
    s = model.score(two_views)
    assert s.shape == (2,)


def test_partial_cca_removes_confound_effect() -> None:
    """PartialCCA recovers a shared signal even when a strong confound dominates."""
    rng = np.random.default_rng(0)
    n = 200
    z = rng.standard_normal((n, 2))
    confound = rng.standard_normal((n, 1))
    x1 = (
        z @ rng.standard_normal((2, 6))
        + confound @ rng.standard_normal((1, 6)) * 5.0
        + 0.1 * rng.standard_normal((n, 6))
    )
    x2 = (
        z @ rng.standard_normal((2, 6))
        + confound @ rng.standard_normal((1, 6)) * 5.0
        + 0.1 * rng.standard_normal((n, 6))
    )
    model = PartialCCA(latent_dimensions=2).fit([x1, x2], partials=confound)
    z1, z2 = model.transform([x1, x2], partials=confound)
    corrs = np.array(
        [np.corrcoef(z1[:, d], z2[:, d])[0, 1] for d in range(z1.shape[1])]
    )
    assert np.all(corrs > 0.5), (
        f"Expected residual correlation after deconfounding, got {corrs}"
    )


# ---------------------------------------------------------------------------
# GRCCA
# ---------------------------------------------------------------------------


def test_grcca_fit_transform(two_views: list[np.ndarray]) -> None:
    """GRCCA fits and transforms using feature groups."""
    rng = np.random.default_rng(2)
    groups1 = rng.integers(0, 3, size=two_views[0].shape[1])
    groups2 = rng.integers(0, 3, size=two_views[1].shape[1])
    k = 2
    model = GRCCA(latent_dimensions=k, c=0.5).fit(
        two_views, feature_groups=[groups1, groups2]
    )
    result = model.transform(two_views)
    assert len(result) == 2
    for arr, view in zip(result, two_views):
        assert arr.shape == (view.shape[0], k)


def test_grcca_weights_shape_matches_original_features(
    two_views: list[np.ndarray],
) -> None:
    """GRCCA weights_ operate on the original (un-augmented) feature space."""
    rng = np.random.default_rng(2)
    groups1 = rng.integers(0, 3, size=two_views[0].shape[1])
    groups2 = rng.integers(0, 3, size=two_views[1].shape[1])
    model = GRCCA(latent_dimensions=1, c=[0.5, 0.0]).fit(
        two_views, feature_groups=[groups1, groups2]
    )
    for w, view in zip(model.weights_, two_views):
        assert w.shape == (view.shape[1], 1)


def test_grcca_zero_c_matches_mcca(two_views: list[np.ndarray]) -> None:
    """GRCCA with c=0 reduces to plain MCCA."""
    k = 2
    s_grcca = GRCCA(latent_dimensions=k, c=0.0).fit(two_views).score(two_views)
    s_mcca = MCCA(latent_dimensions=k, pca=False).fit(two_views).score(two_views)
    np.testing.assert_allclose(s_grcca, s_mcca, atol=1e-6)


def test_grcca_default_feature_groups_warns_when_c_nonzero(
    two_views: list[np.ndarray],
) -> None:
    """GRCCA warns when c>0 but no feature_groups are provided."""
    with pytest.warns(UserWarning, match="feature_groups"):
        GRCCA(latent_dimensions=1, c=0.5).fit(two_views)


def test_grcca_three_views(three_views: list[np.ndarray]) -> None:
    """GRCCA fits on more than two views."""
    rng = np.random.default_rng(3)
    groups = [rng.integers(0, 2, size=v.shape[1]) for v in three_views]
    model = GRCCA(latent_dimensions=1, c=0.3).fit(three_views, feature_groups=groups)
    result = model.transform(three_views)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# CCAR3
# ---------------------------------------------------------------------------


def test_ccar3_fit_transform(two_views: list[np.ndarray]) -> None:
    """CCAR3 fits and transforms in both the low-dim and high-dim regimes."""
    k = 2
    for highdim in (False, True):
        model = CCAR3(latent_dimensions=k, highdim=highdim).fit(two_views)
        result = model.transform(two_views)
        assert len(result) == 2
        for arr, view in zip(result, two_views):
            assert arr.shape == (view.shape[0], k)


def test_ccar3_rejects_three_views(three_views: list[np.ndarray]) -> None:
    """CCAR3 raises ValueError when given 3 views."""
    with pytest.raises(ValueError):
        CCAR3(latent_dimensions=1).fit(three_views)


def test_ccar3_highdim_zero_penalty_matches_lowdim(
    correlated_views: list[np.ndarray],
) -> None:
    """With lambda_=0, the ADMM solver converges to the closed-form B."""
    k = 2
    s_lowdim = (
        CCAR3(latent_dimensions=k, highdim=False, ledoit_wolf=False)
        .fit(correlated_views)
        .score(correlated_views)
    )
    s_highdim = (
        CCAR3(
            latent_dimensions=k,
            highdim=True,
            ledoit_wolf=False,
            lambda_=0.0,
            tol=1e-8,
        )
        .fit(correlated_views)
        .score(correlated_views)
    )
    np.testing.assert_allclose(s_highdim, s_lowdim, atol=1e-4)


def test_ccar3_sparsity_zeroes_rows(two_views: list[np.ndarray]) -> None:
    """A moderate lambda_ drives some rows of the X weights to zero, not all."""
    model = CCAR3(
        latent_dimensions=2,
        highdim=True,
        lambda_=0.3,
        ledoit_wolf=False,
        tol=1e-8,
    ).fit(two_views)
    row_norms = np.linalg.norm(model.weights_[0], axis=1)
    assert np.any(row_norms < 1e-3)
    assert np.any(row_norms > 1e-2)


def test_ccar3_sparsity_at_default_tol(correlated_views: list[np.ndarray]) -> None:
    """Regression test for the ADMM solver's returned variable.

    The solver must return the row-thresholded variable (Z), not the smooth
    working variable (B), or `lambda_` has no effect on sparsity at the
    default `tol=1e-4` -- B only converges *towards* Z within `tol` in an
    aggregate Frobenius sense, so individual rows of B stay generically
    nonzero (just small) well past that tolerance, while Z has exact zero
    rows by construction of the group soft-threshold applied to it. Uses the
    library's default `tol` deliberately, unlike test_ccar3_sparsity_zeroes_rows
    above (tol=1e-8): that tighter tolerance can mask this bug by converging
    B close enough to Z anyway.
    """
    dense = CCAR3(latent_dimensions=2, lambda_=0.0, ledoit_wolf=False).fit(
        correlated_views
    )
    sparse = CCAR3(latent_dimensions=2, lambda_=0.05, ledoit_wolf=False).fit(
        correlated_views
    )

    dense_row_norms = np.linalg.norm(dense.weights_[0], axis=1)
    sparse_row_norms = np.linalg.norm(sparse.weights_[0], axis=1)

    assert np.all(dense_row_norms > 1e-8), "lambda_=0 should not zero any rows"
    assert np.any(sparse_row_norms == 0.0), "a moderate lambda_ should zero some rows"
    assert np.any(sparse_row_norms > 1e-8), "and leave others exactly as nonzero"


def test_ccar3_score_shape(two_views: list[np.ndarray]) -> None:
    """CCAR3's score returns an array of shape (latent_dimensions,)."""
    k = 2
    model = CCAR3(latent_dimensions=k).fit(two_views)
    s = model.score(two_views)
    assert s.shape == (k,)
