"""Tests for ElasticNetCCA."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.sparse import ElasticNetCCA


def _make_model(latent_dimensions: int = 1, **kwargs: object) -> ElasticNetCCA:
    return ElasticNetCCA(latent_dimensions=latent_dimensions, **kwargs)


# ---------------------------------------------------------------------------
# fit completes
# ---------------------------------------------------------------------------


def test_two_view_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes on two-view data without error."""
    model = _make_model()
    fitted = model.fit(two_views_small)
    assert fitted is model


def test_three_view_fit_completes(three_views_small: list[np.ndarray]) -> None:
    """Fit completes on three-view data without error."""
    model = _make_model()
    fitted = model.fit(three_views_small)
    assert fitted is model


# ---------------------------------------------------------------------------
# transform output shapes
# ---------------------------------------------------------------------------


def test_transform_shapes_training_data(two_views_small: list[np.ndarray]) -> None:
    """Transform on training data returns (n_samples, latent_dimensions) arrays."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    result = model.transform(two_views_small)
    assert len(result) == 2
    n = two_views_small[0].shape[0]
    for arr in result:
        assert arr.shape == (n, k)


def test_transform_on_test_data(two_views_small: list[np.ndarray]) -> None:
    """Transform returns correct shapes for new (unseen) test samples."""
    rng = np.random.default_rng(99)
    test_views = [rng.standard_normal((10, 5)), rng.standard_normal((10, 5))]
    k = 1
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    result = model.transform(test_views)
    for arr in result:
        assert arr.shape == (10, k)


# ---------------------------------------------------------------------------
# fit_transform consistency
# ---------------------------------------------------------------------------


def test_fit_transform_consistency(two_views_small: list[np.ndarray]) -> None:
    """fit_transform equals fit().transform() numerically."""
    m1 = _make_model(random_state=0)
    m2 = _make_model(random_state=0)
    result_ft = m1.fit_transform(two_views_small)
    result_sep = m2.fit(two_views_small).transform(two_views_small)
    for a, b in zip(result_ft, result_sep):
        np.testing.assert_allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# score shape and range
# ---------------------------------------------------------------------------


def test_score_shape(two_views_small: list[np.ndarray]) -> None:
    """Score returns array of shape (latent_dimensions,)."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    s = model.score(two_views_small)
    assert s.shape == (k,)


def test_score_values_in_range(two_views_small: list[np.ndarray]) -> None:
    """Score values lie in [-1, 1]."""
    model = _make_model().fit(two_views_small)
    s = model.score(two_views_small)
    assert np.all(s >= -1.0 - 1e-9)
    assert np.all(s <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# weights is a real, linear weight matrix (unlike GAMCCA/GaussianProcessCCA/TreeCCA)
# ---------------------------------------------------------------------------


def test_weights_not_fitted_raises() -> None:
    """Accessing weights before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = ElasticNetCCA()
    with pytest.raises(NotFittedError):
        _ = model.weights


def test_weights_shapes_and_matches_transform(
    two_views_small: list[np.ndarray],
) -> None:
    """weights are real (p_i, k) arrays and transform(v) == centred(v) @ weights."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    weights = model.weights
    assert len(weights) == 2
    for w, v in zip(weights, two_views_small):
        assert w.shape == (v.shape[1], k)

    transformed = model.transform(two_views_small)
    for v, w, t, mean in zip(two_views_small, weights, transformed, model.means_):
        np.testing.assert_allclose((v - mean) @ w, t, atol=1e-8)


# ---------------------------------------------------------------------------
# get_factor_loadings / pairwise_correlations shapes
# ---------------------------------------------------------------------------


def test_get_factor_loadings_shapes(two_views_small: list[np.ndarray]) -> None:
    """get_factor_loadings returns (n_features_i, k) arrays."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    loadings = model.get_factor_loadings(two_views_small)
    assert len(loadings) == 2
    for loading, view in zip(loadings, two_views_small):
        assert loading.shape == (view.shape[1], k)


def test_pairwise_correlations_shape(two_views_small: list[np.ndarray]) -> None:
    """pairwise_correlations returns (n_views, n_views, k)."""
    k = 1
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    corrs = model.pairwise_correlations(two_views_small)
    assert corrs.shape == (2, 2, k)


# ---------------------------------------------------------------------------
# center=False
# ---------------------------------------------------------------------------


def test_center_false(two_views_small: list[np.ndarray]) -> None:
    """ElasticNetCCA works with center=False."""
    model = _make_model(center=False)
    model.fit(two_views_small)
    result = model.transform(two_views_small)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


def test_elasticnetcca_finds_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """ElasticNetCCA finds substantial correlation on correlated views."""
    model = ElasticNetCCA(latent_dimensions=1, alpha=0.01, random_state=0)
    s = model.fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


def test_objective_decreases_monotonically(
    correlated_views: list[np.ndarray],
) -> None:
    """Every coordinate-descent sweep strictly lowers the penalised EY objective."""
    from cca_zoo._utils._ey import ey_loss

    objs = []
    for n_iter in range(1, 11):
        model = ElasticNetCCA(
            latent_dimensions=1,
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=n_iter,
            tol=1e-300,
            random_state=0,
        )
        model.fit(correlated_views)
        reps = model.transform(correlated_views)
        penalty = sum(
            model.alpha * model.l1_ratio * np.sum(np.abs(w))
            + 0.5 * model.alpha * (1 - model.l1_ratio) * np.sum(w**2)
            for w in model.weights
        )
        objs.append(ey_loss(reps)["objective"] + penalty)
    assert np.all(np.diff(objs) <= 1e-9), objs


def test_higher_alpha_increases_sparsity(
    correlated_views: list[np.ndarray],
) -> None:
    """Increasing alpha (with l1_ratio > 0) should not decrease sparsity."""
    n_nonzero = []
    for alpha in [0.001, 0.1, 1.0]:
        model = ElasticNetCCA(
            latent_dimensions=1, alpha=alpha, l1_ratio=0.9, random_state=0
        )
        model.fit(correlated_views)
        n_nonzero.append(
            sum((np.abs(w) > 1e-10).sum() for w in model.weights)
        )
    assert n_nonzero[0] >= n_nonzero[1] >= n_nonzero[2]


# ---------------------------------------------------------------------------
# sklearn compatibility spot-checks (full suite covered by test_sklearn_compat.py)
# ---------------------------------------------------------------------------


def test_clone_and_get_params_roundtrip() -> None:
    """clone()/get_params() round-trip correctly (sklearn BaseEstimator contract)."""
    from sklearn.base import clone

    model = ElasticNetCCA(latent_dimensions=2, alpha=0.3, l1_ratio=0.4, random_state=0)
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()


def test_invalid_l1_ratio_raises() -> None:
    """l1_ratio outside [0, 1] is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        ElasticNetCCA(l1_ratio=1.5)._validate_params()
