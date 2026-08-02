"""Tests for TreeCCA.

All tests are marked slow and require xgboost (an optional extra, not part
of the base ``dev`` install).
"""

from __future__ import annotations

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost", reason="xgboost is not installed")

from cca_zoo.tree import TreeCCA

pytestmark = pytest.mark.slow


def _make_model(latent_dimensions: int = 1, **kwargs: object) -> TreeCCA:
    kwargs.setdefault("n_estimators", 5)
    return TreeCCA(latent_dimensions=latent_dimensions, **kwargs)


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
    assert len(model.boosters_) == 3


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
    assert len(result) == 2
    for arr in result:
        assert arr.shape == (10, k)


def test_transform_shapes_three_views(three_views_small: list[np.ndarray]) -> None:
    """Transform on three-view data returns one array per view."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(three_views_small)
    result = model.transform(three_views_small)
    assert len(result) == 3
    n = three_views_small[0].shape[0]
    for arr in result:
        assert arr.shape == (n, k)


# ---------------------------------------------------------------------------
# fit_transform consistency
# ---------------------------------------------------------------------------


def test_fit_transform_consistency(two_views_small: list[np.ndarray]) -> None:
    """fit_transform equals fit().transform() numerically."""
    m1 = _make_model()
    m2 = _make_model()
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
# get_params / set_params roundtrip
# ---------------------------------------------------------------------------


def test_get_params_roundtrip() -> None:
    """get_params returns the correct parameter values."""
    model = TreeCCA(latent_dimensions=2, n_estimators=7, max_depth=3)
    params = model.get_params()
    assert params["latent_dimensions"] == 2
    assert params["n_estimators"] == 7
    assert params["max_depth"] == 3


def test_set_params_roundtrip() -> None:
    """set_params correctly updates model parameters."""
    model = TreeCCA(latent_dimensions=1)
    model.set_params(latent_dimensions=3)
    assert model.latent_dimensions == 3


# ---------------------------------------------------------------------------
# weights is not implemented
# ---------------------------------------------------------------------------


def test_weights_not_fitted_raises() -> None:
    """Accessing weights before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = TreeCCA()
    with pytest.raises(NotFittedError):
        _ = model.weights


def test_weights_raises_not_implemented(two_views_small: list[np.ndarray]) -> None:
    """Accessing weights after fitting raises NotImplementedError."""
    model = _make_model().fit(two_views_small)
    with pytest.raises(NotImplementedError, match="boosters_"):
        _ = model.weights


# ---------------------------------------------------------------------------
# get_factor_loadings shapes
# ---------------------------------------------------------------------------


def test_get_factor_loadings_shapes(two_views_small: list[np.ndarray]) -> None:
    """get_factor_loadings returns (n_features_i, k) arrays."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    loadings = model.get_factor_loadings(two_views_small)
    assert len(loadings) == 2
    for loading, view in zip(loadings, two_views_small):
        assert loading.shape == (view.shape[1], k)


# ---------------------------------------------------------------------------
# pairwise_correlations shape
# ---------------------------------------------------------------------------


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
    """TreeCCA works with center=False."""
    model = _make_model(center=False)
    model.fit(two_views_small)
    result = model.transform(two_views_small)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# gauss_seidel toggle
# ---------------------------------------------------------------------------


def test_jacobi_variant_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes with gauss_seidel=False (Jacobi updates)."""
    model = _make_model(gauss_seidel=False).fit(two_views_small)
    result = model.transform(two_views_small)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# boosters_ attribute
# ---------------------------------------------------------------------------


def test_boosters_attribute_shape(two_views_small: list[np.ndarray]) -> None:
    """boosters_ has one list of k boosters per view."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    assert len(model.boosters_) == 2
    for view_boosters in model.boosters_:
        assert len(view_boosters) == k
        for booster in view_boosters:
            assert isinstance(booster, xgboost.Booster)


# ---------------------------------------------------------------------------
# backend selection
# ---------------------------------------------------------------------------


def test_invalid_backend_raises(two_views_small: list[np.ndarray]) -> None:
    """An unrecognised backend raises ValueError."""
    model = _make_model(backend="not-a-backend")
    with pytest.raises(ValueError, match="backend must be"):
        model.fit(two_views_small)


def test_lightgbm_backend_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes end-to-end with backend='lightgbm'."""
    lightgbm = pytest.importorskip("lightgbm", reason="lightgbm is not installed")
    k = 2
    model = _make_model(latent_dimensions=k, backend="lightgbm").fit(two_views_small)
    result = model.transform(two_views_small)
    n = two_views_small[0].shape[0]
    for arr in result:
        assert arr.shape == (n, k)
    for view_boosters in model.boosters_:
        assert len(view_boosters) == k
        for booster in view_boosters:
            assert isinstance(booster, lightgbm.Booster)


def test_lightgbm_backend_missing_raises_import_error(
    two_views_small: list[np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """backend='lightgbm' without the lightgbm package raises ImportError."""
    import cca_zoo.tree._treecca as treecca_module

    monkeypatch.setattr(treecca_module, "_LGBM_AVAILABLE", False)
    model = _make_model(backend="lightgbm")
    with pytest.raises(ImportError, match="lightgbm"):
        model.fit(two_views_small)


def test_lightgbm_backend_fit_transform_consistency(
    two_views_small: list[np.ndarray],
) -> None:
    """fit_transform equals fit().transform() for the lightgbm backend."""
    pytest.importorskip("lightgbm", reason="lightgbm is not installed")
    m1 = _make_model(backend="lightgbm")
    m2 = _make_model(backend="lightgbm")
    result_ft = m1.fit_transform(two_views_small)
    result_sep = m2.fit(two_views_small).transform(two_views_small)
    for a, b in zip(result_ft, result_sep):
        np.testing.assert_allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


def test_treecca_finds_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """TreeCCA finds substantial correlation on views with shared latent structure."""
    model = TreeCCA(latent_dimensions=1, n_estimators=60, max_depth=3, random_state=0)
    s = model.fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


def test_treecca_finds_correlation_on_three_correlated_views() -> None:
    """TreeCCA (multiview) finds substantial correlation on 3 correlated views."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 1))
    views = [
        z @ rng.standard_normal((1, 5)) + 0.1 * rng.standard_normal((200, 5))
        for _ in range(3)
    ]
    model = TreeCCA(latent_dimensions=1, n_estimators=300, max_depth=3, random_state=0)
    s = model.fit(views).score(views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"
