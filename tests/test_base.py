"""Tests for the BaseModel abstract class in cca_zoo._base."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from cca_zoo._base import BaseModel
from cca_zoo.linear._cca import CCA

# ---------------------------------------------------------------------------
# Concrete minimal subclass for testing abstract interface
# ---------------------------------------------------------------------------


class _MinimalModel(BaseModel):
    """Minimal concrete subclass that stores identity weight matrices."""

    def fit(self, views: list, y: None = None) -> _MinimalModel:
        """Fit by storing identity-like weight matrices."""
        validated = self._setup_fit(views)
        self.weights_ = [np.eye(v.shape[1], self.latent_dimensions) for v in validated]
        return self


# ---------------------------------------------------------------------------
# Abstract method enforcement
# ---------------------------------------------------------------------------


def test_cannot_instantiate_base_model() -> None:
    """BaseModel cannot be instantiated directly (abstract class)."""
    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]


def test_minimal_subclass_instantiates() -> None:
    """A concrete subclass with fit implemented can be instantiated."""
    model = _MinimalModel(latent_dimensions=2)
    assert model.latent_dimensions == 2


# ---------------------------------------------------------------------------
# validate_views error paths
# ---------------------------------------------------------------------------


def test_validate_views_raises_on_single_view() -> None:
    """validate_views raises ValueError when fewer than 2 views are passed."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 5))
    with pytest.raises(ValueError, match="At least 2 views"):
        _MinimalModel().fit([x])


def test_validate_views_raises_on_inconsistent_samples() -> None:
    """validate_views raises ValueError when views differ in number of samples."""
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((10, 5))
    x2 = rng.standard_normal((12, 5))
    with pytest.raises(ValueError, match="same number of samples"):
        _MinimalModel().fit([x1, x2])


# ---------------------------------------------------------------------------
# Centering behaviour
# ---------------------------------------------------------------------------


def test_center_true_subtracts_means(two_views: list[np.ndarray]) -> None:
    """When center=True, means_ are stored and fit data is mean-subtracted."""
    model = _MinimalModel(center=True).fit(two_views)
    for v, m in zip(two_views, model.means_):
        np.testing.assert_allclose(m, v.mean(axis=0), rtol=1e-10)


def test_center_false_means_are_zeros(two_views: list[np.ndarray]) -> None:
    """When center=False, means_ are zero arrays."""
    model = _MinimalModel(center=False).fit(two_views)
    for m, v in zip(model.means_, two_views):
        np.testing.assert_array_equal(m, np.zeros(v.shape[1]))


# ---------------------------------------------------------------------------
# weights property raises NotFittedError before fit
# ---------------------------------------------------------------------------


def test_weights_raises_before_fit() -> None:
    """Accessing .weights before fit raises NotFittedError."""
    model = _MinimalModel()
    with pytest.raises(NotFittedError):
        _ = model.weights


def test_weights_accessible_after_fit(two_views: list[np.ndarray]) -> None:
    """Accessing .weights after fit returns a list of arrays."""
    model = _MinimalModel(latent_dimensions=1).fit(two_views)
    w = model.weights
    assert isinstance(w, list)
    assert len(w) == len(two_views)


# ---------------------------------------------------------------------------
# transform raises NotFittedError before fit
# ---------------------------------------------------------------------------


def test_transform_raises_before_fit(two_views: list[np.ndarray]) -> None:
    """Calling transform before fit raises NotFittedError."""
    model = _MinimalModel()
    with pytest.raises(NotFittedError):
        model.transform(two_views)


# ---------------------------------------------------------------------------
# fit_transform consistency
# ---------------------------------------------------------------------------


def test_fit_transform_equals_fit_then_transform(
    two_views: list[np.ndarray],
) -> None:
    """fit_transform output must equal fit().transform() numerically."""
    result_ft = _MinimalModel(latent_dimensions=1).fit_transform(two_views)
    model = _MinimalModel(latent_dimensions=1).fit(two_views)
    result_sep = model.transform(two_views)
    for ft, sep in zip(result_ft, result_sep):
        np.testing.assert_allclose(ft, sep, rtol=1e-12)


# ---------------------------------------------------------------------------
# score and pairwise_correlations
# ---------------------------------------------------------------------------


def test_score_shape(two_views: list[np.ndarray]) -> None:
    """Score returns shape (latent_dimensions,)."""
    k = 2
    model = CCA(latent_dimensions=k).fit(two_views)
    s = model.score(two_views)
    assert s.shape == (k,)


def test_score_values_in_range(correlated_views: list[np.ndarray]) -> None:
    """All score values must be in [-1, 1]."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    s = model.score(correlated_views)
    assert np.all(s >= -1.0 - 1e-9)
    assert np.all(s <= 1.0 + 1e-9)


def test_pairwise_correlations_shape(two_views: list[np.ndarray]) -> None:
    """pairwise_correlations returns shape (n_views, n_views, latent_dimensions)."""
    k = 2
    model = CCA(latent_dimensions=k).fit(two_views)
    corrs = model.pairwise_correlations(two_views)
    assert corrs.shape == (2, 2, k)


def test_pairwise_correlations_diagonal_is_one(two_views: list[np.ndarray]) -> None:
    """Diagonal entries of pairwise_correlations should be 1 (self-correlation)."""
    model = CCA(latent_dimensions=1).fit(two_views)
    corrs = model.pairwise_correlations(two_views)
    np.testing.assert_allclose(corrs[0, 0, :], 1.0, atol=1e-10)
    np.testing.assert_allclose(corrs[1, 1, :], 1.0, atol=1e-10)


def test_average_pairwise_correlations_equals_score(
    two_views: list[np.ndarray],
) -> None:
    """average_pairwise_correlations and score should return the same values."""
    model = CCA(latent_dimensions=2).fit(two_views)
    np.testing.assert_allclose(
        model.average_pairwise_correlations(two_views),
        model.score(two_views),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# get_factor_loadings
# ---------------------------------------------------------------------------


def test_get_factor_loadings_shapes(two_views: list[np.ndarray]) -> None:
    """get_factor_loadings returns one array per view with shape (n_features, k)."""
    k = 2
    model = CCA(latent_dimensions=k).fit(two_views)
    loadings = model.get_factor_loadings(two_views)
    assert len(loadings) == len(two_views)
    for loading, view in zip(loadings, two_views):
        assert loading.shape == (view.shape[1], k)


# ---------------------------------------------------------------------------
# sklearn get_params / set_params roundtrip
# ---------------------------------------------------------------------------


def test_get_params_set_params_roundtrip() -> None:
    """get_params / set_params roundtrip for BaseModel subclass."""
    model = _MinimalModel(latent_dimensions=3, center=False)
    params = model.get_params()
    assert params["latent_dimensions"] == 3
    assert params["center"] is False
    model2 = _MinimalModel()
    model2.set_params(**params)
    assert model2.latent_dimensions == 3
    assert model2.center is False


# ---------------------------------------------------------------------------
# n_views_ and n_features_in_ metadata
# ---------------------------------------------------------------------------


def test_metadata_set_after_fit(two_views: list[np.ndarray]) -> None:
    """n_views_, n_features_in_, and n_samples_ are set correctly after fit."""
    model = _MinimalModel().fit(two_views)
    assert model.n_views_ == 2
    assert model.n_features_in_ == [10, 8]
    assert model.n_samples_ == 50
