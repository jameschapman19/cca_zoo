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
# predict
# ---------------------------------------------------------------------------


def test_predict_raises_before_fit(two_views: list[np.ndarray]) -> None:
    """Calling predict before fit raises NotFittedError."""
    model = CCA()
    with pytest.raises(NotFittedError):
        model.predict(two_views)


def test_predict_wrong_length_raises(two_views: list[np.ndarray]) -> None:
    """Predict raises ValueError when views has the wrong length."""
    model = CCA().fit(two_views)
    with pytest.raises(ValueError, match="Expected 2 views"):
        model.predict([two_views[0]])


def test_predict_all_none_raises(two_views: list[np.ndarray]) -> None:
    """Predict raises ValueError when every view is None."""
    model = CCA().fit(two_views)
    with pytest.raises(ValueError, match="At least one view"):
        model.predict([None, None])


def test_predict_mismatched_samples_raises(two_views: list[np.ndarray]) -> None:
    """Predict raises ValueError when observed views disagree on n_samples."""
    model = CCA().fit(two_views)
    with pytest.raises(ValueError, match="same number of samples"):
        model.predict([two_views[0], two_views[1][:5]])


def test_predict_wrong_n_features_raises(two_views: list[np.ndarray]) -> None:
    """Predict raises ValueError when an observed view has the wrong width."""
    model = CCA().fit(two_views)
    with pytest.raises(ValueError, match="expected 10"):
        model.predict([two_views[0][:, :3], None])


def test_predict_output_shapes(two_views: list[np.ndarray]) -> None:
    """Predict returns one reconstruction per view, each matching its input shape."""
    model = CCA(latent_dimensions=2).fit(two_views)
    preds = model.predict([two_views[0], None])
    assert len(preds) == 2
    assert preds[0].shape == two_views[0].shape
    assert preds[1].shape == (two_views[0].shape[0], two_views[1].shape[1])


def test_predict_reconstructs_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """Predict cross-reconstructs a view from the other with high fidelity.

    Regression test for the CCA forward-model ambiguity discussed in #182:
    a naive ``scores @ weights.T`` reconstruction is only correct when the
    data is pre-whitened, so this checks the actual, unwhitened, correlated
    fixture data reconstructs well via the least-squares loadings instead.
    """
    model = CCA(latent_dimensions=2).fit(correlated_views)
    x2_pred = model.predict([correlated_views[0], None])[1]
    corr = np.corrcoef(x2_pred.ravel(), correlated_views[1].ravel())[0, 1]
    assert corr > 0.9


def test_predict_self_reconstruction_beats_naive_weights_reconstruction(
    correlated_views: list[np.ndarray],
) -> None:
    """The fitted loadings reconstruct better than a naive weights.T formula.

    On heterogeneous, unwhitened per-feature scales -- the case reported in
    #182 -- ``scores @ weights.T`` is a poor inverse of ``transform`` for
    CCA. Rescale the fixture's views to heterogeneous per-feature scales and
    check predict's least-squares loadings noticeably outperform the naive
    formula.
    """
    rng = np.random.default_rng(1)
    scales = [rng.uniform(0.5, 20, size=v.shape[1]) for v in correlated_views]
    views = [v * s for v, s in zip(correlated_views, scales)]
    model = CCA(latent_dimensions=2).fit(views)

    x2_pred = model.predict([views[0], None])[1]
    corr_predict = np.corrcoef(x2_pred.ravel(), views[1].ravel())[0, 1]

    z_hat = (views[0] - model.means_[0]) @ model.weights_[0]
    naive_pred = z_hat @ model.weights_[1].T + model.means_[1]
    corr_naive = np.corrcoef(naive_pred.ravel(), views[1].ravel())[0, 1]

    assert corr_predict > corr_naive


def test_predict_ignores_extra_observed_views(two_views: list[np.ndarray]) -> None:
    """Passing an observed (not None) target view doesn't change its shape."""
    model = CCA(latent_dimensions=2).fit(two_views)
    both_observed = model.predict(two_views)
    one_observed = model.predict([two_views[0], None])
    assert both_observed[1].shape == one_observed[1].shape


# ---------------------------------------------------------------------------
# inverse_transform
# ---------------------------------------------------------------------------


def test_inverse_transform_raises_before_fit(two_views: list[np.ndarray]) -> None:
    """Calling inverse_transform before fit raises NotFittedError."""
    model = CCA()
    with pytest.raises(NotFittedError):
        model.inverse_transform(two_views)


def test_inverse_transform_wrong_length_raises(two_views: list[np.ndarray]) -> None:
    """inverse_transform raises ValueError when scores has the wrong length."""
    model = CCA(latent_dimensions=2).fit(two_views)
    scores = model.transform(two_views)
    with pytest.raises(ValueError, match="Expected 2 score arrays"):
        model.inverse_transform([scores[0]])


def test_inverse_transform_wrong_n_latent_dims_raises(
    two_views: list[np.ndarray],
) -> None:
    """inverse_transform raises ValueError when a score array has the wrong width."""
    model = CCA(latent_dimensions=2).fit(two_views)
    scores = model.transform(two_views)
    with pytest.raises(ValueError, match="expected latent_dimensions=2"):
        model.inverse_transform([scores[0][:, :1], scores[1]])


def test_inverse_transform_output_shapes(two_views: list[np.ndarray]) -> None:
    """inverse_transform returns one reconstruction per view, matching its width."""
    model = CCA(latent_dimensions=2).fit(two_views)
    scores = model.transform(two_views)
    approx = model.inverse_transform(scores)
    assert len(approx) == 2
    for a, view in zip(approx, two_views):
        assert a.shape == view.shape


def test_inverse_transform_round_trips_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """inverse_transform(transform(views)) approximately recovers views."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    scores = model.transform(correlated_views)
    approx = model.inverse_transform(scores)
    for a, view in zip(approx, correlated_views):
        corr = np.corrcoef(a.ravel(), view.ravel())[0, 1]
        assert corr > 0.9


def test_inverse_transform_round_trips_on_heterogeneous_scales(
    correlated_views: list[np.ndarray],
) -> None:
    """The round trip holds even on unwhitened, heterogeneously-scaled views.

    Unlike predict's cross-view reconstruction, inverse_transform never
    needs to correct for the CCA forward-model whitening subtlety from
    #182/#195, since it regresses each view onto its own score rather than
    a consensus score borrowed from another view -- this checks that
    holds even when features are rescaled to very different magnitudes.
    """
    rng = np.random.default_rng(2)
    scales = [rng.uniform(0.5, 20, size=v.shape[1]) for v in correlated_views]
    views = [v * s for v, s in zip(correlated_views, scales)]
    model = CCA(latent_dimensions=2).fit(views)
    scores = model.transform(views)
    approx = model.inverse_transform(scores)
    for a, view in zip(approx, views):
        corr = np.corrcoef(a.ravel(), view.ravel())[0, 1]
        assert corr > 0.9


def test_inverse_transform_differs_from_predict(
    correlated_views: list[np.ndarray],
) -> None:
    """inverse_transform and predict answer different questions.

    inverse_transform reconstructs a view from its own score;
    predict reconstructs it from other views' scores. On two correlated
    but distinct views, they should not give numerically identical
    reconstructions of view 2.
    """
    model = CCA(latent_dimensions=2).fit(correlated_views)
    scores = model.transform(correlated_views)
    via_inverse_transform = model.inverse_transform(scores)[1]
    via_predict = model.predict([correlated_views[0], None])[1]
    assert not np.allclose(via_inverse_transform, via_predict)


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
