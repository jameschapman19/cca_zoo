"""Tests for OrthogonalMatchingPursuitCCA."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.sparse import OrthogonalMatchingPursuitCCA


def _make_model(
    latent_dimensions: int = 1, **kwargs: object
) -> OrthogonalMatchingPursuitCCA:
    return OrthogonalMatchingPursuitCCA(latent_dimensions=latent_dimensions, **kwargs)


# ---------------------------------------------------------------------------
# fit completes
# ---------------------------------------------------------------------------


def test_two_view_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes on two-view data without error."""
    model = _make_model(n_nonzero_coefs=2)
    fitted = model.fit(two_views_small)
    assert fitted is model


def test_three_view_fit_completes(three_views_small: list[np.ndarray]) -> None:
    """Fit completes on three-view data without error."""
    model = _make_model(n_nonzero_coefs=2)
    fitted = model.fit(three_views_small)
    assert fitted is model


# ---------------------------------------------------------------------------
# transform output shapes / weights
# ---------------------------------------------------------------------------


def test_transform_shapes_training_data(two_views_small: list[np.ndarray]) -> None:
    """Transform on training data returns (n_samples, latent_dimensions) arrays."""
    k = 2
    model = _make_model(latent_dimensions=k, n_nonzero_coefs=2).fit(two_views_small)
    result = model.transform(two_views_small)
    assert len(result) == 2
    n = two_views_small[0].shape[0]
    for arr in result:
        assert arr.shape == (n, k)


def test_weights_shapes_and_matches_transform(
    two_views_small: list[np.ndarray],
) -> None:
    """Weights are real (p_i, k) arrays and transform(v) == centred(v) @ weights."""
    k = 2
    model = _make_model(latent_dimensions=k, n_nonzero_coefs=2).fit(two_views_small)
    weights = model.weights_
    assert len(weights) == 2
    for w, v in zip(weights, two_views_small):
        assert w.shape == (v.shape[1], k)

    transformed = model.transform(two_views_small)
    for v, w, t, mean in zip(two_views_small, weights, transformed, model.means_):
        np.testing.assert_allclose((v - mean) @ w, t, atol=1e-8)


def test_weights_not_fitted_raises() -> None:
    """Transform before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = OrthogonalMatchingPursuitCCA()
    with pytest.raises(NotFittedError):
        model.transform([np.ones((3, 2)), np.ones((3, 2))])


# ---------------------------------------------------------------------------
# fit_transform consistency
# ---------------------------------------------------------------------------


def test_fit_transform_consistency(two_views_small: list[np.ndarray]) -> None:
    """fit_transform equals fit().transform() numerically."""
    m1 = _make_model(n_nonzero_coefs=2, random_state=0)
    m2 = _make_model(n_nonzero_coefs=2, random_state=0)
    result_ft = m1.fit_transform(two_views_small)
    result_sep = m2.fit(two_views_small).transform(two_views_small)
    for a, b in zip(result_ft, result_sep):
        np.testing.assert_allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def test_score_shape(two_views_small: list[np.ndarray]) -> None:
    """Score is one float, as sklearn expects."""
    k = 2
    model = _make_model(latent_dimensions=k, n_nonzero_coefs=2).fit(two_views_small)
    s = model.score(two_views_small)
    assert isinstance(s, float)


def test_score_values_in_range(two_views_small: list[np.ndarray]) -> None:
    """Score values lie in [-1, 1]."""
    model = _make_model(n_nonzero_coefs=2).fit(two_views_small)
    s = model.score(two_views_small)
    assert np.all(s >= -1.0 - 1e-9)
    assert np.all(s <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# center=False
# ---------------------------------------------------------------------------


def test_center_false(two_views_small: list[np.ndarray]) -> None:
    """OrthogonalMatchingPursuitCCA works with center=False."""
    model = _make_model(n_nonzero_coefs=2, center=False)
    model.fit(two_views_small)
    result = model.transform(two_views_small)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Correctness / cardinality
# ---------------------------------------------------------------------------


def test_omp_finds_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """OrthogonalMatchingPursuitCCA finds substantial correlation on correlated data."""
    model = OrthogonalMatchingPursuitCCA(
        latent_dimensions=1, n_nonzero_coefs=3, random_state=0
    )
    s = model.fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


def test_active_feature_count_matches_budget(
    correlated_views: list[np.ndarray],
) -> None:
    """Each view has exactly n_nonzero_coefs active (nonzero-row) features."""
    budget = 3
    model = OrthogonalMatchingPursuitCCA(
        latent_dimensions=2, n_nonzero_coefs=budget, random_state=0
    )
    model.fit(correlated_views)
    for w in model.weights_:
        n_active = int(np.sum(np.linalg.norm(w, axis=1) > 1e-10))
        assert n_active == budget


def test_per_view_budget_list(correlated_views: list[np.ndarray]) -> None:
    """A list of per-view budgets is honoured independently per view."""
    budgets = [2, 4]
    model = OrthogonalMatchingPursuitCCA(
        latent_dimensions=1, n_nonzero_coefs=budgets, random_state=0
    )
    model.fit(correlated_views)
    for w, budget in zip(model.weights_, budgets):
        n_active = int(np.sum(np.abs(w.ravel()) > 1e-10))
        assert n_active == budget


def test_budget_exceeding_n_features_is_capped(
    correlated_views: list[np.ndarray],
) -> None:
    """A budget larger than a view's feature count is capped, not an error."""
    n_features = correlated_views[0].shape[1]
    model = OrthogonalMatchingPursuitCCA(
        latent_dimensions=1, n_nonzero_coefs=n_features + 100, random_state=0
    )
    model.fit(correlated_views)
    n_active = int(np.sum(np.abs(model.weights_[0].ravel()) > 1e-10))
    assert n_active <= n_features


def test_default_n_nonzero_coefs_is_ten_percent(
    correlated_views: list[np.ndarray],
) -> None:
    """With n_nonzero_coefs=None, each view defaults to max(1, n_features // 10)."""
    model = OrthogonalMatchingPursuitCCA(latent_dimensions=1, random_state=0)
    model.fit(correlated_views)
    for w, v in zip(model.weights_, correlated_views):
        expected = max(1, v.shape[1] // 10)
        n_active = int(np.sum(np.abs(w.ravel()) > 1e-10))
        assert n_active == expected


def test_wrong_length_budget_list_raises(correlated_views: list[np.ndarray]) -> None:
    """A per-view budget list whose length doesn't match n_views raises."""
    model = OrthogonalMatchingPursuitCCA(n_nonzero_coefs=[1, 2, 3])
    with pytest.raises(ValueError):
        model.fit(correlated_views)


def test_nonpositive_budget_raises(correlated_views: list[np.ndarray]) -> None:
    """A non-positive n_nonzero_coefs raises."""
    model = OrthogonalMatchingPursuitCCA(n_nonzero_coefs=0)
    with pytest.raises(ValueError):
        model.fit(correlated_views)


# ---------------------------------------------------------------------------
# sklearn compatibility spot-checks
# ---------------------------------------------------------------------------


def test_clone_and_get_params_roundtrip() -> None:
    """clone()/get_params() round-trip correctly (sklearn BaseEstimator contract)."""
    from sklearn.base import clone

    model = OrthogonalMatchingPursuitCCA(
        latent_dimensions=2, n_nonzero_coefs=3, random_state=0
    )
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()


def test_invalid_max_iter_raises() -> None:
    """max_iter below 1 is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        OrthogonalMatchingPursuitCCA(max_iter=0)._validate_params()
