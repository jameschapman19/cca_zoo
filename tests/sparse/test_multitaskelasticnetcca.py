"""Tests for MultiTaskElasticNetCCA."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.sparse import MultiTaskElasticNetCCA


def _make_model(latent_dimensions: int = 2, **kwargs: object) -> MultiTaskElasticNetCCA:
    return MultiTaskElasticNetCCA(latent_dimensions=latent_dimensions, **kwargs)


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
# transform output shapes / weights
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


def test_weights_shapes_and_matches_transform(
    two_views_small: list[np.ndarray],
) -> None:
    """Weights are real (p_i, k) arrays and transform(v) == centred(v) @ weights."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
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

    model = MultiTaskElasticNetCCA()
    with pytest.raises(NotFittedError):
        model.transform([np.ones((3, 2)), np.ones((3, 2))])


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
# score
# ---------------------------------------------------------------------------


def test_score_shape(two_views_small: list[np.ndarray]) -> None:
    """Score is one float, as sklearn expects."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    s = model.score(two_views_small)
    assert isinstance(s, float)


def test_score_values_in_range(two_views_small: list[np.ndarray]) -> None:
    """Score values lie in [-1, 1]."""
    model = _make_model().fit(two_views_small)
    s = model.score(two_views_small)
    assert np.all(s >= -1.0 - 1e-9)
    assert np.all(s <= 1.0 + 1e-9)


# ---------------------------------------------------------------------------
# center=False
# ---------------------------------------------------------------------------


def test_center_false(two_views_small: list[np.ndarray]) -> None:
    """MultiTaskElasticNetCCA works with center=False."""
    model = _make_model(center=False)
    model.fit(two_views_small)
    result = model.transform(two_views_small)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


def test_multitask_finds_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """MultiTaskElasticNetCCA finds substantial correlation on correlated views."""
    model = MultiTaskElasticNetCCA(latent_dimensions=1, alpha=0.01, random_state=0)
    s = model.fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


def test_objective_decreases_monotonically(
    correlated_views: list[np.ndarray],
) -> None:
    """Every coordinate-descent sweep does not increase the penalised EY objective."""
    from cca_zoo._utils._ey import _group_penalty, ey_loss

    objs = []
    for n_iter in range(1, 8):
        model = MultiTaskElasticNetCCA(
            latent_dimensions=2,
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=n_iter,
            tol=1e-300,
            random_state=0,
        )
        model.fit(correlated_views)
        reps = model.transform(correlated_views)
        n_views = len(correlated_views)
        penalty = _group_penalty(
            model.weights_, [model.alpha] * n_views, [model.l1_ratio] * n_views
        )
        objs.append(ey_loss(reps)["objective"] + penalty)
    assert np.all(np.diff(objs) <= 1e-8), objs


def test_row_sparsity_is_joint_across_components(
    correlated_views: list[np.ndarray],
) -> None:
    """A feature's row is either active in every component or in none."""
    model = MultiTaskElasticNetCCA(
        latent_dimensions=2, alpha=0.3, l1_ratio=0.9, random_state=0
    )
    model.fit(correlated_views)
    for w in model.weights_:
        active_per_component = np.abs(w) > 1e-10  # (p, k) boolean
        # For every row, either all components are active or none are.
        row_any = active_per_component.any(axis=1)
        row_all = active_per_component.all(axis=1)
        np.testing.assert_array_equal(row_any, row_all)


def test_higher_alpha_increases_sparsity(
    correlated_views: list[np.ndarray],
) -> None:
    """Increasing alpha (with l1_ratio > 0) should not decrease row sparsity."""
    n_active_rows = []
    for alpha in [0.001, 0.1, 1.0]:
        model = MultiTaskElasticNetCCA(
            latent_dimensions=2, alpha=alpha, l1_ratio=0.9, random_state=0
        )
        model.fit(correlated_views)
        n_active_rows.append(
            sum(int(np.sum(np.linalg.norm(w, axis=1) > 1e-10)) for w in model.weights_)
        )
    assert n_active_rows[0] >= n_active_rows[1] >= n_active_rows[2]


def test_per_view_alpha_list_gives_sparser_penalised_view(
    correlated_views: list[np.ndarray],
) -> None:
    """A per-view alpha list applies a stronger penalty to only one view."""
    model = MultiTaskElasticNetCCA(
        latent_dimensions=2, alpha=[0.001, 1.0], l1_ratio=0.9, random_state=0
    ).fit(correlated_views)
    n_active_rows = [
        int(np.sum(np.linalg.norm(w, axis=1) > 1e-10)) for w in model.weights_
    ]
    assert n_active_rows[1] < n_active_rows[0]


def test_per_view_alpha_wrong_length_raises(
    two_views_small: list[np.ndarray],
) -> None:
    """A per-view alpha list must have one entry per view."""
    with pytest.raises(ValueError, match="alpha"):
        MultiTaskElasticNetCCA(alpha=[0.1, 0.2, 0.3]).fit(two_views_small)


# ---------------------------------------------------------------------------
# sklearn compatibility spot-checks
# ---------------------------------------------------------------------------


def test_clone_and_get_params_roundtrip() -> None:
    """clone()/get_params() round-trip correctly (sklearn BaseEstimator contract)."""
    from sklearn.base import clone

    model = MultiTaskElasticNetCCA(
        latent_dimensions=2, alpha=0.3, l1_ratio=0.4, random_state=0
    )
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()


def test_invalid_l1_ratio_raises() -> None:
    """l1_ratio outside [0, 1] is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        MultiTaskElasticNetCCA(l1_ratio=1.5)._validate_params()
