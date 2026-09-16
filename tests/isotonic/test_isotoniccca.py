"""Tests for IsotonicCCA.

Like GAMCCA, IsotonicCCA has no optional dependency (built entirely on
scikit-learn's IsotonicRegression, already required by cca_zoo), so these
tests run unconditionally.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.isotonic import IsotonicCCA
from cca_zoo.isotonic._isotoniccca import _IsotonicEncoder


def _make_model(latent_dimensions: int = 1, **kwargs: object) -> IsotonicCCA:
    return IsotonicCCA(latent_dimensions=latent_dimensions, **kwargs)


# ---------------------------------------------------------------------------
# _IsotonicEncoder: the sign-bug regression test
# ---------------------------------------------------------------------------


def test_encoder_boosts_toward_a_residual_target_without_diverging() -> None:
    """Boosting toward a proper (shrinking) residual target converges and stays put.

    Regression test for a genuine sign bug: the encoder's accumulation
    used to add ``learning_rate * (isotonic fit of the raw gradient)``,
    which is gradient *ascent*, not descent -- unlike TreeCCA's XGBoost/
    LightGBM backends, whose Newton-step leaf values are already
    correctly signed by the library itself, nothing here negates the
    gradient automatically, so it has to happen in `boost` directly. With
    the bug, the accumulated prediction grew without bound (verified via
    the EY loss's own `objective` getting worse, monotonically, the more
    boosting rounds were run) instead of converging.
    """
    rng = np.random.default_rng(0)
    n = 200
    z = rng.standard_normal(n)
    X = np.column_stack([z + 0.1 * rng.standard_normal(n) for _ in range(4)])
    target = z.reshape(-1, 1)

    enc = _IsotonicEncoder(X, k=1, out_of_bounds="clip")
    boost_rng = np.random.default_rng(0)
    mses = []
    for round_ in range(200):
        residual = target - enc.predict()
        enc.boost(-residual, learning_rate=0.05, subsample=0.5, rng=boost_rng)
        if round_ + 1 in (20, 100, 200):
            mses.append(float(np.mean((enc.predict()[:, 0] - z) ** 2)))

    assert mses[0] < 0.1, f"expected early convergence, got MSE {mses[0]}"
    # Once converged, more rounds shouldn't blow it back up.
    assert mses[-1] < 5 * mses[0], (
        f"prediction diverged with more rounds: MSE went {mses[0]} -> {mses[-1]}"
    )


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
    assert len(model.encoders_) == 3


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


# get_params/set_params roundtrip behaviour is exercised generically for
# every model in the package (including IsotonicCCA) by
# tests/test_sklearn_compat.py.


# ---------------------------------------------------------------------------
# weights is not implemented
# ---------------------------------------------------------------------------


def test_weights_not_fitted_raises() -> None:
    """Accessing weights before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = IsotonicCCA()
    with pytest.raises(NotFittedError):
        _ = model.weights


def test_weights_raises_not_implemented(two_views_small: list[np.ndarray]) -> None:
    """Accessing weights after fitting raises NotImplementedError."""
    model = _make_model().fit(two_views_small)
    with pytest.raises(NotImplementedError, match="shape_function"):
        _ = model.weights


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
    """IsotonicCCA works with center=False."""
    model = _make_model(center=False)
    model.fit(two_views_small)
    result = model.transform(two_views_small)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# encoders_ attribute
# ---------------------------------------------------------------------------


def test_encoders_attribute_shape(two_views_small: list[np.ndarray]) -> None:
    """encoders_ has one encoder per view, each producing k-dim output."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    assert len(model.encoders_) == 2
    for enc in model.encoders_:
        assert enc.k == k
        assert enc.predict().shape == (two_views_small[0].shape[0], k)


# ---------------------------------------------------------------------------
# shape_function
# ---------------------------------------------------------------------------


def test_shape_function_shape(two_views_small: list[np.ndarray]) -> None:
    """shape_function evaluates one feature's additive term at given points."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    x_grid = np.linspace(-2, 2, 7)
    term = model.shape_function(view=0, feature=1, x=x_grid)
    assert term.shape == (7, k)


def test_shape_function_sums_to_prediction(two_views_small: list[np.ndarray]) -> None:
    """Summed shape_function terms reproduce the encoder's raw prediction."""
    model = _make_model(latent_dimensions=1).fit(two_views_small)
    view = two_views_small[0]
    total = sum(model.shape_function(0, j, view[:, j]) for j in range(view.shape[1]))
    np.testing.assert_allclose(total, model.encoders_[0].predict(), atol=1e-6)


def test_shape_function_not_fitted_raises() -> None:
    """Calling shape_function before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = IsotonicCCA()
    with pytest.raises(NotFittedError):
        model.shape_function(0, 0, np.array([0.0]))


def test_shape_function_is_monotonic(correlated_views: list[np.ndarray]) -> None:
    """Every fitted shape function is monotonic in its own feature -- the whole point.

    Each round's isotonic term shares a fixed per-feature direction (see
    ``_IsotonicEncoder``), so the *sum* over every boosting round should
    still be monotonic -- unlike GAMCCA's unconstrained splines.
    """
    model = IsotonicCCA(latent_dimensions=1, n_estimators=40, random_state=0).fit(
        correlated_views
    )
    x_grid = np.linspace(-3, 3, 100)
    for view in range(2):
        for feature in range(correlated_views[view].shape[1]):
            term = model.shape_function(view, feature, x_grid)[:, 0]
            diffs = np.diff(term)
            assert np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9), (
                f"shape_function(view={view}, feature={feature}) is not monotonic"
            )


# ---------------------------------------------------------------------------
# Correctness / stability
# ---------------------------------------------------------------------------


def test_isotoniccca_finds_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """IsotonicCCA finds substantial correlation on views with shared structure."""
    model = IsotonicCCA(latent_dimensions=1, random_state=0)
    s = model.fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


def test_more_rounds_does_not_diverge() -> None:
    """Held-out correlation on a monotonic relationship shouldn't collapse.

    Regression test at the model level for the same sign bug
    ``test_encoder_boosts_toward_a_residual_target_without_diverging``
    catches at the encoder level: before the fix, score *worsened*
    monotonically with more boosting rounds (even going negative) instead
    of improving/plateauing.
    """
    rng = np.random.default_rng(0)
    n_train, n_test = 300, 300
    z_train = rng.standard_normal(n_train)
    z_test = rng.standard_normal(n_test)

    def views(z: np.ndarray) -> list[np.ndarray]:
        x1 = np.column_stack([z + 0.1 * rng.standard_normal(len(z)) for _ in range(5)])
        x2 = np.column_stack(
            [z**3 + 0.1 * rng.standard_normal(len(z)) for _ in range(5)]
        )
        return [x1, x2]

    train_views = views(z_train)
    test_views = views(z_test)

    scores = {}
    for n_estimators in (30, 300):
        model = IsotonicCCA(
            latent_dimensions=1, n_estimators=n_estimators, random_state=0
        ).fit(train_views)
        scores[n_estimators] = model.score(test_views)[0]

    assert scores[300] > scores[30] - 0.1, (
        f"score got worse with more rounds: {scores[30]:.2f} -> {scores[300]:.2f}"
    )
