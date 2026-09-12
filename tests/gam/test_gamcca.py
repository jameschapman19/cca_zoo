"""Tests for GAMCCA.

Unlike TreeCCA, GAMCCA has no optional dependency (it is built entirely on
scikit-learn's SplineTransformer/Ridge, already required by cca_zoo), so
these tests run unconditionally.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import SplineTransformer

from cca_zoo.gam import GAMCCA


def _make_model(latent_dimensions: int = 1, **kwargs: object) -> GAMCCA:
    kwargs.setdefault("n_estimators", 10)
    return GAMCCA(latent_dimensions=latent_dimensions, **kwargs)


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
# every model in the package (including GAMCCA) by
# tests/test_sklearn_compat.py.


# ---------------------------------------------------------------------------
# weights is not implemented
# ---------------------------------------------------------------------------


def test_weights_not_fitted_raises() -> None:
    """Accessing weights before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = GAMCCA()
    with pytest.raises(NotFittedError):
        _ = model.weights


def test_weights_raises_not_implemented(two_views_small: list[np.ndarray]) -> None:
    """Accessing weights after fitting raises NotImplementedError."""
    model = _make_model().fit(two_views_small)
    with pytest.raises(NotImplementedError, match="shape_function"):
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
    """GAMCCA works with center=False."""
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


def test_encoder_basis_is_sklearn_spline_transformer(
    two_views_small: list[np.ndarray],
) -> None:
    """The per-view spline basis is an actual fitted SplineTransformer.

    Confirms basis construction is delegated to scikit-learn rather than
    reimplemented.
    """
    model = _make_model().fit(two_views_small)
    for enc in model.encoders_:
        assert isinstance(enc._spline, SplineTransformer)


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
    """Summed shape_function terms reproduce the encoder's raw prediction.

    Summing every feature's shape_function at the training values should
    reproduce the encoder's raw (base-margin-free) training prediction.
    """
    model = _make_model(latent_dimensions=1).fit(two_views_small)
    view = two_views_small[0]
    total = sum(model.shape_function(0, j, view[:, j]) for j in range(view.shape[1]))
    np.testing.assert_allclose(total, model.encoders_[0].predict(), atol=1e-6)


def test_shape_function_not_fitted_raises() -> None:
    """Calling shape_function before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = GAMCCA()
    with pytest.raises(NotFittedError):
        model.shape_function(0, 0, np.array([0.0]))


# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


def test_gamcca_finds_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """GAMCCA finds substantial correlation on views with shared latent structure."""
    model = GAMCCA(latent_dimensions=1, n_estimators=100, random_state=0)
    s = model.fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


def test_gamcca_finds_correlation_on_three_correlated_views() -> None:
    """GAMCCA (multiview) finds substantial correlation on 3 correlated views."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 1))
    views = [
        z @ rng.standard_normal((1, 5)) + 0.1 * rng.standard_normal((200, 5))
        for _ in range(3)
    ]
    model = GAMCCA(latent_dimensions=1, n_estimators=150, random_state=0)
    s = model.fit(views).score(views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


@pytest.mark.slow
def test_gamcca_outperforms_linear_and_tree_on_smooth_nonmonotonic_data() -> None:
    """GAMCCA beats rCCA and TreeCCA on a held-out, smooth-but-nonlinear task.

    View 1 is a noisy linear copy of a shared latent factor ``z``; view 2 is
    a noisy linear copy of ``z ** 2`` — a smooth but *non-monotonic* (even)
    transform, chosen so that ``corr(z, z**2) ~ 0`` for symmetric ``z``.  No
    linear combination of view 1's raw features can align with view 2 (so
    ``rCCA`` is expected to fail), while a per-view nonlinear encoder that
    (approximately) learns the "square" transform recovers near-perfect
    cross-view correlation. GAMCCA's B-spline basis represents a quadratic
    almost exactly, so it should reach a given held-out correlation in far
    fewer boosting rounds than TreeCCA's step-function tree ensembles, and
    should out-generalise TreeCCA even at a matched round budget.

    Marked slow since it also requires TreeCCA's optional ``xgboost``
    dependency, not part of the base ``dev`` install.
    """
    pytest.importorskip("xgboost", reason="xgboost is not installed")
    from cca_zoo.linear import rCCA
    from cca_zoo.tree import TreeCCA

    rng = np.random.default_rng(0)
    n_train, n_test, p, noise = 500, 500, 5, 0.3
    n = n_train + n_test
    z = rng.standard_normal(n)
    X1 = np.column_stack([z + noise * rng.standard_normal(n) for _ in range(p)])
    X2 = np.column_stack([z**2 + noise * rng.standard_normal(n) for _ in range(p)])
    X1_tr, X1_te = X1[:n_train], X1[n_train:]
    X2_tr, X2_te = X2[:n_train], X2[n_train:]

    n_estimators = 150

    gam = GAMCCA(latent_dimensions=1, n_estimators=n_estimators, random_state=0)
    gam_test = gam.fit([X1_tr, X2_tr]).score([X1_te, X2_te])[0]

    tree = TreeCCA(
        latent_dimensions=1, n_estimators=n_estimators, max_depth=5, random_state=0
    )
    tree_test = tree.fit([X1_tr, X2_tr]).score([X1_te, X2_te])[0]

    rcca = rCCA(latent_dimensions=1, c=[0.3, 0.3])
    rcca_test = rcca.fit([X1_tr, X2_tr]).score([X1_te, X2_te])[0]

    assert gam_test > 0.9, (
        f"Expected GAMCCA to recover the relationship, got {gam_test}"
    )
    assert gam_test > tree_test + 0.2, (
        f"Expected GAMCCA ({gam_test}) to clearly beat TreeCCA ({tree_test}) "
        f"at a matched budget of {n_estimators} boosting rounds"
    )
    assert gam_test > rcca_test + 0.5, (
        f"Expected GAMCCA ({gam_test}) to clearly beat linear rCCA ({rcca_test})"
    )
