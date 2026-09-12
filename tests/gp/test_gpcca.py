"""Tests for GPCCA.

Like GAMCCA, GPCCA has no optional dependency (it is built entirely on
scikit-learn's GaussianProcessRegressor, already required by cca_zoo), so
these tests run unconditionally.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor

from cca_zoo.gp import GPCCA
from cca_zoo.gp._gpcca import _GpEncoder, _SparseGpEncoder


def _make_model(latent_dimensions: int = 1, **kwargs: object) -> GPCCA:
    return GPCCA(latent_dimensions=latent_dimensions, **kwargs)


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
# return_std uncertainty output
# ---------------------------------------------------------------------------


def test_transform_return_std_shapes_and_positive(
    two_views_small: list[np.ndarray],
) -> None:
    """transform(..., return_std=True) returns matching-shape, positive stds."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    means, stds = model.transform(two_views_small, return_std=True)
    n = two_views_small[0].shape[0]
    assert len(means) == 2
    assert len(stds) == 2
    for mean, std in zip(means, stds):
        assert mean.shape == (n, k)
        assert std.shape == (n, k)
        assert np.all(std > 0)


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
# every model in the package (including GPCCA) by
# tests/test_sklearn_compat.py.


# ---------------------------------------------------------------------------
# weights is not implemented
# ---------------------------------------------------------------------------


def test_weights_not_fitted_raises() -> None:
    """Accessing weights before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = GPCCA()
    with pytest.raises(NotFittedError):
        _ = model.weights


def test_weights_raises_not_implemented(two_views_small: list[np.ndarray]) -> None:
    """Accessing weights after fitting raises NotImplementedError."""
    model = _make_model().fit(two_views_small)
    with pytest.raises(NotImplementedError, match="Gaussian process"):
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
    """GPCCA works with center=False."""
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


def test_encoder_models_are_sklearn_gaussian_process_regressor(
    two_views_small: list[np.ndarray],
) -> None:
    """Each fitted per-component model is an actual GaussianProcessRegressor.

    Confirms the GP fit is delegated to scikit-learn rather than
    reimplemented.
    """
    model = _make_model().fit(two_views_small)
    for enc in model.encoders_:
        for m in enc.models_:
            assert isinstance(m, GaussianProcessRegressor)


# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


def test_gpcca_finds_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """GPCCA finds substantial correlation on views with shared latent structure."""
    model = GPCCA(latent_dimensions=1, random_state=0)
    s = model.fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


def test_gpcca_finds_correlation_on_three_correlated_views() -> None:
    """GPCCA (multiview) finds substantial correlation on 3 correlated views."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 1))
    views = [
        z @ rng.standard_normal((1, 5)) + 0.1 * rng.standard_normal((200, 5))
        for _ in range(3)
    ]
    model = GPCCA(latent_dimensions=1, random_state=0)
    s = model.fit(views).score(views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


# ---------------------------------------------------------------------------
# sparse (inducing-point) approximation
# ---------------------------------------------------------------------------


def test_sparse_fit_completes_and_shapes(two_views_small: list[np.ndarray]) -> None:
    """n_inducing < n_samples switches to the sparse DTC encoder and fits."""
    k = 2
    n = two_views_small[0].shape[0]
    model = _make_model(latent_dimensions=k, n_inducing=n // 2).fit(two_views_small)
    for enc in model.encoders_:
        assert isinstance(enc, _SparseGpEncoder)
    result = model.transform(two_views_small)
    assert len(result) == 2
    for arr in result:
        assert arr.shape == (n, k)


def test_sparse_return_std_shapes_and_positive(
    two_views_small: list[np.ndarray],
) -> None:
    """Sparse transform(..., return_std=True) returns positive, matching-shape stds."""
    k = 2
    n = two_views_small[0].shape[0]
    model = _make_model(latent_dimensions=k, n_inducing=n // 2).fit(two_views_small)
    means, stds = model.transform(two_views_small, return_std=True)
    for mean, std in zip(means, stds):
        assert mean.shape == (n, k)
        assert std.shape == (n, k)
        assert np.all(std > 0)


def test_n_inducing_at_least_n_samples_falls_back_to_exact(
    two_views_small: list[np.ndarray],
) -> None:
    """n_inducing >= n_samples is equivalent to exact (dense) GP inference."""
    n = two_views_small[0].shape[0]
    model = _make_model(n_inducing=10 * n).fit(two_views_small)
    for enc in model.encoders_:
        assert isinstance(enc, _GpEncoder)
        assert not isinstance(enc, _SparseGpEncoder)


def test_sparse_finds_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """The sparse approximation still recovers substantial correlation."""
    n = correlated_views[0].shape[0]
    model = GPCCA(latent_dimensions=1, random_state=0, n_inducing=max(10, n // 3))
    s = model.fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


@pytest.mark.slow
def test_sparse_scales_to_large_sample_sizes() -> None:
    """Sparse GPCCA fits at a sample size that would defeat exact GP inference.

    Exact GP inference redoes an O(n^3) Cholesky factorisation at every
    Newton step of every inner/outer round, which would be impractically
    slow here; the sparse approximation should still recover the
    underlying correlation.
    """
    rng = np.random.default_rng(0)
    n_train, n_test, noise = 4000, 500, 0.3
    n = n_train + n_test
    z = rng.standard_normal(n)
    X1 = np.column_stack([z + noise * rng.standard_normal(n) for _ in range(3)])
    X2 = np.column_stack([z**2 + noise * rng.standard_normal(n) for _ in range(3)])
    X1_tr, X1_te = X1[:n_train], X1[n_train:]
    X2_tr, X2_te = X2[:n_train], X2[n_train:]

    model = GPCCA(latent_dimensions=1, random_state=0, n_inducing=100)
    test_corr = model.fit([X1_tr, X2_tr]).score([X1_te, X2_te])[0]
    assert test_corr > 0.7, (
        f"Expected substantial held-out correlation, got {test_corr}"
    )


@pytest.mark.slow
def test_gpcca_outperforms_others_on_genuine_interaction() -> None:
    """GPCCA beats GAMCCA, TreeCCA, and rCCA on a genuine feature-interaction task.

    View 1 is two noisy independent factors ``u, v``; view 2 is a noisy
    copy of their *interaction* ``u * v`` -- not additively separable into
    a function of ``u`` plus a function of ``v``. GAMCCA's additive-spline
    encoder structurally cannot represent this; ``TreeCCA`` can only
    approximate it via multivariate splits, and at its default
    ``colsample_bytree`` a single tree is often starved of joint access to
    both features. GPCCA's joint (non-additive) RBF kernel represents the
    interaction directly.

    Marked slow since it also requires TreeCCA's optional ``xgboost``
    dependency, not part of the base ``dev`` install.
    """
    pytest.importorskip("xgboost", reason="xgboost is not installed")
    from cca_zoo.gam import GAMCCA
    from cca_zoo.linear import rCCA
    from cca_zoo.tree import TreeCCA

    rng = np.random.default_rng(0)
    n_train, n_test, noise = 300, 300, 0.2
    n = n_train + n_test
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    interaction = u * v
    X1 = np.column_stack([u, v]) + noise * rng.standard_normal((n, 2))
    X2 = np.column_stack([interaction, interaction]) + noise * rng.standard_normal(
        (n, 2)
    )
    X1_tr, X1_te = X1[:n_train], X1[n_train:]
    X2_tr, X2_te = X2[:n_train], X2[n_train:]

    gp = GPCCA(latent_dimensions=1, random_state=0)
    gp_test = gp.fit([X1_tr, X2_tr]).score([X1_te, X2_te])[0]

    gam = GAMCCA(latent_dimensions=1, random_state=0)
    gam_test = gam.fit([X1_tr, X2_tr]).score([X1_te, X2_te])[0]

    tree = TreeCCA(latent_dimensions=1, n_estimators=150, max_depth=5, random_state=0)
    tree_test = tree.fit([X1_tr, X2_tr]).score([X1_te, X2_te])[0]

    rcca = rCCA(latent_dimensions=1, c=[0.3, 0.3])
    rcca_test = rcca.fit([X1_tr, X2_tr]).score([X1_te, X2_te])[0]

    assert gp_test > 0.7, f"Expected GPCCA to recover the interaction, got {gp_test}"
    assert gp_test > gam_test + 0.05, (
        f"Expected GPCCA ({gp_test}) to beat additive GAMCCA ({gam_test}) "
        f"on a genuine feature interaction"
    )
    assert gp_test > tree_test + 0.05, (
        f"Expected GPCCA ({gp_test}) to beat TreeCCA ({tree_test}) at "
        f"TreeCCA's default colsample_bytree"
    )
    assert gp_test > rcca_test + 0.3, (
        f"Expected GPCCA ({gp_test}) to clearly beat linear rCCA ({rcca_test})"
    )
