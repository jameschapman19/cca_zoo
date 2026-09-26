"""Tests for XGBoostCCA and LightGBMCCA (the TreeCCA family).

All tests are marked slow and require xgboost (an optional extra, not part
of the base ``dev`` install).
"""

from __future__ import annotations

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost", reason="xgboost is not installed")

from cca_zoo.metrics import factor_loadings, pairwise_correlations
from cca_zoo.tree import CatBoostCCA, LightGBMCCA, XGBoostCCA
from cca_zoo.tree._treecca import TreeCCA

pytestmark = pytest.mark.slow


def _make_model(latent_dimensions: int = 1, **kwargs: object) -> XGBoostCCA:
    kwargs.setdefault("n_estimators", 5)
    return XGBoostCCA(latent_dimensions=latent_dimensions, **kwargs)


# ---------------------------------------------------------------------------
# TreeCCA itself is an abstract base, not instantiable
# ---------------------------------------------------------------------------


def test_treecca_base_class_not_instantiable() -> None:
    """TreeCCA is an abstract base; only its subclasses can be constructed."""
    with pytest.raises(TypeError):
        TreeCCA()


def test_treecca_subclasses_share_the_base_class() -> None:
    """XGBoostCCA, LightGBMCCA, and CatBoostCCA share the TreeCCA base class."""
    assert issubclass(XGBoostCCA, TreeCCA)
    assert issubclass(LightGBMCCA, TreeCCA)
    assert issubclass(CatBoostCCA, TreeCCA)


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


# get_params/set_params roundtrip behaviour is exercised generically for
# every model in the package (including XGBoostCCA and LightGBMCCA) by
# tests/test_sklearn_compat.py.


# ---------------------------------------------------------------------------
# weights is not implemented
# ---------------------------------------------------------------------------


def test_weights_not_fitted_raises() -> None:
    """Transform before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = XGBoostCCA()
    with pytest.raises(NotFittedError):
        model.transform([np.ones((3, 2)), np.ones((3, 2))])


# ---------------------------------------------------------------------------
# factor_loadings shapes
# ---------------------------------------------------------------------------


def test_get_factor_loadings_shapes(two_views_small: list[np.ndarray]) -> None:
    """factor_loadings returns (n_features_i, k) arrays."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    loadings = factor_loadings(two_views_small, model.transform(two_views_small))
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
    corrs = pairwise_correlations(model.transform(two_views_small))
    assert corrs.shape == (2, 2, k)


# ---------------------------------------------------------------------------
# center=False
# ---------------------------------------------------------------------------


def test_center_false(two_views_small: list[np.ndarray]) -> None:
    """XGBoostCCA works with center=False."""
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
# Per-view parameters
# ---------------------------------------------------------------------------


def test_per_view_n_estimators_list(two_views_small: list[np.ndarray]) -> None:
    """A per-view n_estimators list stops boosting the exhausted view early."""
    model = _make_model(n_estimators=[2, 6]).fit(two_views_small)
    assert model.boosters_[0][0].num_boosted_rounds() == 2
    assert model.boosters_[1][0].num_boosted_rounds() == 6


def test_per_view_max_depth_list(two_views_small: list[np.ndarray]) -> None:
    """A per-view max_depth list gives each view's trees a different max depth.

    Deeper trees have more nodes, so node count is used as a proxy: this
    xgboost version's `trees_to_dataframe()` has no `Depth` column.
    """
    model = _make_model(n_estimators=20, max_depth=[1, 6], min_child_weight=1).fit(
        two_views_small
    )
    n_nodes_shallow = len(model.boosters_[0][0].trees_to_dataframe())
    n_nodes_deep = len(model.boosters_[1][0].trees_to_dataframe())
    assert n_nodes_shallow < n_nodes_deep


def test_per_view_n_estimators_wrong_length_raises(
    two_views_small: list[np.ndarray],
) -> None:
    """A per-view n_estimators list must have one entry per view."""
    with pytest.raises(ValueError, match="n_estimators"):
        _make_model(n_estimators=[5, 6, 7]).fit(two_views_small)


# ---------------------------------------------------------------------------
# LightGBMCCA
# ---------------------------------------------------------------------------


def test_lightgbm_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes end-to-end with LightGBMCCA."""
    lightgbm = pytest.importorskip("lightgbm", reason="lightgbm is not installed")
    k = 2
    model = LightGBMCCA(latent_dimensions=k, n_estimators=5).fit(two_views_small)
    result = model.transform(two_views_small)
    n = two_views_small[0].shape[0]
    for arr in result:
        assert arr.shape == (n, k)
    for view_boosters in model.boosters_:
        assert len(view_boosters) == k
        for booster in view_boosters:
            assert isinstance(booster, lightgbm.Booster)


def test_lightgbm_missing_raises_import_error(
    two_views_small: list[np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """LightGBMCCA without the lightgbm package installed raises ImportError."""
    import cca_zoo.tree._treecca as treecca_module

    monkeypatch.setattr(treecca_module, "_LGBM_AVAILABLE", False)
    model = LightGBMCCA(n_estimators=5)
    with pytest.raises(ImportError, match="lightgbm"):
        model.fit(two_views_small)


def test_lightgbm_fit_transform_consistency(
    two_views_small: list[np.ndarray],
) -> None:
    """fit_transform equals fit().transform() for LightGBMCCA."""
    pytest.importorskip("lightgbm", reason="lightgbm is not installed")
    m1 = LightGBMCCA(n_estimators=5)
    m2 = LightGBMCCA(n_estimators=5)
    result_ft = m1.fit_transform(two_views_small)
    result_sep = m2.fit(two_views_small).transform(two_views_small)
    for a, b in zip(result_ft, result_sep):
        np.testing.assert_allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------------
# CatBoostCCA
# ---------------------------------------------------------------------------


def test_catboost_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes end-to-end with CatBoostCCA."""
    pytest.importorskip("catboost", reason="catboost is not installed")
    k = 2
    model = CatBoostCCA(latent_dimensions=k, n_estimators=5).fit(two_views_small)
    result = model.transform(two_views_small)
    n = two_views_small[0].shape[0]
    for arr in result:
        assert arr.shape == (n, k)
    for view_boosters in model.boosters_:
        assert len(view_boosters) == k


def test_catboost_missing_raises_import_error(
    two_views_small: list[np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CatBoostCCA without the catboost package installed raises ImportError."""
    import cca_zoo.tree._treecca as treecca_module

    monkeypatch.setattr(treecca_module, "_CATBOOST_AVAILABLE", False)
    model = CatBoostCCA(n_estimators=5)
    with pytest.raises(ImportError, match="catboost"):
        model.fit(two_views_small)


def test_catboost_fit_transform_consistency(
    two_views_small: list[np.ndarray],
) -> None:
    """fit_transform equals fit().transform() for CatBoostCCA."""
    pytest.importorskip("catboost", reason="catboost is not installed")
    m1 = CatBoostCCA(n_estimators=5)
    m2 = CatBoostCCA(n_estimators=5)
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
    """XGBoostCCA finds substantial correlation on correlated views."""
    model = XGBoostCCA(
        latent_dimensions=1, n_estimators=60, max_depth=3, random_state=0
    )
    s = model.fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


def test_treecca_finds_correlation_on_three_correlated_views() -> None:
    """XGBoostCCA (multiview) finds substantial correlation on 3 correlated views."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 1))
    views = [
        z @ rng.standard_normal((1, 5)) + 0.1 * rng.standard_normal((200, 5))
        for _ in range(3)
    ]
    model = XGBoostCCA(
        latent_dimensions=1, n_estimators=300, max_depth=3, random_state=0
    )
    s = model.fit(views).score(views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


def _held_out_pair(kind: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Train/test views sharing one latent signal in their first feature."""

    def views(seed: int) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(1000)
        first = z if kind == "linear" else np.sin(2 * z)
        return [
            np.column_stack(
                [first + 0.3 * rng.standard_normal(1000)]
                + [rng.standard_normal(1000) for _ in range(5)]
            ),
            np.column_stack(
                [z + 0.3 * rng.standard_normal(1000)]
                + [rng.standard_normal(1000) for _ in range(5)]
            ),
        ]

    return views(0), views(1)


@pytest.mark.parametrize("cls", [XGBoostCCA, LightGBMCCA], ids=["xgb", "lgbm"])
@pytest.mark.parametrize(("kind", "floor"), [("linear", 0.8), ("sin", 0.6)])
def test_defaults_learn_the_shared_signal(cls: type, kind: str, floor: float) -> None:
    """At its defaults the model recovers a shared signal on held-out data.

    Regression test: with a unit-variance random start and gradients
    renormalised to a fixed small size every round, the boosters' learned
    part stayed a fraction of a random projection they could not undo, and
    held-out correlation on a plain linear signal was about 0.14.
    """
    train, test = _held_out_pair(kind)
    assert cls(random_state=0).fit(train).score(test) > floor
