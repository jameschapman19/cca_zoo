"""Tests for XGBoostCCA and LightGBMCCA (the TreeCCA family).

All tests are marked slow and require xgboost (an optional extra, not part
of the base ``dev`` install).
"""

from __future__ import annotations

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost", reason="xgboost is not installed")

from cca_zoo.tree import CatBoostCCA, LightGBMCCA, XGBoostCCA

pytestmark = pytest.mark.slow


def _make_model(n_components: int = 1, **kwargs: object) -> XGBoostCCA:
    kwargs.setdefault("n_estimators", 5)
    kwargs.setdefault("random_state", 0)
    return XGBoostCCA(n_components=n_components, **kwargs)


# ---------------------------------------------------------------------------
# gauss_seidel toggle
# ---------------------------------------------------------------------------


def test_jacobi_variant_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes with gauss_seidel=False (Jacobi updates)."""
    model = _make_model(gauss_seidel=False).fit(two_views_small)
    result = model.transform(two_views_small)
    assert len(result) == 2


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


# ---------------------------------------------------------------------------
# LightGBMCCA
# ---------------------------------------------------------------------------


def test_lightgbm_missing_raises_import_error(
    two_views_small: list[np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """LightGBMCCA without the lightgbm package installed raises ImportError."""
    import cca_zoo.tree._treecca as treecca_module

    monkeypatch.setattr(treecca_module, "_LGBM_AVAILABLE", False)
    model = LightGBMCCA(n_estimators=5, random_state=0)
    with pytest.raises(ImportError, match="lightgbm"):
        model.fit(two_views_small)


# ---------------------------------------------------------------------------
# CatBoostCCA
# ---------------------------------------------------------------------------


def test_catboost_missing_raises_import_error(
    two_views_small: list[np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CatBoostCCA without the catboost package installed raises ImportError."""
    import cca_zoo.tree._treecca as treecca_module

    monkeypatch.setattr(treecca_module, "_CATBOOST_AVAILABLE", False)
    model = CatBoostCCA(n_estimators=5, random_state=0)
    with pytest.raises(ImportError, match="catboost"):
        model.fit(two_views_small)


# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


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
