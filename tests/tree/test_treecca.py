"""TreeCCA: a gradient-boosted tree encoder per view."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xgboost")

import cca_zoo.tree._treecca as treecca
from cca_zoo._utils._ey import ey_cross_covariance
from cca_zoo.tree import CatBoostCCA, LightGBMCCA, XGBoostCCA

pytestmark = pytest.mark.slow


def _views(seed: int) -> list[np.ndarray]:
    """A sin(2z) signal in one feature of view 1, z in one feature of view 2."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(1000)
    return [
        np.column_stack(
            [signal + 0.3 * rng.standard_normal(1000), rng.standard_normal((1000, 5))]
        )
        for signal in (np.sin(2 * z), z)
    ]


@pytest.mark.parametrize("cls", [XGBoostCCA, LightGBMCCA])
def test_recovers_a_nonlinear_signal_at_its_defaults(cls: type) -> None:
    """At its defaults the model relates sin(2z) to z on held-out data."""
    assert cls(random_state=0).fit(_views(0)).score(_views(1)) > 0.6


def test_components_come_in_order_of_reward() -> None:
    """The rotated outputs have a diagonal, descending EY reward."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((300, 3)) * [3, 2, 1]
    views = [
        z @ rng.standard_normal((3, p)) + rng.standard_normal((300, p)) for p in (6, 5)
    ]
    model = XGBoostCCA(n_components=3, n_estimators=20, random_state=0).fit(views)
    reward, _ = ey_cross_covariance(model.transform(views))
    assert np.all(np.diff(np.diag(reward)) <= 1e-8)
    np.testing.assert_allclose(reward - np.diag(np.diag(reward)), 0.0, atol=1e-8)


def test_n_estimators_per_view(two_views_small: list[np.ndarray]) -> None:
    """Each view boosts for its own number of rounds."""
    model = XGBoostCCA(n_estimators=[2, 6], random_state=0).fit(two_views_small)
    assert [b[0].num_boosted_rounds() for b in model.boosters_] == [2, 6]


def test_max_depth_per_view(two_views_small: list[np.ndarray]) -> None:
    """Deeper trees in one view have more nodes than stumps in the other."""
    model = XGBoostCCA(
        n_estimators=20, max_depth=[1, 6], min_child_weight=1, random_state=0
    ).fit(two_views_small)
    nodes = [len(b[0].trees_to_dataframe()) for b in model.boosters_]
    assert nodes[0] < nodes[1]


@pytest.mark.parametrize(
    ("cls", "flag", "package"),
    [
        (LightGBMCCA, "_LGBM_AVAILABLE", "lightgbm"),
        (CatBoostCCA, "_CATBOOST_AVAILABLE", "catboost"),
    ],
)
def test_missing_backend_names_the_package(
    cls: type,
    flag: str,
    package: str,
    two_views_small: list[np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without its optional backend, a model says which package to install."""
    monkeypatch.setattr(treecca, flag, False)
    with pytest.raises(ImportError, match=package):
        cls().fit(two_views_small)
