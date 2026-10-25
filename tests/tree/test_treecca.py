"""TreeCCA: a gradient-boosted tree encoder per view."""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytest.importorskip("xgboost")

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
    ("cls", "package"), [(LightGBMCCA, "lightgbm"), (CatBoostCCA, "catboost")]
)
def test_missing_backend_names_the_package(
    cls: type,
    package: str,
    two_views_small: list[np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without its optional backend, a model's fit says which package is missing."""
    monkeypatch.setitem(sys.modules, package, None)
    with pytest.raises(ImportError, match=package):
        cls().fit(two_views_small)


def _with_missing(views: list[np.ndarray], rate: float, seed: int) -> list[np.ndarray]:
    """Copies of ``views`` with a fraction ``rate`` of entries set to NaN."""
    rng = np.random.default_rng(seed)
    return [np.where(rng.random(v.shape) < rate, np.nan, v) for v in views]


@pytest.mark.parametrize("cls", [XGBoostCCA, LightGBMCCA, CatBoostCCA])
def test_missing_values_are_accepted(
    cls: type, two_views_small: list[np.ndarray]
) -> None:
    """NaN entries pass through fit, transform, predict and inverse_transform."""
    views = _with_missing(two_views_small, 0.2, 0)
    model = cls(n_components=2, n_estimators=5, random_state=0).fit(views)
    scores = model.transform(views)
    assert all(np.isfinite(z).all() for z in scores)
    assert all(np.isfinite(p).all() for p in model.predict([views[0], views[1]]))
    assert all(np.isfinite(x).all() for x in model.inverse_transform(scores))
    assert np.isfinite(model.score(views))


@pytest.mark.parametrize("cls", [XGBoostCCA, LightGBMCCA, CatBoostCCA])
def test_infinity_is_rejected(cls: type, two_views_small: list[np.ndarray]) -> None:
    """Only NaN is a missing value: infinities still raise."""
    views = [v.copy() for v in two_views_small]
    views[0][0, 0] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        cls(n_estimators=2).fit(views)


def test_a_nonlinear_signal_survives_missing_values() -> None:
    """With 30% of entries missing the signal is still recovered on held-out data."""
    train, test = _with_missing(_views(0), 0.3, 1), _with_missing(_views(1), 0.3, 2)
    assert XGBoostCCA(random_state=0).fit(train).score(test) > 0.4


def test_missingness_itself_can_carry_signal() -> None:
    """A feature seen only through whether it is missing still relates the views."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal(2000)
    a = np.column_stack(
        [
            np.where(z > 0, np.nan, rng.standard_normal(2000)),
            rng.standard_normal((2000, 3)),
        ]
    )
    b = np.column_stack([z, rng.standard_normal((2000, 3))])
    model = XGBoostCCA(random_state=0).fit([a[:1500], b[:1500]])
    assert model.score([a[1500:], b[1500:]]) > 0.5
