"""The multiview estimator contract, checked on every model in the package.

``transform``, ``predict`` and ``inverse_transform`` all map through each
model's own per-view encoder (``BaseModel._transform_view``), so every model
— linear, kernel, spline, tree, GP, manifold or probabilistic — must honour
them identically. ``feature_importances_`` is checked here too: one
non-negative array per view summing to one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cca_zoo._base import BaseModel
from tests.test_sklearn_compat import _MODEL_CLASSES

# Constructor arguments that keep each fit small; every other model runs on
# its defaults.
_FAST: dict[str, dict[str, Any]] = {
    "ProbabilisticCCA": {"num_warmup": 20, "num_samples": 20},
    "VariationalBayesCCA": {"num_steps": 50},
    "GFA": {"max_iter": 50},
    "XGBoostCCA": {"n_estimators": 5},
    "LightGBMCCA": {"n_estimators": 5},
    "CatBoostCCA": {"n_estimators": 5},
    "ProjectionPursuitCCA": {"n_restarts": 1, "max_iter": 5},
    "StochasticCCAEY": {"max_iter": 5},
}


def _views(seed: int, shift: float = 0.0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((60, 1))
    return [
        shift + z @ rng.standard_normal((1, p)) + 0.5 * rng.standard_normal((60, p))
        for p in (4, 3)
    ]


def _make(cls: type[BaseModel]) -> BaseModel:
    model = cls(latent_dimensions=1, **_FAST.get(cls.__name__, {}))
    if "random_state" in model.get_params():
        model.set_params(random_state=0)
    return model


def _fit(
    cls: type[BaseModel], views: list[np.ndarray], model: BaseModel | None = None
) -> BaseModel:
    model = _make(cls) if model is None else model
    if cls.__name__ == "PartialCCA":
        partials = np.random.default_rng(1).standard_normal((len(views[0]), 1))
        return model.fit(views, partials=partials)
    return model.fit(views)


_IDS = [c.__name__ for c in _MODEL_CLASSES]


@pytest.mark.parametrize("cls", _MODEL_CLASSES, ids=_IDS)
def test_transform_predict_inverse_transform(cls: type[BaseModel]) -> None:
    """Every model transforms per view and reconstructs from any observed view."""
    views = _views(0, shift=5.0)
    model = _fit(cls, views)
    scores = model.transform(views)
    assert len(scores) == len(views)
    assert all(s.shape == (60, 1) and np.all(np.isfinite(s)) for s in scores)
    for missing in range(len(views)):
        observed: list[np.ndarray | None] = list(views)
        observed[missing] = None
        reconstructed = model.predict(observed)
        assert [r.shape for r in reconstructed] == [v.shape for v in views]
        assert all(np.all(np.isfinite(r)) for r in reconstructed)
    inverted = model.inverse_transform(scores)
    assert [r.shape for r in inverted] == [v.shape for v in views]


@pytest.mark.parametrize("cls", _MODEL_CLASSES, ids=_IDS)
def test_refit_leaves_no_stale_state(cls: type[BaseModel]) -> None:
    """After a refit, predictions match a fresh fit on the new data."""
    first, second = _views(0), _views(1, shift=3.0)
    model = _fit(cls, first)
    model.predict([first[0], None])
    model.inverse_transform(model.transform(first))
    _fit(cls, second, model)
    fresh = _fit(cls, second)
    np.testing.assert_allclose(
        model.predict([second[0], None])[1],
        fresh.predict([second[0], None])[1],
        atol=1e-6,
    )


@pytest.mark.parametrize("cls", _MODEL_CLASSES, ids=_IDS)
def test_score_is_a_float(cls: type[BaseModel]) -> None:
    """``score`` follows sklearn's contract: one float, higher is better."""
    views = _views(0)
    assert isinstance(_fit(cls, views).score(views), float)


@pytest.mark.parametrize("cls", _MODEL_CLASSES, ids=_IDS)
def test_feature_importances(cls: type[BaseModel]) -> None:
    """One non-negative array per view, each summing to one.

    A view whose embedding uses no feature at all (a sparse model can zero
    one out) gets all zeros instead, as sklearn's tree models do.
    """
    views = _views(0)
    importances = _fit(cls, views).feature_importances_
    assert [imp.shape for imp in importances] == [(v.shape[1],) for v in views]
    for imp in importances:
        assert np.all(imp >= 0)
        assert imp.sum() == pytest.approx(1.0) or not imp.any()


def test_linear_importance_matches_the_permutation_definition() -> None:
    """Var(x_j) * sum_k w_jk^2 is half the mean squared permutation change."""
    from cca_zoo.linear import CCA

    rng = np.random.default_rng(0)
    z = rng.standard_normal((20000, 1))
    views = [
        z @ rng.standard_normal((1, p)) + rng.standard_normal((20000, p))
        for p in (4, 3)
    ]
    model = CCA(latent_dimensions=2).fit(views)
    closed_form = model._feature_importances()
    permuted = model._permutation_importances()
    for exact, estimate in zip(closed_form, permuted):
        np.testing.assert_allclose(estimate, 2 * exact, rtol=0.05)


def _signal_in_first_feature(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    z = rng.standard_normal(n)
    first = np.sin(2 * z)
    return [
        np.column_stack(
            [first + 0.2 * rng.standard_normal(n)]
            + [rng.standard_normal(n) for _ in range(3)]
        ),
        np.column_stack(
            [z + 0.2 * rng.standard_normal(n)]
            + [rng.standard_normal(n) for _ in range(3)]
        ),
    ]


@pytest.mark.parametrize(
    "name", ["rCCA", "GAMCCA", "MARSCCA", "XGBoostCCA", "KCCA", "GaussianProcessCCA"]
)
def test_importance_finds_the_signal_feature(name: str) -> None:
    """Each importance family ranks the one informative feature first."""
    classes = {c.__name__: c for c in _MODEL_CLASSES}
    if name not in classes:
        pytest.skip(f"{name}'s optional dependency is not installed")
    cls = classes[name]
    kwargs = {"kernel": "rbf"} if name == "KCCA" else {}
    model = cls(latent_dimensions=1, **kwargs)
    if "random_state" in model.get_params():
        model.set_params(random_state=0)
    importances = model.fit(_signal_in_first_feature(300)).feature_importances_
    assert [int(np.argmax(imp)) for imp in importances] == [0, 0]
