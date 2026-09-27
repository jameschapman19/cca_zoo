"""The multiview contract, checked on every model in the package.

scikit-learn's own checks (``test_estimator_checks``) cover each model as an
estimator of one array. These cover what is specific to several views:
``transform``, ``predict`` and ``inverse_transform`` through each model's
per-view encoder, the number and shapes of views, parameter validation and
``feature_importances_per_view_``.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._param_validation import InvalidParameterError

import cca_zoo._base
from cca_zoo._base import BaseModel
from tests._helpers import MODEL_CLASSES, make_model

_IDS = [c.__name__ for c in MODEL_CLASSES]
_TWO_VIEW_ONLY = {"CCA", "RidgeCCA", "PLS", "CCAR3", "ECCA"}


def _views(
    seed: int, shift: float = 0.0, n_views: int = 2, n: int = 60, noise: float = 0.5
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 1))
    return [
        shift + z @ rng.standard_normal((1, p)) + noise * rng.standard_normal((n, p))
        for p in (4, 3, 5)[:n_views]
    ]


def _fit(model: BaseModel, views: list[np.ndarray]) -> BaseModel:
    if type(model).__name__ == "PartialCCA":
        partials = np.random.default_rng(1).standard_normal((len(views[0]), 1))
        return model.fit(views, partials=partials)
    return model.fit(views)


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=_IDS)
def test_transform_predict_inverse_transform(cls: type[BaseModel]) -> None:
    """Every model transforms per view and reconstructs from any observed view."""
    views = _views(0, shift=5.0)
    model = _fit(make_model(cls), views)
    scores = model.transform(views)
    assert all(s.shape == (60, 1) and np.all(np.isfinite(s)) for s in scores)
    for missing in range(len(views)):
        observed: list[np.ndarray | None] = list(views)
        observed[missing] = None
        reconstructed = model.predict(observed)
        assert [r.shape for r in reconstructed] == [v.shape for v in views]
        assert all(np.all(np.isfinite(r)) for r in reconstructed)
    inverted = model.inverse_transform(scores)
    assert [r.shape for r in inverted] == [v.shape for v in views]


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=_IDS)
def test_number_of_views(cls: type[BaseModel]) -> None:
    """Two-view methods reject a third view by name; the rest take it."""
    views = _views(0, n_views=3)
    model = make_model(cls)
    if cls.__name__ in _TWO_VIEW_ONLY:
        with pytest.raises(ValueError, match=f"{cls.__name__} requires exactly 2"):
            _fit(model, views)
    else:
        assert len(_fit(model, views).transform(views)) == 3


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=_IDS)
def test_new_views_must_match_the_fitted_shapes(cls: type[BaseModel]) -> None:
    """A wrong number of views, width or sample count raises."""
    views = _views(0)
    model = _fit(make_model(cls), views)
    with pytest.raises(ValueError, match="Expected 2 views"):
        model.transform(views[:1])
    with pytest.raises(ValueError, match="View 1 has 4 features"):
        model.transform([views[0], views[0]])
    with pytest.raises(ValueError, match="same number of samples"):
        model.transform([views[0], views[1][:10]])
    with pytest.raises(ValueError, match="View 0 has 3 features"):
        model.predict([views[1], None])
    with pytest.raises(ValueError, match="Expected 2 views"):
        model.predict([views[0]])
    with pytest.raises(ValueError, match="same number of samples"):
        model.predict([views[0], views[1][:10]])


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=_IDS)
def test_every_parameter_is_validated(cls: type[BaseModel]) -> None:
    """Each constructor parameter has a constraint that a nonsense value fails."""
    model = make_model(cls)
    params = model.get_params()
    assert set(params) <= set(cls._parameter_constraints)
    for name in params:
        invalid = make_model(cls).set_params(**{name: object()})
        with pytest.raises(InvalidParameterError, match=name):
            _fit(invalid, _views(0))


# Settings for the models whose quick test settings are too short to converge.
_ADEQUATE = {
    "ProbabilisticCCA": {"n_warmup": 100, "n_posterior_samples": 100},
    "XGBoostCCA": {"n_estimators": 50},
    "LightGBMCCA": {"n_estimators": 50},
    "CatBoostCCA": {"n_estimators": 50},
}


@pytest.mark.filterwarnings("error::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=_IDS)
def test_recovers_a_shared_signal(cls: type[BaseModel]) -> None:
    """At its defaults, each model finds a strong one-dimensional shared signal."""
    views = _views(0, n=200, noise=0.3)
    model = cls(**_ADEQUATE.get(cls.__name__, {}))
    if "random_state" in model.get_params():
        model.set_params(random_state=0)
    assert _fit(model, views).score(views) > 0.8


_ITERATIVE = [c for c in MODEL_CLASSES if "max_iter" in make_model(c).get_params()]
# ECCA and CCAR3 solve their default alpha=0 by least squares, without iterating.
_ITERATING = {"ECCA": {"alpha": 0.1}, "CCAR3": {"alpha": 0.1}}


@pytest.mark.parametrize("cls", _ITERATIVE, ids=[c.__name__ for c in _ITERATIVE])
def test_stopping_at_max_iter_warns(cls: type[BaseModel]) -> None:
    """A fit cut short by max_iter warns, and n_iter_ reports the iterations run."""
    model = make_model(cls).set_params(max_iter=1, **_ITERATING.get(cls.__name__, {}))
    with pytest.warns(ConvergenceWarning):
        _fit(model, _views(0))
    assert np.max(model.n_iter_) == 1


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=_IDS)
def test_center_false(cls: type[BaseModel]) -> None:
    """Uncentred models keep zero means and still transform."""
    views = _views(0)
    model = _fit(make_model(cls).set_params(center=False), views)
    assert all(not m.any() for m in model.means_)
    assert all(np.all(np.isfinite(s)) for s in model.transform(views))


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=_IDS)
def test_refit_leaves_no_stale_state(cls: type[BaseModel]) -> None:
    """After a refit, predictions match a fresh fit on the new data."""
    first, second = _views(0), _views(1, shift=3.0)
    model = _fit(make_model(cls), first)
    model.predict([first[0], None])
    model.inverse_transform(model.transform(first))
    _fit(model, second)
    fresh = _fit(make_model(cls), second)
    np.testing.assert_allclose(
        model.predict([second[0], None])[1],
        fresh.predict([second[0], None])[1],
        atol=1e-6,
    )


# Out-of-sample kernel and manifold embeddings are built from the training rows;
# GaussianProcessCCA uses every row as an inducing point by default.
_KEEPS_TRAINING_VIEWS = {"KCCA", "KGCCA", "KTCCA", "ManifoldCCA", "GaussianProcessCCA"}


def _arrays(obj: object, seen: set[int]) -> list[np.ndarray]:
    """Every array reachable from ``obj``'s attributes and containers."""
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return [obj]
    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, list | tuple):
        children = list(obj)
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("cca_zoo"):
        children = list(vars(obj).values())
    else:
        return []
    return [a for child in children for a in _arrays(child, seen)]


@pytest.mark.parametrize(
    "cls", [c for c in MODEL_CLASSES if c.__name__ not in _KEEPS_TRAINING_VIEWS]
)
def test_fitted_model_keeps_no_training_data(cls: type[BaseModel]) -> None:
    """No array in the fitted model holds a training view, raw or centred."""
    views = _views(0)
    model = _fit(make_model(cls), views)
    training = views + [v - v.mean(axis=0) for v in views]
    for array in _arrays(model, set()):
        assert not any(
            array.shape == v.shape and np.allclose(array, v) for v in training
        )


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=_IDS)
def test_feature_importances(cls: type[BaseModel]) -> None:
    """One non-negative array per view summing to one, or zero if no feature is used."""
    views = _views(0)
    importances = _fit(make_model(cls), views).feature_importances_per_view_
    assert [imp.shape for imp in importances] == [(v.shape[1],) for v in views]
    for imp in importances:
        assert np.all(imp >= 0)
        assert imp.sum() == pytest.approx(1.0) or not imp.any()


def test_linear_importance_matches_the_permutation_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Var(x_j) * sum_k w_jk^2 is half the mean squared permutation change."""
    from cca_zoo.linear import CCA

    monkeypatch.setattr(cca_zoo._base, "_PERMUTATION_SAMPLES", 20000)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((20000, 1))
    views = [
        z @ rng.standard_normal((1, p)) + rng.standard_normal((20000, p))
        for p in (4, 3)
    ]
    model = CCA(n_components=2).fit(views)
    centred = [v - m for v, m in zip(views, model.means_)]
    closed_form = model._feature_importances(centred)
    permuted = model._permutation_importances(centred)
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
    "name",
    ["RidgeCCA", "GAMCCA", "MARSCCA", "XGBoostCCA", "KCCA", "GaussianProcessCCA"],
)
def test_importance_finds_the_signal_feature(name: str) -> None:
    """Each importance family ranks the one informative feature first."""
    classes = {c.__name__: c for c in MODEL_CLASSES}
    if name not in classes:
        pytest.skip(f"{name}'s optional dependency is not installed")
    cls = classes[name]
    kwargs = {"kernel": "rbf"} if name == "KCCA" else {}
    model = cls(n_components=1, **kwargs)
    if "random_state" in model.get_params():
        model.set_params(random_state=0)
    importances = model.fit(_signal_in_first_feature(300)).feature_importances_per_view_
    assert [int(np.argmax(imp)) for imp in importances] == [0, 0]
