"""Helpers shared across test modules."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from cca_zoo._base import BaseModel
from cca_zoo.datasets import make_joint_data
from cca_zoo.metrics import average_pairwise_correlations, pairwise_correlations

_MODULES = [
    "cca_zoo.linear",
    "cca_zoo.nonparametric",
    "cca_zoo.tree",
    "cca_zoo.gam",
    "cca_zoo.gp",
    "cca_zoo.probabilistic",
    "cca_zoo.sparse",
    "cca_zoo.stochastic",
]

# Modules whose models need the slow optional backends.
SLOW_MODULES = {"cca_zoo.tree", "cca_zoo.probabilistic"}

# Constructor arguments that keep each fit small; other models use defaults.
_FAST: dict[str, dict[str, Any]] = {
    "ProbabilisticCCA": {"n_warmup": 10, "n_posterior_samples": 10},
    "VariationalBayesCCA": {"n_iter": 20, "n_posterior_samples": 10},
    "GFA": {"max_iter": 50, "n_posterior_samples": 10},
    "XGBoostCCA": {"n_estimators": 3},
    "LightGBMCCA": {"n_estimators": 3},
    "CatBoostCCA": {"n_estimators": 3},
    "ProjectionPursuitCCA": {"n_init": 1, "max_iter": 5},
    "StochasticCCAEY": {"max_iter": 5},
    "SAR": {"n_alphas": 10, "max_iter": 20},
    "MultiTaskElasticNetCCA": {"max_iter": 20},
    "ElasticNetCCA": {"max_iter": 20},
    "TrimmedCCA": {"n_init": 2},
}


def _discover() -> list[type[BaseModel]]:
    """Every public model; a module missing its optional backend exports none."""
    classes = []
    for name in _MODULES:
        module = importlib.import_module(name)
        for export in getattr(module, "__all__", []):
            obj = getattr(module, export)
            if isinstance(obj, type) and issubclass(obj, BaseModel):
                classes.append(obj)
    return classes


MODEL_CLASSES = _discover()


def make_model(cls: type[BaseModel]) -> BaseModel:
    """A quick-fitting, seeded instance of ``cls``."""
    model = cls(**_FAST.get(cls.__name__, {}))
    if "random_state" in model.get_params():
        model.set_params(random_state=0)
    return model


def canonical_correlations(model: object, views: list[np.ndarray]) -> np.ndarray:
    """Per-dimension mean pairwise canonical correlation of a fitted model."""
    return average_pairwise_correlations(pairwise_correlations(model.transform(views)))  # type: ignore[attr-defined]


def linear_views(
    seed: int = 0,
    n: int = 100,
    widths: tuple[int, ...] = (6, 5),
    n_components: int = 2,
    noise: float = 1.0,
    shift: float = 0.0,
) -> list[np.ndarray]:
    """Views of shared Gaussian factors, by the package's own simulator.

    One view per width; ``noise`` is each feature's noise standard deviation,
    and ``shift`` is added to every entry, for data that needs centring.
    """
    views = make_joint_data(
        n_samples=n,
        n_features=list(widths),
        n_views=len(widths),
        n_components=n_components,
        signal_to_noise=noise**-2,
        random_state=seed,
    )
    return [v + shift for v in views]


def ordered_views(
    seed: int = 0,
    n: int = 300,
    widths: tuple[int, ...] = (6, 5),
    scales: tuple[float, ...] = (3.0, 2.0, 1.0),
    noise: float = 1.0,
) -> list[np.ndarray]:
    """Views of shared factors of decreasing scale, so canonical correlations differ.

    ``linear_views``' factors are exchangeable, which leaves the order of the
    components to chance.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, len(scales))) * np.asarray(scales)
    return [
        z @ rng.standard_normal((len(scales), p)) + noise * rng.standard_normal((n, p))
        for p in widths
    ]


def fit(model: BaseModel, views: list[np.ndarray]) -> BaseModel:
    """Fit any model, giving PartialCCA the conditioning variable it needs."""
    if type(model).__name__ == "PartialCCA":
        partials = np.random.default_rng(1).standard_normal((len(views[0]), 1))
        return model.fit(views, partials=partials)
    return model.fit(views)


def slow_marks(cls: type[BaseModel]) -> list[pytest.MarkDecorator]:
    """``[pytest.mark.slow]`` for a model with a slow backend, else none."""
    return (
        [pytest.mark.slow] if cls.__module__.rsplit(".", 1)[0] in SLOW_MODULES else []
    )


def model_params(classes: list[type[BaseModel]]) -> list[Any]:
    """Parametrize over models, marking those with slow backends ``slow``."""
    return [pytest.param(c, id=c.__name__, marks=slow_marks(c)) for c in classes]


def assert_same_scores_as(
    scores: list[np.ndarray], reference: list[np.ndarray], atol: float = 1e-6
) -> None:
    """Each component is perfectly correlated with the reference's, up to sign."""
    for a, b in zip(scores, reference):
        for j in range(b.shape[1]):
            assert abs(np.corrcoef(a[:, j], b[:, j])[0, 1]) > 1 - atol


def principal_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between the column spans of a and b."""
    qa = np.linalg.qr(a - a.mean(axis=0))[0]
    qb = np.linalg.qr(b - b.mean(axis=0))[0]
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def assert_same_subspace(
    a: list[np.ndarray], b: list[np.ndarray], atol: float = 1e-3
) -> None:
    """Each view's scores in ``a`` span the same subspace as in ``b``."""
    for x, y in zip(a, b):
        np.testing.assert_allclose(principal_cosines(x, y), 1.0, atol=atol)
