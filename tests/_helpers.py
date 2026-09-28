"""Helpers shared across test modules."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from cca_zoo._base import BaseModel
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


def linear_views(seed: int, n: int) -> list[np.ndarray]:
    """Two views of two shared Gaussian factors, with noise of mixed scale."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    return [
        z @ rng.standard_normal((2, 6)) + rng.standard_normal((n, 6)),
        z @ rng.standard_normal((2, 5)) + rng.standard_normal((n, 5)),
    ]


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
