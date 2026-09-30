"""Every hyperparameter changes the fit.

A parameter that is accepted, validated and documented but has no effect is a
bug that no property of the output reveals: GRCCA's ``mu=0`` was silently
``mu=1``. Each model is fitted at a base setting and again with one
hyperparameter moved, and the scores must differ. Parameters that only steer
the solver, and those that take effect only alongside another setting, are
named with the reason.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import RBF

from cca_zoo._base import BaseModel
from tests._helpers import MODEL_CLASSES, SLOW_MODULES, linear_views

# Parameters that change how a solution is reached, not which: at convergence
# the fit does not depend on them.
_SOLVER = {
    "max_iter",
    "tol",
    "n_iter",
    "n_warmup",
    "n_posterior_samples",
    "admm_iter",
    "max_trials",
    "stop_probability",
    "n_iter_no_change",
    "n_init",
    "n_alphas",
    "random_state",
    "init",
}
_MODEL_SOLVER = {
    ("MCCA", "pca"): "solving in the principal-component basis is exact",
    ("ADMMCCA", "rho"): "ADMM's step size, not its problem",
    ("StochasticCCAEY", "learning_rate"): "the step size, not the problem",
    ("VariationalBayesCCA", "learning_rate"): "the step size, not the problem",
}

# The model under test is fitted with these, so each parameter has a setting
# in which it acts.
_BASE: dict[str, dict[str, Any]] = {
    "GRCCA": {"shrinkage": 0.5, "feature_groups": [np.arange(6) % 2, np.arange(5) % 2]},
    "IPLSCCA": {"alpha": 0.05},
    "ProjectionPursuitCCA": {"n_init": 1, "max_iter": 20},
    "KCCA": {"kernel": "poly"},
    "KGCCA": {"kernel": "poly"},
    "KTCCA": {"kernel": "poly"},
    "GFA": {"n_components": 3, "max_iter": 2000},
    "ProbabilisticCCA": {"n_warmup": 50, "n_posterior_samples": 50},
    "VariationalBayesCCA": {"n_iter": 300},
    "CatBoostCCA": {"n_estimators": 20},
    "LightGBMCCA": {"n_estimators": 20},
    "XGBoostCCA": {"n_estimators": 20},
}

# A parameter that acts only alongside another: the setting it needs.
_WITH: dict[tuple[str, str], dict[str, Any]] = {
    ("ManifoldCCA", "reg"): {"method": "lle"},
    ("ManifoldCCA", "gamma"): {"affinity": "rbf"},
}

# The second value of each parameter, by name, then by model.
_ALTERNATIVE: dict[str, Any] = {
    "shrinkage": 0.5,
    "mu": 1.0,
    "ledoit_wolf": "flip",
    "min_samples": 0.6,
    "residual_threshold": 0.5,
    "support_fraction": 0.55,
    "reg": 0.1,
    "delta": 1.0,
    "coef0": 3.0,
    "degree": 2,
    "gamma": 0.3,
    "kernel": "rbf",
    "view_weights": [1.0, 4.0],
    "affinity": "rbf",
    "method": "lle",
    "n_neighbors": 5,
    "n_operator_components": 4,
    "colsample_bytree": 0.5,
    "gauss_seidel": "flip",
    "learning_rate": 0.4,
    "max_depth": 2,
    "min_child_weight": 5,
    "n_estimators": 5,
    "subsample": 0.5,
    "k": 5,
    "m": 3,
    "sp": 10.0,
    "endspan": 3,
    "minspan": 3,
    "nk": 3,
    "nprune": 2,
    "thresh": 0.1,
    "n_inducing": 20,
    "drop_k": "flip",
    "l1_ratio": 0.2,
    "l1_bound": 0.5,
    "n_nonzero_coefs": 2,
    "span": 2,
    "batch_size": 16,
    "positive": "flip",
}
_MODEL_ALTERNATIVE: dict[tuple[str, str], Any] = {
    ("GRCCA", "shrinkage"): 0.9,
    ("GaussianProcessCCA", "kernel"): RBF(0.3),
    ("MARSCCA", "degree"): 2,
    ("GraphicalLassoCCA", "alpha"): 0.5,
    ("MARSCCA", "alpha"): 1.0,
    ("GaussianProcessCCA", "alpha"): 1.0,
    ("ElasticNetCCA", "alpha"): 0.05,
    ("MultiTaskElasticNetCCA", "alpha"): 0.05,
    ("IPLSCCA", "alpha"): 0.2,
    ("ADMMCCA", "alpha"): 0.5,
    ("ParkhomenkoCCA", "alpha"): 0.5,
    ("CCAR3", "alpha"): 0.1,
    ("ECCA", "alpha"): 0.1,
    ("KCCA", "shrinkage"): 0.6,
    ("KGCCA", "shrinkage"): 0.6,
    ("KTCCA", "shrinkage"): 0.6,
    ("RANSACCCA", "shrinkage"): 0.6,
    ("TrimmedCCA", "shrinkage"): 0.6,
}

# Not scalar settings: covered by the model's own tests.
_STRUCTURAL = {"n_components", "center", "feature_groups", "kernel_params"}


def _cases() -> list[Any]:
    cases = []
    for cls in MODEL_CLASSES:
        name = cls.__name__
        slow = cls.__module__.rsplit(".", 1)[0] in SLOW_MODULES
        for param in cls().get_params():
            if (
                param in _SOLVER
                or param in _STRUCTURAL
                or (name, param) in _MODEL_SOLVER
            ):
                continue
            marks = [pytest.mark.slow] if slow else []
            cases.append(pytest.param(cls, param, id=f"{name}-{param}", marks=marks))
    return cases


def _scores(model: BaseModel, views: list[np.ndarray]) -> list[np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if type(model).__name__ == "PartialCCA":
            partials = np.random.default_rng(1).standard_normal((len(views[0]), 1))
            model.fit(views, partials=partials)
        else:
            model.fit(views)
    return [np.abs(np.asarray(s)) for s in model.transform(views)]


@pytest.mark.parametrize(("cls", "param"), _cases())
def test_the_parameter_changes_the_fit(cls: type[BaseModel], param: str) -> None:
    """Moving the parameter from its base value changes the scores."""
    name = cls.__name__
    views = linear_views(0, 150)
    base = {**_BASE.get(name, {}), **_WITH.get((name, param), {})}
    if "random_state" in cls().get_params():
        base["random_state"] = 0
    model = cls(**base)
    alternative = _MODEL_ALTERNATIVE.get((name, param), _ALTERNATIVE.get(param))
    assert alternative is not None, f"no second value for {name}.{param}"
    if alternative == "flip":
        alternative = not model.get_params()[param]
    moved = cls(**{**base, param: alternative})
    for a, b in zip(_scores(model, views), _scores(moved, views)):
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-6):
            return
    pytest.fail(f"{name}.{param}={alternative!r} fits as {model.get_params()[param]!r}")
