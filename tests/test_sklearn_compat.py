"""Generic sklearn-estimator-contract checks for every BaseModel subclass.

``sklearn.utils.estimator_checks.check_estimator`` can't run wholesale here:
cca_zoo models fit on ``list[ArrayLike]`` (multiview), not a single 2-D
array, and are tagged accordingly (``no_validation=True``,
``input_tags.two_d_array=False`` — see ``BaseModel.__sklearn_tags__``), so
sklearn's own check-estimator machinery skips them entirely rather than
constructing invalid test data.

A handful of individual checks never touch ``fit``/``predict`` at all —
they only exercise the constructor, ``get_params``/``set_params``, and
``repr`` — and those work identically for a multiview estimator. Running
them here, parametrized over every model in the package, replaces
hand-written per-model get_params/set_params roundtrip tests with the
actual checks scikit-learn ships and maintains, while covering every
model rather than a hand-picked handful.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from sklearn.utils.estimator_checks import (
    check_estimator_repr,
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_set_params,
)

from cca_zoo._base import BaseModel

_MODULE_NAMES = [
    "cca_zoo.linear",
    "cca_zoo.nonparametric",
    "cca_zoo.tree",
    "cca_zoo.probabilistic",
]


def _discover_model_classes() -> list[type[BaseModel]]:
    """Collect every public BaseModel subclass exported by cca_zoo.

    Modules gated behind an optional dependency (``cca_zoo.tree``,
    ``cca_zoo.probabilistic``) expose an empty ``__all__`` rather than
    raising ImportError when that dependency is missing (see their
    ``__init__.py``), so they simply contribute nothing here in that case.
    """
    classes: list[type[BaseModel]] = []
    for module_name in _MODULE_NAMES:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", []):
            obj = getattr(module, name)
            if isinstance(obj, type) and issubclass(obj, BaseModel):
                classes.append(obj)
    return classes


_MODEL_CLASSES = _discover_model_classes()
_CHECKS = [
    check_no_attributes_set_in_init,
    check_get_params_invariance,
    check_set_params,
    check_estimator_repr,
]


@pytest.mark.parametrize(
    "ModelClass", _MODEL_CLASSES, ids=[c.__name__ for c in _MODEL_CLASSES]
)
@pytest.mark.parametrize("check", _CHECKS, ids=[c.__name__ for c in _CHECKS])
def test_sklearn_estimator_contract(ModelClass: type[BaseModel], check: Any) -> None:
    """Every model satisfies sklearn's constructor/get_params/repr contract."""
    check(ModelClass.__name__, ModelClass())


def test_discovered_at_least_the_core_modules() -> None:
    """Sanity check that discovery actually found the always-available models."""
    names = {c.__name__ for c in _MODEL_CLASSES}
    assert {"CCA", "rCCA", "MCCA", "GCCA", "TCCA", "CCAR3"} <= names
