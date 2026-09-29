"""Models behind an optional extra name it when the extra is missing."""

from __future__ import annotations

import importlib
import importlib.util

import pytest


@pytest.mark.parametrize(
    ("module", "name", "package", "extra"),
    [
        ("cca_zoo.tree", "XGBoostCCA", "xgboost", "tree"),
        ("cca_zoo.probabilistic", "ProbabilisticCCA", "numpyro", "probabilistic"),
        ("cca_zoo.deep", "DCCA", "lightning", "deep"),
        ("cca_zoo.model_selection", "OptunaSearchCV", "optuna_integration", "optuna"),
    ],
)
def test_a_missing_extra_is_named(
    module: str, name: str, package: str, extra: str
) -> None:
    """Importing a model without its extra says which extra to install."""
    if importlib.util.find_spec(package) is not None:
        pytest.skip(f"{package} is installed")
    with pytest.raises(ImportError, match=rf"cca-zoo\[{extra}\]"):
        getattr(importlib.import_module(module), name)
