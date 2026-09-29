"""Models behind an optional extra name it when the extra is missing."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module", "name", "extra"),
    [
        ("cca_zoo.tree", "XGBoostCCA", "tree"),
        ("cca_zoo.probabilistic", "ProbabilisticCCA", "probabilistic"),
        ("cca_zoo.deep", "DCCA", "deep"),
        ("cca_zoo.model_selection", "OptunaSearchCV", "optuna"),
    ],
)
def test_a_missing_extra_is_named(module: str, name: str, extra: str) -> None:
    """Importing a model without its extra says which extra to install.

    Decided by the import itself: optuna_integration's lazy module has no
    ``__spec__``, so ``find_spec`` fails once anything has imported it.
    """
    try:
        getattr(importlib.import_module(module), name)
    except ImportError as error:
        assert f"cca-zoo[{extra}]" in str(error)
    else:
        pytest.skip(f"the {extra} extra is installed")
