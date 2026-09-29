"""Gradient-boosted-tree CCA; requires the ``tree`` extra."""

from __future__ import annotations

import importlib.util

# TreeCCA (the abstract base shared by XGBoostCCA/LightGBMCCA/CatBoostCCA) stays
# importable for type hints and subclassing but is intentionally left out of
# __all__/docs, since it cannot be instantiated directly.
_xgboost_available = importlib.util.find_spec("xgboost") is not None

if _xgboost_available:
    from cca_zoo.tree._treecca import CatBoostCCA, LightGBMCCA, XGBoostCCA
    from cca_zoo.tree._treecca import TreeCCA as TreeCCA

    __all__ = ["CatBoostCCA", "LightGBMCCA", "XGBoostCCA"]
else:
    __all__ = []

    def __getattr__(name: str) -> object:
        if name in {"CatBoostCCA", "LightGBMCCA", "XGBoostCCA", "TreeCCA"}:
            raise ImportError(
                f"{name} requires the tree extra: pip install 'cca-zoo[tree]'."
            )
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
