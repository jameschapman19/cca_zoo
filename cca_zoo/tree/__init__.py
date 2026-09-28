"""Gradient-boosted-tree CCA; requires the ``tree`` extra."""

from __future__ import annotations

import importlib.util

_xgboost_available = importlib.util.find_spec("xgboost") is not None

if _xgboost_available:
    from cca_zoo.tree._treecca import CatBoostCCA, LightGBMCCA, XGBoostCCA
    from cca_zoo.tree._treecca import TreeCCA as TreeCCA

    __all__ = ["CatBoostCCA", "LightGBMCCA", "XGBoostCCA"]
else:
    __all__ = []
# TreeCCA (the abstract base shared by XGBoostCCA/LightGBMCCA/CatBoostCCA) stays
# importable for type hints and subclassing but is intentionally left out of
# __all__/docs, since it cannot be instantiated directly.
