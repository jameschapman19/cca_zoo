"""Tree-based nonlinear CCA methods.

This module is only available when ``xgboost`` is installed. Import errors
are deferred to usage time rather than raised at import of ``cca_zoo``.
"""

from __future__ import annotations

import importlib.util

_xgboost_available = importlib.util.find_spec("xgboost") is not None

if _xgboost_available:
    from cca_zoo.tree._treecca import LightGBMCCA, XGBoostCCA
    from cca_zoo.tree._treecca import TreeCCA as TreeCCA

    __all__ = ["XGBoostCCA", "LightGBMCCA"]
else:
    __all__ = []
# TreeCCA (the abstract base shared by XGBoostCCA/LightGBMCCA) stays importable
# for type hints and subclassing but is intentionally left out of __all__/docs,
# since it cannot be instantiated directly.
