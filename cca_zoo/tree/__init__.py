"""Tree-based nonlinear CCA methods.

This module is only available when ``xgboost`` is installed. Import errors
are deferred to usage time rather than raised at import of ``cca_zoo``.
"""

from __future__ import annotations

import importlib.util

_xgboost_available = importlib.util.find_spec("xgboost") is not None

if _xgboost_available:
    from cca_zoo.tree._treecca import TreeCCA

    __all__ = ["TreeCCA"]
else:
    __all__ = []
