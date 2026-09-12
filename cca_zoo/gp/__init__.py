"""Gaussian-process (GP) nonlinear CCA methods."""

from __future__ import annotations

from cca_zoo.gp._gpcca import GPCCA as GPCCA
from cca_zoo.gp._gpcca import GaussianProcessCCA

__all__ = ["GaussianProcessCCA"]
# Deprecated alias GPCCA stays importable for backward compatibility but is
# intentionally left out of __all__/docs.
