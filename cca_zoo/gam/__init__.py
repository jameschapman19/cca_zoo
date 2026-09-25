"""Spline-based nonlinear CCA methods: generalized additive models (GAM) and MARS."""

from __future__ import annotations

from cca_zoo.gam._gamcca import GAMCCA
from cca_zoo.gam._marscca import MARSCCA

__all__ = ["GAMCCA", "MARSCCA"]
