"""Probabilistic CCA methods using MCMC via numpyro.

This module is only available when ``numpyro`` and ``jax`` are installed.
Import errors are deferred to usage time.
"""

from __future__ import annotations

import importlib.util

_numpyro_available = importlib.util.find_spec("numpyro") is not None
_jax_available = importlib.util.find_spec("jax") is not None

if _numpyro_available and _jax_available:
    from cca_zoo.probabilistic._pcca import ProbabilisticCCA

    __all__ = ["ProbabilisticCCA"]
else:
    __all__ = []
