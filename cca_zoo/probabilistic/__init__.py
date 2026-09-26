"""Probabilistic CCA.

:class:`GFA` needs no extra dependencies; :class:`ProbabilisticCCA` and
:class:`VariationalBayesCCA` require the ``probabilistic`` extra.
"""

from __future__ import annotations

import importlib.util

from cca_zoo.probabilistic._gfa import GFA

_numpyro_available = importlib.util.find_spec("numpyro") is not None
_jax_available = importlib.util.find_spec("jax") is not None

if _numpyro_available and _jax_available:
    from cca_zoo.probabilistic._pcca import ProbabilisticCCA
    from cca_zoo.probabilistic._vbcca import VariationalBayesCCA

    __all__ = ["GFA", "ProbabilisticCCA", "VariationalBayesCCA"]
else:
    __all__ = ["GFA"]
