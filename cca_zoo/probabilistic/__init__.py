"""Probabilistic (Bayesian) CCA methods.

``GFA`` is a closed-form coordinate-ascent variational algorithm with no
dependencies beyond numpy/scikit-learn, and is always available.
``ProbabilisticCCA`` and ``VariationalBayesCCA`` perform MCMC/black-box
variational inference via ``numpyro`` and are only available when
``numpyro`` and ``jax`` are installed; import errors for those two are
deferred to usage time.
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
