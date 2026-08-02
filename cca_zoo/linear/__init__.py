"""Linear CCA methods.

This module provides classical linear multiview CCA algorithms ranging from
the standard two-view CCA and PLS to multiset and generalised variants, as
well as sparse/regularised iterative methods and gradient-descent methods
suited to high-dimensional or streaming data.
"""

from ._cca import CCA
from ._gcca import GCCA
from ._iterative import (
    PLS_ALS,
    SCCA_ADMM,
    SCCA_IPLS,
    SCCA_PMD,
    ElasticCCA,
    ParkhomenkoCCA,
    SCCA_Span,
)
from ._mcca import MCCA
from ._pls import PLS
from ._rcca import rCCA
from ._tcca import TCCA
from .gradient import CCA_EY, MCCA_EY, PLS_EY

__all__ = [
    # Exact eigendecomposition
    "CCA",
    "rCCA",
    "PLS",
    "MCCA",
    "GCCA",
    "TCCA",
    # Gradient descent (high-dimensional / streaming)
    "PLS_EY",
    "CCA_EY",
    "MCCA_EY",
    # Sparse / regularised ALS
    "SCCA_PMD",
    "SCCA_ADMM",
    "SCCA_IPLS",
    "SCCA_Span",
    "ElasticCCA",
    "ParkhomenkoCCA",
    "PLS_ALS",
]
