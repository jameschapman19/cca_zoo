"""
Linear CCA methods
"""
from ._gcca import GCCA
from ._gradient import CCA_EY, CCA_GHA, CCA_SVD, CCA_TraceNorm, PLS_EY, PLSStochasticPower
from ._grcca import GRCCA
from ._iterative import (
    PLS_ALS,
    SCCA_IPLS,
    SPLS,
    # AltMaxVar,
    ElasticCCA,
    SCCA_Parkhomenko,
    SCCA_Span,
)
from ._mcca import CCA, MCCA, rCCA
from ._partialcca import PartialCCA
from ._pcacca import PCACCA
from ._pls import MPLS, PLS
from ._prcca import PRCCA
from ._tcca import TCCA

__all__ = [
    "MCCA",
    "CCA",
    "rCCA",
    "PLS",
    "MPLS",
    "GCCA",
    "GRCCA",
    "PartialCCA",
    "PRCCA",
    "TCCA",
    "PCACCA",
    "SCCA_IPLS",
    "ElasticCCA",
    "PLS_ALS",
    "SPLS",
    "SCCA_Parkhomenko",
    "SCCA_Span",
    "CCA_EY",
    "PLS_EY",
    "CCA_GHA",
    "CCA_SVD",
    "CCA_TraceNorm",
    "PLSStochasticPower",
]
